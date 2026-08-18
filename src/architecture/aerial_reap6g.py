"""
Aerial-REAP-6G: Kinematic-RF SNN Architecture for 6G UAV Swarm Beamswitching
==============================================================================

Implements the forward-pass architecture described in the technical blueprint:

  2. Input Layer      -> Event-driven sensor fusion (IMU delta + RSS delta)
                          with adaptive gain control
  3. Membrane Dynamics -> Leaky Integrate-and-Fire (LIF)
  4. Output Layer      -> First-to-spike Winner-Take-All (WTA) with lateral
                          inhibition, mapped to relative beam-steering actions

R-STDP training (Section 5) and the flight/RF simulator are OUT OF SCOPE for
this module by design -- this is the architecture only, for offline
validation with synthetic sequences.

Framework: snnTorch (torch backend)
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import snntorch as snn


# ---------------------------------------------------------------------------
# Beam action codebook (Section 4: Relative Spatial Mapping)
# ---------------------------------------------------------------------------
BEAM_ACTIONS = [
    "hold",
    "shift_up",
    "shift_down",
    "shift_left",
    "shift_right",
    "widen_beam",
]
NUM_ACTIONS = len(BEAM_ACTIONS)


@dataclass
class ArchitectureConfig:
    quat_dim: int = 4          
    rss_dim: int = 1           
    hidden_dim: int = 32       
    num_actions: int = NUM_ACTIONS

    # Lowered from 0.9 to 0.5. Forces the membrane to bleed off noise rapidly.
    beta: float = 0.5          
    v_th: float = 0.05          
    reset_mechanism: str = "zero"  

    gain_eps: float = 1e-3     
    gain_window: int = 10      
    gain_clip: float = 50.0    
    noise_floor_var: float = 0.01  
    
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Step 1: Adaptive Gain Controller
#   G_t \propto 1 / Variance(ΔRSS)
# ---------------------------------------------------------------------------
class AdaptiveGainController(nn.Module):
    """
    Computes G_t from a rolling window of past ΔRSS values.

    Implementation notes:
    - Variance is computed over a trailing window (gain_window) rather than the
      full history, so the gain tracks *local* channel volatility (per the
      blueprint's intent: "scales inversely with channel volatility").
    - noise_floor_var replaces a bare numerical epsilon: it represents the
      RF sensor's baseline ΔRSS variance during genuinely calm flight (i.e.
      measurement noise, not a maneuver/fade event). Flooring the variance
      at this calibrated level means G_t only rises when volatility exceeds
      what's attributable to normal noise -- without it, 1/Variance(ΔRSS)
      is scale-sensitive and pins near the gain ceiling even in calm
      segments, since variance of small deltas is quadratically tiny.
    - gain_clip still bounds worst-case current spikes as a hard safety rail,
      but with the noise floor in place it should rarely be the binding
      constraint during calm flight.
    """

    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        self.window = cfg.gain_window
        self.noise_floor_var = cfg.noise_floor_var
        self.clip = cfg.gain_clip

    def forward(self, delta_rss_seq: torch.Tensor) -> torch.Tensor:
        """
        delta_rss_seq: (T, B) tensor of ΔRSS values across the full sequence
                        so far (or a pre-windowed buffer).
        returns:       (T, B) tensor of G_t, one gain value per timestep.
        """
        T, B = delta_rss_seq.shape
        gains = torch.empty(T, B, device=delta_rss_seq.device)

        for t in range(T):
            lo = max(0, t - self.window + 1)
            window = delta_rss_seq[lo : t + 1]  # (w, B)
            if window.shape[0] < 2:
                # Not enough samples yet for a meaningful variance estimate;
                # fall back to the noise floor itself -> gain of 1.0.
                var = torch.full((B,), self.noise_floor_var, device=delta_rss_seq.device)
            else:
                var = window.var(dim=0, unbiased=False)
            floored_var = torch.clamp(var, min=self.noise_floor_var)
            gain = self.noise_floor_var / floored_var
            gains[t] = torch.clamp(gain, max=self.clip)

        return gains


# ---------------------------------------------------------------------------
# Step 2: Fused LIF Current Layer
#   I_t = G_t * (W_kin Δq_t + W_rf ΔRSS_t) + b
# ---------------------------------------------------------------------------
class FusedCurrentLayer(nn.Module):
    """
    Learned linear fusion of kinematic and RF deltas, projected up to
    hidden_dim to drive the LIF population.

    DEVIATION FROM THE ORIGINAL BLUEPRINT EQUATION (documented deliberately):
    The original spec applies G_t to BOTH terms:
        I_t = G_t * (W_kin*Δq_t + W_rf*ΔRSS_t) + b
    Empirically, this is self-defeating: G_t ∝ 1/Variance(ΔRSS) collapses
    toward zero exactly when ΔRSS is volatile -- which is precisely the
    onset of a fade event, i.e. the RF signal the RF pathway exists to
    detect. Applying G_t to the RF term suppresses the fade signal at the
    moment it matters most.

    The gain controller's STATED purpose (blueprint Sec. 2) is "to prevent
    neuron saturation during violent maneuvers" -- a kinematic concern.
    We therefore apply G_t only to the kinematic term as a saturation guard,
    and let the RF term pass through at unit gain so fade events aren't
    self-suppressed:

        I_t = (G_t * W_kin*Δq_t) + (W_rf*ΔRSS_t) + b

    W_kin and W_rf remain separate weight matrices so each pathway's
    contribution can still be inspected/ablated independently.
    """

    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        self.W_kin = nn.Linear(cfg.quat_dim, cfg.hidden_dim, bias=False)
        self.W_rf = nn.Linear(cfg.rss_dim, cfg.hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.hidden_dim))

    def forward(
        self,
        delta_q: torch.Tensor,     # (T, B, quat_dim)
        delta_rss: torch.Tensor,   # (T, B, rss_dim)
        gain: torch.Tensor,        # (T, B) -- applied to kinematic term only
    ) -> torch.Tensor:
        kin_term = self.W_kin(delta_q)              # (T, B, hidden_dim)
        rf_term = self.W_rf(delta_rss)               # (T, B, hidden_dim)
        gated_kin = gain.unsqueeze(-1) * kin_term    # saturation guard on IMU path only
        return gated_kin + rf_term + self.bias


# ---------------------------------------------------------------------------
# Step 3: LIF Membrane (snnTorch)
#   U_t = β U_{t-1} + I_t
# ---------------------------------------------------------------------------
class LIFMembrane(nn.Module):
    """
    Thin wrapper around snntorch.Leaky mapping hidden_dim currents down to
    num_actions membrane potentials -- one membrane per candidate beam action.
    """

    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_actions)
        self.lif = snn.Leaky(
            beta=cfg.beta,
            threshold=cfg.v_th,
            reset_mechanism=cfg.reset_mechanism,
        )

    def forward(self, i_t: torch.Tensor, mem: torch.Tensor):
        """
        i_t: (B, hidden_dim) current at this single timestep
        mem: (B, num_actions) previous membrane potential
        returns: spk (B, num_actions), mem (B, num_actions)
        """
        cur = self.fc(i_t)               # (B, num_actions)
        spk, mem = self.lif(cur, mem)    # snnTorch handles U_t = beta*U_{t-1} + I_t
        return spk, mem


# ---------------------------------------------------------------------------
# Step 4: First-to-Spike WTA Output Layer
# ---------------------------------------------------------------------------
class FirstToSpikeWTA(nn.Module):
    """
    Enforces the blueprint's Section 4 semantics explicitly:
      - The FIRST neuron across V_th in the whole sequence fires.
      - Lateral inhibition suppresses all other neurons for the rest of the
        sequence (one decision per inference window -- this models the
        "1-bit hardware interrupt" behavior, not a per-timestep softmax).

    This is intentionally NOT just "take the first spk==1 in the raw spike
    train" -- snnTorch's Leaky layer can produce multiple simultaneous
    spikes at a single timestep across the num_actions dimension if several
    membranes cross threshold together. In that tie case we break the tie
    by raw membrane potential (highest U_t wins), since that's the neuron
    closest to a "confident" decision.
    """

    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        self.num_actions = cfg.num_actions

    def forward(self, spk_seq: torch.Tensor, mem_seq: torch.Tensor):
        """
        spk_seq: (T, B, num_actions) binary spike train over the whole window
        mem_seq: (T, B, num_actions) membrane potentials over the whole window
        returns:
          winner_idx:  (B,) action index of the winning neuron, per batch item
                        (-1 if no neuron ever spiked in the window)
          winner_time: (B,) timestep at which the winning spike occurred
                        (-1 if no spike occurred)
        """
        T, B, N = spk_seq.shape
        winner_idx = torch.full((B,), -1, dtype=torch.long, device=spk_seq.device)
        winner_time = torch.full((B,), -1, dtype=torch.long, device=spk_seq.device)

        for b in range(B):
            fired = False
            for t in range(T):
                active = torch.nonzero(spk_seq[t, b], as_tuple=False).squeeze(-1)
                if active.numel() > 0:
                    if active.numel() == 1:
                        winner_idx[b] = active.item()
                    else:
                        # tie-break: highest membrane potential among co-firing neurons
                        winner_idx[b] = active[torch.argmax(mem_seq[t, b, active])].item()
                    winner_time[b] = t
                    fired = True
                    break  # lateral inhibition: stop scanning, decision is locked
            # if never fired, winner stays -1 -- caller should decide the
            # "hold" fallback policy (e.g. default to BEAM_ACTIONS[0])

        return winner_idx, winner_time


# ---------------------------------------------------------------------------
# Full architecture: wires steps 1-4 together for a forward pass
# ---------------------------------------------------------------------------
class AerialREAP6G(nn.Module):
    def __init__(self, cfg: Optional[ArchitectureConfig] = None):
        super().__init__()
        self.cfg = cfg or ArchitectureConfig()
        self.gain_ctrl = AdaptiveGainController(self.cfg)
        self.current_layer = FusedCurrentLayer(self.cfg)
        self.membrane = LIFMembrane(self.cfg)
        self.wta = FirstToSpikeWTA(self.cfg)

    def forward(self, delta_q: torch.Tensor, delta_rss: torch.Tensor):
        """
        delta_q:   (T, B, 4)  quaternion deltas
        delta_rss: (T, B, 1)  RSS deltas

        returns dict with intermediate tensors for inspection/plotting:
          gain, current, spk_seq, mem_seq, winner_idx, winner_time, winner_action
        """
        T, B, _ = delta_q.shape
        device = delta_q.device

        # Step 1: adaptive gain from ΔRSS history
        gain = self.gain_ctrl(delta_rss.squeeze(-1))  # (T, B)

        # Step 2: fused current I_t
        current = self.current_layer(delta_q, delta_rss, gain)  # (T, B, hidden_dim)

        # Step 3: run the LIF membrane timestep-by-timestep
        mem = torch.zeros(B, self.cfg.num_actions, device=device)
        spk_seq, mem_seq = [], []
        for t in range(T):
            spk, mem = self.membrane(current[t], mem)
            spk_seq.append(spk)
            mem_seq.append(mem)
        spk_seq = torch.stack(spk_seq)  # (T, B, num_actions)
        mem_seq = torch.stack(mem_seq)  # (T, B, num_actions)

        # Step 4: first-to-spike WTA decision
        winner_idx, winner_time = self.wta(spk_seq, mem_seq)
        winner_action = [
            BEAM_ACTIONS[i] if i >= 0 else "hold(no-spike)" for i in winner_idx.tolist()
        ]

        return {
            "gain": gain,
            "current": current,
            "spk_seq": spk_seq,
            "mem_seq": mem_seq,
            "winner_idx": winner_idx,
            "winner_time": winner_time,
            "winner_action": winner_action,
        }