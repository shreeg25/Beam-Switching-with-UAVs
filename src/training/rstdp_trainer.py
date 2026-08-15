"""
Stage 5: Reward-Modulated STDP (R-STDP) Training
=====================================================

Implements blueprint Section 5's online, biologically-inspired learning
rule -- explicitly NOT backpropagation-through-time. Weight updates are
computed directly from spike timing and a scalar reward, with no autograd
graph retained across timesteps.

REWARD FUNCTION (blueprint Section 5)
----------------------------------------
    R = alpha*(delta_SE) + beta*(delta_RSS_post) - gamma*(Omega)

We don't have a live closed-loop simulator where the network's action
actually changes the drone's future trajectory/RSS (that would require
Phase 1-style co-simulation, explicitly out of scope). Instead, reward is
computed by comparing the network's CHOSEN action against the ORACLE's
recommended action at that timestep, using the same channel-model geometry
both were computed from:
  - delta_SE:      Shannon-capacity gain if the network's action is applied
                    instead of holding, using the antenna-gain-implied SNR.
  - delta_RSS_post: proxy for "did this action move toward the oracle's
                    action" -- positive if the network's pick matches or is
                    adjacent to the oracle's pick in the codebook, negative
                    otherwise.
  - Omega:          1 if the network immediately reverses its own previous
                    action (ping-pong), else 0 -- tracked from the
                    network's own action HISTORY, independent of the oracle.

This is a supervised-flavored approximation of the blueprint's live-reward
formulation, documented as such rather than presented as a full closed-loop
result. It's a reasonable substitute given no live flight-in-the-loop
simulator exists yet, and it still exercises the actual R-STDP update rule
on real spike timing.

STDP RULE
---------
Uses eligibility traces (standard technique for reward-MODULATED STDP,
since the reward isn't known until after the relevant spikes have already
occurred): a pre-synaptic trace decays exponentially and is captured into
an eligibility trace at each post-synaptic spike; the eligibility trace
itself decays, and is periodically consolidated into actual weight change
scaled by the reward. This is what makes it call R-STDP rather than plain
Hebbian STDP -- reward arrives asynchronously from the spike-timing
information.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.architecture.aerial_reap6g import AerialREAP6G, ArchitectureConfig, BEAM_ACTIONS
from src.dataset.channel_model import ChannelParams


@dataclass
class RSTDPConfig:
    alpha: float = 1.0      
    beta: float = 0.5       
    # Lowered from 2.0 to 1.0. Allows the network room to make mistakes early on.
    gamma: float = 1.0      

    tau_trace: float = 20.0     
    trace_decay: float = 0.9    
    lr: float = 0.01            
    weight_clip: float = 5.0    

    class_weight_hold: float = 1.0       
    # Increased from 8.0 to 50.0. A correct shift is now a massive, network-altering reward.
    class_weight_shift: float = 50.0      

    exploration_eps_start: float = 0.5   
    exploration_eps_end: float = 0.05    
    # Increased from 2000 to 25000. Exploration now spans the entirety of Epoch 1.
    exploration_decay_steps: int = 25000


class RSTDPTrainer:
    """
    Wraps an AerialREAP6G model and applies reward-modulated STDP updates
    directly to its learnable weight tensors, given a batch of
    (delta_q, delta_rss, oracle_action_idx) windows.

    Only three weight tensors are updated: W_kin, W_rf (FusedCurrentLayer)
    and fc.weight (LIFMembrane's readout). These are the tensors the
    blueprint's Delta_w = R * sum(STDP(Delta_t)) rule applies to -- the
    gain controller and WTA layer have no learnable parameters by design.
    """

    def __init__(self, model: AerialREAP6G, cfg: Optional[RSTDPConfig] = None):
        self.model = model
        self.cfg = cfg or RSTDPConfig()
        self.action_history = []  # network's own past actions, for Omega
        self._n_updates = 0  # drives exploration epsilon decay

    @torch.no_grad()
    def predict(self, delta_q_np: np.ndarray, delta_rss_np: np.ndarray) -> int:
        """
        Pure inference, NO weight updates and NO exploration override.
        Used for evaluation, where we want to measure what the network
        actually learned, not exploration-assisted behavior. Returns the
        action index (defaults to 'hold' if the network never fires).
        """
        delta_q = torch.from_numpy(delta_q_np).float().unsqueeze(1)
        delta_rss = torch.from_numpy(delta_rss_np).float().unsqueeze(1)
        out = self.model(delta_q, delta_rss)
        winner_idx = out["winner_idx"][0].item()
        return winner_idx if winner_idx >= 0 else BEAM_ACTIONS.index("hold")

    def _current_exploration_eps(self) -> float:
        cfg = self.cfg
        frac = min(1.0, self._n_updates / cfg.exploration_decay_steps)
        return cfg.exploration_eps_start + frac * (cfg.exploration_eps_end - cfg.exploration_eps_start)

    # -- reward computation ------------------------------------------------
    def _compute_reward(
        self, chosen_action_idx: int, oracle_action_idx: int, delta_rss_window: np.ndarray
    ) -> float:
        cfg = self.cfg

        # delta_SE proxy: Shannon capacity is monotonic in SNR/RSS, so a
        # correct (oracle-matching) action gets positive delta_SE credit,
        # scaled by how much RSS was actually fluctuating in this window
        # (a correction matters more when the channel was genuinely volatile).
        rss_volatility = float(np.std(delta_rss_window))
        correct = (chosen_action_idx == oracle_action_idx)
        delta_se = rss_volatility if correct else -rss_volatility

        # delta_RSS_post proxy: reward being in the right NEIGHBORHOOD of the
        # oracle action even if not exact (adjacent shifts in BEAM_ACTIONS
        # ordering are more forgivable than a wrong-direction shift).
        idx_distance = abs(chosen_action_idx - oracle_action_idx)
        delta_rss_post = 1.0 / (1.0 + idx_distance)

        # Omega: ping-pong penalty from the network's OWN action history.
        omega = 0.0
        if len(self.action_history) >= 2:
            if chosen_action_idx == self.action_history[-2] and chosen_action_idx != self.action_history[-1]:
                omega = 1.0

        reward = cfg.alpha * delta_se + cfg.beta * delta_rss_post - cfg.gamma * omega

        # Class-imbalance compensation: scale reward magnitude up when the
        # oracle action is a real shift, not hold -- otherwise R-STDP mostly
        # reinforces "always hold" since that's correct ~99.5% of the time
        # by raw frequency (see RSTDPConfig.class_weight_shift docstring).
        is_shift = oracle_action_idx != BEAM_ACTIONS.index("hold")
        reward *= cfg.class_weight_shift if is_shift else cfg.class_weight_hold

        return reward

    # -- STDP weight update --------------------------------------------------
    def _stdp_update(
        self,
        delta_q: torch.Tensor,      # (T, 4)
        delta_rss: torch.Tensor,    # (T, 1)
        spk_seq: torch.Tensor,      # (T, num_actions)
        reward: float,
    ):
        """
        Eligibility-trace R-STDP update for W_kin, W_rf, and fc.weight.
        Runs with torch.no_grad() throughout -- this is NOT backprop; no
        gradient graph is built or used.
        """
        cfg = self.cfg
        T = delta_q.shape[0]

        with torch.no_grad():
            # Pre-synaptic traces for the two input pathways, and for the
            # hidden-layer activity feeding the readout fc layer.
            trace_kin = torch.zeros(delta_q.shape[1])
            trace_rf = torch.zeros(delta_rss.shape[1])
            trace_hidden = torch.zeros(self.model.cfg.hidden_dim)

            elig_fc = torch.zeros_like(self.model.membrane.fc.weight)

            for t in range(T):
                # decay and accumulate pre-synaptic traces
                trace_kin = cfg.trace_decay * trace_kin + delta_q[t]
                trace_rf = cfg.trace_decay * trace_rf + delta_rss[t]

                # hidden activity: recompute the same fused-current forward
                # pass this timestep would have produced (cheap, no grad),
                # needed as the "pre-synaptic" signal for the fc layer.
                gain_t = torch.ones(1)  # gain is a scalar multiplier already applied
                                          # upstream in the real forward pass; for the
                                          # trace we use unit gain as a simplification,
                                          # since STDP eligibility only needs RELATIVE
                                          # pre-synaptic activity, not the exact current.
                hidden_t = (
                    gain_t * self.model.current_layer.W_kin(delta_q[t])
                    + self.model.current_layer.W_rf(delta_rss[t])
                )
                trace_hidden = cfg.trace_decay * trace_hidden + hidden_t

                spikes_t = spk_seq[t]  # (num_actions,)
                if spikes_t.sum() > 0:
                    # Post-synaptic spike occurred: capture current hidden-
                    # layer trace into the fc eligibility tensor via outer
                    # product (this is what "STDP(Delta_t)" reduces to in an
                    # eligibility-trace formulation -- correlate recent
                    # pre-synaptic activity with this post-synaptic spike).
                    elig_fc += torch.outer(spikes_t, trace_hidden)

            # Apply reward-modulated update. W_kin/W_rf feed the shared hidden
            # layer, not per-action neurons directly, so their eligibility is
            # accumulated from the SAME hidden-trace correlation used for fc,
            # projected back through fc's connectivity (a standard credit-
            # assignment approximation for a 2-layer spiking readout without
            # backprop).
            hidden_credit = elig_fc.sum(dim=0)  # (hidden_dim,) how much each
                                                  # hidden unit contributed to firing
            elig_kin = torch.outer(hidden_credit, trace_kin)
            elig_rf = torch.outer(hidden_credit, trace_rf)

            self.model.current_layer.W_kin.weight += cfg.lr * reward * elig_kin
            self.model.current_layer.W_rf.weight += cfg.lr * reward * elig_rf
            self.model.membrane.fc.weight += cfg.lr * reward * elig_fc

            # Hard weight clipping -- STDP has no built-in normalization, so
            # repeated same-sign rewards can otherwise grow weights unboundedly.
            self.model.current_layer.W_kin.weight.clamp_(-cfg.weight_clip, cfg.weight_clip)
            self.model.current_layer.W_rf.weight.clamp_(-cfg.weight_clip, cfg.weight_clip)
            self.model.membrane.fc.weight.clamp_(-cfg.weight_clip, cfg.weight_clip)

    # -- one training step on a single window --------------------------------
    def train_on_window(self, delta_q_np: np.ndarray, delta_rss_np: np.ndarray, oracle_actions_np: np.ndarray) -> dict:
        """
        delta_q_np:       (T, 4)
        delta_rss_np:     (T, 1)
        oracle_actions_np: (T,) int, oracle action index at each timestep

        Runs one forward pass, determines the network's chosen action for
        this window, computes reward against the window's FINAL oracle
        label (the decision that mattered most, since WTA only fires once
        per window), and applies the STDP update.

        EXPLORATION: with probability exploration_eps (decaying over
        training), the STDP update credits the ORACLE's action-neuron
        instead of whichever neuron the network naturally fired first. This
        is necessary because STDP only builds an eligibility trace for
        neurons that actually spike -- a correct-but-never-firing neuron has
        no mechanism to ever get reinforced otherwise (found during
        validation: an untrained net could get permanently stuck on one
        wrong action with zero path to learning the right one). The
        exploration spike uses a REAL post-synaptic timestep (the window's
        natural winner_time, or the last timestep if nothing fired), so the
        credited eligibility trace is still built from real activity in
        this window, not synthetic data.
        """
        delta_q = torch.from_numpy(delta_q_np).float().unsqueeze(1)     # (T, 1, 4)
        delta_rss = torch.from_numpy(delta_rss_np).float().unsqueeze(1)  # (T, 1, 1)

        out = self.model(delta_q, delta_rss)
        winner_idx = out["winner_idx"][0].item()
        winner_time = out["winner_time"][0].item()
        natural_action_idx = winner_idx if winner_idx >= 0 else BEAM_ACTIONS.index("hold")

        oracle_action_idx = int(oracle_actions_np[-1])  # final timestep's label

        eps = self._current_exploration_eps()
        explored = np.random.random() < eps
        spk_seq_for_update = out["spk_seq"].squeeze(1).clone()  # (T, num_actions)

        if explored and oracle_action_idx != natural_action_idx:
            # Credit the oracle's neuron at the same timestep the network
            # would have committed its own decision (or the final timestep
            # if the network never fired at all), so the STDP update still
            # uses this window's real pre-synaptic trace up to that point.
            credit_t = winner_time if winner_time >= 0 else spk_seq_for_update.shape[0] - 1
            spk_seq_for_update[credit_t] = 0.0
            spk_seq_for_update[credit_t, oracle_action_idx] = 1.0
            reported_action_idx = oracle_action_idx  # this window is "about" reinforcing the oracle action
        else:
            reported_action_idx = natural_action_idx

        reward = self._compute_reward(reported_action_idx, oracle_action_idx, delta_rss_np.squeeze(-1))

        self._stdp_update(
            delta_q.squeeze(1), delta_rss.squeeze(1), spk_seq_for_update, reward,
        )
        self._n_updates += 1

        self.action_history.append(natural_action_idx)  # Omega tracks the network's REAL behavior,
                                                           # not exploration-forced credit
        if len(self.action_history) > 100:
            self.action_history.pop(0)

        return {
            "reward": reward,
            "explored": bool(explored and oracle_action_idx != natural_action_idx),
            "exploration_eps": eps,
            "natural_action": BEAM_ACTIONS[natural_action_idx],
            "oracle_action": BEAM_ACTIONS[oracle_action_idx],
            "correct": natural_action_idx == oracle_action_idx,  # network's OWN (unforced) correctness
            "fired": winner_idx >= 0,
        }
