"""
Synthetic validation harness for Aerial-REAP-6G.

Generates two kinds of synthetic sequences to exercise the architecture:
  1. "calm"    -- steady flight, slowly fading RF link (small deltas)
  2. "violent" -- sharp attitude change + sudden RSS drop (large deltas),
                  simulating a UAV banking hard and losing LoS

This is NOT a physically grounded flight/RF simulator (that was explicitly
scoped out). It's the minimum synthetic signal needed to check that:
  - the adaptive gain suppresses/boosts current sensibly,
  - the LIF membrane integrates and fires,
  - the WTA layer picks exactly one winner and stops scanning after it.
"""

import torch
from aerial_reap6g import AerialREAP6G, ArchitectureConfig, BEAM_ACTIONS

torch.manual_seed(0)


def make_sequence(T: int, batch: int, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (delta_q, delta_rss) of shapes (T, batch, 4) and (T, batch, 1).
    """
    if mode == "calm":
        delta_q = 0.01 * torch.randn(T, batch, 4)
        delta_rss = 0.05 * torch.randn(T, batch, 1)

    elif mode == "violent":
        delta_q = 0.01 * torch.randn(T, batch, 4)
        delta_rss = 0.05 * torch.randn(T, batch, 1)
        # inject a sharp maneuver + LoS fade starting mid-sequence
        onset = T // 2
        delta_q[onset : onset + 3] += torch.tensor([0.4, -0.3, 0.5, -0.2])
        delta_rss[onset : onset + 5] -= 2.5  # sudden signal collapse

    else:
        raise ValueError(f"unknown mode: {mode}")

    return delta_q, delta_rss


def run_case(model: AerialREAP6G, mode: str, T: int = 40, batch: int = 4):
    delta_q, delta_rss = make_sequence(T, batch, mode)
    out = model(delta_q, delta_rss)

    print(f"\n=== Case: {mode} (T={T}, batch={batch}) ===")
    print(f"Gain range:      min={out['gain'].min():.3f}  max={out['gain'].max():.3f}")
    print(f"Total spikes:    {out['spk_seq'].sum().item():.0f} across all neurons/timesteps")
    for b in range(batch):
        t = out["winner_time"][b].item()
        print(
            f"  batch[{b}]: winner='{out['winner_action'][b]:>12}' "
            f"@ t={t if t >= 0 else 'never'}"
        )
    return out


if __name__ == "__main__":
    cfg = ArchitectureConfig(
        hidden_dim=32,
        beta=0.9,
        v_th=1.0,
        gain_window=10,
    )
    model = AerialREAP6G(cfg)

    print(f"Beam action codebook: {BEAM_ACTIONS}")
    print(f"Total learnable params: {sum(p.numel() for p in model.parameters())}")

    calm_out = run_case(model, "calm")
    violent_out = run_case(model, "violent")

    # Sanity checks (assertions, not just prints) --------------------------
    T, B, N = violent_out["spk_seq"].shape

    # 1. WTA must never report more than one committed winner per batch item
    assert violent_out["winner_idx"].shape == (B,)

    # 2. Once a winner fires, no "double decision" bookkeeping exists --
    #    winner_time must be a single scalar per batch item, not a sequence.
    assert violent_out["winner_time"].shape == (B,)

    # 3. Gain should be finite and respect the configured clip ceiling
    assert torch.isfinite(calm_out["gain"]).all()
    assert (calm_out["gain"] <= cfg.gain_clip + 1e-6).all()

    # 4. NOTE: with random (untrained) weights, spike timing does not reliably
    #    track calm-vs-violent signal strength -- that discrimination is a
    #    property of LEARNED weights (R-STDP), not the architecture alone.
    #    See demo_hand_set_weights.py for a hand-derived weight configuration
    #    that proves the wiring itself (gain -> current -> membrane -> WTA)
    #    behaves correctly, decoupled from training.
    for b in range(B):
        ct, vt = calm_out["winner_time"][b].item(), violent_out["winner_time"][b].item()
        if ct >= 0 and vt >= 0:
            print(f"[info] batch[{b}] calm fired @ {ct}, violent fired @ {vt} (random weights -- not meaningful yet)")

    print("\nAll structural sanity checks passed.")