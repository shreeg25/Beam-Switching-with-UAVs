"""
Stage 4: Dataset Assembly
============================

Ties Stages 1-3 together into an actual trainable dataset. For every UAV in
every generated flight, extracts per-timestep (Δq, ΔRSS, oracle_action),
slices into fixed-length windows, splits by SEED (not by row) to prevent a
flight's data leaking across train/val/test, and exports to .npz.

WHY SPLIT BY SEED, NOT BY ROW
-------------------------------
Randomly shuffling individual timesteps/windows across train/test would let
the network see near-identical (or literally overlapping) context from the
same continuous flight in both splits -- an easy way to get misleadingly
good validation numbers that don't reflect real generalization. Splitting
by seed means an entire flight (and everything windowed from it) belongs to
exactly one split.

WHY OVERLAPPING WINDOWS
-------------------------
A single 20s flight at 500Hz is 10,000 timesteps -- one training example.
Slicing into overlapping windows (stride < window length) multiplies the
number of usable examples per flight without needing more simulated flights,
standard practice for sequence data of this kind.
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple
import json

import numpy as np

from src.dataset.swarm_scenario import build_swarm_scenario
from src.dataset.channel_model import ChannelModel, ChannelParams, BaseStation
from src.dataset.beam_codebook import BeamState, apply_action, oracle_best_action, BEAM_ACTIONS


@dataclass
class DatasetConfig:
    window_length: int = 200        # timesteps per window (200 * 2ms = 0.4s of flight)
    window_stride: int = 50         # 75% overlap between consecutive windows
    n_uavs_per_flight: int = 3
    flight_duration_s: float = 20.0
    dt: float = 0.002
    train_seeds: Tuple[int, ...] = tuple(range(0, 20))
    val_seeds: Tuple[int, ...] = tuple(range(20, 25))
    test_seeds: Tuple[int, ...] = tuple(range(25, 30))
    regimes: Tuple[str, ...] = ("calm", "aggressive")
    bs_position: Tuple[float, float, float] = (0.0, 0.0, 5.0)


def _extract_flight_features(
    uav, model: ChannelModel, params: ChannelParams, rng: np.random.Generator
) -> dict:
    """
    Runs the channel model + oracle over one already-simulated UAV flight
    (uav.history from Stage 1), producing per-timestep arrays:
      delta_q   (T-1, 4)
      delta_rss (T-1, 1)  -- RAW, fading-noise-included RSS delta (Stage 2
                              design choice: SNN input stays noisy)
      oracle_action_idx (T-1,) int, index into BEAM_ACTIONS
    T-1 because deltas need a previous timestep; the first timestep has no
    delta and is dropped.
    """
    T = len(uav.history["t"])
    q_arr = np.array(uav.history["q"])   # (T, 4) scipy xyzw order
    p_arr = np.array(uav.history["p"])   # (T, 3)

    rss = np.empty(T)
    beam_state = BeamState()
    oracle_actions = np.empty(T, dtype=int)

    for i in range(T):
        out = model.compute_rss_dbm(p_arr[i], q_arr[i], rng)
        rss[i] = out["rss_dbm"]

        _, _, direction = model._geometry(p_arr[i])
        action, _ = oracle_best_action(
            q_arr[i], direction, beam_state,
            params.antenna_peak_gain_dbi, params.antenna_pattern_exponent,
        )
        beam_state = apply_action(beam_state, action)
        oracle_actions[i] = BEAM_ACTIONS.index(action)

    delta_q = np.diff(q_arr, axis=0)          # (T-1, 4)
    delta_rss = np.diff(rss)[:, None]         # (T-1, 1)
    # Oracle action label for timestep t is the action recommended AT t
    # (post-diff alignment: drop the first entry to match delta arrays).
    oracle_action_idx = oracle_actions[1:]    # (T-1,)

    return {
        "delta_q": delta_q.astype(np.float32),
        "delta_rss": delta_rss.astype(np.float32),
        "oracle_action_idx": oracle_action_idx.astype(np.int64),
        "event_active": np.array(uav.history["event_active"][1:], dtype=bool),
    }


def _window_flight(features: dict, window_length: int, stride: int, meta: dict) -> List[dict]:
    """Slices one flight's feature arrays into overlapping fixed-length windows."""
    T = features["delta_q"].shape[0]
    windows = []
    start = 0
    while start + window_length <= T:
        end = start + window_length
        windows.append({
            "delta_q": features["delta_q"][start:end],
            "delta_rss": features["delta_rss"][start:end],
            "oracle_action_idx": features["oracle_action_idx"][start:end],
            "event_active": features["event_active"][start:end],
            **meta,
        })
        start += stride
    return windows


def _build_split(seeds: Tuple[int, ...], cfg: DatasetConfig) -> List[dict]:
    bs = BaseStation(position=np.array(cfg.bs_position))
    params = ChannelParams()
    model = ChannelModel(params, bs)

    all_windows = []
    for seed in seeds:
        for regime in cfg.regimes:
            result = build_swarm_scenario(
                n_uavs=cfg.n_uavs_per_flight,
                regime=regime,
                duration=cfg.flight_duration_s,
                dt=cfg.dt,
                seed=seed,
            )
            for uav in result.uavs:
                rng = np.random.default_rng(seed * 1000 + uav.uav_id)
                features = _extract_flight_features(uav, model, params, rng)
                meta = {"seed": seed, "regime": regime, "uav_id": uav.uav_id}
                windows = _window_flight(features, cfg.window_length, cfg.window_stride, meta)
                all_windows.extend(windows)
    return all_windows


def build_dataset(cfg: DatasetConfig = None) -> dict:
    """
    Returns {'train': [...], 'val': [...], 'test': [...]}, each a list of
    window dicts. Call save_dataset() to persist to disk.
    """
    cfg = cfg or DatasetConfig()
    return {
        "train": _build_split(cfg.train_seeds, cfg),
        "val": _build_split(cfg.val_seeds, cfg),
        "test": _build_split(cfg.test_seeds, cfg),
    }


def _stack_split(windows: List[dict]) -> dict:
    """Stacks a list of window dicts into batched numpy arrays for .npz export."""
    if not windows:
        return {}
    return {
        "delta_q": np.stack([w["delta_q"] for w in windows]),               # (N, T, 4)
        "delta_rss": np.stack([w["delta_rss"] for w in windows]),           # (N, T, 1)
        "oracle_action_idx": np.stack([w["oracle_action_idx"] for w in windows]),  # (N, T)
        "event_active": np.stack([w["event_active"] for w in windows]),     # (N, T)
        "seed": np.array([w["seed"] for w in windows]),
        "regime": np.array([w["regime"] for w in windows]),
        "uav_id": np.array([w["uav_id"] for w in windows]),
    }


def save_dataset(dataset: dict, out_dir: str, cfg: DatasetConfig) -> dict:
    """Saves each split to <out_dir>/<split>.npz, plus a config/summary JSON.
    Returns the summary dict (also useful for a quick sanity report)."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    summary = {"config": asdict(cfg), "splits": {}}
    for split_name, windows in dataset.items():
        stacked = _stack_split(windows)
        if not stacked:
            summary["splits"][split_name] = {"n_windows": 0}
            continue
        np.savez_compressed(f"{out_dir}/{split_name}.npz", **stacked)

        n_windows = stacked["oracle_action_idx"].shape[0]
        n_hold = (stacked["oracle_action_idx"] == BEAM_ACTIONS.index("hold")).sum()
        n_total_steps = stacked["oracle_action_idx"].size
        summary["splits"][split_name] = {
            "n_windows": int(n_windows),
            "n_flights_seeds": sorted(set(int(s) for s in stacked["seed"])),
            "hold_fraction": float(n_hold / n_total_steps),
            "n_calm": int((stacked["regime"] == "calm").sum()),
            "n_aggressive": int((stacked["regime"] == "aggressive").sum()),
        }

    with open(f"{out_dir}/dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    cfg = DatasetConfig()
    print(f"Building dataset: {len(cfg.train_seeds)} train / {len(cfg.val_seeds)} val / "
          f"{len(cfg.test_seeds)} test seeds, {cfg.n_uavs_per_flight} UAVs/flight, "
          f"regimes={cfg.regimes}")
    dataset = build_dataset(cfg)
    summary = save_dataset(dataset, "data/processed", cfg)
    print(json.dumps(summary["splits"], indent=2))
