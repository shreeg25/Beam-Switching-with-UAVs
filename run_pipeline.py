#!/usr/bin/env python3
"""
Aerial-REAP-6G: End-to-End Pipeline CLI Runner

Automates the full pipeline:
1. (Optional) Synthetic dataset generation and windowing.
2. Architecture initialization or checkpoint restoration.
3. R-STDP training loop over configurable epochs.
4. Model checkpointing (weights, config, trainer state).
5. Evaluation on held-out splits (macro accuracy, confusion matrix, per-class metrics).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch

from src.architecture.aerial_reap6g import AerialREAP6G, ArchitectureConfig
from src.training.rstdp_trainer import RSTDPTrainer, RSTDPConfig
from src.training.evaluate import evaluate, print_eval_report


def set_seed(seed: int) -> None:
    """Enforce deterministic execution across NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset_if_needed(data_dir: Path, force: bool = False) -> None:
    """Execute the Stage 4 dataset sweep if data files are missing or force=True."""
    train_path = data_dir / "train.npz"
    val_path = data_dir / "val.npz"
    test_path = data_dir / "test.npz"

    if force or not (train_path.exists() and val_path.exists() and test_path.exists()):
        print("\n" + "=" * 80)
        print(">>> [STAGE 1-4] Generating Dataset via build_dataset.py...")
        print("=" * 80)
        from src.pipeline import build_dataset
        # Execute dataset builder main routine
        if hasattr(build_dataset, "main"):
            build_dataset.main()
        else:
            import subprocess
            subprocess.run([sys.executable, "-m", "src.pipeline.build_dataset"], check=True)
    else:
        print(f">>> Found existing dataset in '{data_dir}'. Skipping generation.")


def save_checkpoint(
    checkpoint_path: Path,
    model: AerialREAP6G,
    trainer: RSTDPTrainer,
    arch_config: ArchitectureConfig,
    rstdp_config: RSTDPConfig,
    epoch: int,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist model weights, optimizer/trainer state, and metadata to disk.

    Writes to a temporary file first, verifies it by reloading, and only
    then atomically replaces the real checkpoint path. This guards against
    silent corruption (found during real-world testing: a checkpoint saved
    directly to its final path came back as a truncated ~683-byte zip shell
    containing only PyTorch's internal bookkeeping files, no actual model
    weights -- likely caused by the process being interrupted mid-write).
    With this approach, the final checkpoint path either contains a
    verified-complete checkpoint or doesn't exist at all; it's never left
    holding a corrupt partial file.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract model state dict cleanly
    if hasattr(model, "state_dict"):
        model_state = model.state_dict()
    elif hasattr(model, "weights"):
        model_state = model.weights
    else:
        model_state = getattr(model, "__dict__", {})

    # Safely extract trainer state without dragging the unpicklable snnTorch model object with it
    if hasattr(trainer, "state_dict"):
        trainer_state = trainer.state_dict()
    else:
        # Shallow copy the dict to avoid modifying the live trainer instance
        trainer_state = dict(getattr(trainer, "__dict__", {}))
        # Strip out the raw model reference to prevent surrogate gradient pickling errors
        trainer_state.pop("model", None)

    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "arch_config": arch_config.__dict__ if hasattr(arch_config, "__dict__") else dict(arch_config),
        "rstdp_config": rstdp_config.__dict__ if hasattr(rstdp_config, "__dict__") else dict(rstdp_config),
        "trainer_state": trainer_state,
        "metrics": metrics or {},
        "timestamp": time.time(),
    }

    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    try:
        torch.save(checkpoint, tmp_path)

        # Verify by reloading before committing -- catches truncated/corrupt
        # writes immediately instead of discovering them later when someone
        # tries to actually use the checkpoint.
        verify = torch.load(tmp_path, map_location="cpu", weights_only=False)
        required_keys = {"epoch", "model_state", "arch_config", "rstdp_config"}
        missing = required_keys - set(verify.keys())
        if missing:
            raise RuntimeError(f"checkpoint verification failed -- missing keys: {missing}")

        tmp_path.replace(checkpoint_path)  # atomic on POSIX and Windows (same filesystem)
        print(f"\n[Checkpoint] Saved and verified: {checkpoint_path}")
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Checkpoint save FAILED verification and was NOT written to {checkpoint_path}. "
            f"No corrupt file left behind. Original error: {e}"
        ) from e


def load_checkpoint(
    checkpoint_path: Path,
) -> tuple[AerialREAP6G, RSTDPTrainer, ArchitectureConfig, RSTDPConfig, int]:
    """Restore model architecture, synaptic weights, and trainer state from disk."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Checkpoint] Loading state from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    arch_cfg_dict = checkpoint["arch_config"]
    rstdp_cfg_dict = checkpoint["rstdp_config"]

    arch_config = ArchitectureConfig(**arch_cfg_dict)
    rstdp_config = RSTDPConfig(**rstdp_cfg_dict)

# Instantiate and push model to CUDA
    model = AerialREAP6G(arch_config).to(device)
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(checkpoint["model_state"])
    elif hasattr(model, "weights"):
        model.weights = checkpoint["model_state"]

    trainer = RSTDPTrainer(model, rstdp_config)
    if hasattr(trainer, "load_state_dict") and "trainer_state" in checkpoint:
        try:
            trainer.load_state_dict(checkpoint["trainer_state"])
        except Exception:
            pass

    start_epoch = checkpoint.get("epoch", 0)
    print(f"[Checkpoint] Successfully restored model from Epoch {start_epoch}")
    return model, trainer, arch_config, rstdp_config, start_epoch


def run_training(
    model: AerialREAP6G,
    trainer: RSTDPTrainer,
    train_data: Dict[str, np.ndarray],
    epochs: int,
    log_interval: int = 500,
    shift_oversample_factor: int = 15,
) -> None:
    """Execute the R-STDP training loop over the specified number of epochs.

    shift_oversample_factor: how many times each "shift"-labeled window (any
    label != hold) is replayed per epoch, relative to hold windows (replayed
    once each). Found necessary during real-scale validation: at the raw
    ~186:1 hold:shift ratio in this dataset, exploration-forced reinforcement
    alone produced only ~6 real shift-reinforcement events across an entire
    2-epoch/47,040-step run -- nowhere near enough for STDP's eligibility
    trace to build a durable weight change before the next hold-reinforcement
    overwrote it. The network converged to a degenerate always-predict-hold
    policy (99.5% overall accuracy, 0% recall on every real disturbance).
    Oversampling shift windows directly increases their exposure count
    without touching the (now-correct, symmetric) reward function. A factor
    of 15 was chosen as a middle ground -- brings the effective ratio down to
    roughly 12:1 (from ~186:1) without replaying the ~126 distinct shift
    windows so many times per epoch that the network just memorizes them.
    """
    delta_q = train_data["delta_q"]
    delta_rss = train_data["delta_rss"]
    oracle_action = train_data["oracle_action_idx"]
    n_samples = delta_q.shape[0]

    # Identify shift-labeled windows (final timestep's oracle label != hold)
    # for oversampling. Uses BEAM_ACTIONS' "hold" index directly rather than
    # a bare 0, in case the action ordering ever changes.
    from src.architecture.aerial_reap6g import BEAM_ACTIONS
    hold_idx = BEAM_ACTIONS.index("hold")
    final_labels = oracle_action[:, -1]
    shift_mask = final_labels != hold_idx
    hold_indices = np.where(~shift_mask)[0]
    shift_indices = np.where(shift_mask)[0]

    print("\n" + "=" * 80)
    print(f">>> [STAGE 5] Starting R-STDP Training ({epochs} Epochs, {n_samples} Base Samples/Epoch)")
    print(f">>> Class balance: {len(hold_indices)} hold, {len(shift_indices)} shift "
          f"(oversampling shift x{shift_oversample_factor} -> "
          f"{len(hold_indices)}:{len(shift_indices) * shift_oversample_factor} effective ratio "
          f"~{len(hold_indices) / max(1, len(shift_indices) * shift_oversample_factor):.1f}:1)")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Oversampled epoch order: every hold window once, every shift window
        # shift_oversample_factor times, then shuffled together so oversampled
        # examples aren't clustered at the start/end of the epoch.
        oversampled_shift = np.tile(shift_indices, shift_oversample_factor)
        epoch_indices = np.concatenate([hold_indices, oversampled_shift])
        indices = np.random.permutation(epoch_indices)
        n_epoch_samples = len(indices)

        batch_size = 128  # process 128 windows per GPU forward/backward pass instead of
                           # one at a time -- this is what actually uses the GPU (see
                           # train_on_batch's docstring in rstdp_trainer.py for why
                           # per-window .to(device) calls alone don't help)

        for step_start in range(0, n_epoch_samples, batch_size):
            step_end = min(step_start + batch_size, n_epoch_samples)
            batch_idx = indices[step_start:step_end]

            trainer.train_on_batch(
                delta_q[batch_idx],
                delta_rss[batch_idx],
                oracle_action[batch_idx],
            )

            if (step_end % log_interval < batch_size) or step_end == n_epoch_samples:
                progress = (step_end / n_epoch_samples) * 100
                print(
                    f"Epoch [{epoch}/{epochs}] | Step [{step_end}/{n_epoch_samples}] ({progress:.1f}%)",
                    end="\r",
                    flush=True,
                )

        elapsed = time.time() - t0
        print(f"\nEpoch [{epoch}/{epochs}] Completed in {elapsed:.2f}s ({n_epoch_samples / elapsed:.1f} samples/s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aerial-REAP-6G Execution Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Execution Flow Flags
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "train", "eval", "build-data"],
        default="all",
        help="Pipeline phase to execute.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to directory containing train.npz, val.npz, test.npz.",
    )
    parser.add_argument(
        "--force-rebuild-data",
        action="store_true",
        help="Force rerun dataset build sweep even if processed files exist.",
    )

    # Model & Architecture Configuration
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension for LIF-SNN layer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Training Configuration
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs over the dataset.")
    parser.add_argument("--log-interval", type=int, default=500, help="Training progress logging frequency.")
    parser.add_argument(
        "--shift-oversample-factor",
        type=int,
        default=15,
        help="How many times each shift-labeled (non-hold) window is replayed per epoch, "
             "relative to hold windows (once each). Compensates for extreme class imbalance "
             "(~186:1 in this dataset) -- without it, real disturbance events get too few "
             "reinforcement opportunities for R-STDP to learn them at all.",
    )

    # Checkpoint Configuration
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save/load model weights.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="aerial_reap6g_rstdp.pt",
        help="Checkpoint file name.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Direct path to an existing checkpoint to resume or evaluate.",
    )

    # Evaluation Configuration
    parser.add_argument(
        "--eval-split",
        type=str,
        choices=["val", "test"],
        default="val",
        help="Split to run final evaluation on.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    checkpoint_file = args.load_checkpoint or (args.checkpoint_dir / args.checkpoint_name)

    # Mode: Build Data Only
    if args.mode == "build-data":
        build_dataset_if_needed(args.data_dir, force=True)
        return

    # Ensure Data Exists
    if args.mode in ["all", "train", "eval"]:
        build_dataset_if_needed(args.data_dir, force=args.force_rebuild_data)

    # Architecture & Trainer Initialization / Restoration
    if args.load_checkpoint and args.load_checkpoint.exists():
        model, trainer, arch_config, rstdp_config, _ = load_checkpoint(args.load_checkpoint)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Init] Initializing new model on {device}")
        
        arch_config = ArchitectureConfig(hidden_dim=args.hidden_dim)
        rstdp_config = RSTDPConfig()
        
        # Push to GPU immediately
        model = AerialREAP6G(arch_config).to(device)
        trainer = RSTDPTrainer(model, rstdp_config)

    # Mode: Train
    if args.mode in ["all", "train"]:
        train_path = args.data_dir / "train.npz"
        if not train_path.exists():
            raise FileNotFoundError(f"Training dataset missing at {train_path}")

        print(f"\n[Data] Loading training dataset from {train_path}...")
        d_train = np.load(train_path, allow_pickle=True)

        run_training(
            model=model,
            trainer=trainer,
            train_data=d_train,
            epochs=args.epochs,
            log_interval=args.log_interval,
            shift_oversample_factor=args.shift_oversample_factor,
        )

        save_checkpoint(
            checkpoint_path=checkpoint_file,
            model=model,
            trainer=trainer,
            arch_config=arch_config,
            rstdp_config=rstdp_config,
            epoch=args.epochs,
        )

    # Mode: Evaluate
    if args.mode in ["all", "eval"]:
        eval_path = args.data_dir / f"{args.eval_split}.npz"
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation dataset missing at {eval_path}")

        print("\n" + "=" * 80)
        print(f">>> [STAGE 6] Evaluating on '{args.eval_split}' split ({eval_path})...")
        print("=" * 80)

        d_eval = np.load(eval_path, allow_pickle=True)
        eval_result = evaluate(
            trainer=trainer,
            delta_q=d_eval["delta_q"],
            delta_rss=d_eval["delta_rss"],
            oracle_action_idx=d_eval["oracle_action_idx"],
        )

        print_eval_report(eval_result)


if __name__ == "__main__":
    main()