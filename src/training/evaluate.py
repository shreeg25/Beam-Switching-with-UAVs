"""
Stage 6: Evaluation Metrics
==============================

Per-class accuracy and confusion matrix, NOT just raw overall accuracy.

WHY THIS MATTERS
-----------------
The dataset is ~99.5% "hold" (see build_dataset.py's dataset_summary.json).
A model that always predicts "hold" gets ~99.5% raw accuracy while being
completely useless -- it would never fire a real beam correction. Overall
accuracy is actively misleading here; the metric that matters is per-class
(macro-averaged) accuracy/recall, which weighs the rare "shift" classes
equally to the dominant "hold" class regardless of how often each appears.

Evaluation always uses RSTDPTrainer.predict() (pure inference, no weight
updates, no exploration override) so numbers reflect what the network
actually learned, not training-time exploration-assisted behavior.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.architecture.aerial_reap6g import BEAM_ACTIONS
from src.training.rstdp_trainer import RSTDPTrainer


@dataclass
class EvalResult:
    overall_accuracy: float
    macro_accuracy: float               # mean of per-class accuracy -- THE headline number to report
    per_class_accuracy: Dict[str, float]
    per_class_support: Dict[str, int]   # how many examples of each class existed
    confusion_matrix: np.ndarray        # (n_classes, n_classes), rows=true, cols=predicted
    n_windows: int


def evaluate(trainer: RSTDPTrainer, delta_q: np.ndarray, delta_rss: np.ndarray, oracle_action_idx: np.ndarray) -> EvalResult:
    """
    delta_q:           (N, T, 4)
    delta_rss:         (N, T, 1)
    oracle_action_idx: (N, T) -- final timestep of each window used as the label
    """
    n_classes = len(BEAM_ACTIONS)
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    N = delta_q.shape[0]
    for i in range(N):
        true_idx = int(oracle_action_idx[i][-1])
        pred_idx = trainer.predict(delta_q[i], delta_rss[i])
        confusion[true_idx, pred_idx] += 1

    per_class_accuracy = {}
    per_class_support = {}
    for c in range(n_classes):
        support = confusion[c].sum()
        per_class_support[BEAM_ACTIONS[c]] = int(support)
        per_class_accuracy[BEAM_ACTIONS[c]] = float(confusion[c, c] / support) if support > 0 else float("nan")

    overall_accuracy = float(np.trace(confusion) / confusion.sum())
    # macro accuracy: only average over classes that actually appeared in
    # this eval set (nan-safe), since a class with 0 support has no defined
    # accuracy and shouldn't silently zero out the average.
    valid_accs = [v for v in per_class_accuracy.values() if not np.isnan(v)]
    macro_accuracy = float(np.mean(valid_accs)) if valid_accs else float("nan")

    return EvalResult(
        overall_accuracy=overall_accuracy,
        macro_accuracy=macro_accuracy,
        per_class_accuracy=per_class_accuracy,
        per_class_support=per_class_support,
        confusion_matrix=confusion,
        n_windows=N,
    )


def print_eval_report(result: EvalResult):
    print(f"\n=== Evaluation ({result.n_windows} windows) ===")
    print(f"Overall accuracy: {result.overall_accuracy:.4f}  "
          f"(MISLEADING on this dataset -- dominated by the 'hold' class)")
    print(f"Macro accuracy:   {result.macro_accuracy:.4f}  "
          f"(the number that actually reflects rare-class performance)")
    print("\nPer-class:")
    for action in BEAM_ACTIONS:
        acc = result.per_class_accuracy[action]
        support = result.per_class_support[action]
        acc_str = f"{acc:.4f}" if not np.isnan(acc) else "n/a (0 examples)"
        print(f"  {action:12s}: accuracy={acc_str:20s} support={support}")

    print("\nConfusion matrix (rows=true oracle action, cols=network prediction):")
    header = "".join(f"{a[:8]:>10s}" for a in BEAM_ACTIONS)
    print(f"{'':14s}{header}")
    for i, action in enumerate(BEAM_ACTIONS):
        row = "".join(f"{result.confusion_matrix[i,j]:10d}" for j in range(len(BEAM_ACTIONS)))
        print(f"{action:14s}{row}")
