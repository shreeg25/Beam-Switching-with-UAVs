"""
Stage 6: Evaluation Metrics & IEEE Publication Plotting
======================================================

Per-class accuracy, confusion matrix, and 300 DPI IEEE-styled figures.
Automatically exports visual benchmarks to the `notebooks/` directory.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.architecture.aerial_reap6g import BEAM_ACTIONS
from src.training.rstdp_trainer import RSTDPTrainer


@dataclass
class EvalResult:
    overall_accuracy: float
    macro_accuracy: float               # Mean of per-class accuracy
    per_class_accuracy: Dict[str, float]
    per_class_support: Dict[str, int]   # Class occurrences
    confusion_matrix: np.ndarray        # (n_classes, n_classes), rows=true, cols=predicted
    n_windows: int


def _setup_ieee_plot_style() -> None:
    """Configure Matplotlib global RC parameters for IEEE-grade figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 12,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "axes.linewidth": 1.6,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def plot_evaluation_results(
    result: EvalResult,
    output_dir: Path | str = "notebooks",
    prefix: str = "aerial_reap6g",
) -> List[Path]:
    """
    Generate and save IEEE-styled 300 DPI publication plots.
    
    Generates:
    1. Normalized Confusion Matrix Heatmap (with raw count overlays).
    2. Per-Class Accuracy & Support Dual-Axis Bar Chart.
    """
    _setup_ieee_plot_style()
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    generated_files = []

    actions = [str(a) for a in BEAM_ACTIONS]
    n_classes = len(actions)

    # -------------------------------------------------------------------------
    # Plot 1: Normalized Confusion Matrix (Heatmap)
    # -------------------------------------------------------------------------
    conf_mat = result.confusion_matrix.astype(float)
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    
    # Avoid zero division on empty support classes
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_conf = np.divide(conf_mat, row_sums, out=np.zeros_like(conf_mat), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    # Custom annotation formatting: Percentage + (Raw Count)
    annot_matrix = np.empty((n_classes, n_classes), dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            cnt = int(result.confusion_matrix[i, j])
            pct = norm_conf[i, j] * 100.0
            if row_sums[i, 0] > 0:
                annot_matrix[i, j] = f"{pct:.1f}%\n({cnt})"
            else:
                annot_matrix[i, j] = "N/A\n(0)"

    # Vibrant colormap with clear contrast
    sns.heatmap(
        norm_conf,
        annot=annot_matrix,
        fmt="",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Normalized Recall Rate"},
        xticklabels=actions,
        yticklabels=actions,
        linewidths=1.2,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold", "color": "white"},
    )

    # High-contrast annotation coloring for readability
    for text in ax.texts:
        val_str = text.get_text().split("%")[0]
        try:
            val = float(val_str)
            if val > 60.0:
                text.set_color("black")
            else:
                text.set_color("white")
        except ValueError:
            text.set_color("white")

    ax.set_title(
        f"Aerial-REAP-6G Confusion Matrix (Macro Acc: {result.macro_accuracy*100:.2f}%)",
        pad=15,
        color="black",
    )
    ax.set_xlabel("Predicted Beam Action", labelpad=12, color="black")
    ax.set_ylabel("Oracle Ground-Truth Action", labelpad=12, color="black")
    plt.xticks(rotation=30, ha="right", fontweight="bold", color="black")
    plt.yticks(rotation=0, fontweight="bold", color="black")

    cm_file = save_path / f"{prefix}_confusion_matrix_ieee.png"
    fig.savefig(cm_file)
    plt.close(fig)
    generated_files.append(cm_file)

    # -------------------------------------------------------------------------
    # Plot 2: Per-Class Accuracy and Imbalance Support (Dual Axis)
    # -------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    accs = [result.per_class_accuracy[a] * 100.0 if not np.isnan(result.per_class_accuracy[a]) else 0.0 for a in actions]
    supports = [result.per_class_support[a] for a in actions]

    x = np.arange(n_classes)
    width = 0.38

    # Accuracy Bars (Vibrant Electric Blue)
    rects1 = ax1.bar(
        x - width / 2,
        accs,
        width,
        label="Per-Class Accuracy (%)",
        color="#0066CC",
        edgecolor="black",
        linewidth=1.4,
        zorder=3,
    )
    ax1.set_ylabel("Accuracy (%)", color="#0066CC", labelpad=10)
    ax1.set_ylim(0, 108)
    ax1.tick_params(axis="y", labelcolor="#0066CC")
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # Support Bars on Secondary Axis (Vibrant Amber/Orange, Log Scale)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(
        x + width / 2,
        supports,
        width,
        label="Class Support (Log Scale)",
        color="#FF8C00",
        edgecolor="black",
        linewidth=1.4,
        zorder=3,
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Total Samples in Split (Log Scale)", color="#D35400", labelpad=12)
    ax2.tick_params(axis="y", labelcolor="#D35400")

    # Add direct percentage labels on top of accuracy bars
    for rect, acc in zip(rects1, accs):
        h = rect.get_height()
        ax1.annotate(
            f"{acc:.1f}%" if acc > 0 else "0%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(actions, rotation=25, ha="right", fontweight="bold", color="black")
    ax1.set_xlabel("Beam Modification Actions", labelpad=10, color="black")

    ax1.set_title(
        f"Per-Class Accuracy vs. Extreme Imbalance Support (N = {result.n_windows})",
        pad=18,
        color="black",
    )

    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.95, edgecolor="black")

    per_class_file = save_path / f"{prefix}_per_class_performance_ieee.png"
    fig.savefig(per_class_file)
    plt.close(fig)
    generated_files.append(per_class_file)

    print("\n" + "=" * 80)
    print(f">>> [IEEE Plots Saved Successfully at 300 DPI]")
    for f in generated_files:
        print(f"    -> {f.resolve()}")
    print("=" * 80)

    return generated_files


def evaluate(
    trainer: RSTDPTrainer,
    delta_q: np.ndarray,
    delta_rss: np.ndarray,
    oracle_action_idx: np.ndarray,
    save_plots: bool = True,
    output_dir: str = "notebooks",
) -> EvalResult:
    """
    Evaluates the model across time windows and generates IEEE-style figures.
    
    delta_q:           (N, T, 4)
    delta_rss:         (N, T, 1)
    oracle_action_idx: (N, T) -- final timestep of each window used as label
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
    valid_accs = [v for v in per_class_accuracy.values() if not np.isnan(v)]
    macro_accuracy = float(np.mean(valid_accs)) if valid_accs else float("nan")

    result = EvalResult(
        overall_accuracy=overall_accuracy,
        macro_accuracy=macro_accuracy,
        per_class_accuracy=per_class_accuracy,
        per_class_support=per_class_support,
        confusion_matrix=confusion,
        n_windows=N,
    )

    if save_plots:
        plot_evaluation_results(result, output_dir=output_dir)

    return result


def print_eval_report(result: EvalResult) -> None:
    """Prints a formatted evaluation report to stdout."""
    print(f"\n=== Evaluation ({result.n_windows} windows) ===")
    print(f"Overall accuracy: {result.overall_accuracy:.4f}  "
          f"(Dominant 'hold' class makes this metric uninformative)")
    print(f"Macro accuracy:   {result.macro_accuracy:.4f}  "
          f"(Primary metric reflecting rare beam shift corrections)")
    print("\nPer-class breakdown:")
    for action in BEAM_ACTIONS:
        acc = result.per_class_accuracy[action]
        support = result.per_class_support[action]
        acc_str = f"{acc:.4f}" if not np.isnan(acc) else "n/a (0 examples)"
        print(f"  {action:14s}: accuracy={acc_str:20s} support={support}")

    print("\nConfusion Matrix (Rows: Oracle Ground Truth, Columns: Prediction):")
    header = "".join(f"{str(a)[:8]:>10s}" for a in BEAM_ACTIONS)
    print(f"{'':16s}{header}")
    for i, action in enumerate(BEAM_ACTIONS):
        row = "".join(f"{result.confusion_matrix[i, j]:10d}" for j in range(len(BEAM_ACTIONS)))
        print(f"{str(action):16s}{row}")