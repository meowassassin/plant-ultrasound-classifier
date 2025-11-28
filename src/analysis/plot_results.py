# src/analysis/plot_results.py
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.training.common import get_project_paths
from src.datasets.plantsounds import (
    scan_plantsounds,
    make_task_lopo_splits,
    make_label,
)

# --------- Global settings to match paper style ---------
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "gray",
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
    }
)


# ---------------------------------------------------------
# Common path utilities
# ---------------------------------------------------------
def get_paths():
    project_root, _ = get_project_paths()
    exp_root = project_root / "experiments"
    fig_root = exp_root / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)
    return project_root, exp_root, fig_root


# ---------------------------------------------------------
# CSV summary utilities
# ---------------------------------------------------------
def _detect_acc_cols(df: pd.DataFrame) -> Tuple[str, str]:
    acc_candidates = ["test_acc", "best_acc", "best_test_acc", "acc"]
    bal_candidates = ["test_bal_acc", "best_bal_acc", "best_test_bal_acc", "bal_acc"]

    acc_col = next((c for c in acc_candidates if c in df.columns), None)
    bal_col = next((c for c in bal_candidates if c in df.columns), None)

    if acc_col is None or bal_col is None:
        raise ValueError(
            f"Could not find accuracy columns. columns={list(df.columns)}"
        )
    return acc_col, bal_col


def summarize_lopo_csv(path: Path, fold_col: str = "fold") -> Dict:
    """
    Summarize one LOPO result CSV (mean/std + per-fold values).
    """
    df = pd.read_csv(path)

    # Remove summary rows like 'mean' from fold column
    if fold_col in df.columns:
        if df[fold_col].dtype == object:
            df_folds = df[df[fold_col] != "mean"].copy()
        else:
            df_folds = df.copy()
    else:
        df_folds = df.copy()

    acc_col, bal_col = _detect_acc_cols(df_folds)

    acc_vals = df_folds[acc_col].to_numpy()
    bal_vals = df_folds[bal_col].to_numpy()

    return {
        "mean_acc": float(np.mean(acc_vals)),
        "std_acc": float(np.std(acc_vals, ddof=1)),
        "mean_bal": float(np.mean(bal_vals)),
        "std_bal": float(np.std(bal_vals, ddof=1)),
        "acc_vals": acc_vals,
        "bal_vals": bal_vals,
    }


# ---------------------------------------------------------
# Task1 (4 experiments)
# ---------------------------------------------------------
TASK1_KEYS = [
    "task1_tomato_dry_vs_cut",
    "task1_tobacco_dry_vs_cut",
    "task1_dry_tomato_vs_tobacco",
    "task1_cut_tomato_vs_tobacco",
]

TASK1_LABELS = [
    "Tomato dry vs cut",
    "Tobacco dry vs cut",
    "Dry: tomato vs tobacco",
    "Cut: tomato vs tobacco",
]

CLASS_LABELS = {
    "task1_tomato_dry_vs_cut": ("cut", "dry"),
    "task1_tobacco_dry_vs_cut": ("cut", "dry"),
    "task1_dry_tomato_vs_tobacco": ("tomato", "tobacco"),
    "task1_cut_tomato_vs_tobacco": ("tomato", "tobacco"),
    "task2_plant_vs_empty": ("empty pot", "plant"),
    "task3_tomato_vs_greenhouse": ("greenhouse noise", "tomato dry"),
}


def load_task1_results(exp_root: Path):
    base_dir = exp_root / "task1_baseline"
    my_dir = exp_root / "task1_my_model"
    semi_dir = exp_root / "task4_my_model"

    baseline = {}
    my_full = {}
    my_semi = {}

    for key in TASK1_KEYS:
        # Baseline
        f_base = base_dir / f"{key}_lopo_results.csv"
        baseline[key] = summarize_lopo_csv(f_base)

        # MyModel (full label)
        f_my = my_dir / f"{key}_lopo_results.csv"
        my_full[key] = summarize_lopo_csv(f_my)

        # MyModel (Task4, 50% label)
        f_semi = semi_dir / f"task4_{key}_lopo_results.csv"
        my_semi[key] = summarize_lopo_csv(f_semi)

    return baseline, my_full, my_semi


def plot_task1_bar(fig_root: Path, baseline, my_full, my_semi):
    x = np.arange(len(TASK1_KEYS))
    width = 0.25

    base_means = [baseline[k]["mean_bal"] for k in TASK1_KEYS]
    base_stds = [baseline[k]["std_bal"] for k in TASK1_KEYS]
    my_means = [my_full[k]["mean_bal"] for k in TASK1_KEYS]
    my_stds = [my_full[k]["std_bal"] for k in TASK1_KEYS]
    semi_means = [my_semi[k]["mean_bal"] for k in TASK1_KEYS]
    semi_stds = [my_semi[k]["std_bal"] for k in TASK1_KEYS]

    # Improved color palette
    colors = ["#3274A1", "#E1812C", "#3A923A"]  # Blue, Orange, Green

    fig, ax = plt.subplots(figsize=(12, 6))

    # Clip error bars at y=1.0
    base_yerr = np.array(base_stds)
    my_yerr = np.array(my_stds)
    semi_yerr = np.array(semi_stds)

    # Clip upper error bars to not exceed 1.0
    base_yerr_clipped = np.minimum(base_yerr, 1.0 - np.array(base_means))
    my_yerr_clipped = np.minimum(my_yerr, 1.0 - np.array(my_means))
    semi_yerr_clipped = np.minimum(semi_yerr, 1.0 - np.array(semi_means))

    # Improved error bar style
    bar1 = ax.bar(x - width, base_means, width, yerr=base_yerr_clipped,
                  capsize=5, color=colors[0], label="Baseline CNN",
                  edgecolor="black", linewidth=1.2, alpha=0.85,
                  error_kw={"linewidth": 2, "ecolor": "black", "capthick": 2})
    bar2 = ax.bar(x, my_means, width, yerr=my_yerr_clipped,
                  capsize=5, color=colors[1], label="MyModel (full)",
                  edgecolor="black", linewidth=1.2, alpha=0.85,
                  error_kw={"linewidth": 2, "ecolor": "black", "capthick": 2})
    bar3 = ax.bar(x + width, semi_means, width, yerr=semi_yerr_clipped,
                  capsize=5, color=colors[2], label="MyModel (50% label)",
                  edgecolor="black", linewidth=1.2, alpha=0.85,
                  error_kw={"linewidth": 2, "ecolor": "black", "capthick": 2})

    # Improved chance level line
    ax.axhline(0.5, color="red", linestyle="--", linewidth=2,
               label="Chance level", alpha=0.7, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(TASK1_LABELS, rotation=25, ha="right", fontsize=12)
    ax.set_ylabel("Balanced accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0.4, 1.02)  # Set y-axis maximum to 1.0
    ax.set_title("Task1: Plant-Plant Classification (LOPO)",
                fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)

    # Improved y-axis tick marks
    ax.set_yticks(np.arange(0.4, 1.1, 0.1))

    fig.tight_layout(pad=1.5)
    fig.savefig(fig_root / "task1_balanced_accuracy_bar.png", dpi=300, bbox_inches="tight", pad_inches=0.2)


def plot_task1_box(fig_root: Path, baseline, my_full):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    axes = axes.reshape(-1)

    # Color definitions
    colors = ["#3274A1", "#E1812C"]

    for idx, key in enumerate(TASK1_KEYS):
        ax = axes[idx]
        data = [baseline[key]["bal_vals"], my_full[key]["bal_vals"]]

        # Improved boxplot style
        bp = ax.boxplot(data, tick_labels=["Baseline", "MyModel"],
                       patch_artist=True, widths=0.6,
                       boxprops=dict(linewidth=1.5, edgecolor="black"),
                       whiskerprops=dict(linewidth=1.5, color="black"),
                       capprops=dict(linewidth=1.5, color="black"),
                       medianprops=dict(linewidth=2.5, color="red"),
                       flierprops=dict(marker='o', markerfacecolor='gray',
                                      markersize=6, linestyle='none',
                                      markeredgecolor='black', alpha=0.6))

        # Set box colors
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Additional handling for cases where all values are identical (std=0)
        for i, (vals, color) in enumerate([(baseline[key]["bal_vals"], colors[0]),
                                             (my_full[key]["bal_vals"], colors[1])]):
            if np.std(vals) < 0.001:  # Almost all values are identical
                mean_val = np.mean(vals)
                # Display as thick horizontal line
                ax.plot([i+0.7, i+1.3], [mean_val, mean_val],
                       color=color, linewidth=4, alpha=0.9, zorder=5)
                # Add text annotation
                ax.text(i+1, mean_val + 0.03, f'All {mean_val:.3f}',
                       ha='center', va='bottom', fontsize=9,
                       fontweight='bold', color=color)

        ax.axhline(0.5, color="red", linestyle="--", linewidth=2,
                  alpha=0.5, zorder=0)
        ax.set_title(TASK1_LABELS[idx], fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Balanced accuracy", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis='both', labelsize=11)

    fig.suptitle("Task1: Fold-wise Balanced Accuracy Distribution",
                fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig.savefig(fig_root / "task1_balanced_accuracy_boxplot.png", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------
# Task2 / Task3
# ---------------------------------------------------------
def load_task23_results(exp_root: Path):
    # Task2
    t2_base = summarize_lopo_csv(
        exp_root / "task2_baseline" / "task2_plant_vs_empty_lopo_results.csv"
    )
    t2_my = summarize_lopo_csv(
        exp_root
        / "task2_my_model"
        / "task2_plant_vs_empty_lopo_results_labeled1.0.csv"
    )

    # Task3
    t3_base = summarize_lopo_csv(
        exp_root / "task3_baseline" / "task3_tomato_vs_greenhouse_lopo_results.csv"
    )
    t3_my = summarize_lopo_csv(
        exp_root
        / "task3_my_model"
        / "task3_tomato_vs_greenhouse_lopo_results_labeled1.0.csv"
    )

    return (t2_base, t2_my), (t3_base, t3_my)


def plot_task23_bar(fig_root: Path, t2, t3):
    tasks = ["Task2: plant vs empty pot", "Task3: tomato vs greenhouse noise"]
    base_means = [t2[0]["mean_bal"], t3[0]["mean_bal"]]
    base_stds = [t2[0]["std_bal"], t3[0]["std_bal"]]
    my_means = [t2[1]["mean_bal"], t3[1]["mean_bal"]]
    my_stds = [t2[1]["std_bal"], t3[1]["std_bal"]]

    x = np.arange(len(tasks))
    width = 0.35

    # Color palette
    colors = ["#3274A1", "#E1812C"]

    # Clip error bars at y=1.0
    base_yerr_clipped = np.minimum(base_stds, 1.0 - np.array(base_means))
    my_yerr_clipped = np.minimum(my_stds, 1.0 - np.array(my_means))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, base_means, width, yerr=base_yerr_clipped,
           capsize=5, color=colors[0], label="Baseline CNN",
           edgecolor="black", linewidth=1.2, alpha=0.85,
           error_kw={"linewidth": 2, "ecolor": "black", "capthick": 2})
    ax.bar(x + width / 2, my_means, width, yerr=my_yerr_clipped,
           capsize=5, color=colors[1], label="MyModel",
           edgecolor="black", linewidth=1.2, alpha=0.85,
           error_kw={"linewidth": 2, "ecolor": "black", "capthick": 2})

    ax.axhline(0.5, color="red", linestyle="--", linewidth=2,
               label="Chance level", alpha=0.7, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15, ha="right", fontsize=12)
    ax.set_ylabel("Balanced accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0.4, 1.02)  # Set y-axis maximum to 1.0
    ax.set_yticks(np.arange(0.4, 1.1, 0.1))
    ax.set_title("Task2 / Task3: Plant vs Noise (LOPO)",
                fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)

    fig.tight_layout(pad=1.5)
    fig.savefig(fig_root / "task2_task3_balanced_accuracy_bar.png", dpi=300, bbox_inches="tight", pad_inches=0.2)


# ---------------------------------------------------------
# Task4 (emphasizing 50% label effect)
# ---------------------------------------------------------
def plot_task4_semi_effect(fig_root: Path, baseline, my_full, my_semi):
    """
    Visualize semi-supervised learning effect with a simple line graph
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.reshape(-1)

    colors = ["#3274A1", "#E1812C", "#3A923A"]

    for idx, key in enumerate(TASK1_KEYS):
        ax = axes[idx]

        # Prepare data
        label_fractions = [50, 100]

        base_mean = baseline[key]["mean_bal"]
        full_mean = my_full[key]["mean_bal"]
        semi_mean = my_semi[key]["mean_bal"]

        full_std = my_full[key]["std_bal"]
        semi_std = my_semi[key]["std_bal"]

        # Baseline horizontal line
        ax.axhline(base_mean, color=colors[0], linestyle='--',
                  linewidth=2, alpha=0.7, label='Baseline', zorder=1)

        # MyModel: 50% -> 100% label
        mymodel_means = [semi_mean, full_mean]
        mymodel_stds = [semi_std, full_std]

        # Line plot with error bars
        ax.errorbar(label_fractions, mymodel_means,
                   yerr=mymodel_stds,
                   color=colors[1], linewidth=2.5, marker='o', markersize=8,
                   capsize=6, capthick=2, elinewidth=2,
                   label='MyModel', alpha=0.9, zorder=3)

        # Display values (simple)
        ax.text(50, semi_mean - 0.05, f'{semi_mean:.3f}',
               ha='center', va='top', fontsize=9, color=colors[1])
        ax.text(100, full_mean + 0.03, f'{full_mean:.3f}',
               ha='center', va='bottom', fontsize=9, color=colors[1])

        # Chance level
        ax.axhline(0.5, color='red', linestyle=':', linewidth=1.5,
                  alpha=0.5, label='Chance', zorder=0)

        ax.set_xlabel('Label Fraction (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Balanced Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(TASK1_LABELS[idx], fontsize=12, fontweight='bold', pad=10)
        ax.set_xlim(40, 110)
        ax.set_ylim(0.5, 1.05)
        ax.set_xticks([50, 100])
        ax.set_xticklabels(['50%', '100%'])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('Task4: Semi-Supervised Learning Effect',
                fontsize=15, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig.savefig(fig_root / "task4_label_fraction_effect.png", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------
# Confusion matrix reconstruction (using acc + bal_acc + label count)
# ---------------------------------------------------------
def _bruteforce_tp_tn(np_pos: int, np_neg: int, acc: float, bal: float) -> Tuple[int, int]:
    """
    Given N_pos, N_neg, accuracy, and balanced accuracy,
    reconstruct confusion matrix by brute-force search for (TP, TN).
    """
    best_tp, best_tn = 0, 0
    best_err = float("inf")
    total = np_pos + np_neg

    for tp in range(np_pos + 1):
        for tn in range(np_neg + 1):
            a = (tp + tn) / total
            tpr = tp / np_pos if np_pos > 0 else 0.0
            tnr = tn / np_neg if np_neg > 0 else 0.0
            b = 0.5 * (tpr + tnr)
            err = (a - acc) ** 2 + (b - bal) ** 2
            if err < best_err:
                best_err = err
                best_tp, best_tn = tp, tn

    return best_tp, best_tn


def reconstruct_confusion_for_model(task_name: str, csv_path: Path) -> np.ndarray:
    """
    Generate a combined 2x2 confusion matrix across all LOPO folds for one task + one model (CSV).
    """
    project_root, plantsounds_root = get_project_paths()
    metas = scan_plantsounds(str(plantsounds_root))
    splits = make_task_lopo_splits(metas, task_name)

    df = pd.read_csv(csv_path)
    # Remove 'mean' row from folds
    if "fold" in df.columns and df["fold"].dtype == object:
        df_folds = df[df["fold"] != "mean"].copy()
    else:
        df_folds = df.copy()

    acc_col, bal_col = _detect_acc_cols(df_folds)

    cm = np.zeros((2, 2), dtype=int)

    for fold_idx, (_, test_idx) in enumerate(splits):
        if fold_idx >= len(df_folds):
            break

        row = df_folds.iloc[fold_idx]
        acc = float(row[acc_col])
        bal = float(row[bal_col])

        # Per-fold label count
        labels = [make_label(metas[i], task_name) for i in test_idx]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos

        # If only one class exists, balanced acc is ambiguous → exclude from confusion
        if n_pos == 0 or n_neg == 0:
            continue

        tp, tn = _bruteforce_tp_tn(n_pos, n_neg, acc, bal)
        fp = n_neg - tn
        fn = n_pos - tp

        # rows=true, cols=pred  [ [TN, FP],[FN, TP] ]
        cm[0, 0] += tn
        cm[0, 1] += fp
        cm[1, 0] += fn
        cm[1, 1] += tp

    return cm


def compute_confusions_task1(exp_root: Path):
    base_dir = exp_root / "task1_baseline"
    my_dir = exp_root / "task1_my_model"

    base_cm = {}
    my_cm = {}

    for key in TASK1_KEYS:
        base_cm[key] = reconstruct_confusion_for_model(
            key, base_dir / f"{key}_lopo_results.csv"
        )
        my_cm[key] = reconstruct_confusion_for_model(
            key, my_dir / f"{key}_lopo_results.csv"
        )

    return base_cm, my_cm


def compute_confusions_task23(exp_root: Path):
    base_cm = {}
    my_cm = {}

    # Task2
    base_cm["task2_plant_vs_empty"] = reconstruct_confusion_for_model(
        "task2_plant_vs_empty",
        exp_root / "task2_baseline" / "task2_plant_vs_empty_lopo_results.csv",
    )
    my_cm["task2_plant_vs_empty"] = reconstruct_confusion_for_model(
        "task2_plant_vs_empty",
        exp_root
        / "task2_my_model"
        / "task2_plant_vs_empty_lopo_results_labeled1.0.csv",
    )

    # Task3
    base_cm["task3_tomato_vs_greenhouse"] = reconstruct_confusion_for_model(
        "task3_tomato_vs_greenhouse",
        exp_root
        / "task3_baseline"
        / "task3_tomato_vs_greenhouse_lopo_results.csv",
    )
    my_cm["task3_tomato_vs_greenhouse"] = reconstruct_confusion_for_model(
        "task3_tomato_vs_greenhouse",
        exp_root
        / "task3_my_model"
        / "task3_tomato_vs_greenhouse_lopo_results_labeled1.0.csv",
    )

    return base_cm, my_cm


def _plot_single_confusion(ax, cm: np.ndarray, title: str, task_name: str):
    # Use improved colormap (better contrast)
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest", vmin=0)

    # If confusion matrix is all zeros (each fold in LOPO contains only one class)
    is_empty = cm.sum() == 0

    if is_empty:
        # Add explanatory text to empty confusion matrix
        ax.text(0.5, 0.5,
                "N/A\n(Each fold contains\nonly one class)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray", style="italic",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                         edgecolor="gray", alpha=0.8))
    else:
        # Number annotations - auto-adjust text color based on background
        max_val = cm.max()
        for i in range(2):
            for j in range(2):
                val = int(cm[i, j])
                # White text for dark background, black for light background
                text_color = "white" if cm[i, j] > max_val / 2 else "black"
                ax.text(j, i, val, ha="center", va="center",
                       color=text_color, fontsize=14, fontweight="bold")

    classes = CLASS_LABELS.get(task_name, ("neg", "pos"))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred\n{c}" for c in classes], fontsize=10, fontweight="bold")
    ax.set_yticklabels([f"True\n{c}" for c in classes], fontsize=10, fontweight="bold")
    ax.set_title(title, pad=10, fontsize=12, fontweight="bold")

    # Add colorbar (only if matrix is not empty)
    if not is_empty:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)


def plot_confusions_task1(fig_root: Path, base_cm, my_cm):
    # Filter out empty confusion matrices (single-class folds)
    valid_items = []
    for i, key in enumerate(TASK1_KEYS):
        # Check if both baseline and mymodel have non-empty confusion matrices
        if base_cm[key].sum() > 0 and my_cm[key].sum() > 0:
            valid_items.append((i, key))

    if len(valid_items) == 0:
        print("[WARN] No valid confusion matrices to plot for Task1")
        return

    # Create subplot with only valid rows
    fig, axes = plt.subplots(len(valid_items), 2, figsize=(9, 2.4 * len(valid_items)))
    if len(valid_items) == 1:
        axes = axes.reshape(1, 2)
    else:
        axes = axes.reshape(len(valid_items), 2)

    for plot_idx, (orig_idx, key) in enumerate(valid_items):
        _plot_single_confusion(
            axes[plot_idx, 0],
            base_cm[key],
            title=f"{TASK1_LABELS[orig_idx]} – Baseline",
            task_name=key,
        )
        _plot_single_confusion(
            axes[plot_idx, 1],
            my_cm[key],
            title=f"{TASK1_LABELS[orig_idx]} – MyModel",
            task_name=key,
        )

    fig.suptitle("Task1: confusion matrices (LOPO pooled)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(fig_root / "task1_confusion_matrices.png", dpi=300)


def plot_confusions_task23(fig_root: Path, base_cm, my_cm):
    keys = ["task2_plant_vs_empty", "task3_tomato_vs_greenhouse"]
    titles = ["Task2: plant vs empty pot", "Task3: tomato vs greenhouse noise"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    axes = axes.reshape(2, 2)

    for i, key in enumerate(keys):
        _plot_single_confusion(
            axes[i, 0],
            base_cm[key],
            title=f"{titles[i]} – Baseline",
            task_name=key,
        )
        _plot_single_confusion(
            axes[i, 1],
            my_cm[key],
            title=f"{titles[i]} – MyModel",
            task_name=key,
        )

    fig.suptitle("Task2 / Task3: confusion matrices (LOPO pooled)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(fig_root / "task2_task3_confusion_matrices.png", dpi=300)


# ---------------------------------------------------------
# Paired scatter (baseline vs MyModel, per-fold basis)
# ---------------------------------------------------------
def plot_task1_paired_scatter(fig_root: Path, baseline, my_full):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.reshape(-1)

    for idx, key in enumerate(TASK1_KEYS):
        ax = axes[idx]
        x_vals = baseline[key]["bal_vals"]
        y_vals = my_full[key]["bal_vals"]

        # Improved scatter plot
        ax.scatter(x_vals, y_vals, s=80, alpha=0.7, c="#3274A1",
                  edgecolors="black", linewidth=1.2, zorder=3)

        vmin = min(x_vals.min(), y_vals.min(), 0.4)
        vmax = max(x_vals.max(), y_vals.max(), 1.0)

        # Improved identity line
        ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=2,
               color="red", alpha=0.6, label="y = x", zorder=1)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--", zorder=0)

        ax.set_xlabel("Baseline balanced acc", fontsize=12, fontweight="bold")
        ax.set_ylabel("MyModel balanced acc", fontsize=12, fontweight="bold")
        ax.set_title(TASK1_LABELS[idx], fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(vmin - 0.02, vmax + 0.02)
        ax.set_ylim(vmin - 0.02, vmax + 0.02)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis='both', labelsize=10)

        # Add legend
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle("Task1: Fold-wise Baseline vs MyModel Comparison",
                fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig.savefig(fig_root / "task1_paired_scatter.png", dpi=300, bbox_inches="tight")


def plot_task23_paired_scatter(fig_root: Path, t2, t3):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Task2
    x_vals = t2[0]["bal_vals"]
    y_vals = t2[1]["bal_vals"]
    ax = axes[0]
    ax.scatter(x_vals, y_vals, s=80, alpha=0.7, c="#3274A1",
              edgecolors="black", linewidth=1.2, zorder=3)
    vmin = min(x_vals.min(), y_vals.min(), 0.4)
    vmax = max(x_vals.max(), y_vals.max(), 1.0)
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=2,
           color="red", alpha=0.6, label="y = x", zorder=1)
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.set_title("Task2: plant vs empty pot", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Baseline balanced acc", fontsize=12, fontweight="bold")
    ax.set_ylabel("MyModel balanced acc", fontsize=12, fontweight="bold")
    ax.set_xlim(vmin - 0.02, vmax + 0.02)
    ax.set_ylim(vmin - 0.02, vmax + 0.02)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Task3
    x_vals = t3[0]["bal_vals"]
    y_vals = t3[1]["bal_vals"]
    ax = axes[1]
    ax.scatter(x_vals, y_vals, s=80, alpha=0.7, c="#E1812C",
              edgecolors="black", linewidth=1.2, zorder=3)
    vmin = min(x_vals.min(), y_vals.min(), 0.4)
    vmax = max(x_vals.max(), y_vals.max(), 1.0)
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=2,
           color="red", alpha=0.6, label="y = x", zorder=1)
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.set_title("Task3: tomato vs greenhouse noise", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Baseline balanced acc", fontsize=12, fontweight="bold")
    ax.set_ylabel("MyModel balanced acc", fontsize=12, fontweight="bold")
    ax.set_xlim(vmin - 0.02, vmax + 0.02)
    ax.set_ylim(vmin - 0.02, vmax + 0.02)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle("Task2 / Task3: Baseline vs MyModel Comparison",
                fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(fig_root / "task2_task3_paired_scatter.png", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------
# Additional visualization: dry vs cut single-class fold results
# ---------------------------------------------------------
def load_fold_level_data_with_metadata(exp_root: Path, task_name: str, model_type: str):
    """
    Load fold-level results with plant metadata.

    Returns:
        List of (fold_idx, accuracy, plant_id, condition) tuples
    """
    from src.training.common import get_project_paths

    # Determine CSV path
    if model_type == "baseline":
        csv_path = exp_root / "task1_baseline" / f"{task_name}_lopo_results.csv"
    elif model_type == "mymodel":
        csv_path = exp_root / "task1_my_model" / f"{task_name}_lopo_results.csv"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if not csv_path.exists():
        return []

    # Load CSV
    df = pd.read_csv(csv_path)
    df_folds = df[df["fold"] != "mean"].copy()

    # Get metadata for this task to map fold -> condition
    project_root, plantsounds_root = get_project_paths()
    metas = scan_plantsounds(str(plantsounds_root))
    splits = make_task_lopo_splits(metas, task_name)

    results = []
    for fold_idx in range(len(splits)):
        if fold_idx >= len(df_folds):
            break

        row = df_folds.iloc[fold_idx]
        acc_col, bal_col = _detect_acc_cols(df_folds)
        accuracy = float(row[bal_col])

        # Get test indices for this fold
        _, test_idx = splits[fold_idx]
        if len(test_idx) == 0:
            continue

        # Get condition from first test sample (all should be same in single-class fold)
        test_meta = metas[test_idx[0]]
        condition = test_meta.stress  # 'dry' or 'cut'
        plant_id = test_meta.plant_id

        results.append((fold_idx, accuracy, plant_id, condition))

    return results


def plot_fold_accuracy_dry(fig_root: Path, exp_root: Path):
    """Plot fold-wise accuracy for dry samples (tomato and tobacco)"""
    tasks = [
        ("task1_tomato_dry_vs_cut", "Tomato dry vs cut"),
        ("task1_tobacco_dry_vs_cut", "Tobacco dry vs cut"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for task_idx, (task_name, task_label) in enumerate(tasks):
        ax = axes[task_idx]

        # Load data
        base_data = load_fold_level_data_with_metadata(exp_root, task_name, "baseline")
        my_data = load_fold_level_data_with_metadata(exp_root, task_name, "mymodel")

        if not base_data or not my_data:
            print(f"[WARN] No data for {task_name}, skipping visualization")
            continue

        # Separate dry samples
        base_dry = [(f, a) for f, a, p, c in base_data if c == 'dry']
        my_dry = [(f, a) for f, a, p, c in my_data if c == 'dry']

        if base_dry:
            folds_dry, accs_dry = zip(*base_dry)
            jitter = np.random.RandomState(42).uniform(-0.01, 0.01, len(accs_dry))
            accs_dry_jittered = np.array(accs_dry) + jitter
            ax.scatter(folds_dry, accs_dry_jittered, marker='o', s=50,
                       color='#3274A1', alpha=0.6, edgecolors='black',
                       linewidth=0.5, label='Baseline (dry)', zorder=3)
            ax.plot(folds_dry, accs_dry, color='#3274A1', linewidth=1.5,
                    alpha=0.3, zorder=1)
        if my_dry:
            folds_dry, accs_dry = zip(*my_dry)
            jitter = np.random.RandomState(43).uniform(-0.01, 0.01, len(accs_dry))
            accs_dry_jittered = np.array(accs_dry) + jitter
            ax.scatter(folds_dry, accs_dry_jittered, marker='s', s=50,
                       color='#E1812C', alpha=0.6, edgecolors='black',
                       linewidth=0.5, label='MyModel (dry)', zorder=3)
            ax.plot(folds_dry, accs_dry, color='#E1812C', linewidth=1.5,
                    alpha=0.3, zorder=1)

        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Fold (dry plants)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(f'{task_label}\nFold-wise Accuracy (Dry)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Fold-wise Accuracy: Dry Samples',
                fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(fig_root / "task1_fold_accuracy_dry.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_root / 'task1_fold_accuracy_dry.png'}")


def plot_fold_accuracy_cut(fig_root: Path, exp_root: Path):
    """Plot fold-wise accuracy for cut samples (tomato and tobacco)"""
    tasks = [
        ("task1_tomato_dry_vs_cut", "Tomato dry vs cut"),
        ("task1_tobacco_dry_vs_cut", "Tobacco dry vs cut"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for task_idx, (task_name, task_label) in enumerate(tasks):
        ax = axes[task_idx]

        # Load data
        base_data = load_fold_level_data_with_metadata(exp_root, task_name, "baseline")
        my_data = load_fold_level_data_with_metadata(exp_root, task_name, "mymodel")

        if not base_data or not my_data:
            print(f"[WARN] No data for {task_name}, skipping visualization")
            continue

        # Separate cut samples
        base_cut = [(f, a) for f, a, p, c in base_data if c == 'cut']
        my_cut = [(f, a) for f, a, p, c in my_data if c == 'cut']

        if base_cut:
            folds_cut, accs_cut = zip(*base_cut)
            jitter = np.random.RandomState(44).uniform(-0.01, 0.01, len(accs_cut))
            accs_cut_jittered = np.array(accs_cut) + jitter
            ax.scatter(folds_cut, accs_cut_jittered, marker='o', s=50,
                       color='#3274A1', alpha=0.6, edgecolors='black',
                       linewidth=0.5, label='Baseline (cut)', zorder=3)
            ax.plot(folds_cut, accs_cut, color='#3274A1', linewidth=1.5,
                    alpha=0.3, zorder=1)
        if my_cut:
            folds_cut, accs_cut = zip(*my_cut)
            jitter = np.random.RandomState(45).uniform(-0.01, 0.01, len(accs_cut))
            accs_cut_jittered = np.array(accs_cut) + jitter
            ax.scatter(folds_cut, accs_cut_jittered, marker='s', s=50,
                       color='#E1812C', alpha=0.6, edgecolors='black',
                       linewidth=0.5, label='MyModel (cut)', zorder=3)
            ax.plot(folds_cut, accs_cut, color='#E1812C', linewidth=1.5,
                    alpha=0.3, zorder=1)

        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Fold (cut plants)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(f'{task_label}\nFold-wise Accuracy (Cut)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Fold-wise Accuracy: Cut Samples',
                fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(fig_root / "task1_fold_accuracy_cut.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_root / 'task1_fold_accuracy_cut.png'}")


def plot_classwise_accuracy_bar(fig_root: Path, exp_root: Path):
    """Plot class-wise accuracy bar chart (dry vs cut)"""
    tasks = [
        ("task1_tomato_dry_vs_cut", "Tomato dry vs cut"),
        ("task1_tobacco_dry_vs_cut", "Tobacco dry vs cut"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for task_idx, (task_name, task_label) in enumerate(tasks):
        ax = axes[task_idx]

        # Load data
        base_data = load_fold_level_data_with_metadata(exp_root, task_name, "baseline")
        my_data = load_fold_level_data_with_metadata(exp_root, task_name, "mymodel")

        if not base_data or not my_data:
            print(f"[WARN] No data for {task_name}, skipping visualization")
            continue

        # Separate by condition
        base_dry = [(f, a) for f, a, p, c in base_data if c == 'dry']
        base_cut = [(f, a) for f, a, p, c in base_data if c == 'cut']
        my_dry = [(f, a) for f, a, p, c in my_data if c == 'dry']
        my_cut = [(f, a) for f, a, p, c in my_data if c == 'cut']

        base_dry_acc = np.mean([a for _, a in base_dry]) if base_dry else 0
        base_cut_acc = np.mean([a for _, a in base_cut]) if base_cut else 0
        my_dry_acc = np.mean([a for _, a in my_dry]) if my_dry else 0
        my_cut_acc = np.mean([a for _, a in my_cut]) if my_cut else 0

        base_dry_std = np.std([a for _, a in base_dry]) if base_dry else 0
        base_cut_std = np.std([a for _, a in base_cut]) if base_cut else 0
        my_dry_std = np.std([a for _, a in my_dry]) if my_dry else 0
        my_cut_std = np.std([a for _, a in my_cut]) if my_cut else 0

        x = np.arange(2)
        width = 0.35

        # Clip error bars at y=1.0
        base_yerr = np.array([base_dry_std, base_cut_std])
        my_yerr = np.array([my_dry_std, my_cut_std])
        base_means_arr = np.array([base_dry_acc, base_cut_acc])
        my_means_arr = np.array([my_dry_acc, my_cut_acc])

        base_yerr_clipped = np.minimum(base_yerr, 1.0 - base_means_arr)
        my_yerr_clipped = np.minimum(my_yerr, 1.0 - my_means_arr)

        ax.bar(x - width/2, [base_dry_acc, base_cut_acc], width,
               yerr=base_yerr_clipped, capsize=5,
               color='#3274A1', label='Baseline', edgecolor='black',
               linewidth=1.2, alpha=0.85,
               error_kw={"linewidth": 2, "ecolor": "black", "capthick": 2})
        ax.bar(x + width/2, [my_dry_acc, my_cut_acc], width,
               yerr=my_yerr_clipped, capsize=5,
               color='#E1812C', label='MyModel', edgecolor='black',
               linewidth=1.2, alpha=0.85,
               error_kw={"linewidth": 2, "ecolor": "black", "capthick": 2})

        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['Dry samples', 'Cut samples'], fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(f'{task_label}\nClass-wise Accuracy', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim(0.4, 1.02)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Class-wise Accuracy: Dry vs Cut',
                fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(fig_root / "task1_classwise_accuracy_bar.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_root / 'task1_classwise_accuracy_bar.png'}")


def plot_sequential_performance(fig_root: Path, exp_root: Path):
    """Plot sequential performance with condition colors"""
    tasks = [
        ("task1_tomato_dry_vs_cut", "Tomato dry vs cut"),
        ("task1_tobacco_dry_vs_cut", "Tobacco dry vs cut"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for task_idx, (task_name, task_label) in enumerate(tasks):
        ax = axes[task_idx]

        # Load data
        base_data = load_fold_level_data_with_metadata(exp_root, task_name, "baseline")
        my_data = load_fold_level_data_with_metadata(exp_root, task_name, "mymodel")

        if not base_data or not my_data:
            print(f"[WARN] No data for {task_name}, skipping visualization")
            continue

        # Plot all folds in order, colored by condition
        all_base = sorted(base_data, key=lambda x: x[0])
        all_my = sorted(my_data, key=lambda x: x[0])

        # Add jitter to separate overlapping points at same fold
        rng_base = np.random.RandomState(46 + task_idx)
        rng_my = np.random.RandomState(47 + task_idx)

        for i, (fold, acc, pid, cond) in enumerate(all_base):
            color = '#87CEEB' if cond == 'dry' else '#FFB6C1'  # light blue / light pink
            # Add small horizontal jitter and vertical jitter
            fold_jitter = fold + rng_base.uniform(-0.2, 0.0)
            acc_jitter = acc + rng_base.uniform(-0.01, 0.01)
            ax.scatter(fold_jitter, acc_jitter, s=60, c=color, edgecolors='#3274A1',
                       linewidth=1.2, alpha=0.7, marker='o', zorder=3)

        for i, (fold, acc, pid, cond) in enumerate(all_my):
            color = '#FFD700' if cond == 'dry' else '#FF8C00'  # gold / dark orange
            # Add small horizontal jitter (opposite direction) and vertical jitter
            fold_jitter = fold + rng_my.uniform(0.0, 0.2)
            acc_jitter = acc + rng_my.uniform(-0.01, 0.01)
            ax.scatter(fold_jitter, acc_jitter, s=60, c=color, edgecolors='#E1812C',
                       linewidth=1.2, alpha=0.7, marker='s', zorder=3)

        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#87CEEB', edgecolor='#3274A1', label='Baseline (dry)'),
            Patch(facecolor='#FFB6C1', edgecolor='#3274A1', label='Baseline (cut)'),
            Patch(facecolor='#FFD700', edgecolor='#E1812C', label='MyModel (dry)'),
            Patch(facecolor='#FF8C00', edgecolor='#E1812C', label='MyModel (cut)'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Fold (Plant ID)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(f'{task_label}\nSequential Performance', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Sequential Performance by Condition',
                fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(fig_root / "task1_sequential_performance.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_root / 'task1_sequential_performance.png'}")


def plot_per_plant_heatmap(fig_root: Path, exp_root: Path):
    """
    Option 3: Per-plant performance heatmap
    Display performance of each plant × condition as heatmap
    """
    tasks = [
        ("task1_tomato_dry_vs_cut", "Tomato"),
        ("task1_tobacco_dry_vs_cut", "Tobacco"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.reshape(2, 2)

    for task_idx, (task_name, species) in enumerate(tasks):
        for model_idx, model_type in enumerate(["baseline", "mymodel"]):
            ax = axes[task_idx, model_idx]

            data = load_fold_level_data_with_metadata(exp_root, task_name, model_type)

            if not data:
                ax.text(0.5, 0.5, "No data", ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f"{species} - {model_type.capitalize()}",
                           fontsize=12, fontweight='bold')
                continue

            # Organize data into dict: plant_id -> {dry: acc, cut: acc}
            plant_data = {}
            for fold, acc, pid, cond in data:
                if pid not in plant_data:
                    plant_data[pid] = {'dry': None, 'cut': None}
                plant_data[pid][cond] = acc

            # Create matrix
            plant_ids = sorted(plant_data.keys())
            matrix = []
            for pid in plant_ids:
                dry_acc = plant_data[pid]['dry'] if plant_data[pid]['dry'] is not None else np.nan
                cut_acc = plant_data[pid]['cut'] if plant_data[pid]['cut'] is not None else np.nan
                matrix.append([dry_acc, cut_acc])

            matrix = np.array(matrix)

            # Plot heatmap
            im = ax.imshow(matrix.T, cmap='RdYlGn', aspect='auto',
                          interpolation='nearest', vmin=0, vmax=1)

            # Annotations
            for i, pid in enumerate(plant_ids):
                for j, cond in enumerate(['dry', 'cut']):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val < 0.5 else 'black'
                        ax.text(i, j, f'{val:.2f}', ha='center', va='center',
                               color=text_color, fontsize=8, fontweight='bold')
                    else:
                        ax.text(i, j, 'N/A', ha='center', va='center',
                               color='gray', fontsize=8, style='italic')

            ax.set_xticks(range(len(plant_ids)))
            ax.set_xticklabels([f'P{pid}' for pid in plant_ids],
                              rotation=45, ha='right', fontsize=8)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Dry', 'Cut'], fontsize=10, fontweight='bold')
            ax.set_xlabel('Plant ID', fontsize=10, fontweight='bold')

            model_label = "Baseline CNN" if model_type == "baseline" else "MyModel"
            ax.set_title(f"{species} - {model_label}",
                        fontsize=12, fontweight='bold', pad=10)

            # Colorbar
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax, label='Accuracy')

    fig.suptitle('Per-Plant Performance Heatmap: Dry vs Cut Tasks',
                fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig.savefig(fig_root / "task1_per_plant_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_root / 'task1_per_plant_heatmap.png'}")


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main():
    _, exp_root, fig_root = get_paths()

    # Task1
    t1_base, t1_my, t1_semi = load_task1_results(exp_root)
    plot_task1_bar(fig_root, t1_base, t1_my, t1_semi)
    plot_task1_box(fig_root, t1_base, t1_my)
    plot_task4_semi_effect(fig_root, t1_base, t1_my, t1_semi)

    # Task2 / Task3
    t2, t3 = load_task23_results(exp_root)
    plot_task23_bar(fig_root, t2, t3)

    # Paired scatter
    plot_task1_paired_scatter(fig_root, t1_base, t1_my)
    plot_task23_paired_scatter(fig_root, t2, t3)

    # Single-class fold analysis (dry vs cut tasks) - individual plots
    # Generate these first as they only need CSV results, not raw data
    print("Generating individual single-class fold analysis plots...")
    plot_fold_accuracy_dry(fig_root, exp_root)
    plot_fold_accuracy_cut(fig_root, exp_root)
    plot_classwise_accuracy_bar(fig_root, exp_root)
    plot_sequential_performance(fig_root, exp_root)

    # Confusion matrices (requires raw data, may fail if not available)
    try:
        base_cm1, my_cm1 = compute_confusions_task1(exp_root)
        plot_confusions_task1(fig_root, base_cm1, my_cm1)

        base_cm23, my_cm23 = compute_confusions_task23(exp_root)
        plot_confusions_task23(fig_root, base_cm23, my_cm23)
    except FileNotFoundError as e:
        print(f"[WARN] Skipping confusion matrices (raw data not available): {e}")

    print(f"Figures saved to: {fig_root}")


if __name__ == "__main__":
    main()
