# validate_plot_consistency.py
"""
Detailed validation of data consistency between Boxplot and Paired Scatter
"""
import pandas as pd
import numpy as np
from pathlib import Path


def check_fold_pairing(task_name, exp_root):
    """
    Verify that folds are correctly paired between Baseline and MyModel
    """
    print(f"\n{'='*80}")
    print(f"Checking fold pairing for: {task_name}")
    print(f"{'='*80}")

    base_path = exp_root / "task1_baseline" / f"{task_name}_lopo_results.csv"
    my_path = exp_root / "task1_my_model" / f"{task_name}_lopo_results.csv"

    if not base_path.exists() or not my_path.exists():
        print(f"  [WARN] Missing files")
        return

    df_base = pd.read_csv(base_path)
    df_my = pd.read_csv(my_path)

    # Remove mean row
    df_base_folds = df_base[df_base["fold"] != "mean"].copy()
    df_my_folds = df_my[df_my["fold"] != "mean"].copy()

    # Check fold indices
    base_folds = df_base_folds["fold"].to_numpy()
    my_folds = df_my_folds["fold"].to_numpy()

    print(f"  Baseline folds: {len(base_folds)}")
    print(f"  MyModel folds: {len(my_folds)}")

    if len(base_folds) != len(my_folds):
        print(f"  [ERROR] Fold count mismatch!")
        return False

    if not np.array_equal(base_folds, my_folds):
        print(f"  [ERROR] Fold indices don't match!")
        print(f"    Baseline: {base_folds[:10]}")
        print(f"    MyModel: {my_folds[:10]}")
        return False

    # Detect columns
    if "best_bal_acc" in df_base_folds.columns:
        base_bal_col = "best_bal_acc"
    elif "best_test_bal_acc" in df_base_folds.columns:
        base_bal_col = "best_test_bal_acc"
    else:
        print(f"  [ERROR] Cannot find bal_acc column in baseline")
        return False

    if "best_bal_acc" in df_my_folds.columns:
        my_bal_col = "best_bal_acc"
    elif "best_test_bal_acc" in df_my_folds.columns:
        my_bal_col = "best_test_bal_acc"
    else:
        print(f"  [ERROR] Cannot find bal_acc column in mymodel")
        return False

    base_bal = df_base_folds[base_bal_col].to_numpy()
    my_bal = df_my_folds[my_bal_col].to_numpy()

    print(f"  ✓ Fold indices match")
    print(f"  ✓ Both have {len(base_folds)} folds")

    # Show some example pairs
    print(f"\n  Example fold pairs (fold, baseline_acc, mymodel_acc):")
    for i in range(min(10, len(base_folds))):
        print(f"    Fold {base_folds[i]}: {base_bal[i]:.4f} -> {my_bal[i]:.4f}")

    # Check for anomalies
    print(f"\n  Anomaly check:")

    # Count perfect scores
    base_perfect = np.sum(base_bal == 1.0)
    my_perfect = np.sum(my_bal == 1.0)
    print(f"    Perfect scores (1.0): Baseline={base_perfect}/{len(base_bal)}, MyModel={my_perfect}/{len(my_bal)}")

    # Count zero scores
    base_zero = np.sum(base_bal == 0.0)
    my_zero = np.sum(my_bal == 0.0)
    if base_zero > 0 or my_zero > 0:
        print(f"    Zero scores (0.0): Baseline={base_zero}/{len(base_bal)}, MyModel={my_zero}/{len(my_bal)}")
        print(f"    [WARN] Zero scores detected - might indicate failed folds")

    # Count chance level
    base_chance = np.sum(base_bal == 0.5)
    my_chance = np.sum(my_bal == 0.5)
    if base_chance > len(base_bal) * 0.2 or my_chance > len(my_bal) * 0.2:
        print(f"    Chance-level scores (0.5): Baseline={base_chance}/{len(base_bal)}, MyModel={my_chance}/{len(my_bal)}")
        print(f"    [INFO] Many chance-level scores - might be single-class folds")

    # Check std
    if np.std(my_bal, ddof=1) < 0.01:
        print(f"    [WARN] MyModel std very low ({np.std(my_bal, ddof=1):.4f}) - all values similar?")
        unique_vals = np.unique(my_bal)
        print(f"    Unique MyModel values: {unique_vals[:10]}")

    return True


def main():
    exp_root = Path("experiments")

    tasks = [
        "task1_tomato_dry_vs_cut",
        "task1_tobacco_dry_vs_cut",
        "task1_dry_tomato_vs_tobacco",
        "task1_cut_tomato_vs_tobacco",
    ]

    print("=" * 80)
    print("PLOT DATA CONSISTENCY VALIDATION")
    print("=" * 80)

    all_ok = True
    for task in tasks:
        result = check_fold_pairing(task, exp_root)
        if result is False:
            all_ok = False

    print("\n" + "=" * 80)
    if all_ok:
        print("✓ All validations passed!")
    else:
        print("✗ Some validations failed!")
    print("=" * 80)

    # Additional check: verify that boxplot and scatter use same data
    print("\n\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
For Paired Scatter plots:
  - Each point represents one fold
  - x-axis: Baseline balanced accuracy for that fold
  - y-axis: MyModel balanced accuracy for that fold
  - Points above y=x line: MyModel performs better
  - Points below y=x line: Baseline performs better
  - Same number of points as in boxplot

For Boxplots:
  - Shows distribution of fold-wise accuracies
  - Box: 25th-75th percentile (IQR)
  - Red line: median
  - Whiskers: 1.5*IQR from box edges
  - Outliers: points beyond whiskers
  - Two boxes per task: Baseline (left), MyModel (right)

Common issues to check:
  1. Zero scores (0.0): Might indicate failed training or single-class folds
  2. All scores at 1.0: Suspiciously perfect - check if task is too easy
  3. Many scores at 0.5: Indicates chance-level - single-class folds or model not learning
  4. Mismatch in fold counts: Data loading error
""")


if __name__ == "__main__":
    main()
