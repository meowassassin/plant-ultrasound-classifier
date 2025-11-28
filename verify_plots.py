# verify_plots.py
"""
Script to verify data consistency between Boxplot and Paired Scatter
"""
import pandas as pd
import numpy as np
from pathlib import Path


def verify_task1_data():
    """Data verification for Task1's 4 tasks"""

    exp_root = Path("experiments")

    tasks = [
        "task1_tomato_dry_vs_cut",
        "task1_tobacco_dry_vs_cut",
        "task1_dry_tomato_vs_tobacco",
        "task1_cut_tomato_vs_tobacco",
    ]

    print("=" * 80)
    print("Task1 Data Verification")
    print("=" * 80)

    for task in tasks:
        print(f"\n{'='*80}")
        print(f"Task: {task}")
        print(f"{'='*80}")

        # Baseline
        base_path = exp_root / "task1_baseline" / f"{task}_lopo_results.csv"
        if base_path.exists():
            df_base = pd.read_csv(base_path)
            df_base_folds = df_base[df_base["fold"] != "mean"].copy()

            # Detect columns
            if "best_bal_acc" in df_base_folds.columns:
                bal_col = "best_bal_acc"
            elif "best_test_bal_acc" in df_base_folds.columns:
                bal_col = "best_test_bal_acc"
            else:
                print(f"  [ERROR] Cannot find balanced accuracy column in {base_path}")
                continue

            base_bal = df_base_folds[bal_col].to_numpy()

            print(f"\n  Baseline:")
            print(f"    File: {base_path.name}")
            print(f"    Column: {bal_col}")
            print(f"    N folds: {len(base_bal)}")
            print(f"    Mean: {np.mean(base_bal):.4f}")
            print(f"    Std: {np.std(base_bal, ddof=1):.4f}")
            print(f"    Min: {np.min(base_bal):.4f}")
            print(f"    Max: {np.max(base_bal):.4f}")
            print(f"    First 5 values: {base_bal[:5]}")
        else:
            print(f"  [WARN] Baseline file not found: {base_path}")

        # MyModel
        my_path = exp_root / "task1_my_model" / f"{task}_lopo_results.csv"
        if my_path.exists():
            df_my = pd.read_csv(my_path)
            df_my_folds = df_my[df_my["fold"] != "mean"].copy()

            # Detect columns
            if "best_bal_acc" in df_my_folds.columns:
                bal_col = "best_bal_acc"
            elif "best_test_bal_acc" in df_my_folds.columns:
                bal_col = "best_test_bal_acc"
            else:
                print(f"  [ERROR] Cannot find balanced accuracy column in {my_path}")
                continue

            my_bal = df_my_folds[bal_col].to_numpy()

            print(f"\n  MyModel:")
            print(f"    File: {my_path.name}")
            print(f"    Column: {bal_col}")
            print(f"    N folds: {len(my_bal)}")
            print(f"    Mean: {np.mean(my_bal):.4f}")
            print(f"    Std: {np.std(my_bal, ddof=1):.4f}")
            print(f"    Min: {np.min(my_bal):.4f}")
            print(f"    Max: {np.max(my_bal):.4f}")
            print(f"    First 5 values: {my_bal[:5]}")

            # Check if fold counts match
            if len(base_bal) != len(my_bal):
                print(f"\n  [WARN] Fold count mismatch! Baseline: {len(base_bal)}, MyModel: {len(my_bal)}")
        else:
            print(f"  [WARN] MyModel file not found: {my_path}")


def verify_task23_data():
    """Task2/3 data verification"""

    exp_root = Path("experiments")

    print("\n\n" + "=" * 80)
    print("Task2/3 Data Verification")
    print("=" * 80)

    # Task2
    print(f"\n{'='*80}")
    print("Task2: plant vs empty pot")
    print(f"{'='*80}")

    t2_base_path = exp_root / "task2_baseline" / "task2_plant_vs_empty_lopo_results.csv"
    t2_my_path = exp_root / "task2_my_model" / "task2_plant_vs_empty_lopo_results_labeled1.0.csv"

    if t2_base_path.exists():
        df = pd.read_csv(t2_base_path)
        df_folds = df[df["fold"] != "mean"].copy()

        if "best_bal_acc" in df_folds.columns:
            bal_col = "best_bal_acc"
        elif "best_test_bal_acc" in df_folds.columns:
            bal_col = "best_test_bal_acc"
        else:
            bal_col = "bal_acc"

        bal = df_folds[bal_col].to_numpy()
        print(f"\n  Baseline:")
        print(f"    N folds: {len(bal)}")
        print(f"    Mean: {np.mean(bal):.4f}")
        print(f"    Std: {np.std(bal, ddof=1):.4f}")

    if t2_my_path.exists():
        df = pd.read_csv(t2_my_path)
        df_folds = df[df["fold"] != "mean"].copy()

        if "best_bal_acc" in df_folds.columns:
            bal_col = "best_bal_acc"
        elif "best_test_bal_acc" in df_folds.columns:
            bal_col = "best_test_bal_acc"
        else:
            bal_col = "bal_acc"

        bal = df_folds[bal_col].to_numpy()
        print(f"\n  MyModel:")
        print(f"    N folds: {len(bal)}")
        print(f"    Mean: {np.mean(bal):.4f}")
        print(f"    Std: {np.std(bal, ddof=1):.4f}")

    # Task3
    print(f"\n{'='*80}")
    print("Task3: tomato vs greenhouse noise")
    print(f"{'='*80}")

    t3_base_path = exp_root / "task3_baseline" / "task3_tomato_vs_greenhouse_lopo_results.csv"
    t3_my_path = exp_root / "task3_my_model" / "task3_tomato_vs_greenhouse_lopo_results_labeled1.0.csv"

    if t3_base_path.exists():
        df = pd.read_csv(t3_base_path)
        df_folds = df[df["fold"] != "mean"].copy()

        if "best_bal_acc" in df_folds.columns:
            bal_col = "best_bal_acc"
        elif "best_test_bal_acc" in df_folds.columns:
            bal_col = "best_test_bal_acc"
        else:
            bal_col = "bal_acc"

        bal = df_folds[bal_col].to_numpy()
        print(f"\n  Baseline:")
        print(f"    N folds: {len(bal)}")
        print(f"    Mean: {np.mean(bal):.4f}")
        print(f"    Std: {np.std(bal, ddof=1):.4f}")

    if t3_my_path.exists():
        df = pd.read_csv(t3_my_path)
        df_folds = df[df["fold"] != "mean"].copy()

        if "best_bal_acc" in df_folds.columns:
            bal_col = "best_bal_acc"
        elif "best_test_bal_acc" in df_folds.columns:
            bal_col = "best_test_bal_acc"
        else:
            bal_col = "bal_acc"

        bal = df_folds[bal_col].to_numpy()
        print(f"\n  MyModel:")
        print(f"    N folds: {len(bal)}")
        print(f"    Mean: {np.mean(bal):.4f}")
        print(f"    Std: {np.std(bal, ddof=1):.4f}")


if __name__ == "__main__":
    verify_task1_data()
    verify_task23_data()

    print("\n" + "=" * 80)
    print("Verification complete!")
    print("=" * 80)
