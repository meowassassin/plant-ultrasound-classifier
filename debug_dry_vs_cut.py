"""
Debug why tomato dry vs cut confusion matrix is empty
"""
from pathlib import Path
import pandas as pd
from src.training.common import get_project_paths
from src.datasets.plantsounds import scan_plantsounds, make_task_lopo_splits, make_label

project_root, data_root = get_project_paths()
exp_root = project_root / "experiments"

task_name = "task1_tomato_dry_vs_cut"
csv_path = exp_root / "task1_baseline" / f"{task_name}_lopo_results.csv"

# Load CSV
df = pd.read_csv(csv_path)
print(f"Total folds in CSV: {len(df) - 1}")  # -1 for mean row

# Load raw data to check labels
metas = scan_plantsounds(data_root)
splits = make_task_lopo_splits(metas, task_name)

print(f"\nTotal LOPO splits: {len(splits)}")
print("\nChecking class distribution in each fold:")

single_class_count = 0
both_class_count = 0

for fold_idx, (train_idx, test_idx) in enumerate(splits):
    labels = [make_label(metas[i], task_name) for i in test_idx]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if fold_idx < 10 or (n_pos == 0 or n_neg == 0):
        status = "SINGLE CLASS" if (n_pos == 0 or n_neg == 0) else "BOTH CLASSES"
        print(f"  Fold {fold_idx}: n_pos={n_pos}, n_neg={n_neg} -> {status}")

    if n_pos == 0 or n_neg == 0:
        single_class_count += 1
    else:
        both_class_count += 1

print(f"\n" + "=" * 70)
print(f"Summary:")
print(f"  Folds with BOTH classes: {both_class_count}")
print(f"  Folds with SINGLE class: {single_class_count}")
print(f"  Total folds: {len(splits)}")
print(f"\nThis is why confusion matrix is empty!")
print(f"Current code skips single-class folds, so if all folds have single class,")
print(f"the confusion matrix becomes empty (sum=0).")
print("=" * 70)
