"""
Debug script to check which confusion matrices are being generated
"""
from pathlib import Path
from src.training.common import get_project_paths
from src.analysis.plot_results import compute_confusions_task1, TASK1_KEYS, TASK1_LABELS

project_root, _ = get_project_paths()
exp_root = project_root / "experiments"

try:
    base_cm, my_cm = compute_confusions_task1(exp_root)

    print("=" * 70)
    print("CONFUSION MATRIX VALIDATION")
    print("=" * 70)

    for i, key in enumerate(TASK1_KEYS):
        label = TASK1_LABELS[i]
        base_sum = base_cm[key].sum()
        my_sum = my_cm[key].sum()

        print(f"\n[{i+1}] {label} ({key}):")
        print(f"    Baseline CM sum: {base_sum:.0f}")
        print(f"    MyModel CM sum:  {my_sum:.0f}")

        if base_sum > 0 and my_sum > 0:
            print(f"    ✓ WILL BE PLOTTED")
            print(f"    Baseline CM:\n{base_cm[key]}")
            print(f"    MyModel CM:\n{my_cm[key]}")
        else:
            print(f"    ✗ SKIPPED (empty confusion matrix)")

    print("\n" + "=" * 70)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
