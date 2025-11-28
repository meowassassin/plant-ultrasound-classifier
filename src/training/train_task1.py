# src/training/train_task1.py
from src.training.common import run_lopo_for_task


def main():
    """
    Task1 (Acoustic box) - 논문 실험과 동일한 네 가지 binary task:

      1) Tomato:  dry vs cut
      2) Tobacco: dry vs cut
      3) Dry:     tomato vs tobacco
      4) Cut:     tomato vs tobacco
    """
    run_lopo_for_task(
        "task1_tomato_dry_vs_cut",
        experiments_subdir="task1_baseline",
    )
    run_lopo_for_task(
        "task1_tobacco_dry_vs_cut",
        experiments_subdir="task1_baseline",
    )
    run_lopo_for_task(
        "task1_dry_tomato_vs_tobacco",
        experiments_subdir="task1_baseline",
    )
    run_lopo_for_task(
        "task1_cut_tomato_vs_tobacco",
        experiments_subdir="task1_baseline",
    )


if __name__ == "__main__":
    main()
