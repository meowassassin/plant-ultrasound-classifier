# src/training/train_task2.py
from src.training.common import run_lopo_for_task


def main():
    """
    Task2: Plant vs Empty Pot (박스 & 온실 잡음 포함)
    """
    run_lopo_for_task(
        "task2_plant_vs_empty",
        experiments_subdir="task2_baseline",
    )


if __name__ == "__main__":
    main()
