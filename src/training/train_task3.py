# src/training/train_task3.py
from src.training.common import run_lopo_for_task


def main():
    """
    Task3: Tomato drought (greenhouse) vs Greenhouse background noises
           = 온실 환경에서의 일반화 실험
    """
    run_lopo_for_task(
        "task3_tomato_vs_greenhouse",
        experiments_subdir="task3_baseline",
    )


if __name__ == "__main__":
    main()
