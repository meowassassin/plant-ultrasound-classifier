# src/training/common.py
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.plantsounds import (
    scan_plantsounds,
    make_task_lopo_splits,
    PlantSoundsDataset,
)
from src.models.baseline_cnn import BaselineCNN


# ----------------------
# 전역 설정 값
# ----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64

# 논문 Keras 코드는 num_epochs = 50 이지만,
# 여기서 EPOCHS 값은 실험 편의에 맞춰 자유롭게 바꿔 써도 된다.
EPOCHS = 10

LR = 5e-4

TARGET_SR = 500000
TARGET_LEN = 1000  # 2 ms @ 500 kHz -> 1000 샘플

SEED = 42


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_bce_loss(criterion: nn.Module) -> bool:
    return isinstance(criterion, (nn.BCEWithLogitsLoss, nn.BCELoss))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    공통 1 epoch 학습 루프.

    - BCEWithLogitsLoss:
        logits: [B] 또는 [B, 1]
        y:      [B] (0/1)
    - CrossEntropyLoss (혹시 쓸 일이 있다면):
        logits: [B, C]
        y:      [B] (0 ~ C-1)
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    use_bce = _is_bce_loss(criterion)

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)

        if use_bce:
            logits_flat = logits.view(-1)
            loss = criterion(logits_flat, y.float())
        else:
            loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(1, n_samples)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
) -> Tuple[float, float, float]:
    """
    공통 평가 루프.

    반환:
        (avg_loss, accuracy, balanced_accuracy)

    - BCEWithLogitsLoss:
        logits: [B] or [B,1]
        pred:   (logits >= 0) -> {0,1}
    - CrossEntropyLoss:
        logits: [B, C]
        pred:   argmax(logits, dim=1)
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    correct = 0

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    use_bce = _is_bce_loss(criterion)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            if use_bce:
                logits_flat = logits.view(-1)
                loss = criterion(logits_flat, y.float())
            else:
                loss = criterion(logits, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            # --- 예측 ---
            if use_bce:
                logits_flat = logits.view(-1)
                preds = (logits_flat >= 0).long()
            else:
                if logits.dim() == 1:
                    preds = (logits >= 0).long()
                elif logits.shape[1] == 1:
                    preds = (logits[:, 0] >= 0).long()
                else:
                    preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()

            y_np = y.cpu().numpy().astype(int)
            p_np = preds.cpu().numpy().astype(int)
            for t, p in zip(y_np, p_np):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    conf[int(t), int(p)] += 1

    avg_loss = total_loss / max(1, n_samples)
    acc = correct / max(1, n_samples)

    # --- Balanced accuracy (class-wise recall의 평균) ---
    recalls: List[float] = []
    for c in range(num_classes):
        tp = conf[c, c]
        fn = conf[c, :].sum() - tp
        denom = tp + fn
        if denom > 0:
            recalls.append(tp / denom)

    bal_acc = float(sum(recalls) / len(recalls)) if recalls else 0.0
    return avg_loss, acc, bal_acc


def get_project_paths() -> Tuple[Path, Path]:
    """
    (project_root, plantsounds_root)를 반환.

    common.py 위치가
        <project_root>/src/training/common.py
    라고 가정하면,
        project_root = parents[2]
    """
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]
    plantsounds_root = project_root / "data" / "raw" / "PlantSounds"
    return project_root, plantsounds_root


def _compute_pos_weight_from_dataset(
    dataset: PlantSoundsDataset,
) -> torch.Tensor | None:
    """
    Keras의 class_weight='balanced'를 흉내 내기 위해,
    train set의 label 분포로부터 pos_weight = n_neg / n_pos 를 계산.

    BCEWithLogitsLoss(pos_weight=pos_weight) 에서
    양성 샘플에만 pos_weight가 곱해진다.
    """
    labels: List[int] = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        if isinstance(y, torch.Tensor):
            y = int(y.item())
        labels.append(int(y))

    labels_np = np.array(labels, dtype=int)
    n_pos = int((labels_np == 1).sum())
    n_neg = int((labels_np == 0).sum())

    if n_pos == 0 or n_neg == 0:
        print("[WARN] 클래스 분포가 한쪽으로 치우쳐 pos_weight를 사용할 수 없습니다.")
        return None

    pos_weight_val = n_neg / n_pos
    print(
        f"    [Class balance] neg={n_neg}, pos={n_pos}, "
        f"pos_weight={pos_weight_val:.4f}"
    )
    return torch.tensor(pos_weight_val, dtype=torch.float32, device=DEVICE)


def run_lopo_for_task(
    task_name: str,
    experiments_subdir: str,
):
    """
    Khait CNN(BaselineCNN)을 이용한 LOPO-CV 실험 (논문 baseline 재현용).

    - LOPO(split): make_task_lopo_splits()
    - 모델: BaselineCNN (CNN_sounds_classifier.py와 동일 구조)
    - 손실: BCEWithLogitsLoss + pos_weight(≈ class_weight='balanced')
    - 옵티마이저: Adam(lr=1e-3)
    """
    set_seed(SEED)
    project_root, plantsounds_root = get_project_paths()
    data_root = plantsounds_root

    print(f"\n===== Task: {task_name} =====")
    print(f"Device: {DEVICE}")
    print(f"DATA_ROOT: {data_root}")

    metas = scan_plantsounds(str(data_root))
    print(f"Total samples found: {len(metas)}")

    splits = make_task_lopo_splits(metas, task_name)
    print(f"Number of LOPO folds: {len(splits)}")

    num_classes = 2  # Task1~3 모두 binary

    fold_results: List[Tuple[float, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(
            f"\n-- Fold {fold_idx + 1}/{len(splits)} "
            f"(train={len(train_idx)}, test={len(test_idx)})"
        )

        # --- Dataset 생성 ---
        train_ds = PlantSoundsDataset(
            root=str(data_root),
            metas=metas,
            indices=train_idx,
            task_name=task_name,
            target_sr=TARGET_SR,
            target_len=TARGET_LEN,
            apply_highpass=True,
        )
        test_ds = PlantSoundsDataset(
            root=str(data_root),
            metas=metas,
            indices=test_idx,
            task_name=task_name,
            target_sr=TARGET_SR,
            target_len=TARGET_LEN,
            apply_highpass=True,
        )

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # --- class_weight ≈ pos_weight 계산 ---
        pos_weight = _compute_pos_weight_from_dataset(train_ds)

        # --- 모델 / 손실 / 옵티마이저 ---
        model = BaselineCNN(input_length=TARGET_LEN).to(DEVICE)

        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_bal_acc = -1.0
        best_acc = 0.0

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            test_loss, test_acc, test_bal_acc = evaluate(
                model, test_loader, criterion, num_classes
            )

            if test_bal_acc > best_bal_acc:
                best_bal_acc = test_bal_acc
                best_acc = test_acc

                ckpt_dir = (
                    project_root
                    / "experiments"
                    / experiments_subdir
                    / "checkpoints"
                )
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"{task_name}_fold{fold_idx}_best.pt"
                torch.save(model.state_dict(), ckpt_path)

            print(
                f"[Fold {fold_idx + 1}/{len(splits)}][Epoch {epoch}/{EPOCHS}] "
                f"train_loss={train_loss:.4f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
                f"test_bal_acc={test_bal_acc:.4f}"
            )

        fold_results.append((best_acc, best_bal_acc))

    # --- Fold별 best 결과 요약 + CSV 저장 ---
    accs = [r[0] for r in fold_results]
    bals = [r[1] for r in fold_results]

    print(
        f"\n>>> Task {task_name}: "
        f"mean_acc={np.mean(accs):.4f} (+/- {np.std(accs):.4f}), "
        f"mean_bal_acc={np.mean(bals):.4f} (+/- {np.std(bals):.4f})"
    )

    import csv

    exp_dir = project_root / "experiments" / experiments_subdir
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_path = exp_dir / f"{task_name}_lopo_results.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_acc", "best_bal_acc"])
        for i, (acc, bal) in enumerate(fold_results):
            writer.writerow([i, float(acc), float(bal)])
        writer.writerow(["mean", float(np.mean(accs)), float(np.mean(bals))])
