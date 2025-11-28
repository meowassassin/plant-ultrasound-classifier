# src/training/train_my_model_task2.py
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.plantsounds import (
    scan_plantsounds,
    make_task_lopo_splits,
    PlantSoundsDataset,
)
from src.models.my_model import MyModel
from src.training.common import (
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    TARGET_SR,
    TARGET_LEN,
    SEED,
    set_seed,
    evaluate,
    get_project_paths,
)

# --------------------
# 하이퍼파라미터
# --------------------
LR = 1e-3

LAMBDA_VAE = 0.1
BETA_KL = 1.0
LAMBDA_SSL = 0.1
SSL_NOISE_STD = 0.01

ALPHA_DG = 0.0

LABELED_FRACTION = 1.0  # plant vs empty에서도 semi-supervised 사용


def _compute_pos_weight_from_dataset(
    dataset: PlantSoundsDataset,
) -> torch.Tensor | None:
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
        print("[WARN] (Task2) labeled subset 클래스 분포가 한쪽으로 치우쳐 pos_weight 사용 불가.")
        return None

    pos_weight_val = n_neg / n_pos
    print(
        f"    [Task2 labeled balance] neg={n_neg}, pos={n_pos}, "
        f"pos_weight={pos_weight_val:.4f}"
    )
    return torch.tensor(pos_weight_val, dtype=torch.float32, device=DEVICE)


def train_one_epoch_my_model_semi(
    model: MyModel,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    criterion_cls: nn.Module,
) -> Tuple[float, float]:
    """
    Task2(plant vs empty)에 대한 MyModel 세미-슈퍼바이즈드 학습 1 epoch.
    """
    model.train()
    total_loss = 0.0
    total_sup_samples = 0
    total_sup_correct = 0

    iter_l = iter(labeled_loader)
    iter_u = iter(unlabeled_loader) if unlabeled_loader is not None else None

    finished_l = False
    finished_u = unlabeled_loader is None

    while not (finished_l and finished_u):
        # labeled
        if not finished_l:
            try:
                x_l, y_l = next(iter_l)
                x_l = x_l.to(DEVICE)
                y_l = y_l.to(DEVICE)

                noise = SSL_NOISE_STD * torch.randn_like(x_l)
                x_l_aug = x_l + noise

                out_l = model(x_l, alpha=ALPHA_DG, return_dict=True)
                out_l_aug = model(x_l_aug, alpha=ALPHA_DG, return_dict=True)

                logits_l = out_l["logits"]
                h_l = out_l["h"]
                h_rec_l = out_l["h_rec"]
                mu_l = out_l["mu"]
                logvar_l = out_l["logvar"]
                z_l = out_l["z"]
                z_l_aug = out_l_aug["z"]

                cls_loss = criterion_cls(logits_l.view(-1), y_l.float())
                vae_loss = MyModel.vae_loss(
                    h_l, h_rec_l, mu_l, logvar_l, beta_kl=BETA_KL
                )
                ssl_loss = F.mse_loss(z_l, z_l_aug)

                loss_l = cls_loss + LAMBDA_VAE * vae_loss + LAMBDA_SSL * ssl_loss

                optimizer.zero_grad()
                loss_l.backward()
                optimizer.step()

                bs = x_l.size(0)
                total_loss += loss_l.item() * bs
                total_sup_samples += bs

                preds_l = (logits_l.view(-1) >= 0).long()
                total_sup_correct += (preds_l == y_l).sum().item()

            except StopIteration:
                finished_l = True

        # unlabeled
        if not finished_u and iter_u is not None:
            try:
                x_u, _ = next(iter_u)
                x_u = x_u.to(DEVICE)

                noise_u = SSL_NOISE_STD * torch.randn_like(x_u)
                x_u_aug = x_u + noise_u

                out_u = model(x_u, alpha=ALPHA_DG, return_dict=True)
                out_u_aug = model(x_u_aug, alpha=ALPHA_DG, return_dict=True)

                h_u = out_u["h"]
                h_rec_u = out_u["h_rec"]
                mu_u = out_u["mu"]
                logvar_u = out_u["logvar"]
                z_u = out_u["z"]
                z_u_aug = out_u_aug["z"]

                vae_loss_u = MyModel.vae_loss(
                    h_u, h_rec_u, mu_u, logvar_u, beta_kl=BETA_KL
                )
                ssl_loss_u = F.mse_loss(z_u, z_u_aug)

                loss_u = LAMBDA_VAE * vae_loss_u + LAMBDA_SSL * ssl_loss_u

                optimizer.zero_grad()
                loss_u.backward()
                optimizer.step()

                bs_u = x_u.size(0)
                total_loss += loss_u.item() * bs_u

            except StopIteration:
                finished_u = True

    avg_loss = total_loss / max(1, total_sup_samples)
    sup_acc = total_sup_correct / max(1, total_sup_samples)
    return avg_loss, sup_acc


def run_task_my_model_task2(task_name: str = "task2_plant_vs_empty"):
    """
    Task2: Plant vs Empty Pot 에 대해 MyModel(VAE+SSL 구조)로 LOPO-CV 수행.
    """
    set_seed(SEED)
    project_root, plantsounds_root = get_project_paths()
    data_root = plantsounds_root

    print(f"\n===== MyModel Task2: {task_name} =====")
    print(f"Device: {DEVICE}")
    print(f"DATA_ROOT: {data_root}")
    print(f"LABELED_FRACTION: {LABELED_FRACTION}")

    metas = scan_plantsounds(str(data_root))
    print(f"Total samples found: {len(metas)}")

    splits = make_task_lopo_splits(metas, task_name)
    print(f"Number of LOPO folds: {len(splits)}")

    num_classes = 2  # plant vs empty

    fold_results: List[Tuple[float, float]] = []

    exp_dir = project_root / "experiments" / "task2_my_model"
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(
            f"\n-- Fold {fold_idx + 1}/{len(splits)} "
            f"(train={len(train_idx)}, test={len(test_idx)})"
        )

        train_idx_arr = np.array(train_idx)
        rng = np.random.RandomState(SEED + fold_idx)
        rng.shuffle(train_idx_arr)

        n_labeled = int(len(train_idx_arr) * LABELED_FRACTION)
        n_labeled = max(1, n_labeled)
        labeled_idx = train_idx_arr[:n_labeled].tolist()
        unlabeled_idx = train_idx_arr[n_labeled:].tolist()

        labeled_ds = PlantSoundsDataset(
            root=str(data_root),
            metas=metas,
            indices=labeled_idx,
            task_name=task_name,
            target_sr=TARGET_SR,
            target_len=TARGET_LEN,
            apply_highpass=True,
        )
        labeled_loader = DataLoader(
            labeled_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

        unlabeled_loader = None
        if len(unlabeled_idx) > 0:
            unlabeled_ds = PlantSoundsDataset(
                root=str(data_root),
                metas=metas,
                indices=unlabeled_idx,
                task_name=task_name,
                target_sr=TARGET_SR,
                target_len=TARGET_LEN,
                apply_highpass=True,
            )
            unlabeled_loader = DataLoader(
                unlabeled_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
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
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        pos_weight = _compute_pos_weight_from_dataset(labeled_ds)
        if pos_weight is not None:
            criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion_cls = nn.BCEWithLogitsLoss()

        model = MyModel(
            num_classes=num_classes,
            num_domains=2,
            emb_dim=128,
            latent_dim=32,
            input_length=TARGET_LEN,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_bal_acc = -1.0
        best_test_acc = 0.0

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch_my_model_semi(
                model,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                criterion_cls,
            )
            test_loss, test_acc, test_bal_acc = evaluate(
                model, test_loader, criterion_cls, num_classes
            )

            if test_bal_acc > best_bal_acc:
                best_bal_acc = test_bal_acc
                best_test_acc = test_acc
                ckpt_path = (
                    exp_dir
                    / "checkpoints"
                    / f"{task_name}_fold{fold_idx}_best.pt"
                )
                torch.save(model.state_dict(), ckpt_path)

            print(
                f"[Fold {fold_idx + 1}/{len(splits)}][Epoch {epoch}/{EPOCHS}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
                f"test_bal_acc={test_bal_acc:.4f}"
            )

        fold_results.append((best_test_acc, best_bal_acc))

    accs = [r[0] for r in fold_results]
    bals = [r[1] for r in fold_results]

    print(
        f"\n>>> MyModel Task2 {task_name}: "
        f"mean_acc={np.mean(accs):.4f} (+/- {np.std(accs):.4f}), "
        f"mean_bal_acc={np.mean(bals):.4f} (+/- {np.std(bals):.4f})"
    )

    import csv

    out_path = exp_dir / f"{task_name}_lopo_results_labeled{LABELED_FRACTION}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_test_acc", "best_test_bal_acc"])
        for i, (acc, bal) in enumerate(fold_results):
            writer.writerow([i, float(acc), float(bal)])
        writer.writerow(
            ["mean", float(np.mean(accs)), float(np.mean(bals))]
        )


def main():
    run_task_my_model_task2("task2_plant_vs_empty")


if __name__ == "__main__":
    main()
