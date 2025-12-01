# src/training/train_my_model_task4.py
from typing import List, Tuple, Optional

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
from src.models.proposed_cnn_model import MyModel
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
LR = 5e-4

LAMBDA_VAE = 0.1
BETA_KL = 1.0
LAMBDA_SSL = 0.1
LAMBDA_DG = 0.1
SSL_NOISE_STD = 0.01

# 각 fold train에서 이 비율만 supervised label로 사용 (나머지는 unlabeled)
LABELED_FRACTION = 0.5


# --------------------
# 유틸: pos_weight 계산
# --------------------
def _compute_pos_weight_from_dataset(
    dataset: PlantSoundsDataset,
    tag: str,
) -> Optional[torch.Tensor]:
    labels: List[int] = []
    for i in range(len(dataset)):
        item = dataset[i]
        if isinstance(item, tuple):
            y = item[1]
        else:
            raise RuntimeError("Dataset __getitem__ unexpected format")

        if isinstance(y, torch.Tensor):
            y_val = int(y.item())
        else:
            y_val = int(y)
        labels.append(y_val)

    labels_np = np.array(labels, dtype=int)
    n_pos = int((labels_np == 1).sum())
    n_neg = int((labels_np == 0).sum())

    if n_pos == 0 or n_neg == 0:
        print(f"[WARN] ({tag}) labeled subset 클래스가 한쪽으로만 있습니다. pos_weight 비활성.")
        return None

    pos_weight_val = n_neg / n_pos
    print(
        f"    [{tag} labeled balance] neg={n_neg}, pos={n_pos}, "
        f"pos_weight={pos_weight_val:.4f}"
    )
    return torch.tensor(pos_weight_val, dtype=torch.float32, device=DEVICE)


# --------------------
# 1 epoch 학습 (Supervised + SSL + DG + Unlabeled)
# --------------------
def train_one_epoch_my_model_ssl_dg(
    model: MyModel,
    labeled_loader: DataLoader,
    unlabeled_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    pos_weight: Optional[torch.Tensor],
) -> Tuple[float, float]:
    model.train()

    if pos_weight is not None:
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        bce = nn.BCEWithLogitsLoss()

    ce_domain = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    # ---------- supervised part ----------
    for batch in labeled_loader:
        if len(batch) == 3:
            x, y, d = batch
            d = d.to(DEVICE)
        else:
            x, y = batch
            d = None

        x = x.to(DEVICE)            # [B, 1, L]
        y = y.to(DEVICE).float()    # [B]

        # 간단 SSL: input noise
        noise = SSL_NOISE_STD * torch.randn_like(x)
        x_aug = x + noise

        out = model(x, alpha=1.0, return_dict=True)
        out_aug = model(x_aug, alpha=1.0, return_dict=True)

        logits = out["logits"].view(-1)  # [B]
        h = out["h"]
        h_rec = out["h_rec"]
        mu = out["mu"]
        logvar = out["logvar"]
        z = out["z"]
        z_aug = out_aug["z"]
        domain_logits = out["domain_logits"]

        # 분류 손실
        cls_loss = bce(logits, y)

        # VAE 손실
        vae_loss = MyModel.vae_loss(h, h_rec, mu, logvar, beta_kl=BETA_KL)

        # SSL 손실 (z 일관성)
        ssl_loss = F.mse_loss(z, z_aug)

        # DG 손실 (domain classifier)
        if d is not None:
            dg_loss = ce_domain(domain_logits, d)
        else:
            dg_loss = torch.tensor(0.0, device=DEVICE)

        loss = (
            cls_loss
            + LAMBDA_VAE * vae_loss
            + LAMBDA_SSL * ssl_loss
            + LAMBDA_DG * dg_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        preds = (logits >= 0).long()
        total_correct += (preds == y.long()).sum().item()

    # ---------- unlabeled part ----------
    if unlabeled_loader is not None:
        for batch in unlabeled_loader:
            if len(batch) == 3:
                x_u, _, d_u = batch
                d_u = d_u.to(DEVICE)
            else:
                x_u, _ = batch
                d_u = None

            x_u = x_u.to(DEVICE)

            noise_u = SSL_NOISE_STD * torch.randn_like(x_u)
            x_u_aug = x_u + noise_u

            out_u = model(x_u, alpha=1.0, return_dict=True)
            out_u_aug = model(x_u_aug, alpha=1.0, return_dict=True)

            h_u = out_u["h"]
            h_rec_u = out_u["h_rec"]
            mu_u = out_u["mu"]
            logvar_u = out_u["logvar"]
            z_u = out_u["z"]
            z_u_aug = out_u_aug["z"]
            domain_logits_u = out_u["domain_logits"]

            vae_loss_u = MyModel.vae_loss(
                h_u, h_rec_u, mu_u, logvar_u, beta_kl=BETA_KL
            )
            ssl_loss_u = F.mse_loss(z_u, z_u_aug)

            if d_u is not None:
                dg_loss_u = ce_domain(domain_logits_u, d_u)
            else:
                dg_loss_u = torch.tensor(0.0, device=DEVICE)

            loss_u = (
                LAMBDA_VAE * vae_loss_u
                + LAMBDA_SSL * ssl_loss_u
                + LAMBDA_DG * dg_loss_u
            )

            optimizer.zero_grad()
            loss_u.backward()
            optimizer.step()

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


# --------------------
# Task4 실행 (task_name 하나에 대해)
# --------------------
def run_task_my_model_task4(task_name: str):
    """
    Task4: Task1 계열 task (4개 중 하나)에 대해,
    각 fold의 train에서 LABELED_FRACTION만 supervised로 쓰고
    나머지는 unlabeled로 쓰는 semi-supervised + SSL + DG 실험.
    """
    set_seed(SEED)
    project_root, plantsounds_root = get_project_paths()
    data_root = plantsounds_root

    print(f"\n===== MyModel Task4: semi-supervised on {task_name} =====")
    print(f"Device: {DEVICE}")
    print(f"DATA_ROOT: {data_root}")
    print(f"LABELED_FRACTION: {LABELED_FRACTION}")

    metas = scan_plantsounds(str(data_root))
    print(f"Total samples found: {len(metas)}")

    # LOPO split은 해당 task_name 기준으로
    splits = make_task_lopo_splits(metas, task_name)
    print(f"Number of LOPO folds: {len(splits)}")

    num_classes = 2  # Task1 계열은 모두 binary

    exp_dir = project_root / "experiments" / "task4_my_model"
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    fold_results: List[Tuple[float, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(
            f"\n-- Fold {fold_idx + 1}/{len(splits)} "
            f"(train={len(train_idx)}, test={len(test_idx)})"
        )

        # ----- train -> labeled / unlabeled split -----
        rng = np.random.RandomState(SEED + fold_idx)
        train_arr = np.array(train_idx)
        rng.shuffle(train_arr)

        n_train = len(train_arr)
        n_labeled = int(round(LABELED_FRACTION * n_train))
        if n_labeled < 1:
            n_labeled = 1
        labeled_indices = train_arr[:n_labeled].tolist()
        unlabeled_indices = train_arr[n_labeled:].tolist()

        # labeled (return_domain=True → DG용 domain label 포함)
        train_labeled_ds = PlantSoundsDataset(
            root=str(data_root),
            metas=metas,
            indices=labeled_indices,
            task_name=task_name,
            target_sr=TARGET_SR,
            target_len=TARGET_LEN,
            apply_highpass=True,
            return_domain=True,
        )
        # unlabeled
        unlabeled_ds = (
            PlantSoundsDataset(
                root=str(data_root),
                metas=metas,
                indices=unlabeled_indices,
                task_name=task_name,
                target_sr=TARGET_SR,
                target_len=TARGET_LEN,
                apply_highpass=True,
                return_domain=True,
            )
            if unlabeled_indices
            else None
        )
        # test (domain label 필요 X)
        test_ds = PlantSoundsDataset(
            root=str(data_root),
            metas=metas,
            indices=test_idx,
            task_name=task_name,
            target_sr=TARGET_SR,
            target_len=TARGET_LEN,
            apply_highpass=True,
            return_domain=False,
        )

        # ----- pos_weight 계산 -----
        pos_weight = _compute_pos_weight_from_dataset(
            train_labeled_ds, f"Task4-{task_name}"
        )

        train_labeled_loader = DataLoader(
            train_labeled_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        unlabeled_loader = (
            DataLoader(
                unlabeled_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
            )
            if unlabeled_ds is not None
            else None
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # ----- 모델/옵티마이저 -----
        model = MyModel(
            num_domains=2,           # box / greenhouse
            emb_dim=128,
            latent_dim=32,
            input_length=TARGET_LEN,
        ).to(DEVICE)

        if pos_weight is not None:
            criterion_eval = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion_eval = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_bal_acc = -1.0
        best_test_acc = 0.0

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch_my_model_ssl_dg(
                model,
                train_labeled_loader,
                unlabeled_loader,
                optimizer,
                pos_weight,
            )
            test_loss, test_acc, test_bal_acc = evaluate(
                model, test_loader, criterion_eval, num_classes
            )

            if test_bal_acc > best_bal_acc:
                best_bal_acc = test_bal_acc
                best_test_acc = test_acc
                ckpt_path = (
                    exp_dir
                    / "checkpoints"
                    / f"task4_{task_name}_fold{fold_idx}_best.pt"
                )
                torch.save(model.state_dict(), ckpt_path)

            print(
                f"[Fold {fold_idx + 1}/{len(splits)}][Epoch {epoch}/{EPOCHS}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
                f"test_bal_acc={test_bal_acc:.4f}"
            )

        fold_results.append((best_test_acc, best_bal_acc))

    # ----- fold 평균 결과 + CSV -----
    accs = [r[0] for r in fold_results]
    bals = [r[1] for r in fold_results]

    print(
        f"\n>>> MyModel Task4 on {task_name}: "
        f"mean_best_acc={np.mean(accs):.4f} (+/- {np.std(accs):.4f}), "
        f"mean_best_bal_acc={np.mean(bals):.4f} (+/- {np.std(bals):.4f})"
    )

    import csv

    out_path = (
        project_root
        / "experiments"
        / "task4_my_model"
        / f"task4_{task_name}_lopo_results.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_test_acc", "best_test_bal_acc"])
        for i, (acc, bal) in enumerate(fold_results):
            writer.writerow([i, float(acc), float(bal)])
        writer.writerow(
            ["mean", float(np.mean(accs)), float(np.mean(bals))]
        )


# --------------------
# main: Task1 네 개 실험을 Task4 설정으로 모두 실행
# --------------------
def main():
    task_list = [
        "task1_tomato_dry_vs_cut",
        "task1_tobacco_dry_vs_cut",
        "task1_dry_tomato_vs_tobacco",
        "task1_cut_tomato_vs_tobacco",
    ]

    for t in task_list:
        run_task_my_model_task4(t)


if __name__ == "__main__":
    main()
