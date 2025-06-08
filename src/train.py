import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast # AMP 쓰는건 맥에서 안됨 -- 메인브렌치도 A100에서 돌리도록
from src.augmentation import cutmix_data, mixup_data
from src.utils.display import (
    print_start, print_epoch_summary, print_validation_accuracy,
    print_success, print_warning, progress_bar, count_parameters
)

def train_model(
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: str,
        num_epochs: int = 30,
        base_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        min_delta: float = 1e-4,
        monitor: str = "val_acc",
        restore_best_weights: bool = True,
        use_cutmix: bool = True,
        use_mixup: bool = True,
        cutmix_prob: float = 0.5,
        mixup_prob: float = 0.5,
        cutmix_alpha: float = 1.0,
        mixup_alpha: float = 0.2,
        warmup_epochs: int = 5,
):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    count_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6),
        ],
        milestones=[warmup_steps],
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=(device.type != "cpu"))

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_ckpt = None
    best_weights = None
    early_stop_counter = 0

    print_start(f"학습 레츠고~ (CutMix: {use_cutmix}, MixUp: {use_mixup})\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0

        use_aug = epoch >= warmup_epochs

        for batch_idx, (features, scalars, labels) in enumerate(train_loader):
            features, scalars, labels = map(lambda x: x.to(device, non_blocking=True), (features, scalars, labels))
            # non_blocking True쓰면 개빠릅니다

            original_labels = labels.clone()
            mixed = False

            if use_aug and (use_cutmix or use_mixup):
                r = np.random.rand()

                if use_cutmix and r < cutmix_prob:
                    features, labels, _, lam = cutmix_data(features, labels, cutmix_alpha, device.type)
                    mixed = True
                elif use_mixup and r < (cutmix_prob + mixup_prob):
                    indices = torch.randperm(features.size(0)).to(device)
                    lam = np.random.beta(mixup_alpha, mixup_alpha)

                    features = lam * features + (1 - lam) * features[indices]
                    scalars = lam * scalars + (1 - lam) * scalars[indices]
                    labels = lam * labels + (1 - lam) * labels[indices]
                    mixed = True

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type != "cpu")):
                logits = model(features, scalars)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if not mixed:
                preds = (logits > 0.0).float()
                train_correct += (preds == labels).sum().item()
                total += labels.size(0)
            else:
                with torch.no_grad():
                    preds = (logits > 0.0).float()
                    train_correct += (preds == original_labels).sum().item()
                    total += original_labels.size(0)

            train_loss += loss.item()

        train_acc = train_correct / total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for features, scalars, labels in val_loader:
                features, scalars, labels = map(lambda x: x.to(device, non_blocking=True), (features, scalars, labels))
                with autocast(enabled=(device.type != "cpu")):
                    logits = model(features, scalars)
                    loss = criterion(logits, labels)

                preds = (logits > 0.0).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        aug_status = f" [Aug: {'ON' if use_aug else 'OFF'}]" if use_cutmix or use_mixup else ""
        print(
            f"[Epoch {epoch+1:02d}]{aug_status} "
            f"Train Loss: {avg_train_loss:.6} | Train Acc: {train_acc:.6f} || "
            f"Val Loss: {avg_val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        if monitor == "val_acc":
            metric = val_acc
            best_metric = best_val_acc
        else:
            metric = -avg_val_loss
            best_metric = -best_val_loss

        if metric - best_metric > min_delta:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_ckpt = os.path.join(save_dir, f"best_epoch{epoch+1:02d}.pth")
            best_weights = model.state_dict() if restore_best_weights else None
            early_stop_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
                "val_loss": avg_val_loss,
                "epoch": epoch + 1,
                "cutmix_used": use_cutmix,
                "mixup_used": use_mixup
            }, best_ckpt)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print_warning("끝;;;;")
                if restore_best_weights and best_weights is not None:
                    model.load_state_dict(best_weights)
                break

    return best_ckpt, best_val_acc