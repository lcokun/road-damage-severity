from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use("Agg")  # no tkinter
import matplotlib.pyplot as plt

from torch.distributions import Beta
from sklearn.metrics import classification_report, confusion_matrix

from src.config import DATA_PROCESSED, IMG_SIZE, BATCH_SIZE, EPOCHS, LR, NUM_WORKERS, SEED, OUTPUTS_DIR
from src.utils import set_seed, get_device, ensure_dir, count_trainable_params, save_text, save_csv_matrix, plot_confusion_matrix
from src.data.dataloaders import get_imagefolder_loaders  # normalize=False by default
from src.models.custom_cnn import SmallCNN
from src.train.train_loops import eval_epoch



def plot_curves(train_accs, val_accs, train_losses, out_dir: Path):
    ensure_dir(out_dir)

    plt.figure()
    plt.plot(range(1, len(train_accs) + 1), train_accs, label="train_acc")
    plt.plot(range(1, len(val_accs) + 1), val_accs, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve (Custom CNN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (Custom CNN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=200)
    plt.close()



# -----------------------
# Focal Loss (hard labels)
# -----------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # optional per-class weights (tensor [C])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")  # [N]
        pt = torch.exp(-ce)  # [N]
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)  # [N]
            loss = at * loss

        return loss.mean()


# -----------------------
# MixUp / CutMix helpers
# -----------------------
def rand_bbox(W: int, H: int, lam: float):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def train_one_epoch_mix(model, loader, optimizer, criterion: nn.Module, device: str,
                        mix_prob: float = 0.7, mixup_alpha: float = 0.4, cutmix_alpha: float = 0.7) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    beta_mixup = Beta(mixup_alpha, mixup_alpha)
    beta_cutmix = Beta(cutmix_alpha, cutmix_alpha)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        r = np.random.rand()
        if r < mix_prob:
            # choose MixUp vs CutMix (50/50)
            use_cutmix = (np.random.rand() < 0.5)
            index = torch.randperm(x.size(0), device=device)
            y2 = y[index]

            if use_cutmix:
                lam = float(beta_cutmix.sample().item())
                _, _, H, W = x.shape
                x1, y1, x2, y2b = rand_bbox(W, H, lam)
                x[:, :, y1:y2b, x1:x2] = x[index, :, y1:y2b, x1:x2]
                lam = 1.0 - ((x2 - x1) * (y2b - y1) / (W * H))
            else:
                lam = float(beta_mixup.sample().item())
                x = lam * x + (1.0 - lam) * x[index]

            logits = model(x)
            loss = lam * criterion(logits, y) + (1.0 - lam) * criterion(logits, y2)

        else:
            logits = model(x)
            loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

    return total_loss / max(n, 1)


def main():
    set_seed(SEED)
    device = get_device()

    # Speed hint for CUDA
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    out_dir = OUTPUTS_DIR / "model_custom"
    ensure_dir(out_dir)

    train_loader, val_loader, test_loader, class_names = get_imagefolder_loaders(
        DATA_PROCESSED, IMG_SIZE, BATCH_SIZE, NUM_WORKERS
    )

    model = SmallCNN(img_size=IMG_SIZE, n_classes=len(class_names)).to(device)
    print(f"[INFO] device={device}")
    print(f"[INFO] classes={class_names}")
    print(f"[INFO] trainable_params={count_trainable_params(model):,}")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, threshold=1e-4
    )

    best_val = -1.0
    best_path = out_dir / "best_custom_cnn.pt"

    train_losses = []
    train_accs = []
    val_accs = []

    patience = 8
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss = train_one_epoch_mix(
            model, train_loader, optimizer, criterion, device,
            mix_prob=0.0
        )

        tr_acc = eval_epoch(model, train_loader, device)[0]
        va_acc, _, _ = eval_epoch(model, val_loader, device)

        # Scheduler step on val accuracy
        scheduler.step(va_acc)

        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | loss={tr_loss:.4f} | train_acc={tr_acc:.4f} | "
            f"val_acc={va_acc:.4f} | lr={cur_lr:.2e} | time={dt:.1f}s"
        )

        if va_acc > best_val + 1e-4:
            best_val = va_acc
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EARLY STOP] no val improvements for {patience} epochs")
                break

    print(f"[DONE] best_val_acc={best_val:.4f} saved={best_path}")

    plot_curves(train_accs, val_accs, train_losses, out_dir)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc, y_true, y_pred = eval_epoch(model, test_loader, device)
    print(f"[TEST] acc={test_acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    save_text(out_dir / "classification_report.txt", report)
    save_csv_matrix(out_dir / "confusion_matrix.csv", cm)
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    print("[SAVED] accuracy_curve.png, loss_curve.png, confusion_matrix.png, classification_report.txt, confusion_matrix.csv")


if __name__ == "__main__":
    main()