from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

from src.config import DATA_PROCESSED, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, SEED, OUTPUTS_DIR
from src.utils import set_seed, get_device, ensure_dir, save_text, save_csv_matrix, plot_confusion_matrix
from src.train.train_loops import train_one_epoch, eval_epoch
from src.data.dataloaders import get_imagefolder_loaders



def plot_acc(train_accs, val_accs, out_dir: Path):
    plt.figure()
    plt.plot(range(1, len(train_accs) + 1), train_accs, label="train_acc")
    plt.plot(range(1, len(val_accs) + 1), val_accs, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve (Transfer Learning)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=200)
    plt.close()



def main():
    set_seed(SEED)
    device = get_device()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    out_dir = OUTPUTS_DIR / "model_transfer_resnet50"
    ensure_dir(out_dir)

    train_loader, val_loader, test_loader, class_names = get_imagefolder_loaders(
        DATA_PROCESSED, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, normalize=True
    )
    num_classes = len(class_names)

    # -------- Model: ResNet50 pretrained --------
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # -------- Stage 1: freeze backbone --------
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val = -1.0
    best_path = out_dir / "best_resnet50.pt"

    train_accs, val_accs = [], []

    def run_epochs(n_epochs: int, tag: str):
        nonlocal best_val
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            tr_acc = eval_epoch(model, train_loader, device)[0]
            va_acc, _, _ = eval_epoch(model, val_loader, device)
            dt = time.time() - t0
            train_accs.append(tr_acc)
            val_accs.append(va_acc)

            print(f"[{tag}] Epoch {epoch:02d}/{n_epochs} | loss={loss:.4f} | train_acc={tr_acc:.4f} | val_acc={va_acc:.4f} | time={dt:.1f}s")

            if va_acc > best_val + 1e-4:
                best_val = va_acc
                torch.save(model.state_dict(), best_path)

    run_epochs(n_epochs=5, tag="freeze")

    # -------- Stage 2: unfreeze last block (layer4) + fc --------
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("layer3.") or name.startswith("layer4.") or name.startswith("fc.")

    # smaller LR for backbone, larger for head
    optimizer = torch.optim.AdamW([
        {"params": model.layer3.parameters(), "lr": 7e-5},
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 5e-4},
    ], weight_decay = 1e-4)

    run_epochs(n_epochs=15, tag="finetune")

    print(f"[DONE] best_val_acc={best_val:.4f} saved={best_path}")
    plot_acc(train_accs, val_accs, out_dir)

    # -------- Test evaluation --------
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc, y_true, y_pred = eval_epoch(model, test_loader, device)
    print(f"[TEST] acc={test_acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    save_text(out_dir / "classification_report.txt", report)
    save_csv_matrix(out_dir / "confusion_matrix.csv", cm)
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()