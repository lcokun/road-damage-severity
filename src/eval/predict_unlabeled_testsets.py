from __future__ import annotations

from pathlib import Path
import csv
import torch
import torch.nn as nn
import numpy as np

from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_EXTRACTED = PROJECT_ROOT / "rdd-dataset"
TEST_SPLITS = ["test1", "test2"]

# Trained TL checkpoint
CKPT_PATH = PROJECT_ROOT / "outputs" / "model_transfer_resnet50" / "best_resnet50.pt"

CLASS_NAMES = ["minor", "normal", "severe"]

OUT_DIR = PROJECT_ROOT / "outputs" / "unlabeled_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(num_classes: int) -> torch.nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict_one(model: torch.nn.Module, img_path: Path):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)  # [1,3,H,W]
    logits = model(x)[0]                  # [C]
    probs = torch.softmax(logits, dim=0)
    conf, pred_idx = torch.max(probs, dim=0)

    topk = torch.topk(probs, k=min(3, probs.numel()))
    top3 = [(CLASS_NAMES[i], float(p)) for p, i in zip(topk.values.cpu().tolist(), topk.indices.cpu().tolist())]

    return int(pred_idx.item()), float(conf.item()), top3


def iter_images(split_dir: Path):
    # Structure (from FileStructure.txt):
    # test1/<Country>/images/*.jpg
    for country_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        img_dir = country_dir / "images"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            yield country_dir.name, img_path


def main():
    print(f"[INFO] device={DEVICE}")
    print(f"[INFO] ckpt={CKPT_PATH}")
    model = load_model(num_classes=len(CLASS_NAMES))

    out_csv = OUT_DIR / "predictions_test1_test2.csv"
    out_summary = OUT_DIR / "summary_counts.csv"

    # Aggregations
    # counts[(split, country, class)] = n
    counts = {}

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "country", "image", "pred_class", "confidence", "top3"])

        total = 0
        for split in TEST_SPLITS:
            split_dir = DATA_EXTRACTED / split
            if not split_dir.exists():
                print(f"[WARN] Missing split folder: {split_dir}")
                continue

            for country, img_path in iter_images(split_dir):
                pred_idx, conf, top3 = predict_one(model, img_path)
                pred_class = CLASS_NAMES[pred_idx]

                top3_str = "; ".join([f"{c}:{p:.4f}" for c, p in top3])
                w.writerow([split, country, img_path.name, pred_class, f"{conf:.6f}", top3_str])

                key = (split, country, pred_class)
                counts[key] = counts.get(key, 0) + 1
                total += 1

        print(f"[DONE] wrote {total} predictions -> {out_csv}")

    # Write summary
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "country", "pred_class", "count"])
        for (split, country, pred_class), n in sorted(counts.items()):
            w.writerow([split, country, pred_class, n])

    print(f"[DONE] wrote summary -> {out_summary}")


if __name__ == "__main__":
    main()