from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms

from src.models.custom_cnn import SmallCNN


# -------------------------
# Config
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Update if your folders differ
DATASET_ROOT_DEFAULT = PROJECT_ROOT / "data_processed_bbox" / "test"

BASELINE_CKPT = PROJECT_ROOT / "outputs" / "model_custom" / "best_custom_cnn.pt"
TL_CKPT = PROJECT_ROOT / "outputs" / "model_transfer_resnet50" / "best_resnet50.pt"

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Must match ImageFolder class order used in training
CLASS_NAMES = ["minor", "normal", "severe"]


# Baseline preprocessing (same as training: ToTensor only)
tf_baseline = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# TL preprocessing (ImageNet normalization)
tf_tl = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_baseline() -> torch.nn.Module:
    if not BASELINE_CKPT.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {BASELINE_CKPT}")
    m = SmallCNN(img_size=IMG_SIZE, n_classes=len(CLASS_NAMES))
    state = torch.load(BASELINE_CKPT, map_location=DEVICE)
    m.load_state_dict(state)
    m.to(DEVICE).eval()
    return m


def load_tl() -> torch.nn.Module:
    if not TL_CKPT.exists():
        raise FileNotFoundError(f"TL checkpoint not found: {TL_CKPT}")
    m = models.resnet50(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, len(CLASS_NAMES))
    state = torch.load(TL_CKPT, map_location=DEVICE)
    m.load_state_dict(state)
    m.to(DEVICE).eval()
    return m


@torch.no_grad()
def predict(model: torch.nn.Module, img: Image.Image, tf) -> tuple[str, float]:
    x = tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = model(x)[0]
    probs = torch.softmax(logits, dim=0)
    conf, idx = torch.max(probs, dim=0)
    return CLASS_NAMES[int(idx.item())], float(conf.item())


def collect_images(dataset_root: Path, per_class: int, seed: int) -> list[tuple[Path, str]]:
    """
    Returns list of (image_path, true_label) sampled from ImageFolder-style:
      dataset_root/minor/*.jpg
      dataset_root/normal/*.jpg
      dataset_root/severe/*.jpg
    """
    random.seed(seed)
    samples = []

    for cls in CLASS_NAMES:
        cls_dir = dataset_root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")

        imgs = sorted([p for p in cls_dir.glob("*.jpg")])
        if not imgs:
            raise RuntimeError(f"No images found in {cls_dir}")

        k = min(per_class, len(imgs))
        picks = random.sample(imgs, k=k)
        samples.extend([(p, cls) for p in picks])

    random.shuffle(samples)
    return samples


def make_grid(samples: list[tuple[Path, str]], out_path: Path, cols: int = 3):
    baseline = load_baseline()
    tl = load_tl()

    n = len(samples)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")

        if i >= n:
            continue

        img_path, true_cls = samples[i]
        img = Image.open(img_path).convert("RGB")

        pred_b, conf_b = predict(baseline, img, tf_baseline)
        pred_t, conf_t = predict(tl, img, tf_tl)

        ax.imshow(img)
        ax.set_title(
            f"True: {true_cls}\n"
            f"Baseline: {pred_b} ({conf_b:.2f})\n"
            f"TL: {pred_t} ({conf_t:.2f})",
            fontsize=10
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVED] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Grid of sampled images with Baseline vs TL predictions.")
    parser.add_argument("--dataset", type=str, default=str(DATASET_ROOT_DEFAULT),
                        help="Path to ImageFolder split, e.g. data_processed_bbox/test")
    parser.add_argument("--per-class", type=int, default=4, help="How many images to sample per class")
    parser.add_argument("--cols", type=int, default=3, help="Grid columns")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "outputs" / "demo_grid_compare.png"),
                        help="Output PNG path")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    samples = collect_images(dataset_root, per_class=args.per_class, seed=args.seed)
    make_grid(samples, out_path=Path(args.out), cols=args.cols)


if __name__ == "__main__":
    main()