"""
Predict road damage severity for a single image.

Usage:
  python src/demo/predict.py path/to/image.jpg --model tl
  python src/demo/predict.py path/to/image.jpg --model baseline
  python src/demo/predict.py path/to/image.jpg --model tl --crop 100 200 600 700
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from src.models.custom_cnn import SmallCNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CKPT = {
    "baseline": PROJECT_ROOT / "outputs" / "model_custom" / "best_custom_cnn.pt",
    "tl":       PROJECT_ROOT / "outputs" / "model_transfer_resnet50" / "best_resnet50.pt",
}

CLASS_NAMES = ["minor", "normal", "severe"]
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_transform(normalize: bool) -> transforms.Compose:
    tf = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
    if normalize:
        tf.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]))
    return transforms.Compose(tf)


def load_model(model_type: str) -> tuple[nn.Module, transforms.Compose]:
    ckpt_path = CKPT[model_type]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if model_type == "baseline":
        model = SmallCNN(img_size=IMG_SIZE, n_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        tf = build_transform(normalize=False)
    else:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        tf = build_transform(normalize=True)

    model.to(DEVICE).eval()
    return model, tf


@torch.no_grad()
def predict_pil(model: nn.Module, tf: transforms.Compose, img: Image.Image):
    x = tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x)[0], dim=0)
    conf, idx = torch.max(probs, dim=0)
    topk = torch.topk(probs, k=len(CLASS_NAMES))
    top3 = [(CLASS_NAMES[i], float(p))
            for p, i in zip(topk.values.cpu().tolist(), topk.indices.cpu().tolist())]
    return CLASS_NAMES[int(idx.item())], float(conf.item()), top3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to image")
    parser.add_argument("--model", choices=["baseline", "tl"], default="tl",
                        help="baseline = SmallCNN, tl = ResNet50 (default: tl)")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Optional crop before prediction: --crop x1 y1 x2 y2")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    model, tf = load_model(args.model)
    print(f"[INFO] model={args.model}  device={DEVICE}  ckpt={CKPT[args.model].name}")

    img = Image.open(img_path).convert("RGB")

    if args.crop:
        img = img.crop(args.crop)
        print(f"[INFO] cropped to {args.crop}")

    pred, conf, top3 = predict_pil(model, tf, img)

    print(f"\nImage   : {img_path.name}")
    print(f"Result  : {pred}  (confidence {conf:.4f})")
    print("Top-3   :", "  ".join(f"{c}:{p:.4f}" for c, p in top3))


if __name__ == "__main__":
    main()
