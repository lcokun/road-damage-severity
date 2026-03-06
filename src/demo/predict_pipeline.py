"""
Full-image pipeline: YOLOv8 detection → ResNet50 severity classification.

Usage:
  python src/demo/predict_pipeline.py path/to/image.jpg
  python src/demo/predict_pipeline.py path/to/image.jpg --conf 0.3 --verbose

Pipeline:
  Full image → YOLOv8 (find damage regions) → crop each box → ResNet50 (classify severity)
  Final result = worst severity across all detections.
  No detections = "normal".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

YOLO_CKPT = PROJECT_ROOT / "outputs" / "yolo" / "detector" / "weights" / "best.pt"
TL_CKPT   = PROJECT_ROOT / "outputs" / "model_transfer_resnet50" / "best_resnet50.pt"

CLASS_NAMES    = ["minor", "normal", "severe"]
SEVERITY_RANK  = {"normal": 0, "minor": 1, "severe": 2}
SEVERE_GATE    = 0.80   # downgrade severe→minor if classifier conf < this
IMG_SIZE       = 224
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_classifier() -> nn.Module:
    if not TL_CKPT.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {TL_CKPT}")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(TL_CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def load_detector():
    if not YOLO_CKPT.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {YOLO_CKPT}")
    from ultralytics import YOLO
    return YOLO(str(YOLO_CKPT))


@torch.no_grad()
def classify_crop(classifier: nn.Module, crop: Image.Image) -> tuple[str, float]:
    x = TF(crop.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(classifier(x)[0], dim=0)
    conf, idx = torch.max(probs, dim=0)
    return CLASS_NAMES[int(idx.item())], float(conf.item())


def run_pipeline(
    img_path: Path,
    det_conf: float = 0.25,
    severe_gate: float = SEVERE_GATE,
    verbose: bool = False,
) -> dict:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    detector  = load_detector()
    classifier = load_classifier()

    results = detector.predict(str(img_path), conf=det_conf, verbose=False)
    boxes = results[0].boxes  # xyxy format

    if boxes is None or len(boxes) == 0:
        if verbose:
            print("[YOLO] No detections — defaulting to 'normal'")
        return {
            "image": img_path.name,
            "detections": 0,
            "final": "normal",
            "regions": [],
        }

    regions = []
    for i, box in enumerate(boxes.xyxy.cpu().tolist()):
        x1, y1, x2, y2 = [int(v) for v in box]
        # clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = img.crop((x1, y1, x2, y2))
        label, conf = classify_crop(classifier, crop)
        det_conf_score = float(boxes.conf[i].item())

        regions.append({
            "box": [x1, y1, x2, y2],
            "det_conf": det_conf_score,
            "severity": label,
            "cls_conf": conf,
        })

        if verbose:
            print(f"  [YOLO box {i+1}] ({x1},{y1})-({x2},{y2})  "
                  f"det={det_conf_score:.3f}  →  {label} ({conf:.3f})")

    # worst severity wins; downgrade severe if classifier confidence is low
    best = max(regions, key=lambda r: (SEVERITY_RANK[r["severity"]], r["cls_conf"]))
    final = best["severity"]
    if final == "severe" and best["cls_conf"] < severe_gate:
        final = "minor"
        if verbose:
            print(f"  [gate] severe downgraded to minor (conf={best['cls_conf']:.3f} < {severe_gate})")

    return {
        "image": img_path.name,
        "detections": len(regions),
        "final": final,
        "regions": regions,
    }


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 + ResNet50 road damage pipeline")
    parser.add_argument("image", type=str, help="Path to full road image")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO detection confidence threshold (default: 0.25)")
    parser.add_argument("--severe-gate", type=float, default=SEVERE_GATE,
                        help=f"Classifier confidence threshold to call severe (default: {SEVERE_GATE})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-box details")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    print(f"[INFO] device={DEVICE}")
    result = run_pipeline(img_path, det_conf=args.conf, severe_gate=args.severe_gate, verbose=args.verbose)

    print(f"\nImage      : {result['image']}")
    print(f"Detections : {result['detections']}")
    print(f"Result     : {result['final']}")

    if result["regions"]:
        print("\nPer-region breakdown:")
        for i, r in enumerate(result["regions"], 1):
            print(f"  [{i}] box={r['box']}  det={r['det_conf']:.3f}  "
                  f"severity={r['severity']} ({r['cls_conf']:.3f})")


if __name__ == "__main__":
    main()
