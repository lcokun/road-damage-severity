"""
Evaluate the YOLOv8 + ResNet50 pipeline on the YOLO val set.

Ground truth is derived from RDD2022 XML annotations using the same severity
mapping used during classifier training:
  D00, D01, D10, D11 → minor
  D20, D40           → severe
  no damage objects  → normal
Image-level label = worst severity across all objects.

Usage:
  python src/eval/eval_pipeline.py                  # full val set
  python src/eval/eval_pipeline.py --limit 200      # quick run
  python src/eval/eval_pipeline.py --conf 0.3
"""
from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

YOLO_VAL_DIR  = PROJECT_ROOT / "yolo_data" / "images" / "val"
RDD_TRAIN_DIR = PROJECT_ROOT / "rdd-dataset" / "train"
YOLO_CKPT     = PROJECT_ROOT / "outputs" / "yolo" / "detector" / "weights" / "best.pt"
TL_CKPT       = PROJECT_ROOT / "outputs" / "model_transfer_resnet50" / "best_resnet50.pt"
OUT_DIR        = PROJECT_ROOT / "outputs" / "pipeline_eval"

SEVERITY_MAP: dict[str, str] = {
    "D00": "minor", "D01": "minor", "D10": "minor", "D11": "minor",
    "D20": "severe", "D40": "severe",
}
CLASS_NAMES   = ["minor", "normal", "severe"]
SEVERITY_RANK = {"normal": 0, "minor": 1, "severe": 2}
IMG_SIZE      = 224
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --------------------------------------------------------------------------- #
# Ground truth from XML                                                        #
# --------------------------------------------------------------------------- #

def gt_from_xml(xml_path: Path) -> str:
    """Return image-level severity from RDD XML annotation."""
    if not xml_path.exists():
        return "normal"
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return "normal"

    worst = "normal"
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        sev = SEVERITY_MAP.get(name)
        if sev and SEVERITY_RANK[sev] > SEVERITY_RANK[worst]:
            worst = sev
    return worst


def resolve_xml(img_name: str) -> Path:
    """
    YOLO val images are named {country}_{original_stem}.jpg (symlinks).
    Recover: country = first segment, original_stem = rest.
    XML lives at rdd-dataset/train/{country}/annotations/xmls/{original_stem}.xml
    """
    stem = Path(img_name).stem                        # e.g. Czech_Czech_000012
    country = stem.split("_")[0]                      # e.g. Czech
    original_stem = stem[len(country) + 1:]           # e.g. Czech_000012
    return RDD_TRAIN_DIR / country / "annotations" / "xmls" / f"{original_stem}.xml"


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #

def load_classifier() -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(TL_CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def load_detector():
    from ultralytics import YOLO
    return YOLO(str(YOLO_CKPT))


@torch.no_grad()
def classify_crop(classifier: nn.Module, crop: Image.Image) -> tuple[str, float]:
    x = TF(crop.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(classifier(x)[0], dim=0)
    conf, idx = torch.max(probs, dim=0)
    return CLASS_NAMES[int(idx.item())], float(conf.item())


def apply_gate(crops: list[tuple[str, float]], severe_gate: float) -> str:
    """
    Worst-severity-wins with an optional confidence gate on severe.
    If the highest-ranked crop is severe but its confidence < severe_gate,
    downgrade to minor (something was detected, just not confidently severe).
    """
    if not crops:
        return "normal"
    # find worst by rank; for ties keep highest confidence
    best = max(crops, key=lambda c: (SEVERITY_RANK[c[0]], c[1]))
    sev, conf = best
    if sev == "severe" and severe_gate < 1.0 and conf < severe_gate:
        return "minor"
    return sev


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #

def predict_image(
    detector,
    classifier: nn.Module,
    img_path: Path,
    det_conf: float,
) -> tuple[list[tuple[str, float]], int]:
    """Returns (list of (severity, conf) per crop, num_detections)."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    results = detector.predict(str(img_path), conf=det_conf, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return [], 0

    crops = []
    for box in boxes.xyxy.cpu().tolist():
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img.crop((x1, y1, x2, y2))
        sev, conf = classify_crop(classifier, crop)
        crops.append((sev, conf))

    return crops, len(boxes)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def print_report(y_true, y_pred, n_images, det_conf, severe_gate):
    report = classification_report(y_true, y_pred, labels=CLASS_NAMES, digits=4)
    cm     = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    gate_str = f"{severe_gate:.2f}" if severe_gate < 1.0 else "none"
    print("\n" + "=" * 60)
    print(f"det_conf={det_conf}  severe_gate={gate_str}")
    print("=" * 60)
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"{'':10}", "  ".join(f"{c:>8}" for c in CLASS_NAMES))
    for cls, row in zip(CLASS_NAMES, cm):
        print(f"{cls:10}", "  ".join(f"{v:>8}" for v in row))
    return report, cm


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 + ResNet50 pipeline")
    parser.add_argument("--conf",  type=float, default=0.25,
                        help="YOLO detection confidence threshold (default: 0.25)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max images to evaluate (default: all)")
    parser.add_argument("--sweep-gate", action="store_true",
                        help="Sweep severe confidence gate thresholds after eval")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] device={DEVICE}  conf={args.conf}")
    print(f"[INFO] loading models...")
    detector   = load_detector()
    classifier = load_classifier()

    val_images = sorted(YOLO_VAL_DIR.glob("*.jpg"))
    if args.limit:
        val_images = val_images[: args.limit]

    print(f"[INFO] evaluating {len(val_images)} images...")

    # Collect raw per-image crop predictions for threshold sweeping
    records = []  # (gt, crops, n_det) where crops = [(severity, conf), ...]

    for i, img_path in enumerate(val_images, 1):
        xml_path = resolve_xml(img_path.name)
        gt = gt_from_xml(xml_path)
        crops, n_det = predict_image(detector, classifier, img_path, args.conf)
        records.append((gt, crops, n_det, img_path.name))

        if i % 100 == 0:
            print(f"  {i}/{len(val_images)}")

    # Baseline (no gate)
    y_true = [r[0] for r in records]
    y_pred = [apply_gate(r[1], severe_gate=1.0) for r in records]
    report, cm = print_report(y_true, y_pred, len(records), args.conf, severe_gate=1.0)

    # Save baseline CSV + report
    csv_path = OUT_DIR / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "true", "predicted", "detections"])
        for (gt, crops, n_det, name), pred in zip(records, y_pred):
            w.writerow([name, gt, pred, n_det])

    report_path = OUT_DIR / "report.txt"
    gate_str = "none"
    report_path.write_text(
        f"Images evaluated : {len(records)}\n"
        f"YOLO conf thresh : {args.conf}\n"
        f"Severe gate      : {gate_str}\n\n"
        + report
        + "\nConfusion matrix (rows=true, cols=pred):\n"
        + f"{'':10}" + "  ".join(f"{c:>8}" for c in CLASS_NAMES) + "\n"
        + "\n".join(
            f"{cls:10}" + "  ".join(f"{v:>8}" for v in row)
            for cls, row in zip(CLASS_NAMES, cm)
        )
    )

    print(f"\n[SAVED] {csv_path}")
    print(f"[SAVED] {report_path}")

    # Threshold sweep
    if args.sweep_gate:
        print("\n--- Severe gate sweep (minor F1 / overall acc) ---")
        print(f"{'gate':>6}  {'minor_f1':>9}  {'acc':>7}  {'macro_f1':>9}")
        for gate in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]:
            yp = [apply_gate(r[1], severe_gate=gate) for r in records]
            rep = classification_report(
                y_true, yp, labels=CLASS_NAMES, digits=4, output_dict=True
            )
            minor_f1 = rep["minor"]["f1-score"]
            acc      = rep["accuracy"]
            macro_f1 = rep["macro avg"]["f1-score"]
            print(f"{gate:>6.2f}  {minor_f1:>9.4f}  {acc:>7.4f}  {macro_f1:>9.4f}")


if __name__ == "__main__":
    main()
