"""
Run the pipeline on an image and save an annotated visualization.

Usage:
  python src/demo/visualize_pipeline.py path/to/image.jpg
  python src/demo/visualize_pipeline.py path/to/image.jpg --out outputs/pipeline_sample.png
  python src/demo/visualize_pipeline.py path/to/image.jpg --conf 0.25 --severe-gate 0.80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running from PMA/ directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image, ImageDraw, ImageFont

from src.demo.predict_pipeline import load_detector, load_classifier, classify_crop
from src.demo.predict_pipeline import SEVERE_GATE, SEVERITY_RANK, CLASS_NAMES, PROJECT_ROOT

COLORS = {
    "minor":  "#f5a623",   # amber
    "severe": "#d0021b",   # red
    "normal": "#7ed321",   # green
}
TEXT_PADDING = 4
BOX_WIDTH = 3


def annotate(
    img_path: Path,
    det_conf: float,
    severe_gate: float,
    out_path: Path,
) -> None:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    detector   = load_detector()
    classifier = load_classifier()

    results = detector.predict(str(img_path), conf=det_conf, verbose=False)
    boxes = results[0].boxes

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", size=max(14, h // 30))
    except OSError:
        font = ImageFont.load_default()

    regions = []
    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes.xyxy.cpu().tolist()):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img.crop((x1, y1, x2, y2))
            sev, conf = classify_crop(classifier, crop)
            regions.append((x1, y1, x2, y2, sev, conf))

        # apply severity gate
        if regions:
            best_sev, best_conf = max(
                ((r[4], r[5]) for r in regions),
                key=lambda rc: (SEVERITY_RANK[rc[0]], rc[1])
            )
            final = best_sev
            if final == "severe" and best_conf < severe_gate:
                final = "minor"
        else:
            final = "normal"

        for x1, y1, x2, y2, sev, conf in regions:
            color = COLORS[sev]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_WIDTH)

            label = f"{sev} {conf:.2f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # label background
            lx1 = x1
            ly1 = max(0, y1 - th - TEXT_PADDING * 2)
            lx2 = x1 + tw + TEXT_PADDING * 2
            ly2 = y1
            draw.rectangle([lx1, ly1, lx2, ly2], fill=color)
            draw.text((lx1 + TEXT_PADDING, ly1 + TEXT_PADDING), label, fill="white", font=font)
    else:
        final = "normal"

    # overall result banner at bottom
    banner_h = max(28, h // 18)
    banner_color = COLORS[final]
    draw.rectangle([0, h - banner_h, w, h], fill=banner_color)
    result_label = f"Result: {final.upper()}  ({len(regions)} region{'s' if len(regions) != 1 else ''} detected)"
    try:
        banner_font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", size=max(12, banner_h - 8))
    except OSError:
        banner_font = font
    draw.text((TEXT_PADDING * 2, h - banner_h + 4), result_label, fill="white", font=banner_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[SAVED] {out_path}")
    print(f"Result : {final}  ({len(regions)} detection{'s' if len(regions) != 1 else ''})")


def main():
    parser = argparse.ArgumentParser(description="Visualize pipeline output on a road image")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--out", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "pipeline_sample.png"),
                        help="Output image path (default: outputs/pipeline_sample.png)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO detection confidence threshold (default: 0.25)")
    parser.add_argument("--severe-gate", type=float, default=SEVERE_GATE,
                        help=f"Severe confidence gate (default: {SEVERE_GATE})")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    annotate(img_path, det_conf=args.conf, severe_gate=args.severe_gate, out_path=Path(args.out))


if __name__ == "__main__":
    main()
