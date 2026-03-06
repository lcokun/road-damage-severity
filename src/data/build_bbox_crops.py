from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

IN_META = {
    "train": PROJECT_ROOT / "outputs" / "metadata_train.csv",
    "val": PROJECT_ROOT / "outputs" / "metadata_val.csv",
    "test": PROJECT_ROOT / "outputs" / "metadata_test.csv",
}

OUT_ROOT = PROJECT_ROOT / "data_processed_bbox"  # new dataset root
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CLASSES = ["normal", "minor", "severe"]

# Your severity mapping (same as before)
SEVERITY_MAP = {
    "D00": "minor",
    "D01": "minor",
    "D10": "minor",
    "D11": "minor",
    "D20": "severe",
    "D40": "severe",
}

PAD_RATIO = 0.15          # padding around bbox (15%)
MIN_CROP_SIZE = 64        # avoid tiny crops


def image_to_xml_path(img_path: Path) -> Path:
    # .../Country/images/XXX.jpg -> .../Country/annotations/xmls/XXX.xml
    parts = list(img_path.parts)
    try:
        idx = parts.index("images")
    except ValueError:
        raise ValueError(f"Expected 'images' in path: {img_path}")
    parts[idx] = "annotations"
    xml_dir = Path(*parts[:idx+1]) / "xmls"
    return xml_dir / (img_path.stem + ".xml")


def parse_bboxes(xml_path: Path):
    # Returns list of (label_name, xmin, ymin, xmax, ymax)
    root = ET.parse(xml_path).getroot()
    out = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bb = obj.find("bndbox")
        if name_el is None or bb is None:
            continue
        name = (name_el.text or "").strip()
        xmin = int(float(bb.findtext("xmin", "0")))
        ymin = int(float(bb.findtext("ymin", "0")))
        xmax = int(float(bb.findtext("xmax", "0")))
        ymax = int(float(bb.findtext("ymax", "0")))
        out.append((name, xmin, ymin, xmax, ymax))
    return out


def choose_bbox(bboxes):
    """
    Choose bbox of highest severity present:
      - if any severe label exists -> pick first severe bbox
      - elif any minor label exists -> pick first minor bbox
      - else -> None (normal/no objects)
    """
    # Tag each bbox with severity
    tagged = []
    for name, x1, y1, x2, y2 in bboxes:
        sev = SEVERITY_MAP.get(name, None)
        tagged.append((sev, name, x1, y1, x2, y2))

    for sev, name, x1, y1, x2, y2 in tagged:
        if sev == "severe":
            return (x1, y1, x2, y2)
    for sev, name, x1, y1, x2, y2 in tagged:
        if sev == "minor":
            return (x1, y1, x2, y2)
    return None


def crop_with_padding(img: Image.Image, bbox, pad_ratio: float):
    W, H = img.size
    x1, y1, x2, y2 = bbox

    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    x1 = max(x1 - pad_w, 0)
    y1 = max(y1 - pad_h, 0)
    x2 = min(x2 + pad_w, W)
    y2 = min(y2 + pad_h, H)

    # ensure minimum crop size
    if (x2 - x1) < MIN_CROP_SIZE:
        cx = (x1 + x2) // 2
        x1 = max(cx - MIN_CROP_SIZE // 2, 0)
        x2 = min(x1 + MIN_CROP_SIZE, W)
    if (y2 - y1) < MIN_CROP_SIZE:
        cy = (y1 + y2) // 2
        y1 = max(cy - MIN_CROP_SIZE // 2, 0)
        y2 = min(y1 + MIN_CROP_SIZE, H)

    return img.crop((x1, y1, x2, y2))


def ensure_out_dirs():
    for split in IN_META.keys():
        for c in CLASSES:
            (OUT_ROOT / split / c).mkdir(parents=True, exist_ok=True)


def process_split(split: str):
    df = pd.read_csv(IN_META[split])
    saved = 0
    skipped = 0

    for _, row in df.iterrows():
        img_path = Path(row["image_path"])
        final_class = str(row["final_class"])

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        if final_class == "normal":
            # No bbox expected, save full image (or center crop if you prefer)
            crop = img
        else:
            xml_path = image_to_xml_path(img_path)
            if not xml_path.exists():
                skipped += 1
                continue

            bboxes = parse_bboxes(xml_path)
            bbox = choose_bbox(bboxes)

            # If bbox missing, fallback to full image
            crop = crop_with_padding(img, bbox, PAD_RATIO) if bbox else img

        out_path = OUT_ROOT / split / final_class / img_path.name
        crop.save(out_path, quality=95)
        saved += 1

    print(f"[{split}] saved={saved}, skipped={skipped}")


def main():
    ensure_out_dirs()
    for split in ["train", "val", "test"]:
        process_split(split)
    print("[DONE] bbox dataset at:", OUT_ROOT)


if __name__ == "__main__":
    main()