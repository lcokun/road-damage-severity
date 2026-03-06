from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
YOLO_DIR = PROJECT_ROOT / "yolo_data"

DAMAGE_CODES = {"D00", "D01", "D10", "D11", "D20", "D40"}

def get_image_size(img_path: Path) -> tuple[int, int]:
    with Image.open(img_path) as img:
        return img.size # (W, H)

def parse_bboxes_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in DAMAGE_CODES:
            continue

        bb = obj.find("bndbox")
        if bb is None:
            continue

        xmin = float(bb.findtext("xmin", "0"))
        ymin = float(bb.findtext("ymin", "0"))
        xmax = float(bb.findtext("xmax", "0"))
        ymax = float(bb.findtext("ymax", "0"))

        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.0, min(1.0, w))
        h  = max(0.0, min(1.0, h))

        if w > 0 and h > 0:
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines

def process_split(csv_path: Path, split: str) -> None:
    img_out = YOLO_DIR / "images" / split
    lbl_out = YOLO_DIR / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    ok, skipped = 0, 0

    for _, row in df.iterrows():
        img_path = Path(row["image_path"])
        if not img_path.exists():
            skipped += 1
            continue

        dest_stem = f"{row['country']}_{img_path.stem}"
        dest_img = img_out / (dest_stem + ".jpg")
        dest_lbl = lbl_out / (dest_stem + ".txt")

        if not dest_img.exists():
            dest_img.symlink_to(img_path.resolve())

        if row["final_class"] == "normal":
            dest_lbl.write_text("")
        else:
            xml_path = (
                img_path.parent.parent / "annotations" / "xmls"
                / (img_path.stem + ".xml")
            )
            if not xml_path.exists():
                dest_lbl.write_text("")
                skipped += 1
                continue

            try:
                img_w, img_h = get_image_size(img_path)
            except Exception:
                skipped += 1
                continue

            label_lines = parse_bboxes_yolo(xml_path, img_w, img_h)
            dest_lbl.write_text("\n".join(label_lines))

        ok += 1

    print(f"[{split}] processed={ok}, skipped={skipped}")

def main():
    process_split(OUTPUTS_DIR / "metadata_train.csv", "train")
    process_split(OUTPUTS_DIR / "metadata_val.csv", "val")

    data_yaml = {
        "path": str(YOLO_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["damage"],
    }
    with (YOLO_DIR / "data.yaml").open("w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"[DONE] yolo_data/ ready at: {YOLO_DIR}")

if __name__ == "__main__":
    main()
