from __future__ import annotations

import shutil
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SEED = 42
random.seed(SEED)

SEVERITY_MAP: Dict[str, str] = {
    "D00": "minor",
    "D01": "minor",
    "D10": "minor",
    "D11": "minor",
    "D20": "severe",
    "D40": "severe",
}
CLASSES = ["normal", "minor", "severe"]

# Dataset location (yours)
DATA_ROOT = PROJECT_ROOT / "rdd-dataset"
TRAIN_DIR = DATA_ROOT / "train"

# Outputs
OUT_ROOT = PROJECT_ROOT / "data_processed"
OUT_TRAIN = OUT_ROOT / "train"
OUT_VAL = OUT_ROOT / "val"
OUT_TEST = OUT_ROOT / "test"

OUT_META_DIR = PROJECT_ROOT / "outputs"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-9


@dataclass
class Sample:
    image_path: Path
    country: str
    raw_labels: List[str]
    final_class: str


def parse_label_map_pbtxt(pbtxt_path: Path) -> Dict[int, str]:
    if not pbtxt_path.exists():
        raise FileNotFoundError(f"label_map.pbtxt not found: {pbtxt_path}")

    text = pbtxt_path.read_text(encoding="utf-8", errors="ignore")
    items = []
    cur_id = None
    cur_name = None

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("id:"):
            cur_id = int(line.split(":", 1)[1].strip())
        elif line.startswith("name:"):
            val = line.split(":", 1)[1].strip().strip("'").strip('"')
            cur_name = val
        elif line.startswith("}"):
            if cur_id is not None and cur_name is not None:
                items.append((cur_id, cur_name))
            cur_id, cur_name = None, None

    return dict(items)


def parse_xml_labels(xml_path: Path) -> List[str]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        raise ValueError(f"Bad XML {xml_path}: {e}")

    labels = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is not None and name_el.text:
            labels.append(name_el.text.strip())
    return labels


def reduce_to_severity_class(raw_labels: List[str]) -> str:
    mapped = [SEVERITY_MAP.get(lbl) for lbl in raw_labels]
    if "severe" in mapped:
        return "severe"
    if "minor" in mapped:
        return "minor"
    return "normal"


def collect_samples_from_train(train_dir: Path) -> List[Sample]:
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    samples: List[Sample] = []
    for country_dir in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
        country = country_dir.name
        img_dir = country_dir / "images"
        xml_dir = country_dir / "annotations" / "xmls"
        if not img_dir.exists() or not xml_dir.exists():
            continue

        xml_by_stem = {x.stem: x for x in xml_dir.glob("*.xml")}

        for img_path in img_dir.glob("*.jpg"):
            xml_path = xml_by_stem.get(img_path.stem)
            if xml_path is None:
                continue

            raw_labels = parse_xml_labels(xml_path)
            final_class = reduce_to_severity_class(raw_labels)
            samples.append(Sample(img_path, country, raw_labels, final_class))

    if not samples:
        raise RuntimeError("No samples collected, check your extracted folder paths.")
    return samples


def stratified_split(samples: List[Sample]) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    by_class: Dict[str, List[Sample]] = {c: [] for c in CLASSES}
    for s in samples:
        by_class[s.final_class].append(s)

    for c in CLASSES:
        random.shuffle(by_class[c])

    train_set, val_set, test_set = [], [], []
    for c, items in by_class.items():
        n = len(items)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    return train_set, val_set, test_set


def ensure_class_folders(base: Path):
    for c in CLASSES:
        (base / c).mkdir(parents=True, exist_ok=True)


def copy_split(split_samples: List[Sample], out_base: Path, use_symlink: bool = False):
    ensure_class_folders(out_base)
    for s in split_samples:
        dest = out_base / s.final_class / s.image_path.name
        if dest.exists():
            continue
        if use_symlink:
            dest.symlink_to(s.image_path.resolve())
        else:
            shutil.copy2(s.image_path, dest)


def save_metadata_csv(samples: List[Sample], csv_path: Path):
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "country", "raw_labels", "final_class"])
        for s in samples:
            w.writerow([str(s.image_path), s.country, ",".join(s.raw_labels), s.final_class])


def main():
    pbtxt = TRAIN_DIR / "label_map.pbtxt"
    if pbtxt.exists():
        id_to_name = parse_label_map_pbtxt(pbtxt)
        print(f"[OK] Loaded label_map.pbtxt with {len(id_to_name)} labels")
    else:
        print("[WARN] label_map.pbtxt not found under train/. Continue anyway.")

    print("[1/4] Collecting annotated samples from train/ ...")
    samples = collect_samples_from_train(TRAIN_DIR)
    print(f"[OK] Collected {len(samples)} annotated images")

    print("[2/4] Saving metadata table ...")
    save_metadata_csv(samples, OUT_META_DIR / "metadata_all.csv")

    print("[3/4] Stratified split (train/val/test from train only) ...")
    train_s, val_s, test_s = stratified_split(samples)
    print(f"[OK] split sizes: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    save_metadata_csv(train_s, OUT_META_DIR / "metadata_train.csv")
    save_metadata_csv(val_s, OUT_META_DIR / "metadata_val.csv")
    save_metadata_csv(test_s, OUT_META_DIR / "metadata_test.csv")

    print("[4/4] Materializing folder dataset to data_processed/ ...")
    copy_split(train_s, OUT_TRAIN, use_symlink=False)
    copy_split(val_s, OUT_VAL, use_symlink=False)
    copy_split(test_s, OUT_TEST, use_symlink=False)

    print("[DONE] data_processed/ is ready for EDA + training")


if __name__ == "__main__":
    main()