from __future__ import annotations

from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SEED = 42
random.seed(SEED)

DATASET_ROOT = PROJECT_ROOT / "data_processed"
OUT_DIR = PROJECT_ROOT / "outputs" / "eda"

SPLIT = "train"
CLASSES = ["normal", "minor", "severe"]


def sample_images(class_dir: Path, k: int) -> list[Path]:
    imgs = sorted(class_dir.glob("*.jpg"))
    return random.sample(imgs, k=min(k, len(imgs))) if imgs else []


def make_grid(split: str, k_per_class: int, out_path: Path):
    rows = len(CLASSES)
    cols = k_per_class

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = [axes]

    for r, cls in enumerate(CLASSES):
        class_dir = DATASET_ROOT / split / cls
        picks = sample_images(class_dir, cols)

        for c in range(cols):
            ax = axes[r][c] if rows > 1 else axes[c]
            ax.axis("off")
            if c < len(picks):
                img = Image.open(picks[c]).convert("RGB")
                ax.imshow(img)
            if c == 0:
                ax.set_title(cls, fontsize=14, loc="left")

    fig.suptitle(f"Sample Images per Class ({split})", fontsize=16)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved {out_path}")


def main():
    make_grid(SPLIT, k_per_class=6, out_path=OUT_DIR / f"sample_grid_{SPLIT}.png")


if __name__ == "__main__":
    main()