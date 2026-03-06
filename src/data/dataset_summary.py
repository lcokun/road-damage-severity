import os
from pathlib import Path
from PIL import Image

DATASET_ROOT = str(Path(__file__).resolve().parent.parent.parent / "data_processed_bbox")
SPLITS = ["test", "train", "val"]
CLASSES = ["minor", "normal", "severe"]

def count_images(folder):
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

def get_image_size(folder):
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, f)
            with Image.open(img_path) as img:
                return img.size  # (width, height)
    return None

total_images = 0
class_totals = {cls: 0 for cls in CLASSES}
split_totals = {}

image_size = None

print("=== Dataset Summary ===\n")

for split in SPLITS:
    split_count = 0
    print(f"{split.upper()} SET:")
    for cls in CLASSES:
        path = os.path.join(DATASET_ROOT, split, cls)
        count = count_images(path)
        split_count += count
        class_totals[cls] += count
        print(f"  {cls.capitalize():<7}: {count}")

        if image_size is None and count > 0:
            image_size = get_image_size(path)

    split_totals[split] = split_count
    total_images += split_count
    print(f"  Total {split}: {split_count}\n")

print("=== Overall Statistics ===")
print(f"Total images (after cropping): {total_images}\n")

for cls, count in class_totals.items():
    print(f"{cls.capitalize():<7}: {count}")

print("\n=== Split Ratios ===")
for split, count in split_totals.items():
    ratio = (count / total_images) * 100
    print(f"{split.capitalize():<5}: {ratio:.2f}%")

if image_size:
    print("\n=== Input Image Size ===")
    print(f"Width x Height: {image_size[0]} x {image_size[1]}")
else:
    print("\nImage size could not be determined.")
