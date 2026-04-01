"""
GTSRB Dataset Preparation Script
─────────────────────────────────
Downloads the German Traffic Sign Recognition Benchmark,
selects 5 classes, resizes to 32×32, and splits into
data/train / data/test (80/20) in ImageDataGenerator format.

Folder output:
  data/
    train/
      speed_limit_20/   speed_limit_30/   stop/   yield/   no_entry/
    test/
      speed_limit_20/   speed_limit_30/   stop/   yield/   no_entry/
"""

import os
import csv
import shutil
import zipfile
import urllib.request
from pathlib import Path
from collections import defaultdict

import random
random.seed(42)

from PIL import Image

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

# 5 GTSRB class IDs → human-readable folder names
# Full label list: https://benchmark.ini.rub.de/gtsrb_dataset.html
SELECTED_CLASSES = {
    0:  "speed_limit_20",
    1:  "speed_limit_30",
    13: "yield",
    14: "stop",
    17: "no_entry",
}

IMG_SIZE   = (32, 32)
TRAIN_FRAC = 0.80
DATA_DIR   = Path("data")
RAW_DIR    = Path("raw_gtsrb")

# Official GTSRB training archive (single ZIP ~310 MB)
GTSRB_URL  = (
    "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
    "GTSRB_Final_Training_Images.zip"
)
ZIP_PATH   = RAW_DIR / "GTSRB_Final_Training_Images.zip"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1_048_576
        tot = total_size / 1_048_576
        print(f"\r  {pct:5.1f}%  {mb:.1f} / {tot:.1f} MB", end="", flush=True)


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] already downloaded: {dest}")
        return
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def extract(zip_path: Path, extract_to: Path):
    if (extract_to / "GTSRB").exists():
        print(f"  [skip] already extracted: {extract_to / 'GTSRB'}")
        return
    print(f"  Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("  Done.")


def read_csv_annotations(csv_path: Path) -> list[dict]:
    """Parse GTSRB GT annotation CSV (semicolon-separated)."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)
    return rows


def collect_images_per_class(gtsrb_train_dir: Path) -> dict[int, list[Path]]:
    """
    Walk every class sub-folder (00000 … 00042), read the GT CSV,
    and return {class_id: [abs_image_path, …]} for selected classes only.
    """
    result: dict[int, list[Path]] = defaultdict(list)

    for class_folder in sorted(gtsrb_train_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        class_id = int(class_folder.name)
        if class_id not in SELECTED_CLASSES:
            continue

        csv_file = class_folder / f"GT-{class_folder.name}.csv"
        if not csv_file.exists():
            print(f"  [warn] no annotation CSV in {class_folder}, skipping.")
            continue

        annotations = read_csv_annotations(csv_file)
        for ann in annotations:
            img_path = class_folder / ann["Filename"]
            if img_path.exists():
                result[class_id].append(img_path)

    return result


def resize_and_save(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        img.save(dst, format="JPEG", quality=95)


def split_and_copy(images: list[Path], class_name: str, train_frac: float):
    """Shuffle, split, resize, and write images into data/train & data/test."""
    images = images.copy()
    random.shuffle(images)

    split_at    = int(len(images) * train_frac)
    train_imgs  = images[:split_at]
    test_imgs   = images[split_at:]

    for subset, subset_imgs in (("train", train_imgs), ("test", test_imgs)):
        out_dir = DATA_DIR / subset / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(subset_imgs):
            dst = out_dir / f"{class_name}_{i:05d}.jpg"
            resize_and_save(src, dst)

    return len(train_imgs), len(test_imgs)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 56)
    print("  GTSRB Dataset Preparation")
    print("=" * 56)

    # 1. Download
    print("\n[1/4] Downloading dataset …")
    download(GTSRB_URL, ZIP_PATH)

    # 2. Extract
    print("\n[2/4] Extracting archive …")
    extract(ZIP_PATH, RAW_DIR)

    # 3. Collect annotated image paths
    gtsrb_train_dir = RAW_DIR / "GTSRB" / "Final_Training" / "Images"
    if not gtsrb_train_dir.exists():
        raise FileNotFoundError(
            f"Expected GTSRB training images at {gtsrb_train_dir}.\n"
            "Check that the ZIP extracted correctly."
        )

    print("\n[3/4] Collecting images for selected classes …")
    class_images = collect_images_per_class(gtsrb_train_dir)

    for class_id, name in SELECTED_CLASSES.items():
        n = len(class_images.get(class_id, []))
        print(f"  Class {class_id:>2} ({name:<20}) : {n:>5} images")

    # 4. Resize + split + write
    print(f"\n[4/4] Resizing to {IMG_SIZE}, splitting 80/20, writing to data/ …")

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)   # start clean

    totals = {"train": 0, "test": 0}
    for class_id, class_name in SELECTED_CLASSES.items():
        images = class_images.get(class_id, [])
        if not images:
            print(f"  [warn] No images found for class {class_id} ({class_name}), skipping.")
            continue
        n_train, n_test = split_and_copy(images, class_name, TRAIN_FRAC)
        totals["train"] += n_train
        totals["test"]  += n_test
        print(f"  {class_name:<22}  train={n_train:>4}  test={n_test:>4}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  Done!")
    print(f"  Total training images : {totals['train']}")
    print(f"  Total test images     : {totals['test']}")
    print(f"  Output directory      : {DATA_DIR.resolve()}")
    print("=" * 56)
    print("\nFolder structure:")
    for subset in ("train", "test"):
        for class_name in SELECTED_CLASSES.values():
            folder = DATA_DIR / subset / class_name
            n = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
            print(f"  data/{subset}/{class_name}/  ({n} images)")
    print()


if __name__ == "__main__":
    main()