"""
preprocess.py — Stage 1: Image Capture & Data Intake / AI Preprocessing
========================================================================
Walks data/raw/ (handling the extra nesting layer in each top-level folder),
collects up to MAX_PER_CLASS images per category, resizes them to 224×224,
normalises pixel values to [0, 1], and writes the results into
data/processed/train/ and data/processed/val/ mirroring the class-folder
structure expected by Keras' ImageDataGenerator.flow_from_directory().

Run:
    python backend/preprocess.py
"""

import os
import sys
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent.parent
RAW_DIR        = REPO_ROOT / "data" / "raw"
PROCESSED_DIR  = REPO_ROOT / "data" / "processed"
IMG_SIZE       = (224, 224)          # MobileNetV2 expected input
VAL_SPLIT      = 0.20                # 80 / 20 train-val split
MAX_PER_CLASS  = 250                 # cap per TOP-LEVEL class (fast demo)
SEED           = 42
VALID_EXTS     = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Top-level class names as they appear in data/raw/
TOP_CLASSES = ["Biodegradable", "Hazardous", "Non-Recyclable", "Recyclable"]


def collect_images(class_dir: Path, limit: int) -> list[Path]:
    """
    Recursively collect image paths under class_dir regardless of how
    many intermediate sub-folders exist.  Returns a shuffled, capped list.
    """
    images = [
        p for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]
    random.shuffle(images)
    return images[:limit]


def save_processed(src_path: Path, dst_path: Path) -> bool:
    """
    Open src_path, resize to IMG_SIZE, normalise to float32 [0,1], then
    save as PNG (lossless, avoids re-encoding artefacts).
    Returns True on success, False if the image is unreadable.
    """
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize(IMG_SIZE, Image.LANCZOS)

            # Normalise pixel values → float32 [0, 1]
            arr = np.array(img, dtype=np.float32) / 255.0

            # Convert back to uint8 PNG so Keras can load it normally
            out = Image.fromarray((arr * 255).astype(np.uint8))
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            out.save(dst_path, format="PNG")
        return True
    except Exception as exc:
        print(f"  [WARN] Skipping {src_path.name}: {exc}")
        return False


def preprocess():
    random.seed(SEED)
    np.random.seed(SEED)

    # Wipe & recreate processed dirs for a clean run
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    (PROCESSED_DIR / "train").mkdir(parents=True)
    (PROCESSED_DIR / "val").mkdir(parents=True)

    summary = {}

    for cls in TOP_CLASSES:
        cls_raw_dir = RAW_DIR / cls
        if not cls_raw_dir.exists():
            print(f"[SKIP] {cls} — folder not found at {cls_raw_dir}")
            continue

        images = collect_images(cls_raw_dir, MAX_PER_CLASS)
        if not images:
            print(f"[SKIP] {cls} — no images found")
            continue

        # Stratified split
        split_idx  = int(len(images) * (1 - VAL_SPLIT))
        train_imgs = images[:split_idx]
        val_imgs   = images[split_idx:]

        print(f"\n[{cls}]  total={len(images)}  train={len(train_imgs)}  val={len(val_imgs)}")

        ok_train = ok_val = 0

        for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
            for i, src in enumerate(img_list):
                dst = PROCESSED_DIR / split / cls / f"{cls}_{split}_{i:04d}.png"
                if save_processed(src, dst):
                    if split == "train":
                        ok_train += 1
                    else:
                        ok_val += 1

        summary[cls] = {"train": ok_train, "val": ok_val}
        print(f"  Saved → train/{cls}: {ok_train}  val/{cls}: {ok_val}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("Preprocessing complete!")
    print(f"Output directory: {PROCESSED_DIR}")
    print(f"{'Class':<20}  {'Train':>6}  {'Val':>6}")
    print("─" * 38)
    for cls, counts in summary.items():
        print(f"  {cls:<18}  {counts['train']:>6}  {counts['val']:>6}")
    print("═" * 55)


if __name__ == "__main__":
    preprocess()
