"""Build a balanced mineral dataset for reliable classification.

Creates ai-model/dataset_balanced/ with train/val/test splits using
8-12 visually distinct classes with enough samples. This produces a model
that actually differentiates between minerals instead of predicting the
same class for everything.

Usage:
  python scripts/build_balanced_mineral_dataset.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List

# Classes with sufficient samples and visually distinct
# (minerals, rocks, ores - diverse enough for the model to learn)
TARGET_CLASSES = [
    "Calcite", "Fluorite", "Pyrite", "Baryte",  # minerals/ores
    "Granite", "Marble", "Quartzite",  # rocks
    "Limestone", "Basalt", "Gneiss",  # more variety
]

MIN_PER_CLASS = 35
MAX_PER_CLASS = 200
TRAIN_FRAC = 0.75
VAL_FRAC = 0.15
TEST_FRAC = 0.10
SEED = 42


def main() -> None:
    random.seed(SEED)
    source = Path("ai-model/dataset")
    output = Path("ai-model/dataset_balanced")
    output.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    for split in ("train", "val", "test"):
        (output / split).mkdir(exist_ok=True)

    total_copied = 0
    for class_name in TARGET_CLASSES:
        src_dir = source / "train" / class_name
        if not src_dir.exists():
            src_dir = source / "val" / class_name
        if not src_dir.exists():
            print(f"  Skipping {class_name}: not found")
            continue

        images = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]
        if len(images) < MIN_PER_CLASS:
            print(f"  Skipping {class_name}: only {len(images)} images (need {MIN_PER_CLASS})")
            continue

        random.shuffle(images)
        n = min(len(images), MAX_PER_CLASS)
        images = images[:n]

        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        n_test = n - n_train - n_val

        splits: List[tuple[str, int]] = [
            ("train", n_train),
            ("val", n_val),
            ("test", n_test),
        ]
        start = 0
        for split_name, count in splits:
            if count <= 0:
                continue
            end = start + count
            split_images = images[start:end]
            start = end

            out_dir = output / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for src in split_images:
                dst = out_dir / src.name
                if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
                    total_copied += 1

        print(f"  {class_name}: {n} images -> train={n_train}, val={n_val}, test={n_test}")

    print(f"\nDataset built at {output}")
    n_classes = len([d for d in (output / "train").iterdir() if d.is_dir()])
    print(f"Classes: {n_classes}")
    print(f"Total images copied: {total_copied}")


if __name__ == "__main__":
    main()
