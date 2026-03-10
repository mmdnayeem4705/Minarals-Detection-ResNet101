"""Prepare an ImageFolder-style dataset from the Demo object-detection annotations.

The Demo dataset is a Roboflow-style object detection dataset with CSV annotations
(contains fields: filename,width,height,class,xmin,ymin,xmax,ymax).

This script selects a single label per image (the one with the largest bounding box)
and copies the image into `ai-model/dataset/{split}/{class}` so the existing
classification training pipeline can be used without modification.

Example usage:
  python ai-model/prepare_demo_dataset.py \
    --demo_dir Demo \
    --output_dir ai-model/dataset \
    --splits train valid test

This will create the following structure:
  ai-model/dataset/train/<class>/*.jpg
  ai-model/dataset/valid/<class>/*.jpg
  ai-model/dataset/test/<class>/*.jpg

The script also prints class counts per split.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def load_annotations(csv_path: Path) -> Dict[str, Tuple[str, int]]:
    """Load annotations and return best label per image.

    For images with multiple labels, the label corresponding to the largest
    bounding box area is selected.

    Returns:
        mapping from filename -> (class_name, area)
    """

    best: Dict[str, Tuple[str, int]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            class_name = row["class"].strip()
            try:
                xmin = int(float(row["xmin"]))
                ymin = int(float(row["ymin"]))
                xmax = int(float(row["xmax"]))
                ymax = int(float(row["ymax"]))
            except (KeyError, ValueError):
                continue

            area = max(0, xmax - xmin) * max(0, ymax - ymin)
            prev = best.get(filename)
            if prev is None or area > prev[1]:
                best[filename] = (class_name, area)

    return best


def build_imagefolder_from_demo(
    demo_dir: Path,
    output_dir: Path,
    splits: Iterable[str],
    force: bool = False,
) -> None:
    """Convert a Roboflow-style demo dataset into an ImageFolder dataset."""

    output_dir = output_dir.expanduser().resolve()
    demo_dir = demo_dir.expanduser().resolve()

    for split in splits:
        split_dir = demo_dir / split
        annotations_csv = split_dir / "_annotations.csv"
        if not annotations_csv.exists():
            print(f"WARNING: Missing annotations for split '{split}': {annotations_csv}")
            continue

        # Map common split names to ImageFolder expected names
        out_split = "val" if split in {"valid", "val"} else split

        print(f"Preparing split '{split}' (-> {out_split}) using annotations: {annotations_csv}")
        best_labels = load_annotations(annotations_csv)

        split_out_dir = output_dir / out_split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        counts: Counter[str] = Counter()
        copied = 0
        skipped = 0

        for filename, (class_name, _) in best_labels.items():
            src_path = split_dir / filename
            if not src_path.exists():
                skipped += 1
                continue

            dst_class_dir = split_out_dir / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_class_dir / filename

            if dst_path.exists() and not force:
                skipped += 1
            else:
                shutil.copy2(src_path, dst_path)
                copied += 1

            counts[class_name] += 1

        print(f"  Copied {copied} images (skipped {skipped} missing/existing) to {split_out_dir}")
        print(f"  Class distribution ({len(counts)} classes):")
        for class_name, cnt in counts.most_common():
            print(f"    {class_name}: {cnt}")
        print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ImageFolder dataset from Demo object-detection annotations"
    )
    parser.add_argument(
        "--demo_dir",
        type=str,
        default="Demo",
        help="Path to the Demo folder containing train/valid/test splits",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ai-model/dataset",
        help="Output ImageFolder dataset directory",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Splits to prepare (default: train valid test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in the output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_imagefolder_from_demo(
        demo_dir=Path(args.demo_dir),
        output_dir=Path(args.output_dir),
        splits=args.splits,
        force=args.force,
    )


if __name__ == "__main__":
    main()
