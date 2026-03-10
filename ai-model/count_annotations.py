"""Count object instances from Demo annotation CSV files.

This is useful when you want to know how many instances of each class (e.g., minerals)
are present in the dataset (before or after training).

Usage:
  python ai-model/count_annotations.py --annotations Demo/train/_annotations.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def count_objects(annotation_csv: Path) -> Counter[str]:
    counts = Counter()
    with annotation_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row.get("class") or row.get("label")
            if not cls:
                continue
            counts[cls.strip()] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count objects in annotation CSV")
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to the annotation CSV file (e.g., Demo/train/_annotations.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.annotations)
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {csv_path}")

    counts = count_objects(csv_path)
    total = sum(counts.values())

    print(f"Total object annotations: {total}")
    for cls, cnt in counts.most_common():
        print(f"  {cls}: {cnt}")


if __name__ == "__main__":
    main()
