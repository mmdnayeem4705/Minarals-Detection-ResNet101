"""Count predicted class occurrences for a folder of images using a ResNet model."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import torch

# Allow imports when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from inference import load_model, predict_image


def list_images(folder: Path) -> Iterator[Path]:
    """Yield image file paths under *folder* (recursively)."""

    for p in folder.rglob("*.*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count predicted classes for a folder of images."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ai-model/models/resnet101_smartmine.pth",
        help="Path to the trained ResNet checkpoint",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="ai-model/dataset/test",
        help="Folder containing images to run predictions on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string",
    )
    args = parser.parse_args()

    model, class_names = load_model(args.model_path, device=args.device)
    print(f"Loaded model with classes: {class_names}")

    images_path = Path(args.images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Image folder '{images_path}' does not exist")

    counts: Counter[str] = Counter()
    total = 0

    for img_path in list_images(images_path):
        pred = predict_image(str(img_path), model, class_names, device=args.device)
        counts[pred["class_name"]] += 1
        total += 1

    print(f"\nProcessed {total} images.")
    print("Predicted counts per class:")
    for class_name, cnt in counts.most_common():
        print(f"  {class_name}: {cnt}")


if __name__ == "__main__":
    main()
