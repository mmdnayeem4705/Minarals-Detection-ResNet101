"""Download a TensorFlow Lite object detection model + labels.

This script follows "Option 2: Get Model + Labels Together" (recommended) and
saves the model + label file under `ai-model/models/tflite/`.

Usage:
    python ai-model/download_tflite_model.py

After downloading, you can run inference with `ai-model/tflite_detector.py`.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "tflite"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URL = "https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v2/2/default/1?lite-format=tflite"
LABEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_3_2020_05_26/labels.txt"

MODEL_FILE = MODEL_DIR / "model.tflite"
LABEL_FILE = MODEL_DIR / "labels.txt"


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, str(dest))
    print(f"✅ Saved {dest}")


def main() -> None:
    download(MODEL_URL, MODEL_FILE)
    download(LABEL_URL, LABEL_FILE)


if __name__ == "__main__":
    main()
