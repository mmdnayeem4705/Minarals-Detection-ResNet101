import os
import sys
from pathlib import Path

import requests

print("🚀 Starting model download...")

# Create models folder if it doesn't exist
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)
print("📁 Using models folder:", models_dir.resolve())

# Model URL (SSD MobileNet V2) - TensorFlow Hub
model_url = "https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v2/2/default/1?lite-format=tflite"
model_path = models_dir / "model.tflite"

# Labels URL (COCO dataset labels - 90 common objects)
label_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_3_2020_05_26/labels.txt"
label_path = models_dir / "labels.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def download_file(url: str, dest: Path) -> None:
    """Download a file with a browser-like User-Agent to avoid 403 blocks."""
    print(f"📥 Downloading {url} -> {dest}")

    with requests.get(url, headers=HEADERS, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # Quick sanity check: a valid TFLite file begins with b"TFL3"
    if dest.suffix == ".tflite":
        with open(dest, "rb") as f:
            magic = f.read(4)
        if magic != b"TFL3":
            raise RuntimeError(
                f"Downloaded file does not look like a TFLite model (magic={magic!r}).\n"
                "If you are behind a proxy/firewall, try downloading the file manually."
            )


try:
    download_file(model_url, model_path)
    print(f"✅ Model downloaded to: {model_path}")
except Exception as e:
    print(f"❌ Failed to download model: {e}")
    print("Please download the model manually from:")
    print("  https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v2/2/default/1?lite-format=tflite")
    sys.exit(1)

try:
    download_file(label_url, label_path)
    print(f"✅ Labels downloaded to: {label_path}")
except Exception as e:
    print(f"❌ Failed to download labels: {e}")
    print("Please download the labels manually from:")
    print("  https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_3_2020_05_26/labels.txt")
    sys.exit(1)

print("\n✨ Download complete! Files are in the 'models' folder.")
print("📂 Location:", models_dir.resolve())
