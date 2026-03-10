# SmartMine Vision Lab (Flask)

## Setup (Windows)
1. Create & activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Train (ResNet‑101)
The default training script uses **ResNet-101** (recommended for best accuracy). To train:
```powershell
python ai-model/train.py --epochs 40 --batch_size 32
```

If you want a faster/smaller model for quick experimentation, add `--fast` to train ResNet‑18:
```powershell
python ai-model/train.py --fast --epochs 20 --batch_size 32
```

## Run the web UI
```powershell
python flask_mineral_app.py
```

## UI
You can see the UI by opening:

`http://localhost:5002`

### Notes
- The Flask app prefers the **ResNet-101** checkpoint file (`ai-model/models/resnet101_mineral.pth`).
- If you want to force a different checkpoint, set `MINERAL_MODEL_PATH` before starting the app.

---

## Option 2: Use a TensorFlow Lite (Edge) object detection model
This project includes helper scripts to download a TFLite object detector + labels, and run inference using `tflite-support`.

### A) Quick download scripts (recommended)
A ready-made downloader is included in the repo:

```powershell
python download_model.py
```

This will create:
- `models/model.tflite`
- `models/labels.txt`

You can then verify the downloaded model with:

```powershell
python test_model.py
```

### B) Alternative: use the built-in helper modules
If you want to manage the download manually (or keep it inside `ai-model/`), use the existing helper scripts:

```powershell
python ai-model/download_tflite_model.py
```

This saves:
- `ai-model/models/tflite/model.tflite`
- `ai-model/models/tflite/labels.txt`

#### ⚠️ Windows + Python 3.13 users (common failure)
On Windows, `tflite-support` sometimes fails to install because it needs Microsoft Visual C++ Build Tools.
If you ran into a `Microsoft Visual C++ 14.0 or greater is required` error, you have two options:

1) **Install the build tools** (recommended if you want tflite-support)
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - After installing, run:
     ```powershell
     pip install tflite-support
     ```

2) **Use TensorFlow instead (no build tools needed)**
   - Install the CPU version (larger download but prebuilt):
     ```powershell
     pip install tensorflow-cpu
     ```

Once `tflite-support` (or `tensorflow`) is installed, run this to verify:

```powershell
python test_model.py
```

---

And to run a quick inference test (once tflite-support is installed):

```powershell
python -c "from ai_model.tflite_detector import load_detector; d,l = load_detector('ai-model/models/tflite/model.tflite'); print('loaded', len(l), 'labels')"
```

### C) Use it in your UI
See `ai-model/tflite_detector.py` for a simple helper that loads the model and runs detection on a file. If you want, I can help you wire it into `flask_mineral_app.py` so your web UI shows detected objects automatically.
