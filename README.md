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
app.py` so your web UI shows detected objects automatically.
