import sys
from pathlib import Path

# Allow importing from the ai-model package folder
sys.path.insert(0, str(Path(__file__).resolve().parent / "ai-model"))

from tflite_detector import load_detector

print("🔍 Testing TFLite model...")

try:
    model_path = "models/model.tflite"
    detector, labels = load_detector(model_path)
    print("✅ Model loaded successfully!")
    print(f"📋 Labels file contains {len(labels)} classes")
    print("   First 5 labels:", labels[:5])

except Exception as e:
    print(f"❌ Error: {e}")
