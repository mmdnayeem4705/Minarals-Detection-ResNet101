"""Flask UI for image-based mineral/rock classification."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
import sys

from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "ai_model"))

try:
    from inference import load_model, predict_from_bytes  # type: ignore[import-not-found]  # noqa: E402
    INFERENCE_AVAILABLE = True
except Exception:
    INFERENCE_AVAILABLE = False
    load_model = None
    predict_from_bytes = None

# Optional TFLite object detection (uses either tflite-support or tensorflow)
try:
    from ai_model.tflite_detector import draw_detections, detect_image, load_detector  # type: ignore[import-not-found]  # noqa: E402
    TFLITE_AVAILABLE = True
except Exception:
    TFLITE_AVAILABLE = False
    draw_detections = None
    detect_image = None
    load_detector = None


BASE_DIR = Path(__file__).resolve().parent
STATIC_UPLOAD_DIR = BASE_DIR / "static" / "uploads"
STATIC_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "change-me-in-prod")
    app.config["UPLOAD_FOLDER"] = str(STATIC_UPLOAD_DIR)

    # Prefer ResNet-101 by default, fall back to ResNet-18 if the checkpoint is missing.
    default_path = BASE_DIR / "ai_model" / "models" / "resnet101_mineral.pth"
    if not default_path.exists():
        default_path = BASE_DIR / "ai_model" / "models" / "resnet18_mineral.pth"
    model_path_env = os.environ.get("MINERAL_MODEL_PATH", str(default_path))
    device = os.environ.get("MINERAL_MODEL_DEVICE", "cpu")

    try:
        if INFERENCE_AVAILABLE:
            model, class_names = load_model(model_path_env, device=device)
        else:
            model = None
            class_names = []
    except FileNotFoundError:
        model = None
        class_names = []
    except Exception:
        model = None
        class_names = []

    # Optional TFLite object detector (uses tflite-support or tensorflow)
    tflite_detector = None
    tflite_labels: list[str] = []
    tflite_error = None

    if TFLITE_AVAILABLE:
        tflite_model_path = os.environ.get(
            "TFLITE_MODEL_PATH",
            str(BASE_DIR / "models" / "model.tflite"),
        )
        try:
            tflite_detector, tflite_labels = load_detector(tflite_model_path)
        except Exception as exc:  # noqa: BLE001
            tflite_error = str(exc)
    else:
        tflite_error = "TFLite support not available (tflite-support or tensorflow not installed or incompatible)"

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/", methods=["GET", "POST"])
    def index():
        error = None
        prediction_result = None
        uploaded_image_url = None
        detection_results = None
        detection_image_url = None

        if request.method == "POST":
            file = request.files.get("image")
            if not file or file.filename == "":
                error = "Please choose an image file to upload."
            elif not allowed_file(file.filename):
                error = "Unsupported file type. Please upload a PNG, JPG, or WEBP image."
            else:
                safe_name = secure_filename(file.filename)
                timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
                filename = f"{timestamp}-{safe_name}"
                save_path = STATIC_UPLOAD_DIR / filename
                file_bytes = file.read()
                save_path.write_bytes(file_bytes)

                # Always show the uploaded sample
                uploaded_image_url = url_for(
                    "uploaded_file", filename=filename, _external=False
                )

                # Try object detection (TFLite) if available
                if tflite_detector is not None:
                    try:
                        detection_results = detect_image(
                            tflite_detector, tflite_labels, str(save_path)
                        )
                        overlay_path = STATIC_UPLOAD_DIR / f"{timestamp}-detected-{safe_name}"
                        draw_detections(
                            str(save_path),
                            detection_results,
                            str(overlay_path),
                        )
                        detection_image_url = url_for(
                            "uploaded_file", filename=overlay_path.name, _external=False
                        )
                    except Exception as exc:  # noqa: BLE001
                        # Keep running the classic classifier if object detection fails
                        error = (
                            "TFLite detection failed (is the model downloaded?): "
                            f"{exc}"
                        )

                # Fallback: run ResNet classifier if the model is loaded
                if model is not None and (prediction_result is None):
                    try:
                        prediction_result = predict_from_bytes(
                            file_bytes,
                            model=model,
                            class_names=class_names,
                            device=device,
                        )
                    except Exception as exc:  # noqa: BLE001
                        error = f"Failed to run prediction: {exc}"

        return render_template(
            "index.html",
            error=error,
            prediction=prediction_result,
            uploaded_image_url=uploaded_image_url,
            class_names=class_names,
            detection_results=detection_results,
            detection_image_url=detection_image_url,
            tflite_error=tflite_error,
        )

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        return redirect(url_for("static", filename=f"uploads/{filename}"))

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=5002, debug=False)

