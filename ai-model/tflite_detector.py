"""TFLite object detection (Google Edge) helper.

This module supports two inference backends:

1) **tflite-support** (preferred, smaller dependency) - uses the TFLite Task API.
2) **tensorflow** (fallback) - uses the low-level Interpreter (works without MSVC).

Usage example:
    from ai_model.tflite_detector import load_detector, detect_image

    detector, labels = load_detector("ai-model/models/tflite/model.tflite")
    detections = detect_image(detector, labels, "path/to/image.jpg")

The result is a list of detections with class name, score, and bounding box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Prefer tflite-support if available (smaller runtime, optimized pipes)
try:
    from tflite_support.task import core, processor, vision  # type: ignore

    _TFLITE_SUPPORT_AVAILABLE = True
except Exception:
    _TFLITE_SUPPORT_AVAILABLE = False

# Fallback to TensorFlow (includes tflite runtime) when tflite_support isn't available
try:
    import tensorflow as tf  # type: ignore

    _TF_AVAILABLE = True
except Exception:
    _TF_AVAILABLE = False


def load_labels(label_path: str) -> List[str]:
    """Load labels from a text file (one label per line)."""
    path = Path(label_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    return labels


class TfLiteInterpreterDetector:
    """A small wrapper around TensorFlow Lite Interpreter for object detection."""

    def __init__(
        self,
        model_path: str,
        max_results: int = 5,
        score_threshold: float = 0.2,
    ):
        if not _TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is not available. Install it with `pip install tensorflow-cpu`."
            )

        self.model_path = model_path
        self.max_results = max_results
        self.score_threshold = score_threshold

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()[0]
        self.input_index = input_details["index"]
        self.input_shape = tuple(input_details["shape"])
        self.input_dtype = input_details["dtype"]

        self.output_details = self.interpreter.get_output_details()

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        target_size = (self.input_shape[2], self.input_shape[1])
        img = image.resize(target_size)
        arr = np.asarray(img).astype(np.float32)
        # Most SSD MobileNet models expect [0,1] normalization.
        arr = arr / 255.0
        if self.input_dtype == np.uint8:
            arr = (arr * 255).astype(np.uint8)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        input_arr = self._prepare_image(image)
        self.interpreter.set_tensor(self.input_index, input_arr)
        self.interpreter.invoke()

        # Typical SSD outputs: boxes, classes, scores, num
        # We try to detect output order for robustness.
        outputs = {o["name"]: self.interpreter.get_tensor(o["index"]) for o in self.output_details}

        # Heuristic mapping
        boxes = outputs.get("StatefulPartitionedCall:0") or outputs.get("TFLite_Detection_PostProcess") or self.interpreter.get_tensor(self.output_details[0]["index"])
        classes = outputs.get("StatefulPartitionedCall:1") or outputs.get("TFLite_Detection_PostProcess:1") or self.interpreter.get_tensor(self.output_details[1]["index"])
        scores = outputs.get("StatefulPartitionedCall:2") or outputs.get("TFLite_Detection_PostProcess:2") or self.interpreter.get_tensor(self.output_details[2]["index"])

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        detections: List[Dict[str, Any]] = []
        for i in range(min(self.max_results, len(scores))):
            score = float(scores[i])
            if score < self.score_threshold:
                continue

            class_id = int(classes[i])
            ymin, xmin, ymax, xmax = boxes[i]

            detections.append(
                {
                    "class_id": class_id,
                    "score": score,
                    "bounding_box": {
                        "xmin": float(xmin),
                        "ymin": float(ymin),
                        "xmax": float(xmax),
                        "ymax": float(ymax),
                    },
                }
            )

        return detections


def load_detector(
    model_path: str,
    max_results: int = 5,
    score_threshold: float = 0.2,
) -> tuple[Any, List[str]]:
    """Load an object detector and labels.

    Returns:
        (detector, labels)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    labels_path = model_path.with_name("labels.txt")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found next to model: {labels_path}"
        )

    labels = load_labels(str(labels_path))

    if _TFLITE_SUPPORT_AVAILABLE:
        base_options = core.BaseOptions(file_name=str(model_path))
        detection_options = processor.DetectionOptions(
            max_results=max_results, score_threshold=score_threshold
        )
        options = vision.ObjectDetectorOptions(
            base_options=base_options, detection_options=detection_options
        )
        detector = vision.ObjectDetector.create_from_options(options)
        return detector, labels

    if _TF_AVAILABLE:
        detector = TfLiteInterpreterDetector(
            model_path=str(model_path),
            max_results=max_results,
            score_threshold=score_threshold,
        )
        return detector, labels

    raise ImportError(
        "No supported TFLite runtime available. Install `tflite-support` or `tensorflow-cpu`."
    )


def detect_image(
    detector: Any,
    labels: List[str],
    image_path: str,
) -> List[Dict[str, Any]]:
    """Run object detection on a single image.

    This supports either the `tflite_support` Task API detector or a
    lightweight TensorFlow Lite Interpreter wrapper.
    """
    img = Image.open(image_path).convert("RGB")

    # If the detector is the Task API object, it expects a TensorImage.
    if _TFLITE_SUPPORT_AVAILABLE and hasattr(detector, "detect"):
        try:
            input_tensor = vision.TensorImage.create_from_array(np.asarray(img))
            detection_result = detector.detect(input_tensor)
            detections = []
            for obj in detection_result.detections:
                category = obj.classes[0]
                class_id = int(category.class_id)
                score = float(category.score)
                class_name = labels[class_id] if class_id < len(labels) else str(class_id)

                bbox = obj.bounding_box  # Normalized [0,1] coords (xmin, ymin, xmax, ymax)
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "score": score,
                        "bounding_box": {
                            "xmin": float(bbox.origin_x),
                            "ymin": float(bbox.origin_y),
                            "xmax": float(bbox.origin_x + bbox.width),
                            "ymax": float(bbox.origin_y + bbox.height),
                        },
                    }
                )
            return detections
        except Exception:
            # Fall through to the interpreter-based path
            pass

    # Fallback to TensorFlow Lite Interpreter wrapper
    detections = detector.detect(img)

    for det in detections:
        class_id = det.get("class_id")
        det["class_name"] = labels[class_id] if class_id is not None and class_id < len(labels) else str(class_id)

    return detections


def draw_detections(
    image_path: str,
    detections: List[Dict[str, Any]],
    output_path: str,
    line_width: int = 3,
    font_size: int = 16,
) -> str:
    """Draw bounding boxes & labels on an image and save it."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    w, h = img.size
    for det in detections:
        bbox = det["bounding_box"]
        xmin = max(0, min(w, bbox["xmin"] * w))
        ymin = max(0, min(h, bbox["ymin"] * h))
        xmax = max(0, min(w, bbox["xmax"] * w))
        ymax = max(0, min(h, bbox["ymax"] * h))

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=line_width)
        label = f"{det['class_name']} {det['score']:.2f}"
        text_size = draw.textsize(label, font=font)
        text_background = [xmin, ymin - text_size[1] - 4, xmin + text_size[0] + 4, ymin]
        draw.rectangle(text_background, fill="black")
        draw.text((xmin + 2, ymin - text_size[1] - 2), label, fill="white", font=font)

    img.save(output_path)
    return output_path


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run TFLite object detection on an image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--model",
        default="ai-model/models/tflite/model.tflite",
        help="Path to the TFLite model file",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of detection results",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Minimum score threshold for detections",
    )
    args = parser.parse_args()

    detector, labels = load_detector(
        args.model, max_results=args.max_results, score_threshold=args.score_threshold
    )
    detections = detect_image(detector, labels, args.image)
    print(json.dumps(detections, indent=2))
