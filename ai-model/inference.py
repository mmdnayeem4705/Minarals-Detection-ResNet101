"""Inference helpers (load + predict)."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from models.resnet101 import SmartMineResNet101, SmartMineResNet18

_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(
    model_path: str,
    num_classes: int = 4,
    device: str = "cpu",
) -> tuple[SmartMineResNet101, list[str]]:
    """Load checkpoint and return (model, class_names)."""
    checkpoint = torch.load(model_path, map_location=device)
    class_names: list[str] = checkpoint.get(
        "class_names", [str(i) for i in range(num_classes)]
    )
    num_classes = len(class_names)
    model_type = checkpoint.get("model_type", "resnet101")

    if model_type == "resnet18":
        model = SmartMineResNet18(num_classes=num_classes)
    else:
        model = SmartMineResNet101(num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names


def preprocess_image(
    image_path_or_pil: Union[str, Image.Image],
) -> torch.Tensor:
    """Convert a file path or PIL Image to a tensor (1,3,224,224)."""
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    tensor = _INFERENCE_TRANSFORM(img)  # (3, 224, 224)
    return tensor.unsqueeze(0)  # (1, 3, 224, 224)


def predict_image(
    image_path: str,
    model: SmartMineResNet101,
    class_names: list[str],
    device: str = "cpu",
) -> dict:
    """Predict from a file path."""
    tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

    top_idx = int(probs.argmax())
    return {
        "class_name": class_names[top_idx],
        "probability": float(probs[top_idx]),
        "all_probabilities": {
            name: float(probs[i]) for i, name in enumerate(class_names)
        },
    }


def predict_from_bytes(
    image_bytes: bytes,
    model: SmartMineResNet101,
    class_names: list[str],
    device: str = "cpu",
) -> dict:
    """Predict from raw image bytes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _INFERENCE_TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_idx = int(probs.argmax())
    return {
        "class_name": class_names[top_idx],
        "probability": float(probs[top_idx]),
        "all_probabilities": {
            name: float(probs[i]) for i, name in enumerate(class_names)
        },
    }


def generate_gradcam(
    image_tensor: torch.Tensor,
    model: SmartMineResNet101,
    target_layer: torch.nn.Module,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap overlay for *image_tensor*.

    The heatmap is upsampled to 224 × 224 and blended with the original image.

    Args:
        image_tensor: Float tensor of shape ``(1, 3, 224, 224)``.
        model: ``SmartMineResNet101`` instance (kept in eval mode).
        target_layer: The convolutional layer to hook (e.g. ``model.layer4``).

    Returns:
        NumPy array of shape ``(224, 224, 3)`` (uint8 RGB) with the CAM overlay.
    """
    model.eval()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(image_tensor)
        top_class = int(output.argmax(dim=1))
        model.zero_grad()
        output[0, top_class].backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    # Global-average-pool the gradients
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = (weights * activations[0]).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    cam = F.relu(cam)

    # Upsample to input resolution
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Normalise to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    # Convert input image to numpy for overlay
    img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Undo ImageNet normalisation for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.clip(img_np * std + mean, 0.0, 1.0)

    # Apply a red-channel heatmap overlay
    heatmap = np.stack([cam, np.zeros_like(cam), np.zeros_like(cam)], axis=-1)
    overlay = np.clip(img_np * 0.6 + heatmap * 0.4, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)
