"""Thin wrapper around Depth-Anything-V2 Metric for metric depth estimation."""

from __future__ import annotations

import numpy as np


class DepthAnythingEstimator:
    """Loads Depth-Anything-V2 Metric and runs inference on RGB images.

    Uses the HuggingFace transformers API. The metric variant produces
    absolute depth in metres, which FoundationPose requires.
    """

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                 device: str = "cuda") -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch

        self._device = device
        self._torch = torch
        self._proc = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self._model = self._model.to(device).eval()

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        """Return metric depth map (H x W float32, metres).

        Args:
            rgb: uint8 H×W×3 numpy array in RGB order.
        """
        from PIL import Image as PILImage

        h, w = rgb.shape[:2]
        pil = PILImage.fromarray(rgb)
        inputs = self._proc(images=pil, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        result = self._proc.post_process_depth_estimation(
            outputs, target_sizes=[(h, w)]
        )[0]["predicted_depth"]
        return result.cpu().numpy().astype(np.float32)
