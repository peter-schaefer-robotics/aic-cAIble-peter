#!/usr/bin/env python3
"""Quick test: subscribe to center_camera/image, run Depth-Anything-V2, save result.

Run inside the pixi env:
  pixi run pip install transformers timm tokenizers   # one-time
  pixi run python3 scripts/test_depth.py
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def ros_image_to_numpy(msg: Image) -> np.ndarray:
    """Decode sensor_msgs/Image to uint8 H×W×3 RGB — no cv_bridge needed."""
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = data.reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1].copy()
    elif msg.encoding == "rgba8":
        img = data.reshape(msg.height, msg.width, 4)[:, :, :3]
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    return img


class DepthTest(Node):
    def __init__(self):
        super().__init__("depth_test")
        self._done = False

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Device: {device_name}")
        self.get_logger().info(f"Loading {MODEL} …")

        self._proc = AutoImageProcessor.from_pretrained(MODEL)
        self._model = AutoModelForDepthEstimation.from_pretrained(MODEL)
        self._model = self._model.to(device_name).eval()
        self._device = device_name

        self.get_logger().info("Model ready. Waiting for /center_camera/image …")
        self.create_subscription(Image, "/center_camera/image", self._cb, 1)

    def _cb(self, msg: Image):
        if self._done:
            return
        self._done = True

        rgb = ros_image_to_numpy(msg)
        h, w = rgb.shape[:2]
        self.get_logger().info(f"Got image {w}x{h}, running depth estimation …")

        pil = PILImage.fromarray(rgb)
        inputs = self._proc(images=pil, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        depth = self._proc.post_process_depth_estimation(
            outputs, target_sizes=[(h, w)]
        )[0]["predicted_depth"].cpu().numpy().astype(np.float32)

        self.get_logger().info(
            f"Depth range: {depth.min():.3f} – {depth.max():.3f} m"
        )

        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite("depth_output.png", np.hstack([rgb_bgr, depth_color]))
        self.get_logger().info("Saved depth_output.png — done.")


def main():
    rclpy.init()
    node = DepthTest()
    while not node._done:
        rclpy.spin_once(node, timeout_sec=0.1)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
