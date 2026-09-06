#!/usr/bin/env python3
"""Print raw bounding box labels from Gazebo to fix class mapping.

Usage:
  pixi run python3 aic_foundationpose/scripts/debug_bbox_labels.py
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from vision_msgs.msg import Detection2DArray

qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.VOLATILE)

CAMERAS = ["left_camera", "center_camera", "right_camera"]

class LabelPrinter(Node):
    def __init__(self):
        super().__init__("label_printer")
        for cam in CAMERAS:
            self.create_subscription(
                Detection2DArray, f"/{cam}/bounding_boxes",
                lambda msg, c=cam: self._cb(msg, c), qos_reliable)
        self.get_logger().info("Waiting for bounding box messages...")

    def _cb(self, msg, cam):
        if not msg.detections:
            return
        self.get_logger().info(f"\n=== {cam} ({len(msg.detections)} detections) ===")
        for det in msg.detections:
            label = det.results[0].hypothesis.class_id if det.results else "<no results>"
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w  = det.bbox.size_x
            h  = det.bbox.size_y
            self.get_logger().info(f"  label='{label}'  bbox=({cx:.0f},{cy:.0f},{w:.0f}x{h:.0f})")

def main():
    rclpy.init()
    rclpy.spin(LabelPrinter())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
