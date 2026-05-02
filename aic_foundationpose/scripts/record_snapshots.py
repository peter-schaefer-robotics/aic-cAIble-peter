#!/usr/bin/env python3
"""Synchronized RGBD snapshot recorder with ground-truth YOLO labels.

Bounding boxes are derived from the segmentation camera mask + TF object positions.
This gives per-instance boxes regardless of how many objects share the same class.

Saves per snapshot:
  {cam}_rgb.png          RGB image
  {cam}_depth.npy        float32 depth in metres (if available)
  {cam}_depth_vis.png    depth colormap
  {cam}_labels.txt       YOLO detection format  (class cx cy w h)
  {cam}_seg.txt          YOLO segmentation format (class polygon...)
  {cam}_seg_colored.png  Gazebo segmentation colormap
  {cam}_bbox_vis.png     RGB with bounding boxes overlaid
  meta.json              camera intrinsics + extrinsics, EEF, plugs, entrances

Usage:
  pixi run python3 aic_foundationpose/scripts/record_snapshots.py --once
  pixi run python3 aic_foundationpose/scripts/record_snapshots.py --interval 5
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import tf2_ros
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from vision_msgs.msg import Detection2DArray


# ─── constants ────────────────────────────────────────────────────────────────

CAMERAS    = ["left_camera", "center_camera", "right_camera"]
PLUG_FRAMES = ["cable_0", "cable_1", "cable_2", "cable_3", "cable_4"]
EEF_FRAME  = "tool0"
BASE_FRAME = "base_link"

CAMERA_OPTICAL = {
    "left_camera":   "left_camera/optical",
    "center_camera": "center_camera/optical",
    "right_camera":  "right_camera/optical",
}

# YOLO classes
YOLO_CLASSES = {
    0: ("nic_card",  (0,   255, 0)),    # full NIC card PCB
    1: ("sc_port",   (0,   128, 255)),  # SC port socket
    2: ("nic_port",  (255, 200, 0)),    # individual SFP slot entrance
}

# TF frame match rules: (must_contain, must_not_contain, class_id, half_extents_m)
FRAME_RULES = [
    # NIC card body — full PCB box
    ("nic_card_link", ["entrance", "mount_link", "sfp_port"], 0, np.array([0.035, 0.073, 0.012])),
    # Individual SFP port slot entrances
    ("sfp_port_0_link_entrance", [], 2, np.array([0.007, 0.004, 0.002])),
    ("sfp_port_1_link_entrance", [], 2, np.array([0.007, 0.004, 0.002])),
    # SC port socket
    ("sc_port_link", ["entrance", "base"], 1, np.array([0.013, 0.014, 0.005])),
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def ros_image_to_numpy(msg: Image) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = data.reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if msg.encoding == "bgr8" else img.copy()
    if msg.encoding == "rgba8":
        return data.reshape(msg.height, msg.width, 4)[:, :, :3].copy()
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    data_u8 = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == "16UC1":
        return data_u8.view(np.uint16).reshape(msg.height, msg.width).copy()
    return data_u8.reshape(msg.height, msg.width).astype(np.uint16)


def camera_info_to_dict(msg: CameraInfo) -> dict:
    return {"width": msg.width, "height": msg.height,
            "K": list(msg.k), "D": list(msg.d),
            "R": list(msg.r), "P": list(msg.p),
            "distortion_model": msg.distortion_model,
            "frame_id": msg.header.frame_id}


def tf_to_dict(tf: TransformStamped) -> dict:
    t, r = tf.transform.translation, tf.transform.rotation
    return {"parent_frame": tf.header.frame_id, "child_frame": tf.child_frame_id,
            "translation": {"x": t.x, "y": t.y, "z": t.z},
            "rotation":    {"x": r.x, "y": r.y, "z": r.z, "w": r.w}}


def tf_dict_to_matrix(d: dict) -> np.ndarray:
    t = d["translation"];  r = d["rotation"]
    T = np.eye(4)
    T[:3, 3]  = [t["x"], t["y"], t["z"]]
    T[:3, :3] = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]]).as_matrix()
    return T


def _yolo_seg_from_labels(labels, bboxes, img_w, img_h):
    lines = []
    for cls, cx, cy, bw, bh in bboxes:
        x1 = max(0, int((cx - bw/2) * img_w))
        y1 = max(0, int((cy - bh/2) * img_h))
        x2 = min(img_w, int((cx + bw/2) * img_w))
        y2 = min(img_h, int((cy + bh/2) * img_h))
        roi = labels[y1:y2, x1:x2]
        unique, counts = np.unique(roi[roi > 0], return_counts=True)
        if not len(unique):
            continue
        inst_id = unique[np.argmax(counts)]
        mask = (labels == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        eps    = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        pts = " ".join(f"{p[0]/img_w:.6f} {p[1]/img_h:.6f}" for p in approx)
        lines.append(f"{cls} {pts}\n")
    return lines


# ─── recorder ─────────────────────────────────────────────────────────────────

class SnapshotRecorder(Node):
    def __init__(self, output_dir: Path, interval: float):
        super().__init__("snapshot_recorder")
        self._output_dir = output_dir
        self._interval   = interval
        self._last_save  = time.monotonic()
        self._count      = 0
        self._classes_written = False

        self._rgb:        dict[str, np.ndarray] = {}
        self._depth:      dict[str, np.ndarray] = {}
        self._info:       dict[str, dict]       = {}
        self._seglabels:  dict[str, np.ndarray] = {}
        self._segcolored: dict[str, np.ndarray] = {}

        qos_sensor   = qos_profile_sensor_data
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                   durability=DurabilityPolicy.VOLATILE)

        for cam in CAMERAS:
            self.create_subscription(Image, f"/{cam}/image",
                lambda msg, c=cam: self._rgb_cb(msg, c), qos_sensor)
            self.create_subscription(Image, f"/{cam}/depth",
                lambda msg, c=cam: self._depth_cb(msg, c), qos_sensor)
            self.create_subscription(CameraInfo, f"/{cam}/camera_info",
                lambda msg, c=cam: self._info_cb(msg, c), qos_sensor)
            self.create_subscription(Image, f"/{cam}/segmentation/labels_map",
                lambda msg, c=cam: self._seglabels_cb(msg, c), qos_reliable)
            self.create_subscription(Image, f"/{cam}/segmentation/colored_map",
                lambda msg, c=cam: self._segcolored_cb(msg, c), qos_reliable)

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.create_timer(0.5, self._tick)
        self.get_logger().info(f"Recording to {output_dir}  |  interval: {interval}s")

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _rgb_cb(self, msg, cam):       self._rgb[cam]       = ros_image_to_numpy(msg)
    def _depth_cb(self, msg, cam):     self._depth[cam]     = ros_image_to_numpy(msg)
    def _info_cb(self, msg, cam):      self._info[cam]      = camera_info_to_dict(msg)
    def _segcolored_cb(self, msg, cam): self._segcolored[cam] = ros_image_to_numpy(msg)

    def _seglabels_cb(self, msg: Image, cam: str):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding == "16UC1":
            arr = data.view(np.uint16).reshape(msg.height, msg.width).copy()
        elif msg.encoding == "rgb8":
            rgb = data.reshape(msg.height, msg.width, 3)
            arr = (rgb[:,:,0].astype(np.uint32) * 256 + rgb[:,:,1]).astype(np.uint16)
        else:
            arr = data.reshape(msg.height, msg.width).astype(np.uint16)
        self._seglabels[cam] = arr

    # ── TF helpers ────────────────────────────────────────────────────────────

    def _lookup_tf(self, child, parent=BASE_FRAME):
        try:
            tf = self._tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
            return tf_to_dict(tf)
        except Exception:
            return None

    def _all_tf_frames(self):
        try:
            return list(yaml.safe_load(self._tf_buffer.all_frames_as_yaml()) or {})
        except Exception:
            return []

    def _lookup_entrances(self):
        result = {}
        for frame in self._all_tf_frames():
            if "entrance" in frame:
                p = self._lookup_tf(frame)
                if p: result[frame] = p
        return result

    def _lookup_camera_extrinsics(self):
        result = {}
        for cam, optical in CAMERA_OPTICAL.items():
            p = self._lookup_tf(optical)
            if p: result[cam] = p
        return result

    def _lookup_nic_card_frames(self):
        result = {}
        for frame in self._all_tf_frames():
            if "nic_card_link" in frame and "entrance" not in frame:
                p = self._lookup_tf(frame)
                if p: result[frame] = p
        return result

    # ── bbox from TF projection ───────────────────────────────────────────────

    def _detect_objects_from_tf(self, cam: str, img_w: int, img_h: int
                                 ) -> list[tuple[int, float, float, float, float]]:
        """Project 3D bounding boxes of known objects into the image using TF.
        Returns per-instance YOLO bboxes regardless of segmentation."""

        # Use the TF frame reported by camera_info (exact frame Gazebo renders from)
        cam_frame = self._info.get(cam, {}).get("frame_id", CAMERA_OPTICAL[cam])
        cam_tf = self._lookup_tf(cam_frame)
        if cam_tf is None or cam not in self._info:
            return []

        T_base_cam = tf_dict_to_matrix(cam_tf)
        T_cam_base = np.linalg.inv(T_base_cam)
        K = np.array(self._info[cam]["K"], dtype=np.float64).reshape(3, 3)

        all_frames = self._all_tf_frames()
        results = []

        for must, must_not, class_id, half_ext in FRAME_RULES:
            for frame in all_frames:
                if must not in frame:
                    continue
                if any(ex in frame for ex in must_not):
                    continue

                obj_tf = self._lookup_tf(frame)
                if obj_tf is None:
                    continue

                T_base_obj = tf_dict_to_matrix(obj_tf)
                T_cam_obj  = T_cam_base @ T_base_obj

                # Project 8 corners of the 3D bounding box
                he = half_ext
                corners = np.array([
                    [sx*he[0], sy*he[1], sz*he[2], 1.0]
                    for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)
                ])
                pts_cam = (T_cam_obj @ corners.T).T[:, :3]
                # Keep only corners in front of camera
                front = pts_cam[:, 2] > 0.01
                if not front.any():
                    continue
                pts_cam = pts_cam[front]
                proj    = (K @ pts_cam.T).T
                pts_img = proj[:, :2] / proj[:, 2:3]

                x1 = max(0,      int(pts_img[:,0].min()))
                y1 = max(0,      int(pts_img[:,1].min()))
                x2 = min(img_w,  int(pts_img[:,0].max()))
                y2 = min(img_h,  int(pts_img[:,1].max()))

                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                if bw <= 0 or bh <= 0:
                    continue
                # Skip if mostly outside image
                if x2 < 0 or y2 < 0 or x1 >= img_w or y1 >= img_h:
                    continue

                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                results.append((class_id, cx, cy, bw, bh))

        return results

    # ── tick ──────────────────────────────────────────────────────────────────

    def _tick(self):
        now = time.monotonic()
        if now - self._last_save < self._interval:
            return
        missing = [c for c in CAMERAS if c not in self._rgb]
        if missing:
            self.get_logger().warn(f"Waiting for RGB: {missing}")
            return

        self._last_save = now
        self._count    += 1
        stamp    = time.strftime("%Y%m%d_%H%M%S")
        snap_dir = self._output_dir / f"{stamp}_{self._count:04d}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        if not self._classes_written:
            self._write_classes_yaml()
            self._classes_written = True

        total_boxes = 0
        for cam in CAMERAS:
            short = cam.replace("_camera", "")
            rgb   = self._rgb[cam]
            h, w  = rgb.shape[:2]

            cv2.imwrite(str(snap_dir / f"{short}_rgb.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if cam in self._depth:
                d = self._depth[cam]
                np.save(str(snap_dir / f"{short}_depth.npy"), d)
                valid = d[d > 0]
                d_vis = (np.clip((d - valid.min()) / (valid.max() - valid.min() + 1e-6), 0, 1)
                         if valid.size else np.zeros_like(d))
                cv2.imwrite(str(snap_dir / f"{short}_depth_vis.png"),
                            cv2.applyColorMap((d_vis*255).astype(np.uint8), cv2.COLORMAP_INFERNO))

            if cam in self._segcolored:
                cv2.imwrite(str(snap_dir / f"{short}_seg_colored.png"),
                            cv2.cvtColor(self._segcolored[cam], cv2.COLOR_RGB2BGR))

            # ── bounding boxes + segmentation masks from seg + TF ─────────────
            if cam in self._seglabels:
                labels = self._seglabels[cam]
                np.save(str(snap_dir / f"{short}_seg_labels.npy"), labels)

                bboxes = self._detect_objects_from_tf(cam, w, h)
                total_boxes += len(bboxes)

                # YOLO detection labels
                with open(snap_dir / f"{short}_labels.txt", "w") as f:
                    for cls, cx, cy, bw, bh in bboxes:
                        f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                # YOLO segmentation labels
                seg_lines = _yolo_seg_from_labels(labels, bboxes, w, h)
                with open(snap_dir / f"{short}_seg.txt", "w") as f:
                    f.writelines(seg_lines)

                # Visualisation
                vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for cls, cx, cy, bw, bh in bboxes:
                    x1 = int((cx - bw/2) * w);  y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w);  y2 = int((cy + bh/2) * h)
                    name, colour = YOLO_CLASSES.get(cls, ("?", (200,200,200)))
                    cv2.rectangle(vis, (x1,y1), (x2,y2), colour, 2)
                    cv2.putText(vis, name, (x1, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
                cv2.imwrite(str(snap_dir / f"{short}_bbox_vis.png"), vis)

        # TF metadata
        eef_pose          = self._lookup_tf(EEF_FRAME)
        plug_poses        = {f: p for f in PLUG_FRAMES if (p := self._lookup_tf(f))}
        entrance_poses    = self._lookup_entrances()
        nic_card_poses    = self._lookup_nic_card_frames()
        camera_extrinsics = self._lookup_camera_extrinsics()

        meta = {
            "timestamp": stamp, "snapshot_id": self._count,
            "has_depth": bool(self._depth),
            "camera_info": self._info,
            "camera_extrinsics": camera_extrinsics,
            "end_effector": eef_pose,
            "plug_poses": plug_poses,
            "entrance_poses": entrance_poses,
            "nic_card_poses": nic_card_poses,
        }
        with open(snap_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.get_logger().info(
            f"[{self._count}] Saved → {snap_dir}  "
            f"| depth: {'yes' if self._depth else 'no'}  "
            f"| boxes: {total_boxes}  "
            f"| entrances: {len(entrance_poses)}"
        )

    def _write_classes_yaml(self):
        self._output_dir.mkdir(parents=True, exist_ok=True)
        data = {"nc": len(YOLO_CLASSES),
                "names": {cid: info[0] for cid, info in YOLO_CLASSES.items()}}
        with open(self._output_dir / "classes.yaml", "w") as f:
            yaml.dump(data, f)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="rgbd_snapshots")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once",     action="store_true")
    args, _ = parser.parse_known_args()
    if args.once:
        args.interval = 3.0

    rclpy.init()
    node = SnapshotRecorder(Path(args.output), args.interval)
    try:
        if args.once:
            while node._count == 0:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f"Stopped. {node._count} snapshots saved.")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
