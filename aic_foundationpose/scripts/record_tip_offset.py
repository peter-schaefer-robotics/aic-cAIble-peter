#!/usr/bin/env python3
"""Record the offset between SC port entrance and SC tip link across trial runs.

With ground truth on, press Enter at each trial to snapshot the current TF
difference between the plug tip and the nearest SC port entrance.
Saves results to tip_offset_measurements.json and prints statistics.

Usage:
  pixi run python3 aic_foundationpose/scripts/record_tip_offset.py
  pixi run python3 aic_foundationpose/scripts/record_tip_offset.py --auto 20
"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation


BASE_FRAME     = "base_link"
TIP_FRAME      = "cable_0/sc_tip_link"
ENTRANCE_FRAME = "task_board/sc_port_1/sc_port_base_link_entrance"


def lookup(buf, child, parent=BASE_FRAME):
    try:
        s = buf.lookup_transform(parent, child, rclpy.time.Time(),
                                 timeout=rclpy.duration.Duration(seconds=2.0))
        t = s.transform
        return np.array([t.translation.x, t.translation.y, t.translation.z]), \
               np.array([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])
    except Exception as e:
        return None, None


def all_frames(buf):
    try:
        return list(yaml.safe_load(buf.all_frames_as_yaml()) or {})
    except Exception:
        return []


def closest_entrance(buf, tip_pos):
    """Return the SC port entrance frame name, position, orientation and distance."""
    pos, quat = lookup(buf, ENTRANCE_FRAME)
    if pos is None:
        return None, None, None, float("inf")
    dist = np.linalg.norm(tip_pos - pos)
    return ENTRANCE_FRAME, pos, quat, dist


class OffsetRecorder(Node):
    def __init__(self):
        super().__init__("tip_offset_recorder")
        self._buf = tf2_ros.Buffer()
        self._lst = tf2_ros.TransformListener(self._buf, self)

    def snapshot(self):
        tip_pos, tip_quat = lookup(self._buf, TIP_FRAME)
        if tip_pos is None:
            self.get_logger().error(f"Cannot read {TIP_FRAME} — is ground truth on?")
            frames = all_frames(self._buf)
            cable_frames = [f for f in frames if "cable" in f or "tip" in f or "plug" in f]
            self.get_logger().info(f"Cable-related frames: {cable_frames}")
            return None

        ent_frame, ent_pos, ent_quat, dist = closest_entrance(self._buf, tip_pos)
        if ent_pos is None:
            self.get_logger().error(f"Cannot read entrance frame: {ENTRANCE_FRAME}")
            frames = all_frames(self._buf)
            self.get_logger().info(f"ALL frames in TF ({len(frames)}):\n" +
                                   "\n".join(f"  {f}" for f in sorted(frames)))
            return None

        # Translational offset: tip position relative to entrance in base_link
        offset_xyz = tip_pos - ent_pos

        # Rotational offset: R_entrance^{-1} @ R_tip  (tip in entrance frame)
        R_ent  = Rotation.from_quat(ent_quat)
        R_tip  = Rotation.from_quat(tip_quat)
        R_rel  = R_ent.inv() * R_tip
        rpy_rel  = R_rel.as_euler("xyz", degrees=True)
        quat_rel = R_rel.as_quat()   # xyzw

        record = {
            "entrance_frame":  ent_frame,
            "entrance_pos":    ent_pos.tolist(),
            "entrance_quat":   ent_quat.tolist(),
            "tip_pos":         tip_pos.tolist(),
            "tip_quat":        tip_quat.tolist(),
            "offset_xyz":      offset_xyz.tolist(),
            "offset_rpy_deg":  rpy_rel.tolist(),
            "offset_quat":     quat_rel.tolist(),
            "dist_m":          float(dist),
        }
        self.get_logger().info(
            f"entrance: {ent_frame}\n"
            f"  offset XYZ : {offset_xyz.round(4)}  ({np.linalg.norm(offset_xyz)*1000:.1f} mm)\n"
            f"  offset RPY : {rpy_rel.round(2)} deg\n"
            f"  offset quat: {quat_rel.round(5)}"
        )
        return record


def print_stats(records):
    xyz     = np.array([r["offset_xyz"]     for r in records])
    rpys    = np.array([r["offset_rpy_deg"] for r in records])
    quats   = np.array([r["offset_quat"]    for r in records])

    # Circular mean for quaternions (average on the unit sphere)
    q_mean  = quats.mean(axis=0)
    q_mean /= np.linalg.norm(q_mean)
    rpy_mean = Rotation.from_quat(q_mean).as_euler("xyz", degrees=True)

    print(f"\n{'='*60}")
    print(f"  {len(records)} measurements")
    print(f"\n  Translation offset (tip - entrance, in base_link):")
    print(f"    mean  XYZ : {xyz.mean(axis=0).round(5)}")
    print(f"    std   XYZ : {xyz.std(axis=0).round(5)}")
    print(f"\n  Rotation offset (R_entrance^-1 @ R_tip, in entrance frame):")
    print(f"    mean  RPY : {rpys.mean(axis=0).round(3)} deg")
    print(f"    std   RPY : {rpys.std(axis=0).round(3)} deg")

    xyz_m = xyz.mean(axis=0)
    print(f"\n  → hardcode in plug_sc_port.py:")
    print(f"    X_OFFSET   = {xyz_m[0]:.5f}   # {xyz_m[0]*1000:.2f} mm")
    print(f"    Y_OFFSET   = {xyz_m[1]:.5f}   # {xyz_m[1]*1000:.2f} mm")
    print(f"    Z_OFFSET   = {xyz_m[2]:.5f}   # {xyz_m[2]*1000:.2f} mm")
    print(f"    ROT_QUAT   = np.array([{', '.join(f'{v:.6f}' for v in q_mean)}])  # xyzw")
    print(f"    ROT_RPY_DEG= {rpy_mean.round(3).tolist()}  # roll pitch yaw")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", type=int, default=0, metavar="N",
                    help="Auto-sample N times every 2s (default: manual Enter)")
    ap.add_argument("--out", default="tip_offset_measurements.json")
    args = ap.parse_args()

    rclpy.init()
    node = OffsetRecorder()

    # Spin in background so TransformListener receives data continuously
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("Warming TF buffer — waiting for entrance frame …")
    deadline = time.time() + 30.0
    while time.time() < deadline:
        pos, _ = lookup(node._buf, ENTRANCE_FRAME)
        if pos is not None:
            print(f"  entrance frame ready: {pos.round(4)}")
            break
        print("  waiting …", end="\r")
        time.sleep(0.3)
    else:
        print(f"\nERROR: {ENTRANCE_FRAME} not available after 30s. "
              f"Is ground truth on and task board spawned?")
        executor.shutdown(wait=False)
        node.destroy_node(); rclpy.shutdown(); return

    records = []
    out = Path(args.out)

    if args.auto > 0:
        print(f"Auto-sampling {args.auto} times …")
        for i in range(args.auto):
            print(f"\n── sample {i+1}/{args.auto} ──")
            r = node.snapshot()
            if r:
                records.append(r)
            time.sleep(2.0)
            node.spin(20)
    else:
        print(f"Press Enter to record a sample, Ctrl+C to finish.\n")
        i = 0
        try:
            while True:
                input(f"  [Enter] for sample {i+1} … ")
                r = node.snapshot()
                if r:
                    records.append(r)
                    i += 1
        except KeyboardInterrupt:
            pass

    if records:
        print_stats(records)
        out.write_text(json.dumps(records, indent=2))
        print(f"Saved {len(records)} records → {out}")
    else:
        print("No records captured.")

    executor.shutdown(wait=False)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
