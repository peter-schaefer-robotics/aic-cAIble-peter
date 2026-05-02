#!/usr/bin/env python3
"""Detect the closest SC port with YOLOv8 + depth and plug in the cable.

Pipeline:
  1. Capture RGB + depth from all 3 cameras
  2. Run YOLOv8 to detect sc_port bounding boxes
  3. Back-project bbox centre through depth → 3D point in base_link
  4. Match each detection to the nearest sc_port TF entrance frame
  5. Move robot:  pre-grasp  →  grasp  →  insert

Usage:
  pixi run python3 aic_foundationpose/scripts/plug_sc_port.py \
      --model runs/detect/yolov8_runs/aic_detect/weights/best.pt \
      --conf  0.5 \
      --approach-offset 0.10   # metres above entrance before descending
      --insert-depth    0.025  # metres to push in after reaching entrance
      --dry-run                # detect & visualise without moving
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# magenta HSV range (same as find_logo.py)
_MAGENTA_LOW  = np.array([140, 80,  60])
_MAGENTA_HIGH = np.array([170, 255, 255])
import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped, Wrench, WrenchStamped, Vector3
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from aic_control_interfaces.msg import MotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode


# ─── constants ────────────────────────────────────────────────────────────────

CAMERAS        = ["left_camera", "center_camera", "right_camera"]
BASE_FRAME     = "base_link"
EEF_FRAME      = "gripper/tcp"
SC_PORT_CLS    = 0   # class index for sc_port in the trained model
                     # (dataset: 0=sc_port, 1=nic_port)

CAMERA_OPTICAL = {
    "left_camera":   "left_camera/optical",
    "center_camera": "center_camera/optical",
    "right_camera":  "right_camera/optical",
}

RELIABLE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def ros_img_to_numpy(msg: Image) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = data.reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if msg.encoding == "rgb8" else img.copy()
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    if msg.encoding == "16UC1":
        return (np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width).astype(np.float32) / 1000.0)
    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def tf_to_matrix(t: dict) -> np.ndarray:
    tr = t["translation"]; ro = t["rotation"]
    T = np.eye(4)
    T[:3, 3] = [tr["x"], tr["y"], tr["z"]]
    T[:3, :3] = Rotation.from_quat([ro["x"], ro["y"], ro["z"], ro["w"]]).as_matrix()
    return T


def lookup_tf(buf: tf2_ros.Buffer, child: str, parent: str = BASE_FRAME) -> dict | None:
    try:
        s: TransformStamped = buf.lookup_transform(parent, child,
                                                   rclpy.time.Time(),
                                                   timeout=rclpy.duration.Duration(seconds=1.0))
        t = s.transform
        return {
            "translation": {"x": t.translation.x, "y": t.translation.y, "z": t.translation.z},
            "rotation":    {"x": t.rotation.x, "y": t.rotation.y,
                            "z": t.rotation.z, "w": t.rotation.w},
        }
    except Exception:
        return None


def all_tf_frames(buf: tf2_ros.Buffer) -> list[str]:
    try:
        return list(yaml.safe_load(buf.all_frames_as_yaml()) or {})
    except Exception:
        return []


def backproject_pixel(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
    """Camera-frame 3-D point from pixel + depth (depth in metres)."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.array([x, y, depth])


def median_depth(depth_img: np.ndarray, u: int, v: int, r: int = 8) -> float | None:
    """Median depth in a small patch — robust to holes."""
    h, w = depth_img.shape
    patch = depth_img[max(0, v-r):min(h, v+r), max(0, u-r):min(w, u+r)]
    valid = patch[patch > 0.05]
    return float(np.median(valid)) if len(valid) > 5 else None


# ─── robot motion ─────────────────────────────────────────────────────────────

def _make_motion_msg(clock, x, y, z, qx, qy, qz, qw,
                     stiffness: float, damping: float,
                     force_xyz: tuple = (0.0, 0.0, 0.0)) -> MotionUpdate:
    flat = []
    for i in range(6):
        row = [0.0] * 6; row[i] = stiffness; flat.extend(row)
    damp = []
    for i in range(6):
        row = [0.0] * 6; row[i] = damping; damp.extend(row)
    msg = MotionUpdate()
    msg.header = Header()
    msg.header.frame_id = BASE_FRAME
    msg.header.stamp = clock.now().to_msg()
    msg.pose.position.x = float(x);  msg.pose.position.y = float(y)
    msg.pose.position.z = float(z)
    msg.pose.orientation.x = float(qx); msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz); msg.pose.orientation.w = float(qw)
    msg.target_stiffness = flat
    msg.target_damping   = damp
    msg.feedforward_wrench_at_tip = Wrench(
        force=Vector3(x=float(force_xyz[0]),
                      y=float(force_xyz[1]),
                      z=float(force_xyz[2])))
    # Keep feedback gains always zero — taring is not available in evaluation
    # so raw sensor readings include gravity/cable bias which would fight insertion.
    # The controller's built-in gravity compensation handles gravity; we only
    # use feedforward_wrench for directed force.
    msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
    return msg


def send_pose(pub, clock, x, y, z, qx, qy, qz, qw,
              stiffness: float = 500.0, damping: float = 50.0,
              force_xyz: tuple = (0.0, 0.0, 0.0)):
    pub.publish(_make_motion_msg(clock, x, y, z, qx, qy, qz, qw,
                                 stiffness, damping, force_xyz))


def move_slow(node: Node, pub, buf: tf2_ros.Buffer,
              target_xyz: np.ndarray, target_quat: np.ndarray,
              speed: float = 0.03, stiffness: float = 300.0, damping: float = 40.0,
              tol: float = 0.005):
    """Stream interpolated waypoints at `speed` m/s toward target_xyz."""
    tf = lookup_tf(buf, EEF_FRAME)
    if tf is None:
        node.get_logger().warn("Cannot read EEF pose for slow move — sending direct.")
        send_pose(pub, node.get_clock(), *target_xyz, *target_quat, stiffness, damping)
        return

    t   = tf["translation"]
    r   = tf["rotation"]
    start_pos  = np.array([t["x"], t["y"], t["z"]])
    start_quat = np.array([r["x"], r["y"], r["z"], r["w"]])

    dist = np.linalg.norm(target_xyz - start_pos)
    if dist < tol:
        return

    step_m   = 0.002          # 2 mm per waypoint
    n_steps  = max(1, int(dist / step_m))
    dt       = step_m / speed  # seconds between waypoints

    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        pos   = start_pos + alpha * (target_xyz - start_pos)
        # slerp orientation
        q0 = Rotation.from_quat(start_quat)
        q1 = Rotation.from_quat(target_quat)
        q  = Rotation.slerp if hasattr(Rotation, "slerp") else None
        try:
            from scipy.spatial.transform import Slerp
            key_rots  = Rotation.concatenate([q0, q1])
            key_times = [0.0, 1.0]
            slerp = Slerp(key_times, key_rots)
            quat = slerp(alpha).as_quat()
        except Exception:
            quat = target_quat

        send_pose(pub, node.get_clock(), *pos, *quat, stiffness, damping)
        rclpy.spin_once(node, timeout_sec=dt)

    node.get_logger().info(
        f"  moved {dist*100:.1f} cm in {dist/speed:.1f}s at {speed*100:.0f}cm/s")


def wait_for_eef(node: Node, buf: tf2_ros.Buffer,
                 target_xyz: np.ndarray, tol: float = 0.008,
                 timeout: float = 15.0):
    """Spin until tool0 is within tol of target_xyz (or timeout)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        tf = lookup_tf(buf, EEF_FRAME)
        if tf is None:
            continue
        p = tf["translation"]
        dist = np.linalg.norm([p["x"] - target_xyz[0],
                                p["y"] - target_xyz[1],
                                p["z"] - target_xyz[2]])
        if dist < tol:
            return True
    node.get_logger().warn("wait_for_eef timed out")
    return False


# ─── main node ────────────────────────────────────────────────────────────────

class PlugNode(Node):
    def __init__(self, args):
        super().__init__("plug_sc_port")
        self.args = args

        self._tf_buf = tf2_ros.Buffer()
        self._tf_lst = tf2_ros.TransformListener(self._tf_buf, self)

        # camera data buffers
        self._rgb   = {}   # cam → np.ndarray (BGR)
        self._depth = {}   # cam → np.ndarray (float32, metres)
        self._K     = {}   # cam → 3×3 ndarray
        self._frame = {}   # cam → optical frame_id

        for cam in CAMERAS:
            self.create_subscription(Image, f"/{cam}/image",
                lambda m, c=cam: self._cb_rgb(c, m),
                qos_profile_sensor_data)
            self.create_subscription(Image, f"/{cam}/depth",
                lambda m, c=cam: self._cb_depth(c, m),
                qos_profile_sensor_data)
            self.create_subscription(CameraInfo, f"/{cam}/camera_info",
                lambda m, c=cam: self._cb_info(c, m),
                qos_profile_sensor_data)

        self._mode_cli = self.create_client(ChangeTargetMode,
                                            "/aic_controller/change_target_mode")
        self._pose_pub = self.create_publisher(MotionUpdate,
                                               "/aic_controller/pose_commands", 1)

        self._wrench_latest = None
        self.create_subscription(WrenchStamped,
                                 "/fts_broadcaster/wrench",
                                 lambda m: setattr(self, "_wrench_latest", m.wrench),
                                 RELIABLE_QOS)

    # ── camera callbacks ──────────────────────────────────────────────────────

    def _cb_rgb(self, cam, msg):
        try:
            self._rgb[cam] = ros_img_to_numpy(msg)
        except Exception:
            pass

    def _cb_depth(self, cam, msg):
        try:
            self._depth[cam] = ros_img_to_numpy(msg)
        except Exception:
            pass

    def _cb_info(self, cam, msg):
        self._K[cam]     = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._frame[cam] = msg.header.frame_id

    # ── wait for data ─────────────────────────────────────────────────────────

    def _wait_cameras(self, timeout: float = 10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            ready = [c for c in CAMERAS
                     if c in self._rgb and c in self._K]
            if len(ready) >= 1:
                self.get_logger().info(
                    f"Got data from: {ready}  "
                    f"(depth: {[c for c in ready if c in self._depth]})")
                return ready
        # show what arrived vs what's missing
        self.get_logger().error(
            f"rgb={list(self._rgb.keys())}  "
            f"depth={list(self._depth.keys())}  "
            f"K={list(self._K.keys())}")
        return []

    # ── YOLO detection ────────────────────────────────────────────────────────

    def _detect_sc_ports(self, model) -> list[dict]:
        """Detect SC ports across all cameras, back-project to 3-D base_link."""
        detections = []
        for cam in CAMERAS:
            if cam not in self._rgb or cam not in self._depth or cam not in self._K:
                continue

            rgb   = self._rgb[cam]
            K     = self._K[cam]
            frame = self._frame.get(cam, CAMERA_OPTICAL[cam])

            tf_cam = lookup_tf(self._tf_buf, frame)
            if tf_cam is None:
                self.get_logger().warn(f"No TF for {frame}")
                continue
            T_base_cam = tf_to_matrix(tf_cam)

            results = model(rgb, conf=self.args.conf, verbose=False)
            for r in results:
                for box in r.boxes:
                    if int(box.cls.item()) != SC_PORT_CLS:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    u = int((x1 + x2) / 2)
                    v = int((y1 + y2) / 2)
                    depth = self._depth.get(cam)
                    d = median_depth(depth, u, v) if depth is not None else None
                    if d is not None and (d < 0.05 or d > 5.0):
                        d = None

                    if d is not None:
                        p_cam  = backproject_pixel(u, v, d, K)
                        p_base = (T_base_cam @ np.append(p_cam, 1.0))[:3]
                    else:
                        # No depth — project ray and intersect with task-board plane (z≈0.03)
                        ray_dir = np.linalg.inv(K) @ np.array([u, v, 1.0])
                        ray_dir /= np.linalg.norm(ray_dir)
                        # transform ray to base_link
                        R_bc = T_base_cam[:3, :3]
                        t_bc = T_base_cam[:3, 3]
                        ray_base = R_bc @ ray_dir
                        if abs(ray_base[2]) < 1e-6:
                            continue
                        BOARD_Z = 0.03
                        t_scale  = (BOARD_Z - t_bc[2]) / ray_base[2]
                        if t_scale < 0:
                            continue
                        p_base = t_bc + t_scale * ray_base
                        self.get_logger().warn(
                            f"[{cam}] no depth — ray-plane fallback z≈{BOARD_Z}")

                    detections.append({
                        "cam":       cam,
                        "pixel":     (u, v),
                        "depth":     d,
                        "pos_base":  p_base,
                        "conf":      float(box.conf.item()),
                        "bbox":      (x1, y1, x2, y2),
                    })
                    self.get_logger().info(
                        f"[{cam}] sc_port conf={box.conf.item():.2f} "
                        f"3D=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f})")
        return detections

    # ── match to TF entrance ──────────────────────────────────────────────────

    def _find_entrance_pose(self, pos_base: np.ndarray) -> dict | None:
        """Find the sc_port TF frame closest to a 3-D detection.

        Matches the port *body* link (same convention CheatCode uses), e.g.:
          task_board/sc_port_0/sc_port_base_link
        Falls back to the _entrance variant if no body link is found.
        """
        best_dist = 0.15
        best_tf   = None
        frames = all_tf_frames(self._tf_buf)

        # Prefer the actual port link (not entrance) — same as CheatCode
        for suffix_req, suffix_exc in [
            ("sc_port", ["entrance"]),   # body link first
            ("sc_port", []),             # entrance fallback
        ]:
            for frame in frames:
                if suffix_req not in frame:
                    continue
                if any(s in frame for s in suffix_exc):
                    continue
                tf = lookup_tf(self._tf_buf, frame)
                if tf is None:
                    continue
                t = tf["translation"]
                dist = np.linalg.norm(pos_base - np.array([t["x"], t["y"], t["z"]]))
                if dist < best_dist:
                    best_dist = dist
                    best_tf   = tf
            if best_tf is not None:
                break

        if best_tf:
            self.get_logger().info(
                f"Matched port frame: {best_tf.get('child_frame','?')} "
                f"(dist={best_dist:.3f}m)")
        return best_tf

    # ── draw detections ───────────────────────────────────────────────────────

    def _visualise(self, detections: list[dict]):
        for cam in CAMERAS:
            if cam not in self._rgb:
                continue
            vis = self._rgb[cam].copy()
            for det in detections:
                if det["cam"] != cam:
                    continue
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                p = det["pos_base"]
                label = (f"sc_port {det['conf']:.2f} "
                         f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})")
                cv2.putText(vis, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            out = f"/tmp/plug_detect_{cam}.png"
            cv2.imwrite(out, vis)
            self.get_logger().info(f"Saved detection viz → {out}")

    # ── robot motion ──────────────────────────────────────────────────────────

    def _switch_cartesian(self):
        if not self._mode_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("change_target_mode service not available")
            return False
        req = ChangeTargetMode.Request()
        req.target_mode.mode = TargetMode.MODE_CARTESIAN
        fut = self._mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        return fut.result() is not None

    # Fixed transform from gripper/tcp → cable_0/sc_tip_link
    # Measured with ground truth on (2026-05-02).  Stays constant because
    # the cable is rigidly attached to the gripper.
    _T_TCP_TIP_TRANSLATION = np.array([-0.000782, -0.010156,  0.008257])
    _T_TCP_TIP_QUAT        = np.array([ 0.160781, -0.167157,  0.694179,  0.681413])  # xyzw

    # Systematic tip→entrance offset calibrated with record_tip_offset.py (2026-05-02).
    # _TIP_CORRECTION_TIP_FRAME : offset (tip_pos − entrance_pos) expressed in the TIP frame.
    #   Storing in tip frame makes the correction pose-invariant; it is rotated to
    #   base_link at runtime using the current tip orientation.
    #   Empirically the misalignment is almost entirely along the tip X axis.
    # _TIP_CORRECTION_QUAT : R_entrance⁻¹ · R_tip  (xyzw, positive-w convention)
    # The entrance target is shifted by −offset so the tip lands at the entrance.
    _TIP_CORRECTION_TIP_FRAME = np.array([-0.007560,  0.006110,  0.002200])  # tip frame, 9 samples 2026-05-02
    _TIP_CORRECTION_QUAT      = np.array([-0.000404, -0.000714, -0.035740,  0.999360])  # xyzw

    def _find_plug_frame(self) -> str:
        """Always use the hardcoded gripper→tip transform — no ground truth needed."""
        return "_synthetic"

    def _lookup_plug_tf(self, plug_frame: str) -> dict | None:
        """Look up plug tip pose — uses TF when ground truth is on,
        otherwise computes from the hardcoded gripper→tip offset."""
        if plug_frame != "_synthetic":
            return lookup_tf(self._tf_buf, plug_frame)

        # Synthetic: compute T_base_tip = T_base_tcp @ T_tcp_tip
        gripper_tf = lookup_tf(self._tf_buf, EEF_FRAME)
        if gripper_tf is None:
            return None
        T_base_tcp = tf_to_matrix(gripper_tf)
        T_tcp_tip  = np.eye(4)
        T_tcp_tip[:3, 3]  = self._T_TCP_TIP_TRANSLATION
        T_tcp_tip[:3, :3] = Rotation.from_quat(self._T_TCP_TIP_QUAT).as_matrix()
        T_base_tip = T_base_tcp @ T_tcp_tip
        t = T_base_tip[:3, 3]
        q = Rotation.from_matrix(T_base_tip[:3, :3]).as_quat()
        return {
            "translation": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
            "rotation":    {"x": float(q[0]), "y": float(q[1]),
                            "z": float(q[2]), "w": float(q[3])},
        }

    def _calc_gripper_pose(self, port_transform: dict, plug_frame: str,
                           z_offset: float, slerp_frac: float, pos_frac: float,
                           reset_integrator: bool = False) -> tuple | None:
        """CheatCode calc_gripper_pose adapted for MotionUpdate.

        Returns (x, y, z, qx, qy, qz, qw) for gripper/tcp in base_link,
        or None if a TF lookup fails.
        """
        from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

        pt = port_transform["translation"]
        pr = port_transform["rotation"]
        q_port = (pr["w"], pr["x"], pr["y"], pr["z"])   # (w,x,y,z) convention

        plug_tf = self._lookup_plug_tf(plug_frame)
        if plug_tf is None:
            return None
        q_plug = (plug_tf["rotation"]["w"], plug_tf["rotation"]["x"],
                  plug_tf["rotation"]["y"], plug_tf["rotation"]["z"])
        plug_xyz = (plug_tf["translation"]["x"],
                    plug_tf["translation"]["y"],
                    plug_tf["translation"]["z"])

        gripper_tf = lookup_tf(self._tf_buf, EEF_FRAME)
        if gripper_tf is None:
            return None
        q_gripper = (gripper_tf["rotation"]["w"], gripper_tf["rotation"]["x"],
                     gripper_tf["rotation"]["y"], gripper_tf["rotation"]["z"])
        gripper_xyz = (gripper_tf["translation"]["x"],
                       gripper_tf["translation"]["y"],
                       gripper_tf["translation"]["z"])

        # rotation: align plug to port, apply to gripper
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        q_target = quaternion_multiply(q_diff, q_gripper)

        # Fixed rotation around the insertion axis to key the SC plug
        if self.args.z_rot_offset != 0.0:
            angle_rad = np.deg2rad(self.args.z_rot_offset)
            # correctly extract insertion axis = Z-column of port rotation matrix
            R_port = Rotation.from_quat([pr["x"], pr["y"], pr["z"], pr["w"]]).as_matrix()
            axis   = R_port[:, 2]   # port Z-axis in base_link
            s = np.sin(angle_rad / 2);  c = np.cos(angle_rad / 2)
            q_zrot = (c, axis[0]*s, axis[1]*s, axis[2]*s)   # wxyz
            q_target = quaternion_multiply(q_zrot, q_target)

        q_slerp  = quaternion_slerp(q_gripper, q_target, slerp_frac)

        # position: put plug tip at (port_xy, port_z + z_offset)
        plug_tip_offset_z = gripper_xyz[2] - plug_xyz[2]

        tip_x_err = pt["x"] - plug_xyz[0]
        tip_y_err = pt["y"] - plug_xyz[1]

        if reset_integrator:
            self._ix = 0.0;  self._iy = 0.0
        else:
            i_lim = 0.05
            self._ix = float(np.clip(self._ix + tip_x_err, -i_lim, i_lim))
            self._iy = float(np.clip(self._iy + tip_y_err, -i_lim, i_lim))

        i_gain = 0.15
        tgt_x = pt["x"] + i_gain * self._ix
        tgt_y = pt["y"] + i_gain * self._iy
        tgt_z = pt["z"] + z_offset - plug_tip_offset_z

        x = pos_frac * tgt_x + (1 - pos_frac) * gripper_xyz[0]
        y = pos_frac * tgt_y + (1 - pos_frac) * gripper_xyz[1]
        z = pos_frac * tgt_z + (1 - pos_frac) * gripper_xyz[2]

        # q_slerp is (w,x,y,z) → return as (qx,qy,qz,qw)
        return x, y, z, q_slerp[1], q_slerp[2], q_slerp[3], q_slerp[0]

    def _detect_logo_yaw(self) -> float | None:
        """Detect the Intrinsic logo and return its yaw (rad) in base_link.

        Uses the same magenta-detection + X-axis backprojection as find_logo.py.
        The logo and SC ports share the same task-board frame so the yaw gives
        the board orientation → the correct plug Z rotation.
        """
        yaw_samples = []

        for cam in CAMERAS:
            if cam not in self._rgb or cam not in self._K:
                continue
            rgb = self._rgb[cam]
            K   = self._K[cam]
            frame = self._frame.get(cam, CAMERA_OPTICAL[cam])
            tf_cam = lookup_tf(self._tf_buf, frame)
            if tf_cam is None:
                continue
            T_base_cam = tf_to_matrix(tf_cam)
            R_base_cam = T_base_cam[:3, :3]

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, _MAGENTA_LOW, _MAGENTA_HIGH)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            max_area   = max(cv2.contourArea(c) for c in contours)
            logo_parts = [c for c in contours if cv2.contourArea(c) > 0.01 * max_area]
            merged     = np.vstack(logo_parts)
            M          = cv2.moments(merged.reshape(-1, 1, 2))
            if M["m00"] == 0:
                continue

            centroid = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
            rect     = cv2.minAreaRect(merged.reshape(-1, 1, 2))
            corners4 = cv2.boxPoints(rect).astype(np.float64)

            # Find the closest edge to centroid → X axis of logo (thick side)
            min_dist = float("inf")
            slit_dir_2d = None
            for i in range(4):
                p1 = corners4[i];  p2 = corners4[(i + 1) % 4]
                ev = p2 - p1;  el = np.linalg.norm(ev)
                if el < 1:
                    continue
                eu   = ev / el
                perp = (centroid - p1) - np.dot(centroid - p1, eu) * eu
                dist = np.linalg.norm(perp)
                if dist < min_dist:
                    min_dist = dist
                    mid      = (p1 + p2) / 2.0
                    raw      = mid - centroid
                    if np.linalg.norm(raw) > 1:
                        slit_dir_2d = raw / np.linalg.norm(raw)

            if slit_dir_2d is None:
                continue

            # Backproject the 2D X direction through the camera into base_link XY plane
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            # Direction ray for the tip of the arrow (centroid + slit_dir_2d)
            def ray(pt):
                return np.array([(pt[0] - cx) / fx,
                                 (pt[1] - cy) / fy, 1.0])

            r0 = R_base_cam @ ray(centroid)
            r1 = R_base_cam @ ray(centroid + slit_dir_2d * 50)
            x_base = r1 - r0
            x_base[2] = 0.0   # project onto XY plane (flat-plane prior)
            n = np.linalg.norm(x_base)
            if n < 1e-6:
                continue
            x_base /= n

            yaw = float(np.arctan2(x_base[1], x_base[0]))
            yaw_samples.append(yaw)
            self.get_logger().info(
                f"[logo] [{cam}] yaw={np.degrees(yaw):.1f}°")

        if not yaw_samples:
            self.get_logger().warn("[logo] No logo detected in any camera.")
            return None

        # Circular mean
        final_yaw = float(np.arctan2(
            np.mean([np.sin(y) for y in yaw_samples]),
            np.mean([np.cos(y) for y in yaw_samples])))
        self.get_logger().info(
            f"[logo] Final yaw = {np.degrees(final_yaw):.1f}°")
        return final_yaw

    def _log_orientation_check(self, port_tf: dict, plug_frame: str):
        """Print tip vs port Z-axes and required rotation so user can verify."""
        from transforms3d._gohlketransforms import quaternion_multiply
        plug_tf = self._lookup_plug_tf(plug_frame)
        if plug_tf is None:
            self.get_logger().warn("Cannot read plug TF for orientation check")
            return
        pr = port_tf["rotation"];  plr = plug_tf["rotation"]
        R_port = Rotation.from_quat([pr["x"],  pr["y"],  pr["z"],  pr["w"]]).as_matrix()
        R_plug = Rotation.from_quat([plr["x"], plr["y"], plr["z"], plr["w"]]).as_matrix()
        port_z = R_port[:, 2]   # insertion direction of port
        tip_z  = R_plug[:, 2]   # current insertion direction of tip
        dot    = float(np.dot(port_z, tip_z))
        angle  = float(np.degrees(np.arccos(np.clip(dot, -1, 1))))

        self.get_logger().info(
            f"[orientation check]\n"
            f"  tip  Z in base_link : {tip_z.round(3)}\n"
            f"  port Z in base_link : {port_z.round(3)}\n"
            f"  misalignment angle  : {angle:.1f}°\n"
            f"  (q_diff will rotate gripper {angle:.1f}° to align)"
        )

        gripper_tf = lookup_tf(self._tf_buf, EEF_FRAME)
        if gripper_tf is None:
            return
        gr = gripper_tf["rotation"]
        q_port   = (pr["w"],  pr["x"],  pr["y"],  pr["z"])
        q_plug   = (plr["w"], plr["x"], plr["y"], plr["z"])
        q_gripper = (gr["w"], gr["x"], gr["y"], gr["z"])
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        q_diff   = quaternion_multiply(q_port, q_plug_inv)
        q_target = quaternion_multiply(q_diff, q_gripper)
        rpy = Rotation.from_quat(
            [q_target[1], q_target[2], q_target[3], q_target[0]]).as_euler("xyz", degrees=True)
        self.get_logger().info(
            f"  target gripper/tcp RPY: {rpy.round(1)} deg")

    def _measure_wrench_bias(self, n: int = 30) -> np.ndarray:
        """Average n wrench readings at the current pose (no contact) → force bias."""
        samples = []
        t0 = time.time()
        while len(samples) < n and time.time() - t0 < 5.0:
            self._wrench_latest = None
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._wrench_latest is not None:
                w = self._wrench_latest
                samples.append([w.force.x, w.force.y, w.force.z])
        if samples:
            bias = np.mean(samples, axis=0)
            self.get_logger().info(f"Wrench bias: {bias.round(3)} N")
            return bias
        self.get_logger().warn("No wrench data — bias set to zero")
        return np.zeros(3)

    def _plug_in(self, port_tf: dict, plug_frame: str, model):
        """CheatCode insertion policy driven by YOLO port position."""
        self._ix = 0.0;  self._iy = 0.0   # integrators

        # Detect logo yaw → sets plug Z rotation dynamically
        # The logo and SC ports share the task-board frame, so logo yaw
        # directly encodes the board orientation.  --z-rot-offset is a
        # fixed calibration trim on top of that.
        logo_yaw = self._detect_logo_yaw()
        if logo_yaw is not None:
            self.args.z_rot_offset = float(np.degrees(logo_yaw)) + \
                                     self.args.z_rot_offset
            self.get_logger().info(
                f"Logo yaw {np.degrees(logo_yaw):.1f}° → "
                f"z_rot_offset = {self.args.z_rot_offset:.1f}°")

        self._log_orientation_check(port_tf, plug_frame)

        z_offset = self.args.approach_offset

        DT = 0.05   # seconds between commands (20 Hz)

        def ros_sleep():
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(DT)

        def move_to(target_pos: np.ndarray, target_quat: np.ndarray,
                    speed_mps: float, stiffness: float, damping: float,
                    label: str):
            """Stream waypoints at speed_mps m/s by tracking commanded position."""
            # start from where we last commanded (or current TF if first call)
            eef_tf = lookup_tf(self._tf_buf, EEF_FRAME)
            if eef_tf is None:
                self.get_logger().warn(f"{label}: cannot read EEF pose")
                return
            cmd_pos = np.array([eef_tf["translation"]["x"],
                                eef_tf["translation"]["y"],
                                eef_tf["translation"]["z"]])
            cmd_quat = np.array([eef_tf["rotation"]["x"], eef_tf["rotation"]["y"],
                                 eef_tf["rotation"]["z"], eef_tf["rotation"]["w"]])
            step_m = speed_mps * DT   # metres to advance per 50 ms step

            from scipy.spatial.transform import Slerp
            total_dist = np.linalg.norm(target_pos - cmd_pos)
            if total_dist < 1e-4:
                return
            n_steps = max(1, int(np.ceil(total_dist / step_m)))
            self.get_logger().info(
                f"{label}: {total_dist*100:.1f} cm at {speed_mps*100:.0f} cm/s "
                f"→ {n_steps * DT:.1f}s")

            key_rots  = Rotation.concatenate([
                Rotation.from_quat(cmd_quat),
                Rotation.from_quat(target_quat)])
            slerp_fn  = Slerp([0.0, 1.0], key_rots)

            for i in range(1, n_steps + 1):
                alpha     = i / n_steps
                pos       = cmd_pos + alpha * (target_pos - cmd_pos)
                quat      = slerp_fn(alpha).as_quat()
                send_pose(self._pose_pub, self.get_clock(),
                          pos[0], pos[1], pos[2],
                          quat[0], quat[1], quat[2], quat[3],
                          stiffness, damping)
                ros_sleep()

        # ── compute full target orientation once ──────────────────────────────
        # Use calc_gripper_pose at frac=1 to get the final target orientation/pos
        target_pose = self._calc_gripper_pose(
            port_tf, plug_frame, z_offset,
            slerp_frac=1.0, pos_frac=1.0, reset_integrator=True)
        if target_pose is None:
            self.get_logger().error("calc_gripper_pose failed — TF lookup error")
            return
        hover_pos   = np.array(target_pose[:3])
        target_quat = np.array(target_pose[3:])   # xyzw (final rotated orientation)

        # Current gripper orientation
        eef_tf0 = lookup_tf(self._tf_buf, EEF_FRAME)
        current_quat = np.array([eef_tf0["rotation"]["x"], eef_tf0["rotation"]["y"],
                                  eef_tf0["rotation"]["z"], eef_tf0["rotation"]["w"]]) \
                       if eef_tf0 else target_quat

        # Build yaw-only intermediate orientation: keep current roll/pitch, apply target yaw only.
        # ZYX intrinsic order → [yaw, pitch, roll]
        yaw_target, _, _            = Rotation.from_quat(target_quat).as_euler("ZYX")
        _, pitch_curr, roll_curr    = Rotation.from_quat(current_quat).as_euler("ZYX")
        yaw_only_quat = Rotation.from_euler(
            "ZYX", [yaw_target, pitch_curr, roll_curr]).as_quat()

        # ── Phase 1: move to hover adjusting only yaw (roll/pitch unchanged) ─
        self.get_logger().info("Phase 1: moving to hover + adjusting yaw only …")
        move_to(hover_pos, yaw_only_quat,
                speed_mps=self.args.speed,
                stiffness=400.0, damping=80.0,
                label="approach")

        # ── hold hover with yaw-only orientation until settled ────────────────
        self.get_logger().info(f"Holding hover for {self.args.hover_wait:.1f}s …")
        for _ in range(int(self.args.hover_wait / DT)):
            send_pose(self._pose_pub, self.get_clock(),
                      hover_pos[0], hover_pos[1], hover_pos[2],
                      yaw_only_quat[0], yaw_only_quat[1],
                      yaw_only_quat[2], yaw_only_quat[3],
                      300.0, 100.0)
            ros_sleep()

        # ── yaw is now settled — re-run YOLO for fresh port position ─────────
        self.get_logger().info("Yaw settled — re-running YOLO for fresh port position …")
        for _ in range(10):   # flush stale camera frames
            rclpy.spin_once(self, timeout_sec=0.05)
        new_detections = self._detect_sc_ports(model)
        if new_detections:
            eef_tf = lookup_tf(self._tf_buf, EEF_FRAME)
            if eef_tf:
                eef_pos = np.array([eef_tf["translation"]["x"],
                                    eef_tf["translation"]["y"],
                                    eef_tf["translation"]["z"]])
                new_detections.sort(
                    key=lambda d: np.linalg.norm(d["pos_base"] - eef_pos))
            best = new_detections[0]
            self.get_logger().info(
                f"Fresh detection: {best['pos_base'].round(4)} "
                f"conf={best['conf']:.2f}")
            new_entrance = self._find_entrance_pose(best["pos_base"])
            if new_entrance:
                # No tip correction here — applied at descent
                new_entrance["translation"]["x"] = float(best["pos_base"][0])
                new_entrance["translation"]["y"] = float(best["pos_base"][1])
                port_tf = new_entrance
                # recompute hover with updated port (orientation already settled, only pos update)
                updated = self._calc_gripper_pose(
                    port_tf, plug_frame, z_offset,
                    slerp_frac=1.0, pos_frac=1.0, reset_integrator=True)
                if updated:
                    hover_pos   = np.array(updated[:3])
                    target_quat = np.array(updated[3:])
        else:
            self.get_logger().warn("Re-detection found nothing — using original position.")

        # ── rotate into full insertion orientation (roll+pitch+yaw) ──────────
        self.get_logger().info("Rotating gripper into full insertion orientation …")
        move_to(hover_pos, target_quat,
                speed_mps=self.args.speed * 0.5,
                stiffness=300.0, damping=50.0,
                label="orient")

        # ── Nudge to tip-corrected position before descent ───────────────────
        plug_tf_c = self._lookup_plug_tf("_synthetic")
        if plug_tf_c is not None:
            R_tip_c = Rotation.from_quat([plug_tf_c["rotation"]["x"],
                                          plug_tf_c["rotation"]["y"],
                                          plug_tf_c["rotation"]["z"],
                                          plug_tf_c["rotation"]["w"]])
            corr_base = R_tip_c.apply(self._TIP_CORRECTION_TIP_FRAME)
        else:
            corr_base = self._TIP_CORRECTION_TIP_FRAME
        hover_pos_corrected = hover_pos - corr_base
        self.get_logger().info(
            f"Nudging by tip correction {(corr_base*1000).round(2)} mm …")
        move_to(hover_pos_corrected, target_quat,
                speed_mps=self.args.speed * 0.3,
                stiffness=300.0, damping=50.0,
                label="tip_correction")
        hover_pos = hover_pos_corrected

        # ── Phase 2: descend into port ───────────────────────────────────────
        # Phase 2: move hover_pos straight down along insertion axis.
        # Do NOT use _calc_gripper_pose — its dynamic plug-offset breaks after rotation.
        pr       = port_tf["rotation"]
        R_port   = Rotation.from_quat([pr["x"], pr["y"], pr["z"], pr["w"]]).as_matrix()
        ins_axis = R_port[:, 2]   # insertion direction (points into port)
        lat_x    = R_port[:, 0]
        lat_y    = R_port[:, 1]

        self.get_logger().info(
            f"Phase 2 ins_axis={ins_axis.round(3)}  "
            f"hover_pos={hover_pos.round(4)}")

        # ins_axis should point INTO the port (downward for SC port ~ [0,0,-1]).
        # If Z component is positive the axis points UP — flip it.
        if ins_axis[2] > 0:
            self.get_logger().warn(
                "ins_axis points UP — flipping. Check port frame orientation.")
            ins_axis = -ins_axis

        insert_speed  = self.args.speed * 0.15
        step_m        = insert_speed * DT
        total_descent = self.args.approach_offset + self.args.insert_depth
        n_insert      = max(1, int(np.ceil(total_descent / step_m)))
        self.get_logger().info(
            f"Phase 2: inserting — {total_descent*100:.1f} cm "
            f"at {insert_speed*1000:.1f} mm/s → {n_insert * DT:.1f}s")

        for i in range(1, n_insert + 1):
            pos = hover_pos + ins_axis * (i * step_m)
            send_pose(self._pose_pub, self.get_clock(),
                      pos[0], pos[1], pos[2],
                      target_quat[0], target_quat[1],
                      target_quat[2], target_quat[3],
                      40.0, 80.0)
            ros_sleep()

        # ── spiral search + XY wiggle to seat the plug ───────────────────────
        if True:
            from transforms3d._gohlketransforms import quaternion_multiply
            base_pos  = hover_pos + ins_axis * total_descent
            base_quat = target_quat

            # Record EEF position at start of spiral — insertion detected when
            # the EEF drops further along the insertion axis (plug fell in).
            spiral_start_tf = lookup_tf(self._tf_buf, EEF_FRAME)
            spiral_start_z  = spiral_start_tf["translation"]["z"] \
                              if spiral_start_tf else base_pos[2]

            def is_inserted(drop_threshold_m: float = 0.005) -> bool:
                """True if EEF dropped >threshold along insertion axis since spiral start."""
                tf = lookup_tf(self._tf_buf, EEF_FRAME)
                if tf is None:
                    return False
                eef_pos = np.array([tf["translation"]["x"],
                                    tf["translation"]["y"],
                                    tf["translation"]["z"]])
                # drop = movement along ins_axis from spiral start position
                drop = np.dot(
                    np.array([eef_pos[0], eef_pos[1], eef_pos[2]]) -
                    np.array([eef_pos[0], eef_pos[1], spiral_start_z]),
                    ins_axis)
                return drop > drop_threshold_m

            # Measure force bias at current pose (gravity + cable, pre-contact).
            # Add insertion delta on top so the net force on the port is INSERT_FORCE_N.
            INSERT_FORCE_N = 8.0
            bias = self._measure_wrench_bias()
            ins_force = tuple(bias + ins_axis * INSERT_FORCE_N)

            # ── Phase A: spiral search ────────────────────────────────────────
            self.get_logger().info("Spiral search …")
            spiral_steps  = 400          # ~20 s at 50 ms
            max_radius    = 0.005        # 5 mm max radius
            for i in range(spiral_steps):
                if is_inserted():
                    self.get_logger().info("Spiral: plug seated!")
                    break
                alpha  = i / spiral_steps
                r      = max_radius * alpha
                theta  = alpha * 8 * np.pi   # 4 full circles
                offset = lat_x * r * np.cos(theta) + lat_y * r * np.sin(theta)
                pos    = base_pos + offset
                send_pose(self._pose_pub, self.get_clock(),
                          pos[0], pos[1], pos[2],
                          base_quat[0], base_quat[1],
                          base_quat[2], base_quat[3],
                          0.0, 30.0, force_xyz=ins_force)
                ros_sleep()

            # ── Phase B: XY wiggle + rotation + push ─────────────────────────
            if not is_inserted():
                self.get_logger().info("XY wiggle + rotation …")
                n_cycles  = 12
                n_steps   = 30
                xy_amp    = 0.004        # ±4 mm lateral
                rot_amp   = 5.0          # ±5° around insertion axis
                for cycle in range(n_cycles):
                    if is_inserted():
                        self.get_logger().info("Wiggle: plug seated!")
                        break
                    for step in range(n_steps):
                        t = 2 * np.pi * step / n_steps
                        # XY nudge
                        offset = lat_x * xy_amp * np.sin(t) + \
                                 lat_y * xy_amp * np.sin(2 * t)
                        pos = base_pos + offset
                        # rotation wiggle
                        angle_rad = np.deg2rad(rot_amp * np.sin(t))
                        s = np.sin(angle_rad / 2); c = np.cos(angle_rad / 2)
                        q_w  = (c, ins_axis[0]*s, ins_axis[1]*s, ins_axis[2]*s)
                        q_bw = (base_quat[3], base_quat[0],
                                base_quat[1], base_quat[2])
                        q_ow = quaternion_multiply(q_w, q_bw)
                        q_out = np.array([q_ow[1], q_ow[2], q_ow[3], q_ow[0]])
                        send_pose(self._pose_pub, self.get_clock(),
                                  pos[0], pos[1], pos[2],
                                  q_out[0], q_out[1], q_out[2], q_out[3],
                                  0.0, 30.0, force_xyz=ins_force)
                        ros_sleep()

        self.get_logger().info("Inserted! Waiting 5s to stabilise …")
        for _ in range(100):
            ros_sleep()

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self):
        from ultralytics import YOLO
        model = YOLO(self.args.model)

        self.get_logger().info("Waiting for camera data …")
        ready = self._wait_cameras(timeout=15.0)
        if not ready:
            self.get_logger().error("No camera data received. Is the simulation running?")
            return

        # Let TF warm up
        self.get_logger().info("Warming TF buffer …")
        time.sleep(2.0)
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Detect
        self.get_logger().info("Running YOLO detection …")
        detections = self._detect_sc_ports(model)
        self._visualise(detections)

        if not detections:
            self.get_logger().error("No SC ports detected. Check conf threshold or camera view.")
            return

        # Find EEF position to pick closest port
        eef_tf = lookup_tf(self._tf_buf, EEF_FRAME)
        if eef_tf:
            eef_pos = np.array([eef_tf["translation"]["x"],
                                eef_tf["translation"]["y"],
                                eef_tf["translation"]["z"]])
            detections.sort(key=lambda d: np.linalg.norm(d["pos_base"] - eef_pos))
            self.get_logger().info(
                f"Closest SC port at {detections[0]['pos_base']} "
                f"(dist={np.linalg.norm(detections[0]['pos_base'] - eef_pos):.3f}m)")
        else:
            self.get_logger().warn("Could not read EEF pose — picking first detection.")

        target = detections[0]

        # Get orientation from nearest TF entrance frame; use YOLO XY for position
        entrance_tf = self._find_entrance_pose(target["pos_base"])
        if entrance_tf is None:
            self.get_logger().warn(
                "No TF entrance within 15cm — using YOLO position + downward orientation.")
            r_down = Rotation.from_euler("x", 180, degrees=True).as_quat()
            entrance_tf = {
                "translation": {k: float(v) for k, v in
                                zip("xyz", target["pos_base"])},
                "rotation":    {"x": r_down[0], "y": r_down[1],
                                "z": r_down[2], "w": r_down[3]},
            }
        else:
            # Override XY position with YOLO detection (no tip correction here — applied at descent)
            entrance_tf["translation"]["x"] = float(target["pos_base"][0])
            entrance_tf["translation"]["y"] = float(target["pos_base"][1])
            self.get_logger().info(
                f"Port: YOLO XY=({target['pos_base'][0]:.3f},{target['pos_base'][1]:.3f}) "
                f"TF Z={entrance_tf['translation']['z']:.3f}")

        # Find cable plug-tip frame
        plug_frame = self._find_plug_frame()
        self.get_logger().info(f"Using plug frame: {plug_frame}")

        if self.args.dry_run:
            self.get_logger().info(
                f"[DRY RUN] port={entrance_tf['translation']}  plug={plug_frame}")
            self._log_orientation_check(entrance_tf, plug_frame)
            self.get_logger().info("Detection images → /tmp/plug_detect_*.png")
            return

        if not self._switch_cartesian():
            self.get_logger().error("Failed to switch to Cartesian mode.")
            return

        self._plug_in(entrance_tf, plug_frame, model)
        self.get_logger().info("Done. Cable plugged in.")


# ─── entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",           default="runs/detect/yolov8_runs/aic_detect/weights/best.pt")
    ap.add_argument("--conf",            type=float, default=0.5)
    ap.add_argument("--approach-offset", type=float, default=0.08,
                    help="Pre-grasp offset above entrance (m)")
    ap.add_argument("--insert-depth",    type=float, default=0.025,
                    help="How far to push past entrance (m)")
    ap.add_argument("--speed",           type=float, default=0.04,
                    help="Approach speed in m/s (default 2 cm/s)")
    ap.add_argument("--hover-wait",      type=float, default=4.0,
                    help="Seconds to stabilise at hover before inserting (default 3s)")
    ap.add_argument("--z-rot-offset",   type=float, default=0.0,
                    help="Trim rotation (deg) on top of logo yaw to key the SC plug")
    ap.add_argument("--dry-run",         action="store_true",
                    help="Detect only, do not move robot")
    args = ap.parse_args()

    rclpy.init()
    node = PlugNode(args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
