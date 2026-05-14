"""Policy that detects the logo, resolves the target entrance from hardcoded
logo-relative offsets, hovers above it, then hands off to insertion.

Entrance offsets (in logo frame, measured 2026-05-02):
  nic_card_mount_X / sfp_port_Y_link_entrance
  sc_port_X / sc_port_base_link_entrance
"""

import time
import numpy as np
import cv2

from scipy.spatial.transform import Rotation
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import TransformException
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Wrench

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task


# ── hardcoded logo-relative entrance offsets (XYZ in logo frame) ─────────────
# Source: dump_entrances_from_logo.py, 2026-05-02, +2.2° yaw correction applied.
# Key format: "{target_module_name}/{port_name}_link_entrance"

_ENTRANCE_XYZ = {
    "nic_card_mount_0/sfp_port_0_link_entrance": np.array([ 0.3271,  0.0078, 0.2093]),
    "nic_card_mount_0/sfp_port_1_link_entrance": np.array([ 0.3273, -0.0154, 0.2093]),
    "nic_card_mount_1/sfp_port_0_link_entrance": np.array([ 0.2871,  0.0074, 0.2093]),
    "nic_card_mount_1/sfp_port_1_link_entrance": np.array([ 0.2873, -0.0158, 0.2093]),
    "nic_card_mount_2/sfp_port_0_link_entrance": np.array([ 0.2471,  0.0070, 0.2093]),
    "nic_card_mount_2/sfp_port_1_link_entrance": np.array([ 0.2473, -0.0162, 0.2093]),
    "nic_card_mount_3/sfp_port_0_link_entrance": np.array([ 0.2071,  0.0066, 0.2093]),
    "nic_card_mount_3/sfp_port_1_link_entrance": np.array([ 0.2073, -0.0166, 0.2093]),
    "nic_card_mount_4/sfp_port_0_link_entrance": np.array([ 0.1671,  0.0061, 0.2093]),
    "nic_card_mount_4/sfp_port_1_link_entrance": np.array([ 0.1673, -0.0170, 0.2093]),
    "sc_port_0/sc_port_base_link_entrance":      np.array([ 0.1108,  0.0010, 0.0601]),
    "sc_port_1/sc_port_base_link_entrance":      np.array([ 0.0698,  0.0006, 0.0601]),
}

_MAGENTA_LOW  = np.array([140,  80,  60])
_MAGENTA_HIGH = np.array([170, 255, 255])
LOGO_YAW_CORRECTION_DEG = 2.2
APPROACH_OFFSET_M = 0.06   # hover this far above the entrance Z


def _entrance_key(task: Task) -> str | None:
    """Build the lookup key from the task fields."""
    # SFP ports: target_module_name="nic_card_mount_X", port_name="sfp_port_Y"
    # SC  ports: target_module_name="sc_port_X",        port_name="sc_port_base"
    if task.port_type in ("sfp",):
        return f"{task.target_module_name}/{task.port_name}_link_entrance"
    if task.port_type in ("sc",):
        return f"{task.target_module_name}/{task.port_name}_link_entrance"
    # fallback: try direct construction
    return f"{task.target_module_name}/{task.port_name}_link_entrance"


class LogoPortPolicy(Policy):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._node = parent_node

    # ── logo detection ────────────────────────────────────────────────────────

    def _detect_logo_pose(self, obs) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Return (pos_base, quat_base) of the logo or (None, None)."""
        pos_samples, yaw_samples = [], []

        cameras = {
            "left_camera":   ("left_camera/optical",   obs.left_image,   obs.left_camera_info),
            "center_camera": ("center_camera/optical", obs.center_image, obs.center_camera_info),
            "right_camera":  ("right_camera/optical",  obs.right_image,  obs.right_camera_info),
        }

        for cam_name, (optical_frame, img_msg, info_msg) in cameras.items():
            if img_msg is None or info_msg is None:
                continue

            # decode image
            data = np.frombuffer(img_msg.data, dtype=np.uint8)
            if img_msg.encoding in ("rgb8", "bgr8"):
                rgb = data.reshape(img_msg.height, img_msg.width, 3)
                if img_msg.encoding == "rgb8":
                    rgb = rgb[:, :, ::-1].copy()
            else:
                continue

            K = np.array(info_msg.k).reshape(3, 3)
            fx, fy, cx_k, cy_k = K[0,0], K[1,1], K[0,2], K[1,2]

            try:
                tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", optical_frame, Time())
                t = tf.transform
                pos_c = np.array([t.translation.x, t.translation.y, t.translation.z])
                quat_c = np.array([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])
            except TransformException:
                continue

            T = np.eye(4)
            T[:3, 3]  = pos_c
            T[:3, :3] = Rotation.from_quat(quat_c).as_matrix()
            R = T[:3, :3]

            # magenta mask
            bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, _MAGENTA_LOW, _MAGENTA_HIGH)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            max_a  = max(cv2.contourArea(c) for c in cnts)
            parts  = [c for c in cnts if cv2.contourArea(c) > 0.01 * max_a]
            merged = np.vstack(parts)
            M = cv2.moments(merged.reshape(-1, 1, 2))
            if M["m00"] == 0:
                continue
            cx_px = M["m10"] / M["m00"]
            cy_px = M["m01"] / M["m00"]

            # 3-D position via ray-plane fallback (Z = 0.03 m)
            ray = np.linalg.inv(K) @ np.array([cx_px, cy_px, 1.0])
            ray /= np.linalg.norm(ray)
            rb = R @ ray; tb = T[:3, 3]
            if abs(rb[2]) < 1e-6:
                continue
            t_s = (0.03 - tb[2]) / rb[2]
            if t_s < 0:
                continue
            pos_samples.append(tb + t_s * rb)

            # yaw from slit direction
            rect    = cv2.minAreaRect(merged.reshape(-1, 1, 2))
            corners = cv2.boxPoints(rect).astype(np.float64)
            cent    = np.array([cx_px, cy_px])
            best_d  = float("inf"); sd = None
            for i in range(4):
                p1, p2 = corners[i], corners[(i+1) % 4]
                ev = p2 - p1; el = np.linalg.norm(ev)
                if el < 1: continue
                eu = ev / el
                perp = (cent - p1) - np.dot(cent - p1, eu) * eu
                d = np.linalg.norm(perp)
                if d < best_d:
                    best_d = d
                    raw = (p1 + p2) / 2.0 - cent
                    if np.linalg.norm(raw) > 1:
                        sd = raw / np.linalg.norm(raw)
            if sd is None:
                continue

            def ray2(pt):
                return np.array([(pt[0]-cx_k)/fx, (pt[1]-cy_k)/fy, 1.0])
            xb = R @ ray2(cent + sd * 50) - R @ ray2(cent)
            xb[2] = 0.0; n = np.linalg.norm(xb)
            if n < 1e-6: continue
            xb /= n
            yaw_samples.append(float(np.arctan2(xb[1], xb[0])))

        if not pos_samples or not yaw_samples:
            return None, None

        pos     = np.mean(pos_samples, axis=0)
        yaw_raw = float(np.arctan2(
            np.mean([np.sin(y) for y in yaw_samples]),
            np.mean([np.cos(y) for y in yaw_samples])))
        yaw  = yaw_raw + np.radians(LOGO_YAW_CORRECTION_DEG)
        quat = Rotation.from_euler("z", yaw).as_quat()
        return pos, quat

    # ── entrance resolution ───────────────────────────────────────────────────

    def _resolve_entrance(self, task: Task,
                          logo_pos: np.ndarray,
                          logo_quat: np.ndarray) -> np.ndarray | None:
        """Return the entrance XYZ in base_link from the logo pose + table."""
        key = _entrance_key(task)
        if key not in _ENTRANCE_XYZ:
            self._log(f"Unknown entrance key: '{key}'")
            self._log(f"Available keys: {list(_ENTRANCE_XYZ.keys())}")
            return None

        offset_logo = _ENTRANCE_XYZ[key]
        T_base_logo = np.eye(4)
        T_base_logo[:3, 3]  = logo_pos
        T_base_logo[:3, :3] = Rotation.from_quat(logo_quat).as_matrix()
        entrance_base = (T_base_logo @ np.append(offset_logo, 1.0))[:3]
        return entrance_base

    # ── motion helpers ────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self._parent_node.get_logger().info(f"[LogoPortPolicy] {msg}")

    def _current_eef_pos(self) -> np.ndarray | None:
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "gripper/tcp", Time())
            t = tf.transform
            return np.array([t.translation.x, t.translation.y, t.translation.z])
        except TransformException:
            return None

    # ── Policy interface ──────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:

        self._log(f"Task received: module={task.target_module_name} "
                  f"port={task.port_name} plug={task.plug_type}")

        # ── 1. detect logo ────────────────────────────────────────────────────
        send_feedback("Detecting logo …")
        obs = get_observation()
        logo_pos, logo_quat = self._detect_logo_pose(obs)
        if logo_pos is None:
            self._log("Logo not detected — aborting.")
            return False
        yaw_deg = Rotation.from_quat(logo_quat).as_euler("xyz", degrees=True)[2]
        self._log(f"Logo: pos={logo_pos.round(3)}  yaw={yaw_deg:.1f}°")

        # ── 2. resolve entrance ───────────────────────────────────────────────
        entrance_pos = self._resolve_entrance(task, logo_pos, logo_quat)
        if entrance_pos is None:
            return False
        self._log(f"Entrance in base_link: {entrance_pos.round(4)}")

        # hover target: approach_offset above entrance
        hover_pos = entrance_pos.copy()
        hover_pos[2] += APPROACH_OFFSET_M

        # ── 3. move to hover ──────────────────────────────────────────────────
        send_feedback(f"Moving to hover above {task.port_name} …")
        current = self._current_eef_pos()
        if current is None:
            self._log("Cannot read EEF pose.")
            return False

        # get current EEF orientation
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "gripper/tcp", Time())
            r = tf.transform.rotation
            current_quat = np.array([r.x, r.y, r.z, r.w])
        except TransformException:
            self._log("Cannot read EEF orientation.")
            return False

        # compute target EEF yaw from logo yaw + plug-type offset
        # SC: +180°  SFP: +90°
        logo_yaw_deg = Rotation.from_quat(logo_quat).as_euler("xyz", degrees=True)[2]
        yaw_offsets  = {"sc": 90.0, "sfp": 90.0}
        eef_yaw_deg  = logo_yaw_deg + yaw_offsets.get(task.plug_type, 0.0)
        self._log(f"Logo yaw {logo_yaw_deg:.1f}° + {yaw_offsets.get(task.plug_type,0):.0f}° "
                  f"→ EEF yaw {eef_yaw_deg:.1f}°")

        # keep current roll/pitch, apply target yaw only
        _, pitch_curr, roll_curr = Rotation.from_quat(current_quat).as_euler("ZYX")
        target_quat = Rotation.from_euler(
            "ZYX", [np.radians(eef_yaw_deg), pitch_curr, roll_curr]).as_quat()

        from scipy.spatial.transform import Slerp
        n_steps = max(1, int(np.linalg.norm(hover_pos - current) / 0.002))
        key_rots = Rotation.concatenate([
            Rotation.from_quat(current_quat),
            Rotation.from_quat(target_quat),
        ])
        slerp_fn = Slerp([0.0, 1.0], key_rots)

        stiffness = [400.0, 400.0, 400.0, 50.0, 50.0, 50.0]
        damping   = [80.0,  80.0,  80.0,  20.0, 20.0, 20.0]

        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            pos   = current + alpha * (hover_pos - current)
            quat  = slerp_fn(alpha).as_quat()
            mu = MotionUpdate(
                header=Header(
                    frame_id="base_link",
                    stamp=self._parent_node.get_clock().now().to_msg()),
                pose=Pose(
                    position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                    orientation=Quaternion(x=float(quat[0]), y=float(quat[1]),
                                          z=float(quat[2]), w=float(quat[3]))),
                target_stiffness=np.diag(stiffness).flatten().tolist(),
                target_damping=np.diag(damping).flatten().tolist(),
                feedforward_wrench_at_tip=Wrench(
                    force=Vector3(x=0.0, y=0.0, z=0.0),
                    torque=Vector3(x=0.0, y=0.0, z=0.0)),
                wrench_feedback_gains_at_tip=[0.0]*6,
                trajectory_generation_mode=TrajectoryGenerationMode(
                    mode=TrajectoryGenerationMode.MODE_POSITION),
            )
            move_robot(motion_update=mu)
            time.sleep(0.05)

        self._log(f"Arrived above {task.port_name} at {hover_pos.round(4)}")
        send_feedback(f"Hovering above {task.port_name}")
        return True
