"""Extension of LogoPortPolicy that refines the target port position with YOLO
after hovering, before final insertion.

YOLO classes (same model as plug_sc_port.py):
  0 = sc_port
  1 = nic_port  (individual SFP entrance)

Sanity checks:
  - Only accept detections whose X offset from logo matches the target module's
    expected X (within X_TOL). This rejects detections from other NIC mounts.
  - For SFP ports, additionally check Y sign to distinguish port_0 (+Y) from
    port_1 (-Y). Never mix them up.
"""

import time
import numpy as np
import cv2

from scipy.spatial.transform import Rotation
import rclpy
from rclpy.time import Time
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

from aic_example_policies.ros.LogoPortPolicy import (
    LogoPortPolicy,
    _ENTRANCE_XYZ,
    _entrance_key,
    APPROACH_OFFSET_M,
)

# ── YOLO class IDs ─────────────────────────────────────────────────────────────
_CLS_SC  = 0   # sc_port
_CLS_SFP = 1   # nic_port (SFP entrance)

# Tolerance for matching detections to expected logo-frame offsets.
# NIC cards can have ±2 cm translation (shifts Y) and ±10° yaw (shifts X ~3 mm).
# Mounts are 40 mm apart in X → 30 mm tolerance safely separates them.
_X_TOL   = 0.030   # 30 mm — identifies the correct NIC card mount by X
_X_PAIR  = 0.035   # 35 mm — window to collect both SFP ports of the same mount


def _mat4(pos, quat):
    T = np.eye(4)
    T[:3, 3]  = pos
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    return T


def _median_depth(depth_img, u, v, r=8):
    h, w = depth_img.shape
    patch = depth_img[max(0, v-r):min(h, v+r), max(0, u-r):min(w, u+r)]
    valid = patch[patch > 0.05]
    return float(np.median(valid)) if len(valid) > 5 else None


def _ros_img_to_numpy(msg):
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = data.reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if msg.encoding == "rgb8" else img.copy()
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    if msg.encoding == "16UC1":
        return (np.frombuffer(msg.data, dtype=np.uint16)
                .reshape(msg.height, msg.width).astype(np.float32) / 1000.0)
    return None


class LogoYoloPolicy(LogoPortPolicy):
    """LogoPortPolicy + YOLO-based port position refinement after hover."""

    def __init__(self, parent_node,
                 model_path="/home/richard/aic/runs/detect/yolov8_runs/aic_detect/weights/best.pt",
                 yolo_conf=0.35):
        super().__init__(parent_node)
        self._model_path    = model_path
        self._yolo_conf     = yolo_conf
        self._yolo_model    = None
        self._wrench_latest = None
        parent_node.create_subscription(
            __import__('geometry_msgs.msg', fromlist=['WrenchStamped']).WrenchStamped,
            "/fts_broadcaster/wrench_filtered",
            lambda m: setattr(self, "_wrench_latest", m.wrench),
            __import__('rclpy.qos', fromlist=['qos_profile_sensor_data']).qos_profile_sensor_data,
        )

    def _measure_wrench_bias(self, n: int = 30) -> np.ndarray:
        """Average n FTS readings at current pose → gravity + cable bias."""
        samples = []
        t0 = time.time()
        while len(samples) < n and time.time() - t0 < 5.0:
            self._wrench_latest = None
            rclpy.spin_once(self._parent_node, timeout_sec=0.1)
            if self._wrench_latest is not None:
                w = self._wrench_latest
                samples.append([w.force.x, w.force.y, w.force.z])
        if samples:
            bias = np.mean(samples, axis=0)
            self._log(f"Wrench bias: {bias.round(3)} N")
            return bias
        self._log("No FTS data — bias set to zero")
        return np.zeros(3)

    # ── robust logo detection ─────────────────────────────────────────────────

    def _detect_logo_robust(self, get_observation, move_robot, send_feedback,
                            min_pixels: int = 3000,
                            edge_margin: int = 30,
                            max_nudge: int = 10,
                            max_z_steps: int = 10,
                            move_step: float = 0.025,
                            z_step: float = 0.015,
                            agree_deg: float = 2.0,
                            ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Two-phase robust logo detection.

        Phase 1 — centering (up to max_nudge attempts):
          Uses ONLY the center camera. Nudges the robot horizontally toward the
          pink-pixel centroid until the logo is complete and unoccluded.

        Phase 2 — accuracy (up to max_z_steps attempts):
          Uses ALL THREE cameras. Moves the robot closer (lower Z) until all
          cameras agree on yaw within agree_deg degrees.

        Debug images saved to /tmp/logo_yolo_debug/.
        """
        import os; os.makedirs("/tmp/logo_yolo_debug", exist_ok=True)

        # ── Phase 1: center camera only — move directly above the logo ───────
        self._log("Logo phase 1: center camera detection …")
        obs = get_observation()
        pos, _, _, _ = self._logo_detect_cam(
            obs, "center", 1, min_pixels=800, edge_margin=0, tag="p1_center")

        if pos is None:
            self._log("Phase 1: no pink pixels — rising + spiral search …")
            found = self._search_logo_up_then_spiral(
                get_observation, move_robot, send_feedback)
            if not found:
                self._log("Phase 1: logo not found after spiral — aborting")
                return None, None
            obs = get_observation()
            pos, _, _, _ = self._logo_detect_cam(
                obs, "center", 2, min_pixels=800, edge_margin=0, tag="p1_spiral")
            if pos is None:
                self._log("Phase 1: still no pink pixels after spiral — aborting")
                return None, None

        # Sanity-check the ray-plane result is within the robot workspace
        if not (-1.5 < float(pos[0]) < 1.5 and -1.5 < float(pos[1]) < 1.5):
            self._log(f"Phase 1: position {pos.round(3)} out of workspace — aborting")
            return None, None

        send_feedback("Moving above logo …")
        self._log(f"Phase 1: moving above logo XY={pos[:2].round(4)}")
        self._move_above_xy(pos[:2], move_robot)
        self._wait_standstill(min_wait=0.8, timeout=5.0)

        # ── Phase 2: move closer until all cameras agree on yaw ───────────────
        self._log("Logo phase 2: accuracy check with all cameras …")
        for z_attempt in range(1, max_z_steps + 1):
            obs = get_observation()
            pos, quat, yaws = self._logo_detect_all(obs, z_attempt)

            if pos is None:
                self._log(f"Phase 2 attempt {z_attempt}: logo lost — aborting")
                return None, None

            if len(yaws) >= 2:
                # Check if any pair of cameras agrees within agree_deg
                best_pair = None
                best_spread = float("inf")
                for i in range(len(yaws)):
                    for j in range(i+1, len(yaws)):
                        d = abs(np.degrees(yaws[i]) - np.degrees(yaws[j]))
                        d = min(d, 360 - d)
                        if d < best_spread:
                            best_spread = d
                            best_pair = (i, j)
                self._log(f"Phase 2 attempt {z_attempt}: "
                          f"{len(yaws)} cameras, best pair spread={best_spread:.1f}°")
                if best_spread <= agree_deg:
                    # Use mean of the agreeing pair
                    i, j = best_pair
                    pair_yaw = float(np.arctan2(
                        np.mean([np.sin(yaws[i]), np.sin(yaws[j])]),
                        np.mean([np.cos(yaws[i]), np.cos(yaws[j])])))
                    pair_quat = Rotation.from_euler("z", pair_yaw).as_quat()
                    self._log(f"Pair agrees ({best_spread:.1f}° < {agree_deg}°) ✓  "
                              f"yaw={np.degrees(pair_yaw):.1f}°")
                    self._log("Logo debug images → /tmp/logo_yolo_debug/")
                    return pos, pair_quat
                send_feedback(f"No pair agrees ({best_spread:.1f}°) — moving closer …")
            else:
                self._log(f"Phase 2 attempt {z_attempt}: only {len(yaws)} camera(s)")
                if len(yaws) == 1:
                    self._log("Only one camera sees logo — accepting single-camera result")
                    return pos, quat

            self._move_closer_z(z_step, move_robot)
            self._wait_standstill(min_wait=0.5, timeout=4.0)

        self._log(f"Phase 2: accepting best estimate after {max_z_steps} attempts")
        return pos, quat

    # ── shared logo-detection helpers ─────────────────────────────────────────

    _MAG_LOW  = np.array([140,  80,  60])
    _MAG_HIGH = np.array([170, 255, 255])
    _LOGO_YAW_CORR_DEG = 4.5

    def _logo_detect_cam(self, obs, cam_key: str, attempt: int,
                         min_pixels: int, edge_margin: int, tag: str = ""
                         ) -> tuple:
        """Detect logo in a single camera.

        Returns (pos, quat, complete, move_dir).
        """
        _CAM_MAP = {
            "left":   ("left_camera/optical",   obs.left_image,   obs.left_camera_info),
            "center": ("center_camera/optical", obs.center_image, obs.center_camera_info),
            "right":  ("right_camera/optical",  obs.right_image,  obs.right_camera_info),
        }
        optical_frame, img_msg, info_msg = _CAM_MAP[cam_key]
        cam_name = f"{cam_key}_camera"

        pos_samples, yaw_samples, complete_flags = [], [], []
        best_move_dir = None

        cameras = {cam_name: (optical_frame, img_msg, info_msg)}

        for cam_name, (optical_frame, img_msg, info_msg) in cameras.items():
            if img_msg is None or info_msg is None:
                continue
            rgb = _ros_img_to_numpy(img_msg)
            if rgb is None:
                continue
            H, W = rgb.shape[:2]
            K = np.array(info_msg.k).reshape(3, 3)
            fx, fy, cx_k, cy_k = K[0,0], K[1,1], K[0,2], K[1,2]

            try:
                tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", optical_frame, Time())
                t = tf.transform
                pos_c  = np.array([t.translation.x, t.translation.y, t.translation.z])
                quat_c = np.array([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])
            except TransformException:
                continue
            T = _mat4(pos_c, quat_c); R = T[:3, :3]

            bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self._MAG_LOW, self._MAG_HIGH)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

            n_px = int(mask.sum() / 255)

            # ── debug canvas ──────────────────────────────────────────────────
            vis = bgr.copy()
            # pink overlay on mask
            overlay = vis.copy()
            overlay[mask > 0] = (80, 0, 200)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
            cv2.putText(vis, f"attempt {attempt}  pink_px={n_px}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if n_px < 10 or not cnts:
                cv2.putText(vis, "NO LOGO", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                continue

            # pink centroid → direction to move
            M_all = cv2.moments(mask)
            if M_all["m00"] > 0:
                cx_all = int(M_all["m10"] / M_all["m00"])
                cy_all = int(M_all["m01"] / M_all["m00"])
                cv2.circle(vis, (cx_all, cy_all), 6, (0, 180, 255), -1)
                # direction in camera-space XY → base_link
                dx_img = (cx_all - cx_k) / fx
                dy_img = (cy_all - cy_k) / fy
                ray    = R @ np.array([dx_img, dy_img, 1.0])
                ray[2] = 0.0
                if np.linalg.norm(ray[:2]) > 1e-6:
                    best_move_dir = ray[:2] / np.linalg.norm(ray[:2])

            # completeness checks
            max_a  = max(cv2.contourArea(c) for c in cnts)
            parts  = [c for c in cnts if cv2.contourArea(c) > 0.01 * max_a]
            merged = np.vstack(parts)
            bx, by, bw, bh = cv2.boundingRect(merged)

            clipped = (bx < edge_margin or by < edge_margin or
                       bx + bw > W - edge_margin or by + bh > H - edge_margin)

            # Solidity check: arm occlusion makes the logo non-convex.
            # A clean logo is a roughly solid shape (solidity > 0.55).
            hull      = cv2.convexHull(merged)
            hull_area = cv2.contourArea(hull)
            solidity  = float(cv2.contourArea(merged)) / (hull_area + 1e-6)
            occluded  = solidity < 0.20

            complete = (n_px >= min_pixels) and not clipped and not occluded

            if not complete:
                if clipped:     reason = "CLIPPED"
                elif occluded:  reason = f"OCCLUDED (solidity={solidity:.2f})"
                else:           reason = f"TOO SMALL ({n_px}px < {min_pixels})"
            else:
                reason = f"COMPLETE (px={n_px} sol={solidity:.2f})"

            complete_flags.append(complete)
            status_color = (0, 200, 0) if complete else (0, 140, 255)
            status_text  = reason
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), status_color, 2)
            cv2.putText(vis, status_text, (bx, by - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # ── centroid + slit direction (correct orientation) + box (precise angle)
            cv2.drawContours(vis, parts, -1, (255, 180, 0), 2)
            Mc = cv2.moments(merged.reshape(-1, 1, 2))
            if Mc["m00"] == 0:
                continue
            cent = np.array([Mc["m10"]/Mc["m00"], Mc["m01"]/Mc["m00"]])
            cv2.circle(vis, tuple(cent.astype(int)), 8, (0, 255, 0), -1)

            def ray2(pt):
                return np.array([(pt[0]-cx_k)/fx, (pt[1]-cy_k)/fy, 1.0])

            # ── Step 1: slit direction → correct rough yaw ────────────────────
            rect2    = cv2.minAreaRect(merged.reshape(-1, 1, 2))
            corners4 = cv2.boxPoints(rect2).astype(np.float64)
            best_d   = float("inf"); sd = None
            for i in range(4):
                p1, p2 = corners4[i], corners4[(i+1) % 4]
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

            # draw slit direction arrow
            tip = (cent + sd * 70).astype(int)
            cv2.arrowedLine(vis, tuple(cent.astype(int)), tuple(tip),
                            (0, 200, 255), 2, tipLength=0.25)

            xb_slit = R @ ray2(cent + sd * 50) - R @ ray2(cent)
            xb_slit[2] = 0.0
            if np.linalg.norm(xb_slit) < 1e-6:
                continue
            xb_slit /= np.linalg.norm(xb_slit)
            slit_yaw = float(np.arctan2(xb_slit[1], xb_slit[0]))

            # ── Step 2: minAreaRect → precise angle, slit picks the right axis ─
            pts  = cv2.findNonZero(mask).reshape(-1, 2).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            box_corners = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(vis, [box_corners], True, (0, 255, 255), 2)

            _, _, box_angle_deg = rect
            ang0 = np.radians(box_angle_deg)

            # All 4 candidate directions of the two box axes
            candidates = [
                np.array([np.cos(ang0),              np.sin(ang0)]),
                np.array([np.cos(ang0 + np.pi/2),    np.sin(ang0 + np.pi/2)]),
                np.array([np.cos(ang0 + np.pi),      np.sin(ang0 + np.pi)]),
                np.array([np.cos(ang0 + 3*np.pi/2),  np.sin(ang0 + 3*np.pi/2)]),
            ]
            # Pick the one most aligned with the slit direction (same quadrant)
            img_dir = max(candidates, key=lambda v: np.dot(v, sd))

            # Project to base_link for the final precise yaw
            xb_box = R @ ray2(cent + img_dir * 50) - R @ ray2(cent)
            xb_box[2] = 0.0
            if np.linalg.norm(xb_box) > 1e-6:
                xb_box /= np.linalg.norm(xb_box)
                yaw_final = float(np.arctan2(xb_box[1], xb_box[0]))
            else:
                yaw_final = slit_yaw

            cv2.putText(vis,
                        f"slit={np.degrees(slit_yaw):.1f}  refined={np.degrees(yaw_final):.1f}",
                        (10, H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # ── Step 3: 3-D centroid via ray-plane at Z=0.03 m ───────────────
            ray_dir = np.linalg.inv(K) @ np.array([cent[0], cent[1], 1.0])
            ray_dir /= np.linalg.norm(ray_dir)
            rb = R @ ray_dir; tb = T[:3, 3]
            if abs(rb[2]) > 1e-6:
                t_s = (0.03 - tb[2]) / rb[2]
                if t_s > 0:
                    pos_samples.append(tb + t_s * rb)
                    yaw_samples.append(yaw_final)

            cv2.putText(vis, f"px={n_px}  complete={complete}", (10, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if not pos_samples:
            return None, None, False, best_move_dir

        pos     = np.mean(pos_samples, axis=0)
        yaw_raw = float(np.arctan2(
            np.mean([np.sin(y) for y in yaw_samples]),
            np.mean([np.cos(y) for y in yaw_samples])))
        quat = Rotation.from_euler("z", yaw_raw).as_quat()
        return pos, quat, any(complete_flags), best_move_dir

    def _logo_detect_all(self, obs, attempt: int) -> tuple:
        """Detect logo in all three cameras.

        Returns (pos, quat, yaw_list) where yaw_list has one entry per camera
        that successfully detected the logo. Used for inter-camera agreement check.
        """
        all_pos, all_yaw = [], []
        for cam_key in ("left", "center", "right"):
            pos, quat, _, _ = self._logo_detect_cam(
                obs, cam_key, attempt, min_pixels=500, edge_margin=10,
                tag=f"acc{attempt}")
            if pos is not None and quat is not None:
                all_pos.append(pos)
                all_yaw.append(Rotation.from_quat(quat).as_euler("xyz")[2])

        if not all_pos:
            return None, None, []
        pos  = np.mean(all_pos, axis=0)
        yaw  = float(np.arctan2(
            np.mean([np.sin(y) for y in all_yaw]),
            np.mean([np.cos(y) for y in all_yaw])))
        quat = Rotation.from_euler("z", yaw).as_quat()
        return pos, quat, all_yaw

    def _has_pink_pixels(self, obs, min_pixels: int = 800) -> bool:
        """Return True if the center camera sees at least min_pixels magenta pixels."""
        for img_msg in (obs.center_image,):
            if img_msg is None:
                continue
            rgb = _ros_img_to_numpy(img_msg)
            if rgb is None:
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self._MAG_LOW, self._MAG_HIGH)
            if int(mask.sum() / 255) >= min_pixels:
                return True
        return False

    def _search_logo_up_then_spiral(self, get_observation, move_robot, send_feedback,
                                    min_pink_pixels: int = 800,
                                    rise_m: float = 0.15,
                                    sweep_half_m: float = 0.45,
                                    sweep_step_m: float = 0.10) -> bool:
        """Rise 15 cm fast, then sweep Y from +45 cm to -45 cm until pink appears."""
        send_feedback("No logo — rising …")
        self._log(f"Logo search: rising {rise_m*100:.0f} cm")

        # Move up with soft Z (200 stiff / 200 damp in Z)
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "gripper/tcp", Time())
            t = tf.transform.translation
            r = tf.transform.rotation
            pos  = np.array([t.x, t.y, t.z])
            quat = np.array([r.x, r.y, r.z, r.w])
        except TransformException:
            return False

        stiff = np.diag([200.0, 200.0, 200.0, 30.0, 30.0, 30.0]).flatten().tolist()
        damp  = np.diag([200.0, 200.0, 200.0, 15.0, 15.0, 15.0]).flatten().tolist()

        def _send(target_pos):
            cur = pos if not hasattr(_send, "cur") else _send.cur
            n = max(1, int(np.linalg.norm(target_pos - cur) / 0.005))
            for i in range(1, n + 1):
                p = cur + (i / n) * (target_pos - cur)
                move_robot(motion_update=MotionUpdate(
                    header=Header(frame_id="base_link",
                                  stamp=self._parent_node.get_clock().now().to_msg()),
                    pose=Pose(
                        position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                        orientation=Quaternion(x=float(quat[0]), y=float(quat[1]),
                                              z=float(quat[2]), w=float(quat[3]))),
                    target_stiffness=stiff, target_damping=damp,
                    feedforward_wrench_at_tip=Wrench(
                        force=Vector3(x=0.0, y=0.0, z=0.0),
                        torque=Vector3(x=0.0, y=0.0, z=0.0)),
                    wrench_feedback_gains_at_tip=[0.0] * 6,
                    trajectory_generation_mode=TrajectoryGenerationMode(
                        mode=TrajectoryGenerationMode.MODE_POSITION),
                ))
                self.sleep_for(0.05)
            _send.cur = target_pos.copy()
        _send.cur = pos.copy()

        # Rise fast
        _send(pos + np.array([0.0, 0.0, rise_m]))
        self._wait_standstill(min_wait=0.4, timeout=5.0)
        risen = _send.cur.copy()

        # Move to Y start (+45 cm) — check pink during motion too
        send_feedback("Sweeping Y for logo …")
        y_start = risen[1] + sweep_half_m
        y_end   = risen[1] - sweep_half_m
        cur     = risen.copy()

        def _sweep_to(target_y):
            """Move to target_y, checking for pink every 5 commands."""
            nonlocal cur
            target_pos = np.array([risen[0], target_y, risen[2]])
            n = max(1, int(abs(target_pos[1] - cur[1]) / 0.005))
            for i in range(1, n + 1):
                p = cur + (i / n) * (target_pos - cur)
                move_robot(motion_update=MotionUpdate(
                    header=Header(frame_id="base_link",
                                  stamp=self._parent_node.get_clock().now().to_msg()),
                    pose=Pose(
                        position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                        orientation=Quaternion(x=float(quat[0]), y=float(quat[1]),
                                              z=float(quat[2]), w=float(quat[3]))),
                    target_stiffness=stiff, target_damping=damp,
                    feedforward_wrench_at_tip=Wrench(
                        force=Vector3(x=0.0, y=0.0, z=0.0),
                        torque=Vector3(x=0.0, y=0.0, z=0.0)),
                    wrench_feedback_gains_at_tip=[0.0] * 6,
                    trajectory_generation_mode=TrajectoryGenerationMode(
                        mode=TrajectoryGenerationMode.MODE_POSITION),
                ))
                self.sleep_for(0.05)
                if i % 5 == 0:
                    if self._has_pink_pixels(get_observation(), min_pink_pixels):
                        self._log(f"Y sweep: pink found at Y≈{float(p[1]):.3f}")
                        return True
            cur = target_pos.copy()
            return False

        self._log(f"Y sweep: {y_start*100:.0f} cm → {y_end*100:.0f} cm (looping until found)")
        while True:
            if _sweep_to(y_start):
                return True
            if _sweep_to(y_end):
                return True
            self._log("Y sweep: end reached — restarting sweep")

    def _move_above_xy(self, target_xy: np.ndarray, move_robot):
        """Move EEF to be directly above target_xy, keeping current Z and orientation."""
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "gripper/tcp", Time())
            t = tf.transform.translation
            r = tf.transform.rotation
            pos  = np.array([t.x, t.y, t.z])
            quat = np.array([r.x, r.y, r.z, r.w])
        except TransformException:
            return
        target = np.array([target_xy[0], target_xy[1], pos[2]])
        dist   = np.linalg.norm(target - pos)
        n      = max(1, int(dist / 0.002))
        stiffness = [400.0]*3 + [40.0]*3
        damping   = [80.0]*3  + [20.0]*3
        for i in range(1, n + 1):
            alpha = i / n
            p = pos + alpha * (target - pos)
            mu = MotionUpdate(
                header=Header(frame_id="base_link",
                              stamp=self._parent_node.get_clock().now().to_msg()),
                pose=Pose(
                    position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                    orientation=Quaternion(x=float(quat[0]), y=float(quat[1]),
                                          z=float(quat[2]), w=float(quat[3]))),
                target_stiffness=np.diag(stiffness).flatten().tolist(),
                target_damping=np.diag(damping).flatten().tolist(),
                feedforward_wrench_at_tip=Wrench(
                    force=Vector3(x=0.0, y=0.0, z=0.0),
                    torque=Vector3(x=0.0, y=0.0, z=0.0)),
                wrench_feedback_gains_at_tip=[0.0] * 6,
                trajectory_generation_mode=TrajectoryGenerationMode(
                    mode=TrajectoryGenerationMode.MODE_POSITION),
            )
            move_robot(motion_update=mu)
            self.sleep_for(0.05)
        self._log(f"Moved above logo XY: {target.round(4)}")

    def _move_closer_z(self, step: float, move_robot):
        """Move EEF step metres downward (lower Z in base_link)."""
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "gripper/tcp", Time())
            t = tf.transform.translation
            r = tf.transform.rotation
            pos  = np.array([t.x, t.y, t.z])
            quat = np.array([r.x, r.y, r.z, r.w])
        except TransformException:
            return
        target = pos - np.array([0.0, 0.0, step])
        n = max(1, int(step / 0.001))
        stiffness = [300.0]*3 + [30.0]*3
        damping   = [60.0]*3  + [15.0]*3
        for i in range(1, n + 1):
            alpha = i / n
            p = pos + alpha * (target - pos)
            mu = MotionUpdate(
                header=Header(frame_id="base_link",
                              stamp=self._parent_node.get_clock().now().to_msg()),
                pose=Pose(
                    position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                    orientation=Quaternion(x=float(quat[0]), y=float(quat[1]),
                                          z=float(quat[2]), w=float(quat[3]))),
                target_stiffness=np.diag(stiffness).flatten().tolist(),
                target_damping=np.diag(damping).flatten().tolist(),
                feedforward_wrench_at_tip=Wrench(
                    force=Vector3(x=0.0, y=0.0, z=0.0),
                    torque=Vector3(x=0.0, y=0.0, z=0.0)),
                wrench_feedback_gains_at_tip=[0.0] * 6,
                trajectory_generation_mode=TrajectoryGenerationMode(
                    mode=TrajectoryGenerationMode.MODE_POSITION),
            )
            move_robot(motion_update=mu)
            self.sleep_for(0.05)

    def _nudge_toward(self, direction_xy: np.ndarray, step: float, move_robot):
        """Move EEF step metres in direction_xy (base_link XY unit vector)."""
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "gripper/tcp", Time())
            t = tf.transform.translation
            r = tf.transform.rotation
            pos  = np.array([t.x, t.y, t.z])
            quat = np.array([r.x, r.y, r.z, r.w])
        except TransformException:
            return
        target = pos + np.array([direction_xy[0], direction_xy[1], 0.0]) * step
        stiffness = [300.0]*3 + [30.0]*3
        damping   = [60.0]*3  + [15.0]*3
        n = max(1, int(step / 0.005))
        for i in range(1, n + 1):
            alpha = i / n
            p = pos + alpha * (target - pos)
            mu = MotionUpdate(
                header=Header(frame_id="base_link",
                              stamp=self._parent_node.get_clock().now().to_msg()),
                pose=Pose(
                    position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                    orientation=Quaternion(x=float(quat[0]), y=float(quat[1]),
                                          z=float(quat[2]), w=float(quat[3]))),
                target_stiffness=np.diag(stiffness).flatten().tolist(),
                target_damping=np.diag(damping).flatten().tolist(),
                feedforward_wrench_at_tip=Wrench(
                    force=Vector3(x=0.0, y=0.0, z=0.0),
                    torque=Vector3(x=0.0, y=0.0, z=0.0)),
                wrench_feedback_gains_at_tip=[0.0] * 6,
                trajectory_generation_mode=TrajectoryGenerationMode(
                    mode=TrajectoryGenerationMode.MODE_POSITION),
            )
            move_robot(motion_update=mu)
            self.sleep_for(0.05)

    # Static transforms gripper/tcp → tip link (read from TF 2026-05-14)
    _R_TCP_TIP = {
        "sfp": Rotation.from_quat([0.179610,  0.005562, -0.027462,  0.983339]),  # sfp_tip_link
        "sc":  Rotation.from_quat([0.160781, -0.167157,  0.694179,  0.681413]),  # sc_tip_link
    }

    def _load_yolo(self):
        if self._yolo_model is None:
            from ultralytics import YOLO
            self._yolo_model = YOLO(self._model_path)
            self._log(f"YOLO loaded: {self._model_path}")
        return self._yolo_model

    # ── SC port yaw from edge detection ──────────────────────────────────────

    _SC_BLUE_LOW  = np.array([ 85,  60,  60])
    _SC_BLUE_HIGH = np.array([135, 255, 255])

    def _sc_edge_yaw(self, obs, det: dict) -> float | None:
        """Estimate SC port yaw from Hough lines on the blue port body.

        det must have 'pixel' (u, v) and 'cam' keys from _detect_ports_yolo.
        Returns angle in radians (base_link XY convention) or None.
        """
        _CAM_MAP = {
            "left_camera":   obs.left_image,
            "center_camera": obs.center_image,
            "right_camera":  obs.right_image,
        }
        img_msg = _CAM_MAP.get(det["cam"])
        if img_msg is None:
            return None
        rgb = _ros_img_to_numpy(img_msg)
        if rgb is None:
            return None
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Crop ROI around the detection pixel (±40 px)
        u, v = det["pixel"]
        H, W = bgr.shape[:2]
        r = 40
        x1, y1 = max(0, u - r), max(0, v - r)
        x2, y2 = min(W, u + r), min(H, v + r)
        roi = bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Segment blue body
        hsv       = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, self._SC_BLUE_LOW, self._SC_BLUE_HIGH)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE,
                                     np.ones((7, 7), np.uint8))

        # Canny edges inside blue mask
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.bitwise_and(edges, blue_mask)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10,
                                minLineLength=5, maxLineGap=5)
        if lines is None:
            self._log("SC edge yaw: no Hough lines found")
            return None

        # Dominant angle via 1° histogram
        angles = [(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
                  for x1, y1, x2, y2 in lines[:, 0]]
        hist, bins = np.histogram(angles, bins=180, range=(0, 180))
        dominant_deg = float(bins[int(np.argmax(hist))] + 0.5)

        # Save debug image
        import os; os.makedirs("/tmp/logo_yolo_debug", exist_ok=True)
        vis = roi.copy()
        vis[blue_mask == 0] = (vis[blue_mask == 0] * 0.3).astype(np.uint8)
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
        rh, rw = roi.shape[:2]
        cx, cy = rw // 2, rh // 2
        rad = np.radians(dominant_deg)
        dx, dy = int(np.cos(rad) * rw * 0.4), int(np.sin(rad) * rh * 0.4)
        cv2.arrowedLine(vis, (cx - dx, cy - dy), (cx + dx, cy + dy),
                        (0, 80, 255), 2, tipLength=0.2)
        cv2.imwrite(f"/tmp/logo_yolo_debug/sc_edge_{det['cam']}.png", vis)

        self._log(f"SC edge yaw: dominant={dominant_deg:.1f}° (image frame)")

        # Convert image-frame angle to base_link yaw:
        # The dominant_deg is in pixel coords (x right, y down).
        # We need to project it through the camera rotation to base_link XY.
        try:
            cam_frame = det["cam"].replace("_camera", "_camera/optical")
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", cam_frame, __import__('rclpy.time', fromlist=['Time']).Time())
            R = Rotation.from_quat([
                tf.transform.rotation.x, tf.transform.rotation.y,
                tf.transform.rotation.z, tf.transform.rotation.w,
            ]).as_matrix()
            # Image direction vector
            img_dir = np.array([np.cos(np.radians(dominant_deg)),
                                 np.sin(np.radians(dominant_deg)), 0.0])
            base_dir = R @ img_dir
            base_dir[2] = 0.0
            n = np.linalg.norm(base_dir[:2])
            if n < 1e-6:
                return None
            yaw_rad = float(np.arctan2(base_dir[1], base_dir[0]))
            self._log(f"SC edge yaw → base_link: {np.degrees(yaw_rad):.1f}°")
            return yaw_rad
        except Exception as e:
            self._log(f"SC edge yaw TF failed: {e}")
            return None

    # ── YOLO detection ────────────────────────────────────────────────────────

    def _detect_ports_yolo(self, obs, cls_id: int,
                            debug_dir: str = "/tmp/logo_yolo_debug",
                            cam_filter: list[str] | None = None) -> list[dict]:
        """Run YOLO on cameras (all by default), return detections."""
        import os; os.makedirs(debug_dir, exist_ok=True)
        model = self._load_yolo()
        detections = []

        _CLS_NAMES = {_CLS_SC: "sc_port", _CLS_SFP: "sfp_port"}
        _CLS_COLOR = {_CLS_SC: (0, 128, 255), _CLS_SFP: (0, 220, 0)}

        all_cameras = {
            "left_camera":   ("left_camera/optical",   obs.left_image,
                               obs.left_camera_info,   getattr(obs, "left_depth",  None)),
            "center_camera": ("center_camera/optical", obs.center_image,
                               obs.center_camera_info, getattr(obs, "center_depth", None)),
            "right_camera":  ("right_camera/optical",  obs.right_image,
                               obs.right_camera_info,  getattr(obs, "right_depth",  None)),
        }
        cameras = {k: v for k, v in all_cameras.items()
                   if cam_filter is None or k in cam_filter}

        for cam_name, (optical_frame, img_msg, info_msg, depth_msg) in cameras.items():
            if img_msg is None or info_msg is None:
                continue

            rgb = _ros_img_to_numpy(img_msg)
            if rgb is None:
                continue

            K = np.array(info_msg.k).reshape(3, 3)
            fx, fy, cx_k, cy_k = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            try:
                tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", optical_frame, Time())
                t = tf.transform
                pos_c  = np.array([t.translation.x, t.translation.y, t.translation.z])
                quat_c = np.array([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])
            except TransformException:
                continue

            T_base_cam = _mat4(pos_c, quat_c)
            depth_img  = _ros_img_to_numpy(depth_msg) if depth_msg is not None else None

            # Debug canvas (BGR)
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            results = model(rgb, conf=self._yolo_conf, verbose=False)
            # Draw all boxes (any class) in grey first for context
            for r in results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
                    c = int(box.cls.item())
                    clr = _CLS_COLOR.get(c, (180, 180, 180))
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), clr, 2)
                    cv2.putText(vis, f"cls{c} {box.conf.item():.2f}",
                                (bx1, by1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1)

            for r in results:
                for box in r.boxes:
                    if int(box.cls.item()) != cls_id:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    u = int((x1 + x2) / 2)
                    v = int(y1 + (y2 - y1) * 0.7)
                    cv2.circle(vis, (u, v), 8, (0, 0, 255), -1)

                    d = _median_depth(depth_img, u, v) if depth_img is not None else None
                    if d is not None and (d < 0.05 or d > 5.0):
                        d = None

                    if d is not None:
                        p_cam  = np.array([(u - cx_k)*d/fx, (v - cy_k)*d/fy, d])
                        p_base = (T_base_cam @ np.append(p_cam, 1.0))[:3]
                    else:
                        # Ray-plane fallback (board Z ≈ 0.03 m for SC, 0.24 m for SFP)
                        ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
                        ray /= np.linalg.norm(ray)
                        rb = T_base_cam[:3, :3] @ ray
                        tb = T_base_cam[:3, 3]
                        board_z = 0.18 if cls_id == _CLS_SFP else 0.03
                        if abs(rb[2]) < 1e-6:
                            continue
                        t_s = (board_z - tb[2]) / rb[2]
                        if t_s < 0:
                            continue
                        p_base = tb + t_s * rb

                    detections.append({
                        "cam":      cam_name,
                        "pos_base": p_base,
                        "conf":     float(box.conf.item()),
                        "pixel":    (u, v),
                    })
                    # annotate 3D position on debug image
                    label = (f"{_CLS_NAMES.get(cls_id,'?')} "
                             f"({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f})"
                             f"{'[depth]' if d else '[ray]'}")
                    cv2.putText(vis, label, (u + 10, v),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            out = f"{debug_dir}/{cam_name}_yolo.png"
            cv2.imwrite(out, vis)
            self._log(f"Debug image → {out}")

        return detections

    # ── sanity-checked port picker ────────────────────────────────────────────

    def _pick_port(self, detections: list[dict], task: Task,
                   logo_pos: np.ndarray, logo_quat: np.ndarray
                   ) -> tuple[np.ndarray | None, float | None]:
        """
        Filter YOLO detections to the one matching the target port.

        NIC mount identification: X in logo frame (mounts 40 mm apart, ±10° yaw
        shifts X by ~3 mm → 30 mm tolerance safely separates mounts).

        Port 0 vs port 1 (SFP): NIC cards can translate ±2 cm along the rail
        (shifts absolute Y), so absolute Y sign is unreliable. Instead we collect
        ALL ports near the target mount's X, then pick by RELATIVE Y — port 0 is
        always the one with higher Y than port 1, regardless of translation/yaw.
        """
        key = _entrance_key(task)
        if key not in _ENTRANCE_XYZ:
            return None
        expected = _ENTRANCE_XYZ[key]

        T_logo_base = np.linalg.inv(_mat4(logo_pos, logo_quat))

        # Transform all detections to logo frame
        for det in detections:
            det["pos_logo"] = (T_logo_base @ np.append(det["pos_base"], 1.0))[:3]

        # Filter by X: must be within _X_TOL of expected mount X
        near_x = [d for d in detections
                  if abs(d["pos_logo"][0] - expected[0]) <= _X_TOL]
        rejected = [d for d in detections
                    if abs(d["pos_logo"][0] - expected[0]) > _X_TOL]

        if rejected:
            self._log(f"YOLO: {len(rejected)} rejected (X mismatch):")
            for d in rejected:
                self._log(f"  [{d['cam']}] logo_x={d['pos_logo'][0]:.3f} "
                          f"expected_x={expected[0]:.3f}")


        if not near_x:
            self._log("YOLO: no detections near target mount X — keeping static estimate")
            return None, None

        if task.plug_type != "sfp":
            # SC: prefer center camera, fall back to closest X
            center_dets = [d for d in near_x if d["cam"] == "center_camera"]
            if center_dets:
                best_det = min(center_dets, key=lambda d: abs(d["pos_logo"][0] - expected[0]))
            else:
                best_det = min(near_x, key=lambda d: abs(d["pos_logo"][0] - expected[0]))
            self._log(f"YOLO: SC port selected [{best_det['cam']}] "
                      f"logo={best_det['pos_logo'].round(4)}")
            return best_det["pos_base"], None

        # SFP: collect all ports near this mount X (both port_0 and port_1 may appear)
        # Deduplicate: prefer center camera, then highest conf
        deduped = {}
        for d in near_x:
            key_yz = (round(d["pos_logo"][1], 2), round(d["pos_logo"][2], 2))
            if key_yz not in deduped:
                deduped[key_yz] = d
            else:
                existing = deduped[key_yz]
                # prefer center camera; within same camera prefer higher conf
                if (d["cam"] == "center_camera" and existing["cam"] != "center_camera") or \
                   (d["cam"] == existing["cam"] and d["conf"] > existing["conf"]):
                    deduped[key_yz] = d
        pair = list(deduped.values())

        want_port_0 = task.port_name.endswith("sfp_port_0")

        nic_yaw = None   # will be set when both ports found

        if len(pair) >= 2:
            # Both ports visible: port_0 always has higher Y than port_1
            pair.sort(key=lambda d: d["pos_logo"][1], reverse=True)
            p0, p1 = pair[0], pair[1]
            best_det = p0 if want_port_0 else p1

            # NIC card yaw from the port-to-port vector in base_link XY.
            # eef_yaw = atan2(v.y, v.x) is geometrically equivalent to
            # logo_yaw + 90° in the nominal case and auto-corrects for card rotation.
            v = p0["pos_base"][:2] - p1["pos_base"][:2]
            nic_yaw = float(np.arctan2(v[1], v[0]))
            self._log(
                f"YOLO: both SFP ports found — "
                f"port_0 Y={p0['pos_logo'][1]:.3f}  "
                f"port_1 Y={p1['pos_logo'][1]:.3f}  "
                f"NIC yaw={np.degrees(nic_yaw):.1f}°  "
                f"→ picking {'port_0' if want_port_0 else 'port_1'}"
            )

            # Draw port-to-port line + yaw arrow on center camera debug image
            _debug_dir = "/tmp/logo_yolo_debug"
            img_path = f"{_debug_dir}/center_camera_yolo.png"
            img = cv2.imread(img_path)
            if img is not None and "pixel" in p0 and "pixel" in p1:
                pt0 = p0["pixel"]
                pt1 = p1["pixel"]
                # port-to-port line
                cv2.line(img, pt0, pt1, (0, 255, 255), 3)
                cv2.circle(img, pt0, 8, (0, 255, 0), -1)   # port_0 green
                cv2.circle(img, pt1, 8, (0, 0, 255), -1)   # port_1 red
                # yaw arrow from midpoint
                mid = ((pt0[0] + pt1[0]) // 2, (pt0[1] + pt1[1]) // 2)
                arrow_len = 60
                ax = int(mid[0] + arrow_len * np.cos(nic_yaw))
                ay = int(mid[1] + arrow_len * np.sin(nic_yaw))
                cv2.arrowedLine(img, mid, (ax, ay), (0, 165, 255), 3, tipLength=0.3)
                cv2.putText(img, f"NIC yaw {np.degrees(nic_yaw):.1f}deg",
                            (mid[0] - 60, mid[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imwrite(img_path, img)
        else:
            # Only one port visible — use it but warn if Y relative to expected is suspicious
            best_det = pair[0]
            dy = best_det["pos_logo"][1] - expected[1]
            self._log(
                f"YOLO: only one SFP port visible near mount X "
                f"(ΔY from nominal={dy*1000:.1f} mm) — using it, nic_yaw unknown"
            )
            if abs(dy) > 0.025:
                self._log("  WARNING: large ΔY — possible port_0/port_1 mix-up, "
                          "check NIC card translation/yaw")

        self._log(f"YOLO: selected [{best_det['cam']}] "
                  f"base={best_det['pos_base'].round(4)}  "
                  f"logo={best_det['pos_logo'].round(4)}  "
                  f"conf={best_det['conf']:.2f}")

        # Sanity-check summary image: green=accepted, red=rejected
        _debug_dir = "/tmp/logo_yolo_debug"
        for det in rejected:
            if "pixel" in det:
                img_path = f"{_debug_dir}/{det['cam']}_yolo.png"
                img = cv2.imread(img_path)
                if img is not None:
                    cv2.circle(img, det["pixel"], 14, (0, 0, 200), 3)
                    cv2.putText(img, "REJECTED", (det["pixel"][0]+16, det["pixel"][1]+16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
                    cv2.imwrite(img_path, img)
        if "pixel" in best_det:
            img_path = f"{_debug_dir}/{best_det['cam']}_yolo.png"
            img = cv2.imread(img_path)
            if img is not None:
                cv2.circle(img, best_det["pixel"], 14, (0, 200, 0), 3)
                cv2.putText(img, "SELECTED", (best_det["pixel"][0]+16, best_det["pixel"][1]-16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                cv2.imwrite(img_path, img)

        return best_det["pos_base"], nic_yaw

    def _wait_standstill(self, tol_m: float = 0.0003, window: float = 1.0,
                         min_wait: float = 1.5, timeout: float = 12.0):
        """Block until EEF moves less than tol_m over a rolling window, or timeout.

        min_wait: minimum seconds to wait before starting to check (lets the robot
        start decelerating so we don't trigger on the initial stationary state).
        """
        self._log(f"Waiting {min_wait:.1f}s before standstill check …")
        self.sleep_for(min_wait)
        t0 = time.time()
        history = []
        while time.time() - t0 < timeout:
            try:
                tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", "gripper/tcp", Time())
                t = tf.transform.translation
                pos = np.array([t.x, t.y, t.z])
            except TransformException:
                self.sleep_for(0.05)
                continue
            now = time.time()
            history.append((now, pos))
            history = [(ts, p) for ts, p in history if now - ts <= window]
            if len(history) >= 8:
                positions = np.array([p for _, p in history])
                spread = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
                if spread < tol_m:
                    self._log(f"Standstill reached (spread={spread*1000:.2f} mm)")
                    return
            self.sleep_for(0.05)
        self._log("Standstill timeout — proceeding anyway")

    # ── Policy interface ──────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:

        self._log(f"Task: module={task.target_module_name} "
                  f"port={task.port_name} plug={task.plug_type}")

        # ── 1. detect logo (robust: debug images + pursue if partial) ────────
        send_feedback("Detecting logo …")
        logo_pos, logo_quat = self._detect_logo_robust(
            get_observation, move_robot, send_feedback)
        if logo_pos is None:
            self._log("Logo not detected — aborting.")
            return False
        yaw_deg = Rotation.from_quat(logo_quat).as_euler("xyz", degrees=True)[2]
        self._log(f"Logo: pos={logo_pos.round(3)}  yaw={yaw_deg:.1f}°")

        # ── 2. static entrance estimate ───────────────────────────────────────
        entrance_pos = self._resolve_entrance(task, logo_pos, logo_quat)
        if entrance_pos is None:
            return False
        self._log(f"Static entrance: {entrance_pos.round(4)}")

        # ── 3. move to hover with target EEF yaw ─────────────────────────────
        hover_pos = entrance_pos.copy()
        hover_pos[2] += APPROACH_OFFSET_M

        send_feedback(f"Moving to hover above {task.port_name} …")
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "gripper/tcp", Time())
        except TransformException:
            self._log("Cannot read EEF pose."); return False

        t, r = tf.transform.translation, tf.transform.rotation
        current      = np.array([t.x, t.y, t.z])
        current_quat = np.array([r.x, r.y, r.z, r.w])

        # Both SC and SFP: logo yaw + 90° to align with insertion axis
        eef_yaw_deg = yaw_deg + 90.0
        _, pitch, roll = Rotation.from_quat(current_quat).as_euler("ZYX")
        target_quat = Rotation.from_euler(
            "ZYX", [np.radians(eef_yaw_deg), pitch, roll]).as_quat()
        self._log(f"EEF yaw: {yaw_deg:.1f}° + 90° = {eef_yaw_deg:.1f}°")

        from scipy.spatial.transform import Slerp
        stiffness = [400.0, 400.0, 400.0, 50.0, 50.0, 50.0]
        damping   = [80.0,  80.0,  80.0,  20.0, 20.0, 20.0]

        def _move_segment(start, end, q_start, q_end):
            n = max(1, int(np.linalg.norm(end - start) / 0.002))
            slerp = Slerp([0.0, 1.0], Rotation.concatenate([
                Rotation.from_quat(q_start), Rotation.from_quat(q_end)]))
            for i in range(1, n + 1):
                alpha = i / n
                p = start + alpha * (end - start)
                q = slerp(alpha).as_quat()
                move_robot(motion_update=MotionUpdate(
                    header=Header(frame_id="base_link",
                                  stamp=self._parent_node.get_clock().now().to_msg()),
                    pose=Pose(
                        position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                        orientation=Quaternion(x=float(q[0]), y=float(q[1]),
                                              z=float(q[2]), w=float(q[3]))),
                    target_stiffness=np.diag(stiffness).flatten().tolist(),
                    target_damping=np.diag(damping).flatten().tolist(),
                    feedforward_wrench_at_tip=Wrench(
                        force=Vector3(x=0.0, y=0.0, z=0.0),
                        torque=Vector3(x=0.0, y=0.0, z=0.0)),
                    wrench_feedback_gains_at_tip=[0.0] * 6,
                    trajectory_generation_mode=TrajectoryGenerationMode(
                        mode=TrajectoryGenerationMode.MODE_POSITION),
                ))
                self.sleep_for(0.05)

        # Phase A: move XY to hover position at current Z (cable clears cards)
        xy_waypoint = np.array([hover_pos[0], hover_pos[1], current[2]])
        send_feedback("Moving above port …")
        self._log(f"Phase A: XY move at Z={current[2]:.3f}")
        _move_segment(current, xy_waypoint, current_quat, target_quat)
        self._wait_standstill(min_wait=0.4, timeout=6.0)

        # Phase B: descend vertically to hover height
        send_feedback("Descending to hover height …")
        self._log(f"Phase B: descend to Z={hover_pos[2]:.3f}")
        _move_segment(xy_waypoint, hover_pos, target_quat, target_quat)
        self._wait_standstill(min_wait=0.4, timeout=6.0)

        self._log(f"Hovering at {hover_pos.round(4)}")
        cls_id = _CLS_SC if task.plug_type == "sc" else _CLS_SFP

        def _move_to(target_pos, target_q, label):
            n = max(1, int(np.linalg.norm(target_pos - hover_pos) / 0.001))
            eef_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "gripper/tcp", Time())
            cur = np.array([eef_tf.transform.translation.x,
                            eef_tf.transform.translation.y,
                            eef_tf.transform.translation.z])
            n = max(1, int(np.linalg.norm(target_pos - cur) / 0.001))
            for i in range(1, n + 1):
                alpha = i / n
                p = cur + alpha * (target_pos - cur)
                mu = MotionUpdate(
                    header=Header(frame_id="base_link",
                                  stamp=self._parent_node.get_clock().now().to_msg()),
                    pose=Pose(
                        position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                        orientation=Quaternion(x=float(target_q[0]), y=float(target_q[1]),
                                              z=float(target_q[2]), w=float(target_q[3]))),
                    target_stiffness=np.diag(stiffness).flatten().tolist(),
                    target_damping=np.diag(damping).flatten().tolist(),
                    feedforward_wrench_at_tip=Wrench(
                        force=Vector3(x=0.0, y=0.0, z=0.0),
                        torque=Vector3(x=0.0, y=0.0, z=0.0)),
                    wrench_feedback_gains_at_tip=[0.0] * 6,
                    trajectory_generation_mode=TrajectoryGenerationMode(
                        mode=TrajectoryGenerationMode.MODE_POSITION),
                )
                move_robot(motion_update=mu)
                self.sleep_for(0.05)
            self._log(f"{label} → {target_pos.round(4)}")

        # ── 4. YOLO #1: descend 5 cm closer, nudge for NIC cards, detect port ──
        logo_x_axis = Rotation.from_quat(logo_quat).as_matrix()[:, 0]
        send_feedback("YOLO #1 — searching port …")
        self._move_closer_z(0.08, move_robot)
        hover_pos[2] -= 0.08
        if task.plug_type != "sc":
            yolo_view = hover_pos + logo_x_axis * 0.040
            _move_to(yolo_view, target_quat, "nudge +40mm")
        self._wait_standstill()
        self._wait_standstill(min_wait=0.5, timeout=5.0)
        obs = get_observation()
        if task.plug_type == "sc":
            dets1 = self._detect_ports_yolo(obs, cls_id, cam_filter=["center_camera"])
            if not dets1:
                self._log("SC: no center detection — falling back to all cameras")
                dets1 = self._detect_ports_yolo(obs, cls_id)
        else:
            dets1 = self._detect_ports_yolo(obs, cls_id)
        self._log(f"YOLO #1: {len(dets1)} detection(s)")
        refined1, nic_yaw = self._pick_port(dets1, task, logo_pos, logo_quat)
        if refined1 is not None:
            if task.plug_type == "sc":
                # Use edge detection to refine SC port yaw
                best_det = next((d for d in dets1 if d.get("pos_base") is not None
                                 and np.allclose(d["pos_base"], refined1, atol=0.01)), None)
                if best_det is None and dets1:
                    best_det = dets1[0]
                if best_det is not None:
                    edge_yaw = self._sc_edge_yaw(obs, best_det)
                    if edge_yaw is not None:
                        _, pitch, roll = Rotation.from_quat(target_quat).as_euler("ZYX")
                        target_quat = Rotation.from_euler(
                            "ZYX", [edge_yaw, pitch, roll]).as_quat()
                        self._log(f"SC edge yaw {np.degrees(edge_yaw):.1f}° → updated quat")
            elif nic_yaw is not None:
                _, pitch, roll = Rotation.from_quat(target_quat).as_euler("ZYX")
                target_quat = Rotation.from_euler("ZYX", [nic_yaw, pitch, roll]).as_quat()
                self._log(f"NIC yaw {np.degrees(nic_yaw):.1f}° → updated quat")
            hover_pos = np.array([refined1[0], refined1[1], hover_pos[2]])
            self._log(f"YOLO #1 port XY: {hover_pos[:2].round(4)}")

        # ── 5. move directly above detected port and hover ────────────────────
        send_feedback("Hovering above port …")
        _move_to(hover_pos, target_quat, "hover above port")
        self._wait_standstill()

        self._log(f"Final hover pos: {hover_pos.round(4)}")

        # ── 6. adjust orientation for tip offset then push straight down ──────
        R_tcp_tip = self._R_TCP_TIP.get(task.plug_type)
        if R_tcp_tip is not None:
            if task.plug_type == "sc":
                # SC: keep current yaw — only apply pitch (Y) from tip transform.
                # The ~90° Z in R_tcp_tip is already baked into the hover yaw (logo+90°).
                tip_zyx = R_tcp_tip.as_euler("ZYX")
                R_pitch_only = Rotation.from_euler("X", [tip_zyx[1]])
                R_tcp_insert = Rotation.from_quat(target_quat) * R_pitch_only
                insert_quat  = R_tcp_insert.as_quat()
            else:
                # SFP: full tip-to-down calculation
                final_yaw = Rotation.from_quat(target_quat).as_euler("ZYX")[0]
                R_tip_target = (Rotation.from_euler("z", final_yaw) *
                                Rotation.from_euler("x", np.pi))
                R_tcp_insert = R_tip_target * R_tcp_tip.inv()
                insert_quat  = R_tcp_insert.as_quat()
            self._log(f"Insert orientation: tcp_rpy="
                      f"{R_tcp_insert.as_euler('xyz',degrees=True).round(1)}")

            send_feedback("Adjusting to insertion orientation …")
            self._wait_standstill(min_wait=0.3, timeout=4.0)
            n_orient = 60
            slerp_o = Slerp([0.0, 1.0], Rotation.concatenate([
                Rotation.from_quat(target_quat),
                Rotation.from_quat(insert_quat)]))
            for i in range(1, n_orient + 1):
                q = slerp_o(i / n_orient).as_quat()
                mu = MotionUpdate(
                    header=Header(frame_id="base_link",
                                  stamp=self._parent_node.get_clock().now().to_msg()),
                    pose=Pose(
                        position=Point(x=float(hover_pos[0]), y=float(hover_pos[1]),
                                       z=float(hover_pos[2])),
                        orientation=Quaternion(x=float(q[0]), y=float(q[1]),
                                              z=float(q[2]), w=float(q[3]))),
                    target_stiffness=np.diag(stiffness).flatten().tolist(),
                    target_damping=np.diag(damping).flatten().tolist(),
                    feedforward_wrench_at_tip=Wrench(
                        force=Vector3(x=0.0, y=0.0, z=0.0),
                        torque=Vector3(x=0.0, y=0.0, z=0.0)),
                    wrench_feedback_gains_at_tip=[0.0] * 6,
                    trajectory_generation_mode=TrajectoryGenerationMode(
                        mode=TrajectoryGenerationMode.MODE_POSITION),
                )
                move_robot(motion_update=mu)
                self.sleep_for(0.05)
            self._wait_standstill(min_wait=0.5, timeout=5.0)


            # Drive plug straight down with position commands — no force/wrench
            send_feedback("Inserting — pushing down …")
            from geometry_msgs.msg import Twist
            INSERT_SPEED_MPS = 0.01    # 10 mm/s
            INSERT_DEPTH_M   = 0.35 if task.plug_type == "sc" else 0.16  # SC: 35 cm, SFP: 16 cm
            DT               = 0.05
            step_m  = INSERT_SPEED_MPS * DT
            n_steps = max(1, int(INSERT_DEPTH_M / step_m))
            stiff = np.diag([200.0, 200.0, 200.0, 100.0, 100.0, 100.0]).flatten().tolist()
            damp  = np.diag([200.0, 200.0, 200.0,  30.0,  30.0,  30.0]).flatten().tolist()
            self._log(f"Position insert: {INSERT_SPEED_MPS*1000:.0f} mm/s, "
                      f"{INSERT_DEPTH_M*100:.0f} cm max")
            for i in range(1, n_steps + 1):
                pos = hover_pos + np.array([0.0, 0.0, -i * step_m])
                mu = MotionUpdate(
                    header=Header(frame_id="base_link",
                                  stamp=self._parent_node.get_clock().now().to_msg()),
                    pose=Pose(
                        position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                        orientation=Quaternion(x=float(insert_quat[0]),
                                              y=float(insert_quat[1]),
                                              z=float(insert_quat[2]),
                                              w=float(insert_quat[3]))),
                    target_stiffness=stiff,
                    target_damping=damp,
                    feedforward_wrench_at_tip=Wrench(
                        force=Vector3(x=0.0, y=0.0, z=0.0),
                        torque=Vector3(x=0.0, y=0.0, z=0.0)),
                    wrench_feedback_gains_at_tip=[0.0] * 6,
                    trajectory_generation_mode=TrajectoryGenerationMode(
                        mode=TrajectoryGenerationMode.MODE_POSITION),
                )
                move_robot(motion_update=mu)
                self.sleep_for(DT)

        self._log("Waiting 5 s for connector to stabilise …")
        send_feedback("Inserted — stabilising …")
        self.sleep_for(5.0)
        self._log("LogoYoloPolicy.insert_cable() done")
        return True
