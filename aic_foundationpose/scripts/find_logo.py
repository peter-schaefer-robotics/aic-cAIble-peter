#!/usr/bin/env python3
"""Detect the magenta Intrinsic logo and compute its 6-DoF pose in base_link.

Position:  multi-view triangulation across 3 cameras (no depth needed)
Rotation:  solvePnP using known logo physical dimensions

Usage:
  pixi run python3 aic_foundationpose/scripts/find_logo.py \
      --width 0.06 --height 0.03     # logo physical size in metres

Saves logo_detection.png and logo_pose.json.
"""

import argparse
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


CAMERAS = ["left_camera", "center_camera", "right_camera"]

def _ang_dist(a, b):
    """Absolute angular distance between two angles in radians, in [0, π]."""
    d = abs(a - b) % (2 * np.pi)
    return d if d <= np.pi else 2 * np.pi - d
BASE_FRAME = "base_link"

# HSV range for magenta (OpenCV H: 0-180)
MAGENTA_LOW  = np.array([140, 80,  60])
MAGENTA_HIGH = np.array([170, 255, 255])


def ros_image_to_numpy(msg):
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = data.reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if msg.encoding == "bgr8" else img.copy()
    return data.reshape(msg.height, msg.width)


def tf_to_matrix(d):
    t = d["translation"]; r = d["rotation"]
    T = np.eye(4)
    T[:3, 3]  = [t["x"], t["y"], t["z"]]
    T[:3, :3] = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]]).as_matrix()
    return T


def order_corners(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2)
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()],  pts[d.argmin()],
                     pts[s.argmax()],  pts[d.argmax()]], dtype=np.float32)


class LogoFinder(Node):
    def __init__(self, logo_w, logo_h):
        super().__init__("logo_finder")
        self._logo_w = logo_w   # physical width  in metres
        self._logo_h = logo_h   # physical height in metres
        self._rgb  = {}
        self._info = {}
        self._done = False

        qos = qos_profile_sensor_data
        for cam in CAMERAS:
            self.create_subscription(Image,      f"/{cam}/image",
                lambda m, c=cam: self._rgb_cb(m, c), qos)
            self.create_subscription(CameraInfo, f"/{cam}/camera_info",
                lambda m, c=cam: self._info_cb(m, c), qos)

        self._tf_buf = tf2_ros.Buffer()
        self._tf_lst = tf2_ros.TransformListener(self._tf_buf, self)
        self.create_timer(0.5, self._tick)
        self.get_logger().info("Waiting for camera data…")

    def _rgb_cb(self, m, c):  self._rgb[c]  = ros_image_to_numpy(m)
    def _info_cb(self, m, c): self._info[c] = {
        "K":        np.array(m.k, dtype=np.float64).reshape(3, 3),
        "frame_id": m.header.frame_id,
    }

    def _lookup_tf(self, child, parent=BASE_FRAME):
        try:
            tf = self._tf_buf.lookup_transform(parent, child, rclpy.time.Time())
            t, r = tf.transform.translation, tf.transform.rotation
            return {"translation": {"x": t.x, "y": t.y, "z": t.z},
                    "rotation":    {"x": r.x, "y": r.y, "z": r.z, "w": r.w}}
        except Exception:
            return None

    def _proj_matrix(self, cam):
        """Build 3×4 projection matrix P = K @ [R | t] (world→image)."""
        info   = self._info[cam]
        K      = info["K"]
        cam_tf = self._lookup_tf(info["frame_id"])
        if cam_tf is None:
            return None, None
        T_base_cam = tf_to_matrix(cam_tf)
        T_cam_base = np.linalg.inv(T_base_cam)
        R = T_cam_base[:3, :3]
        t = T_cam_base[:3, 3:]
        P = K @ np.hstack([R, t])          # 3×4
        return P, T_base_cam

    def _detect(self, cam):
        """Detect logo robustly against partial occlusion.

        Strategy:
        - Keep ALL magenta fragments above a minimum area (a cable may split the logo)
        - Merge their pixels into one set for centroid + bounding rect
        - Weight confidence by fraction of expected logo area that is visible
        """
        rgb = self._rgb[cam]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, MAGENTA_LOW, MAGENTA_HIGH)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, bgr, 0.0

        # Keep all contours larger than 1% of the largest — merges fragments
        max_area   = max(cv2.contourArea(c) for c in contours)
        logo_parts = [c for c in contours if cv2.contourArea(c) > 0.01 * max_area]

        # Merge all fragments into one point cloud for a stable bounding rect
        merged = np.vstack(logo_parts)
        M      = cv2.moments(merged.reshape(-1, 1, 2))
        if M["m00"] == 0:
            return None, None, bgr, 0.0

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid  = np.array([cx, cy], dtype=np.float64)
        # Confidence = total visible area / largest single contour area
        # (1.0 = fully visible, <1.0 = fragmented / occluded)
        total_area   = sum(cv2.contourArea(c) for c in logo_parts)
        confidence   = min(1.0, total_area / max(max_area, 1))

        # ── X axis: closest rect edge (= thick side) ─────────────────────
        rect     = cv2.minAreaRect(merged.reshape(-1, 1, 2))
        corners4 = cv2.boxPoints(rect).astype(np.float64)

        slit_dir = None;  y_dir = None
        closest_edge = None;  closest_edge_mid = None
        min_dist = float("inf")

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
                closest_edge = (p1, p2)
                closest_edge_mid = (p1 + p2) / 2.0

        if closest_edge_mid is not None:
            raw = closest_edge_mid - centroid
            if np.linalg.norm(raw) > 1:
                slit_dir = raw / np.linalg.norm(raw)

        # ── Y axis: slit position on thick side → top of logo ─────────────
        # Convexity defect = the slit notch
        hull_idx = cv2.convexHull(merged, returnPoints=False)
        hull_idx = np.sort(hull_idx, axis=0)
        defects  = cv2.convexityDefects(merged, hull_idx)
        slit_tip = None

        if defects is not None and closest_edge is not None:
            depths  = defects[:, 0, 3]
            d_idx   = defects[np.argmax(depths), 0]
            slit_tip = merged[d_idx[2]][0].astype(np.float64)
            p1, p2  = closest_edge
            eu      = (p2 - p1) / np.linalg.norm(p2 - p1)
            t       = np.dot(slit_tip - p1, eu)   # projection along edge
            # The slit is in the LOWER HALF → top corner is the FARTHER one
            slit_proj = p1 + t * eu
            d_p1, d_p2 = t, np.linalg.norm(p2 - p1) - t
            top_corner = p1 if d_p1 > d_p2 else p2
            y_raw = top_corner - slit_proj
            if np.linalg.norm(y_raw) > 1:
                y_dir = y_raw / np.linalg.norm(y_raw)

        # ── order 4 rect corners: +X toward closest edge, +Y perpendicular ─
        corners = corners4.copy()   # already computed above

        if slit_dir is not None:
            perp = np.array([-slit_dir[1], slit_dir[0]])
            # Project each corner onto slit (+X) and perp (+Y)
            rel      = corners - centroid
            x_proj   = rel @ slit_dir
            y_proj   = rel @ perp
            # Desired order for PnP: TL=(-x,-y), TR=(+x,-y), BR=(+x,+y), BL=(-x,+y)
            order    = sorted(range(4),
                              key=lambda i: (x_proj[i], y_proj[i]))
            # order = [far-bottom, far-top, near-bottom, near-top]
            #   TL = far-top   (order[1])
            #   TR = near-top  (order[3])
            #   BR = near-bot  (order[2])
            #   BL = far-bot   (order[0])
            corners  = corners[[order[1], order[3], order[2], order[0]]]
        else:
            corners = order_corners(corners)

        # ── debug overlay ─────────────────────────────────────────────────
        vis = bgr.copy()
        cv2.drawContours(vis, logo_parts, -1, (255, 0, 255), 2)
        # Confidence badge
        badge_col = (0, int(255*confidence), int(255*(1-confidence)))
        cv2.putText(vis, f"conf={confidence:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_col, 2)

        if closest_edge is not None:
            # Highlight the closest edge (= reference for X axis) in cyan
            p1c, p2c = closest_edge[0].astype(int), closest_edge[1].astype(int)
            cv2.line(vis, tuple(p1c), tuple(p2c), (255, 255, 0), 4)

        if slit_dir is not None:
            arrow_len = int(min_dist * 0.9)
            tip = (centroid + slit_dir * arrow_len).astype(int)
            cv2.arrowedLine(vis, tuple(centroid.astype(int)), tuple(tip),
                            (0, 0, 255), 3, tipLength=0.25)
            cv2.putText(vis, "+X", tuple(tip + np.array([5, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if y_dir is not None:
            arrow_len_y = int(min_dist * 0.9)
            y_tip = (centroid + y_dir * arrow_len_y).astype(int)
            cv2.arrowedLine(vis, tuple(centroid.astype(int)), tuple(y_tip),
                            (0, 200, 0), 3, tipLength=0.25)
            cv2.putText(vis, "+Y", tuple(y_tip + np.array([5, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        elif slit_dir is not None:
            # fallback: Y perpendicular to X
            perp = np.array([-slit_dir[1], slit_dir[0]])
            y_tip = (centroid + perp * int(min_dist * 0.9)).astype(int)
            cv2.arrowedLine(vis, tuple(centroid.astype(int)), tuple(y_tip),
                            (0, 180, 0), 2, tipLength=0.25)

        if slit_tip is not None:
            cv2.circle(vis, tuple(slit_tip.astype(int)), 7, (0, 128, 255), -1)
            cv2.putText(vis, "slit", tuple(slit_tip.astype(int) + np.array([5, 0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

        # Centroid crosshair
        ci = centroid.astype(int)
        cv2.circle(vis, tuple(ci), 10, (0, 255, 255), -1)
        cv2.line(vis, (ci[0]-15, ci[1]), (ci[0]+15, ci[1]), (0,0,0), 2)
        cv2.line(vis, (ci[0], ci[1]-15), (ci[0], ci[1]+15), (0,0,0), 2)

        return centroid, corners.astype(np.float32), vis, confidence, slit_dir, y_dir

    def _triangulate_point(self, uv_per_cam, conf_map=None):
        """Triangulate a 3-D point (base_link) from ≥2 cameras.
        Pairs weighted by product of confidence scores."""
        if len(uv_per_cam) < 2:
            return None
        cams = list(uv_per_cam.keys())
        pts_weighted, weights = [], []
        for i in range(len(cams)):
            for j in range(i+1, len(cams)):
                c1, c2 = cams[i], cams[j]
                P1, _ = self._proj_matrix(c1)
                P2, _ = self._proj_matrix(c2)
                if P1 is None or P2 is None:
                    continue
                p1 = np.array([[uv_per_cam[c1][0]], [uv_per_cam[c1][1]]], dtype=np.float32)
                p2 = np.array([[uv_per_cam[c2][0]], [uv_per_cam[c2][1]]], dtype=np.float32)
                pt4 = cv2.triangulatePoints(P1, P2, p1, p2)
                pt  = (pt4[:3] / pt4[3]).ravel()
                w   = (conf_map[c1] * conf_map[c2]) if conf_map else 1.0
                pts_weighted.append(pt)
                weights.append(w)
        if not pts_weighted:
            return None
        weights = np.array(weights)
        weights /= weights.sum()
        return np.average(pts_weighted, axis=0, weights=weights)

    def _solve_pose(self, cam, corners_2d, slit_dir_2d, y_dir_2d=None):
        """Compute logo pose using the flat-plane prior (roll=pitch=0).

        Yaw is derived by back-projecting the detected +X image direction into
        base_link and projecting onto the XY plane — no corner matching needed.
        rvec/tvec are still computed via solvePnP for axis drawing only.
        """
        info = self._info[cam]
        K    = info["K"]
        cam_tf = self._lookup_tf(info["frame_id"])
        if cam_tf is None:
            return None, None, None, None
        T_base_cam = tf_to_matrix(cam_tf)
        R_base_cam = T_base_cam[:3, :3]

        # ── yaw from 2D X and Y directions (flat-plane prior) ────────────
        K_inv = np.linalg.inv(K)

        def backproject(d2):
            """2D image direction → unit vector in base_link XY plane."""
            ray = K_inv @ np.array([d2[0], d2[1], 0.0])
            r   = R_base_cam @ ray
            r[2] = 0.0
            n = np.linalg.norm(r)
            return r / n if n > 1e-6 else None

        x_base = backproject(slit_dir_2d) if slit_dir_2d is not None else None
        y_base = backproject(y_dir_2d)    if y_dir_2d    is not None else None

        if x_base is not None and y_base is not None:
            # Build 2×2 rotation matrix from both directions and orthogonalise
            R2 = np.column_stack([x_base[:2], y_base[:2]])
            U, _, Vt = np.linalg.svd(R2)
            R2_clean = U @ Vt
            yaw = np.arctan2(R2_clean[1, 0], R2_clean[0, 0])
        elif x_base is not None:
            yaw = np.arctan2(x_base[1], x_base[0])
        else:
            yaw = 0.0

        quat = Rotation.from_euler("z", yaw).as_quat()

        # ── solvePnP only for rvec/tvec used in drawFrameAxes ────────────
        w, h    = self._logo_w / 2, self._logo_h / 2
        obj_pts = np.array([[-w, -h, 0], [w, -h, 0],
                             [w,  h, 0], [-w,  h, 0]], dtype=np.float64)
        img_pts = corners_2d.astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, None)
        if not ok:
            # Fall back: synthesise rvec/tvec from known yaw + triangulated position
            rvec = Rotation.from_euler("z", yaw).as_rotvec().reshape(3, 1)
            tvec = np.zeros((3, 1))

        return quat, rvec, tvec, K

    def _tick(self):
        if self._done:
            return
        if len(self._rgb) < 3 or len(self._info) < 3:
            return

        detections = {}
        debug_imgs  = []
        for cam in CAMERAS:
            centroid, corners, vis, conf, slit_dir, y_dir = self._detect(cam)
            debug_imgs.append(vis)
            if centroid is not None:
                detections[cam] = (centroid, corners, conf, slit_dir, y_dir)
                self.get_logger().info(
                    f"[{cam}] logo detected at pixel {centroid}  conf={conf:.2f}")
            else:
                self.get_logger().warn(f"[{cam}] no magenta logo detected")

        if len(detections) < 2:
            self.get_logger().error(
                f"Need ≥2 cameras, got {len(detections)}. "
                "Check --width/--height and HSV range.")
            self._done = True
            self._save_debug(debug_imgs)
            return

        # ── position via multi-view triangulation ─────────────────────────
        centroid_uvs = {c: det[0] for c, det in detections.items()}
        conf_map     = {c: det[2] for c, det in detections.items()}
        position = self._triangulate_point(centroid_uvs, conf_map)

        # ── rotation via solvePnP on best camera ─────────────────────────────
        # Pick camera with highest confidence (least occluded) for PnP
        best_cam = max(detections, key=lambda c: detections[c][2])  # highest confidence

        # ── rotation: per-camera yaw, then reject angular outliers ──────────
        axis_len   = self._logo_w * 0.8
        cam_yaws   = {}   # cam → yaw (radians)
        for cam, (_, corners, conf, slit_dir, y_dir) in detections.items():
            q, _, _, _ = self._solve_pose(cam, corners, slit_dir, y_dir)
            if q is not None:
                cam_yaws[cam] = (Rotation.from_quat(q).as_euler("xyz")[2], conf)

        # Circular median for outlier rejection
        if cam_yaws:
            yaw_vals = [v[0] for v in cam_yaws.values()]
            # Circular median: direction with smallest sum of angular distances
            best_ref = min(yaw_vals,
                           key=lambda r: sum(_ang_dist(r, y) for y in yaw_vals))
            OUTLIER_DEG = 45.0

        yaw_sins, yaw_coss, yaw_weights = [], [], []
        for cam, (yaw, conf) in cam_yaws.items():
            dev = np.degrees(_ang_dist(yaw, best_ref))
            w   = conf if dev < OUTLIER_DEG else 0.0
            yaw_sins.append(np.sin(yaw) * w)
            yaw_coss.append(np.cos(yaw) * w)
            yaw_weights.append(w)
            status = "OK" if dev < OUTLIER_DEG else f"REJECTED (dev={dev:.0f}°)"
            self.get_logger().info(
                    f"  [{cam}] yaw={np.degrees(yaw):.1f}°  conf={conf:.2f}  {status}")

        if yaw_weights:
            final_yaw = np.arctan2(sum(yaw_sins), sum(yaw_coss))
            quaternion = Rotation.from_euler("z", final_yaw).as_quat()
        else:
            quaternion = None

        # rvec/tvec for axis drawing — from best camera
        _, rvec, tvec, K_best = self._solve_pose(
            best_cam, detections[best_cam][1],
            detections[best_cam][3], detections[best_cam][4])

        # ── report ────────────────────────────────────────────────────────
        rpy = None
        result = {
            "frame":    "base_link",
            "position": {"x": float(position[0]),
                          "y": float(position[1]),
                          "z": float(position[2])},
        }
        if quaternion is not None:
            rpy = Rotation.from_quat(quaternion).as_euler("xyz", degrees=True)
            result["orientation_quat"] = {"x": float(quaternion[0]),
                                           "y": float(quaternion[1]),
                                           "z": float(quaternion[2]),
                                           "w": float(quaternion[3])}
            result["orientation_rpy_deg"] = {"roll":  float(rpy[0]),
                                              "pitch": float(rpy[1]),
                                              "yaw":   float(rpy[2])}

        self.get_logger().info(
            f"\n{'='*60}\n"
            f"LOGO POSE in base_link  (triangulated from {len(detections)} cameras)\n"
            f"  position : x={position[0]:.4f}  y={position[1]:.4f}  z={position[2]:.4f} m\n"
            + (f"  rotation : roll={rpy[0]:.1f}°  pitch={rpy[1]:.1f}°  yaw={rpy[2]:.1f}°\n"
               if rpy is not None else "  rotation : PnP failed\n")
            + f"{'='*60}"
        )

        with open("logo_pose.json", "w") as f:
            json.dump(result, f, indent=2)
        self.get_logger().info("Saved logo_pose.json")

        if rvec is not None:
            # Build logo pose in base_link from best camera result
            R_logo_cam, _  = cv2.Rodrigues(rvec)
            best_tf        = self._lookup_tf(self._info[best_cam]["frame_id"])
            T_base_bestcam = tf_to_matrix(best_tf)
            # T_base_logo: origin at logo centroid, axes aligned with logo frame
            T_cam_base   = np.linalg.inv(T_base_bestcam)
            R_cam_logo   = R_logo_cam          # solvePnP gives cam←logo rotation
            t_cam_logo   = tvec.ravel()
            # Full logo→base_link transform
            T_logo_cam   = np.eye(4)
            T_logo_cam[:3, :3] = R_cam_logo.T
            T_logo_cam[:3,  3] = -R_cam_logo.T @ t_cam_logo
            T_base_logo  = T_base_bestcam @ T_logo_cam

            for cam in CAMERAS:
                idx = CAMERAS.index(cam)
                info = self._info.get(cam)
                if info is None:
                    continue
                cam_tf = self._lookup_tf(info["frame_id"])
                if cam_tf is None:
                    continue
                T_base_cam = tf_to_matrix(cam_tf)
                T_cam_base = np.linalg.inv(T_base_cam)

                # Logo pose in this camera's frame
                T_cam_logo   = T_cam_base @ T_base_logo
                R_cl         = T_cam_logo[:3, :3]
                t_cl         = T_cam_logo[:3,  3]
                rv_cam, _    = cv2.Rodrigues(R_cl)
                tv_cam       = t_cl.reshape(3, 1)
                K_cam        = info["K"]

                cv2.drawFrameAxes(debug_imgs[idx], K_cam, None,
                                  rv_cam, tv_cam, axis_len, thickness=3)

                def proj(pt, _rv=rv_cam, _tv=tv_cam, _K=K_cam):
                    p, _ = cv2.projectPoints(np.array([pt], dtype=np.float64),
                                             _rv, _tv, _K, None)
                    return tuple(p[0, 0].astype(int))

                cv2.putText(debug_imgs[idx], "X", proj([axis_len, 0, 0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(debug_imgs[idx], "Y", proj([0, axis_len, 0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(debug_imgs[idx], "Z", proj([0, 0, axis_len]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                cv2.putText(debug_imgs[idx], cam.replace("_camera", ""),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Label the centroid dot
        for cam, (centroid, _, _conf, _slit, _ydir) in detections.items():
            idx = CAMERAS.index(cam)
            u, v = int(centroid[0]), int(centroid[1])
            cv2.circle(debug_imgs[idx], (u, v), 8, (0, 255, 0), -1)
            cv2.putText(debug_imgs[idx], "centroid", (u + 10, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self._save_debug(debug_imgs)
        self._done = True

    def _save_debug(self, imgs):
        if not imgs:
            return
        h  = min(i.shape[0] for i in imgs)
        sc = [cv2.resize(i, (int(i.shape[1]*h/i.shape[0]), h)) for i in imgs]
        cv2.imwrite("logo_detection.png", np.hstack(sc))
        self.get_logger().info("Saved logo_detection.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width",  type=float, default=0.06,
                        help="Logo physical width in metres")
    parser.add_argument("--height", type=float, default=0.03,
                        help="Logo physical height in metres")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = LogoFinder(args.width, args.height)
    while not node._done:
        rclpy.spin_once(node, timeout_sec=0.1)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
