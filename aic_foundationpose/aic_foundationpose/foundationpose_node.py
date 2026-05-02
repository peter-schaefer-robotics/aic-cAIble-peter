"""FoundationPose ROS 2 node — 6-DoF pose estimation for the NIC Card.

Pipeline:
  /center_camera/image  ─►  Depth-Anything-V2 Metric  ─►  depth (float32, m)
                                                             │
  /center_camera/image + depth + K  ─►  FoundationPose ─►  /nic_card/pose

Initialisation:
  Call  ros2 service call /foundationpose/initialize std_srvs/srv/Trigger
  Optionally first publish an ROI on /foundationpose/roi (sensor_msgs/RegionOfInterest)
  to constrain the object mask to a known image region.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image, RegionOfInterest
from std_srvs.srv import Trigger
import tf2_ros


class FoundationPoseNode(Node):

    def __init__(self) -> None:
        super().__init__("foundationpose_node")

        # ── parameters ──────────────────────────────────────────────────────
        self.declare_parameter("camera_name", "center_camera")
        self.declare_parameter("mesh_path", "")
        self.declare_parameter("foundationpose_dir", "")
        self.declare_parameter(
            "depth_model",
            "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        )
        self.declare_parameter("refine_iterations", 2)
        self.declare_parameter("register_iterations", 5)
        # NIC card expected depth range from camera (metres)
        self.declare_parameter("object_depth_min", 0.05)
        self.declare_parameter("object_depth_max", 2.0)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("object_frame", "nic_card")
        self.declare_parameter("debug_vis", True)

        cam = self.get_parameter("camera_name").value

        # ── state ────────────────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._rgb: np.ndarray | None = None
        self._rgb_header = None
        self._K: np.ndarray | None = None
        self._roi: tuple[int, int, int, int] | None = None  # x1,y1,x2,y2
        self._initialized = False
        self._depth_est = None  # loaded lazily
        self._fp = None          # FoundationPose instance, loaded lazily
        self._models_ready = False

        # ── mesh ─────────────────────────────────────────────────────────────
        mesh_path = self.get_parameter("mesh_path").value
        self.get_logger().info(f"Loading mesh: {mesh_path}")
        self._mesh = trimesh.load(mesh_path, force="mesh")
        # FoundationPose expects the mesh centred at origin in metres
        self._mesh.apply_translation(-self._mesh.centroid)
        self.get_logger().info(
            f"Mesh loaded: {len(self._mesh.vertices)} vertices, "
            f"bounds {self._mesh.bounds.tolist()}"
        )

        # ── subscriptions ────────────────────────────────────────────────────
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, f"/{cam}/image", self._rgb_cb, qos)
        self.create_subscription(CameraInfo, f"/{cam}/camera_info", self._info_cb, qos)
        self.create_subscription(
            RegionOfInterest, "/foundationpose/roi", self._roi_cb, 10
        )

        # ── publishers ───────────────────────────────────────────────────────
        self._pose_pub = self.create_publisher(PoseStamped, "/nic_card/pose", 10)
        self._debug_pub = self.create_publisher(Image, "/foundationpose/debug_image", 1)

        # ── services ─────────────────────────────────────────────────────────
        self.create_service(Trigger, "/foundationpose/initialize", self._init_srv)

        # ── TF broadcaster ───────────────────────────────────────────────────
        if self.get_parameter("publish_tf").value:
            self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info(
            "FoundationPose node ready.\n"
            "  Optionally publish ROI: ros2 topic pub /foundationpose/roi "
            "sensor_msgs/RegionOfInterest '{x_offset: X, y_offset: Y, width: W, height: H}'\n"
            "  Then initialise:        ros2 service call /foundationpose/initialize "
            "std_srvs/srv/Trigger"
        )

    # ── model loading ────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        if self._models_ready:
            return

        depth_model = self.get_parameter("depth_model").value
        fp_dir = self.get_parameter("foundationpose_dir").value

        self.get_logger().info("Loading Depth-Anything-V2 Metric …")
        from aic_foundationpose.depth_estimator import DepthAnythingEstimator
        self._depth_est = DepthAnythingEstimator(depth_model)

        self.get_logger().info(f"Loading FoundationPose from {fp_dir} …")
        sys.path.insert(0, fp_dir)
        from estimator import FoundationPose          # noqa: PLC0415
        from model_runner import ScorePredictor, PoseRefinePredictor  # noqa: PLC0415
        import nvdiffrast.torch as dr                 # noqa: PLC0415

        weights = Path(fp_dir) / "weights"
        scorer = ScorePredictor()
        scorer.load_weights(str(weights / "scorer.pth"))
        refiner = PoseRefinePredictor()
        refiner.load_weights(str(weights / "refiner.pth"))
        glctx = dr.RasterizeCudaContext()

        self._fp = FoundationPose(
            model_pts=self._mesh.vertices.copy(),
            model_normals=self._mesh.vertex_normals.copy(),
            mesh=self._mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
        )
        self._models_ready = True
        self.get_logger().info("All models loaded.")

    # ── callbacks ────────────────────────────────────────────────────────────

    def _rgb_cb(self, msg: Image) -> None:
        with self._lock:
            self._rgb = self._bridge.imgmsg_to_cv2(msg, "rgb8")
            self._rgb_header = msg.header
        if self._initialized:
            self._track()

    def _info_cb(self, msg: CameraInfo) -> None:
        if self._K is None:
            self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"Camera intrinsics received:\n{self._K}")

    def _roi_cb(self, msg: RegionOfInterest) -> None:
        with self._lock:
            self._roi = (
                int(msg.x_offset),
                int(msg.y_offset),
                int(msg.x_offset + msg.width),
                int(msg.y_offset + msg.height),
            )
        self.get_logger().info(f"ROI set to {self._roi}")

    # ── initialisation service ───────────────────────────────────────────────

    def _init_srv(self, request, response):
        if self._K is None:
            response.success = False
            response.message = "Camera intrinsics not received yet."
            return response
        with self._lock:
            rgb = self._rgb.copy() if self._rgb is not None else None
            roi = self._roi
        if rgb is None:
            response.success = False
            response.message = "No image received yet."
            return response

        try:
            self._load_models()
            self.get_logger().info("Estimating depth for registration …")
            depth = self._depth_est.infer(rgb)
            mask = self._make_mask(depth, roi)

            if mask.sum() < 50:
                response.success = False
                response.message = (
                    f"Object mask too small ({mask.sum()} px). "
                    "Publish an ROI on /foundationpose/roi to constrain the search region."
                )
                self.get_logger().warn(response.message)
                return response

            self.get_logger().info(
                f"Registering with mask ({mask.sum()} px) …"
            )
            pose = self._fp.register(
                K=self._K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=self.get_parameter("register_iterations").value,
            )
            self._initialized = True
            response.success = True
            response.message = f"Initialised. t={pose[:3, 3].tolist()}"
            self.get_logger().info(response.message)

        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f"Initialisation failed: {exc}")

        return response

    # ── mask generation ──────────────────────────────────────────────────────

    def _make_mask(self, depth: np.ndarray, roi: tuple | None) -> np.ndarray:
        """Binary object mask from depth thresholding + optional ROI crop."""
        d_min = self.get_parameter("object_depth_min").value
        d_max = self.get_parameter("object_depth_max").value
        valid = (depth > d_min) & (depth < d_max)

        if roi is not None:
            x1, y1, x2, y2 = roi
            roi_mask = np.zeros_like(valid, dtype=bool)
            roi_mask[y1:y2, x1:x2] = True
            valid = valid & roi_mask

        # Keep only the largest connected component (most likely the object)
        n_labels, labels = cv2.connectedComponents(valid.astype(np.uint8))
        if n_labels <= 1:
            return valid
        sizes = [(labels == i).sum() for i in range(1, n_labels)]
        best = int(np.argmax(sizes)) + 1
        mask = (labels == best).astype(np.uint8)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask.astype(bool)

    # ── tracking ─────────────────────────────────────────────────────────────

    def _track(self) -> None:
        with self._lock:
            rgb = self._rgb.copy()
            header = self._rgb_header

        depth = self._depth_est.infer(rgb)

        try:
            pose = self._fp.track_one(
                rgb=rgb,
                depth=depth,
                K=self._K,
                iteration=self.get_parameter("refine_iterations").value,
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Track failed: {exc}")
            return

        self._publish_pose(pose, header)

        if self.get_parameter("debug_vis").value:
            self._publish_debug(rgb, pose)

    # ── publishing ───────────────────────────────────────────────────────────

    def _publish_pose(self, pose: np.ndarray, header) -> None:
        R = pose[:3, :3]
        t = pose[:3, 3]
        quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]

        msg = PoseStamped()
        msg.header = header
        msg.pose.position.x = float(t[0])
        msg.pose.position.y = float(t[1])
        msg.pose.position.z = float(t[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self._pose_pub.publish(msg)

        if self.get_parameter("publish_tf").value:
            tf_msg = TransformStamped()
            tf_msg.header = header
            tf_msg.child_frame_id = self.get_parameter("object_frame").value
            tf_msg.transform.translation.x = float(t[0])
            tf_msg.transform.translation.y = float(t[1])
            tf_msg.transform.translation.z = float(t[2])
            tf_msg.transform.rotation.x = float(quat[0])
            tf_msg.transform.rotation.y = float(quat[1])
            tf_msg.transform.rotation.z = float(quat[2])
            tf_msg.transform.rotation.w = float(quat[3])
            self._tf_broadcaster.sendTransform(tf_msg)

    def _publish_debug(self, rgb: np.ndarray, pose: np.ndarray) -> None:
        """Project mesh bounding-box corners onto the image as a debug overlay."""
        vis = rgb.copy()
        corners = np.array([
            [self._mesh.bounds[i, 0], self._mesh.bounds[j, 1], self._mesh.bounds[k, 2], 1.0]
            for i in range(2) for j in range(2) for k in range(2)
        ])  # 8 × 4
        pts_cam = (pose @ corners.T).T[:, :3]
        proj = (self._K @ pts_cam.T).T
        pts_img = (proj[:, :2] / proj[:, 2:3]).astype(int)

        for p in pts_img:
            if 0 <= p[0] < vis.shape[1] and 0 <= p[1] < vis.shape[0]:
                cv2.circle(vis, tuple(p), 5, (0, 255, 0), -1)

        # Draw rough bounding box edges (pairs of corners that share two axes)
        edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        h, w = vis.shape[:2]
        for a, b in edges:
            pa, pb = pts_img[a], pts_img[b]
            if (0 <= pa[0] < w and 0 <= pa[1] < h and
                    0 <= pb[0] < w and 0 <= pb[1] < h):
                cv2.line(vis, tuple(pa), tuple(pb), (0, 255, 0), 2)

        self._debug_pub.publish(self._bridge.cv2_to_imgmsg(vis, "rgb8"))


def main() -> None:
    rclpy.init()
    node = FoundationPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
