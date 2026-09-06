#!/usr/bin/env python3
"""Move the robot to a Cartesian pose.

Usage:
  pixi run python3 aic_foundationpose/scripts/move_robot.py --x -0.501 --y -0.175 --z 0.2 --qx 0.7071 --qy 0.7071 --qz 0.0 --qw 0.0
"""

import argparse
import rclpy
from rclpy.node import Node
from aic_control_interfaces.srv import ChangeTargetMode
from aic_control_interfaces.msg import MotionUpdate, TargetMode, TrajectoryGenerationMode
from geometry_msgs.msg import Pose
from std_msgs.msg import Header


class MoveRobot(Node):
    def __init__(self, x, y, z, qx, qy, qz, qw, stiffness=85.0, damping=75.0):
        super().__init__("move_robot")
        self._x, self._y, self._z = x, y, z
        self._qx, self._qy, self._qz, self._qw = qx, qy, qz, qw
        self._stiffness = stiffness
        self._damping = damping

        self._mode_client = self.create_client(ChangeTargetMode, "/aic_controller/change_target_mode")
        self._pose_pub = self.create_publisher(MotionUpdate, "/aic_controller/pose_commands", 1)

    def run(self):
        # Step 1: switch to Cartesian pose mode
        self.get_logger().info("Switching to Cartesian pose mode ...")
        if not self._mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Service /aic_controller/change_target_mode not available")
            return

        req = ChangeTargetMode.Request()
        req.target_mode.mode = TargetMode.MODE_CARTESIAN
        future = self._mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            self.get_logger().error("Mode change service call failed")
            return
        self.get_logger().info("Mode switched.")

        # Step 2: publish pose command
        diag = [self._stiffness, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, self._stiffness, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, self._stiffness, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, self._stiffness, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, self._stiffness, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, self._stiffness]

        damp = [self._damping, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, self._damping, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, self._damping, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, self._damping, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, self._damping, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, self._damping]

        msg = MotionUpdate()
        msg.header = Header()
        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = self._x
        msg.pose.position.y = self._y
        msg.pose.position.z = self._z
        msg.pose.orientation.x = self._qx
        msg.pose.orientation.y = self._qy
        msg.pose.orientation.z = self._qz
        msg.pose.orientation.w = self._qw
        msg.target_stiffness = diag
        msg.target_damping = damp
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION

        self.get_logger().info(
            f"Sending pose: xyz=({self._x}, {self._y}, {self._z})  "
            f"quat=({self._qx}, {self._qy}, {self._qz}, {self._qw})"
        )
        self._pose_pub.publish(msg)
        # publish twice to make sure it arrives
        rclpy.spin_once(self, timeout_sec=0.1)
        self._pose_pub.publish(msg)
        self.get_logger().info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Move robot to Cartesian pose")
    parser.add_argument("--x",   type=float, required=True,  help="Position X (m)")
    parser.add_argument("--y",   type=float, required=True,  help="Position Y (m)")
    parser.add_argument("--z",   type=float, required=True,  help="Position Z (m)")
    parser.add_argument("--qx",  type=float, default=0.7071068, help="Quaternion X")
    parser.add_argument("--qy",  type=float, default=0.7071068, help="Quaternion Y")
    parser.add_argument("--qz",  type=float, default=0.0,       help="Quaternion Z")
    parser.add_argument("--qw",  type=float, default=0.0,       help="Quaternion W")
    parser.add_argument("--stiffness", type=float, default=85.0, help="Cartesian stiffness")
    parser.add_argument("--damping",   type=float, default=75.0, help="Cartesian damping")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = MoveRobot(
        args.x, args.y, args.z,
        args.qx, args.qy, args.qz, args.qw,
        args.stiffness, args.damping,
    )
    node.run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
