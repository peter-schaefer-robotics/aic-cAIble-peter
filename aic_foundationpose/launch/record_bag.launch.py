"""Record a ROS 2 bag with true RGBD data from all three Basler cameras.

Usage (Gazebo must already be running):
  ros2 launch aic_foundationpose record_bag.launch.py

Optional:
  ros2 launch aic_foundationpose record_bag.launch.py output_dir:=/tmp/my_bag
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


TOPICS = [
    # RGB images
    "/left_camera/image",
    "/center_camera/image",
    "/right_camera/image",
    # RGB camera info (intrinsics + distortion)
    "/left_camera/camera_info",
    "/center_camera/camera_info",
    "/right_camera/camera_info",
    # Depth images (float32, metres, same intrinsics as RGB)
    "/left_camera/depth",
    "/center_camera/depth",
    "/right_camera/depth",
    # Depth camera info
    "/left_camera/depth/camera_info",
    "/center_camera/depth/camera_info",
    "/right_camera/depth/camera_info",
    # Robot state
    "/joint_states",
    "/fts_broadcaster/wrench",
    # Transforms
    "/tf",
    "/tf_static",
    # Clock (needed for playback at correct speed)
    "/clock",
]


def generate_launch_description() -> LaunchDescription:
    output_arg = DeclareLaunchArgument(
        "output_dir",
        default_value="rgbd_bag",
        description="Directory name for the recorded bag",
    )

    record = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record",
            "--output", LaunchConfiguration("output_dir"),
            "--compression-mode", "file",
            "--compression-format", "zstd",
            "--use-sim-time",
        ] + TOPICS,
        output="screen",
    )

    return LaunchDescription([output_arg, record])
