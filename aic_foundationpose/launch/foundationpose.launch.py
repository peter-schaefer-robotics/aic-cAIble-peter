"""Launch FoundationPose pose estimation node for the NIC Card.

Usage (standalone, Gazebo already running):
  ros2 launch aic_foundationpose foundationpose.launch.py

Optional overrides:
  ros2 launch aic_foundationpose foundationpose.launch.py \
      camera_name:=left_camera \
      debug_vis:=false
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("aic_foundationpose")
    default_params = str(Path(pkg_share) / "config" / "params.yaml")

    camera_arg = DeclareLaunchArgument(
        "camera_name",
        default_value="center_camera",
        description="Basler camera to use for pose estimation",
    )
    debug_arg = DeclareLaunchArgument(
        "debug_vis",
        default_value="true",
        description="Publish debug overlay image",
    )
    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to params YAML",
    )

    node = Node(
        package="aic_foundationpose",
        executable="foundationpose_node",
        name="foundationpose_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "camera_name": LaunchConfiguration("camera_name"),
                "debug_vis": LaunchConfiguration("debug_vis"),
            },
        ],
    )

    return LaunchDescription([camera_arg, debug_arg, params_arg, node])
