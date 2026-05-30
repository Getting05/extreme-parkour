"""Launch file for MuJoCo simulation + deploy_node (extreme-parkour)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_dir = get_package_share_directory('deploy_cpp')
    default_cfg = os.path.join(pkg_dir, 'config', 'robots', 'mybot_v3_parkour.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('robot_config_file', default_value=default_cfg,
                              description='Path to robot runtime yaml config'),
        DeclareLaunchArgument('sim_pingpong_mode', default_value='false',
                              description='Enable state-triggered ping-pong mode'),

        # MuJoCo sim node (Python)
        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(pkg_dir, 'sim', 'mujoco_sim_node.py'),
                '--robot-config', LaunchConfiguration('robot_config_file'),
            ],
            output='screen',
        ),

        # Deploy node (C++) in sim mode
        Node(
            package='deploy_cpp',
            executable='deploy_node',
            name='deploy_node',
            output='screen',
            parameters=[{
                'robot_config_file': LaunchConfiguration('robot_config_file'),
                'sim_mode': True,
                'sim_pingpong_mode': LaunchConfiguration('sim_pingpong_mode'),
            }],
        ),
    ])
