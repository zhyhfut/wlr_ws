import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('wlr_controller')
    params_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    balance_node = Node(
        package='wlr_controller',
        executable='balance_node',
        name='balance_controller',
        parameters=[params_file],
        output='screen',
    )

    return LaunchDescription([
        balance_node,
    ])
