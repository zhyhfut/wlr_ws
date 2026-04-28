import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_description = get_package_share_directory('wlr_description')
    pkg_gazebo = get_package_share_directory('wlr_gazebo')

    xacro_file = os.path.join(pkg_description, 'urdf', 'wlr_robot.urdf.xacro')
    world_file = os.path.join(pkg_gazebo, 'worlds', 'empty.sdf')

    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen',
    )

    # Gazebo starts RUNNING so controller_manager / spawners work normally
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': ['-r -s -v 4 ', world_file],
        }.items(),
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'wlr_robot',
            '-z', '0.26',
        ],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU',
        ],
        output='screen',
    )

    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    load_effort_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['effort_controller'],
        output='screen',
    )

    pkg_controller = get_package_share_directory('wlr_controller')
    balance_node = Node(
        package='wlr_controller',
        executable='balance_node',
        name='balance_controller',
        parameters=[
            os.path.join(pkg_controller, 'config', 'params.yaml'),
            {'use_sim_time': True},
        ],
        output='screen',
    )

    # Sequential chain: spawn → (JSB + effort in parallel) → balance_node
    # Starting JSB and effort spawners in parallel cuts ~2s from startup.
    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        bridge,
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_entity,
                on_exit=[
                    load_joint_state_broadcaster,
                    load_effort_controller,
                ],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_effort_controller,
                on_exit=[balance_node],
            )
        ),
    ])
