# start MoveIt! for the UR5, including RViz

import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Declare arguments
    arg_ur_type = DeclareLaunchArgument(
            "ur_type",
            default_value="ur5",
            description="Type/series of used UR robot.",
            choices=["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e"],
        )
    arg_robot_ip = DeclareLaunchArgument(
            "robot_ip",
            default_value="xxx.yyy.zzz.www",
            description="IP address by which the robot can be reached.",
        )
    arg_use_fake_hardware = DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="true",
            description="Start robot with fake hardware mirroring command to its states.",
        )
    arg_fake_sensor_commands = DeclareLaunchArgument(
            "fake_sensor_commands",
            default_value="true",
            description="Enable fake command interfaces for sensors used for simple simulations. \
            Used only if 'use_fake_hardware' parameter is true.",
        )
    arg_initial_joint_controller = DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="joint_trajectory_controller",
            description="Initially loaded robot controller.",
            choices=[
                "scaled_joint_trajectory_controller",
                "joint_trajectory_controller",
                "forward_velocity_controller",
                "forward_position_controller",
            ],
        )
    arg_activate_joint_controller = DeclareLaunchArgument(
            "activate_joint_controller",
            default_value="true",
            description="Activate loaded joint controller.",
        )
    arg_use_sim_time = DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation time.",
        )

    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    initial_joint_controller = LaunchConfiguration("initial_joint_controller")
    activate_joint_controller = LaunchConfiguration("activate_joint_controller")
    use_sim_time = LaunchConfiguration("use_sim_time")

    ur_robot_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ur_robot_driver'),
                'launch/ur_control.launch.py')),
            launch_arguments={
                "ur_type": ur_type,
                "robot_ip": robot_ip,
                "use_fake_hardware": use_fake_hardware,
                "fake_sensor_commands": fake_sensor_commands,
                "initial_joint_controller": initial_joint_controller,
                "activate_joint_controller": activate_joint_controller,
                "launch_rviz": "false",
            }.items(),
    )

    ur_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ur_moveit_config'),
                'launch/ur_moveit.launch.py')),
        launch_arguments={"ur_type": ur_type,
                          "use_sim_time": use_sim_time,
                          "launch_rviz": "true"
                          }.items(),
    )

    return LaunchDescription([
        arg_ur_type,
        arg_robot_ip,
        arg_use_fake_hardware,
        arg_fake_sensor_commands,
        arg_initial_joint_controller,
        arg_activate_joint_controller,
        arg_use_sim_time,
        ur_robot_driver_launch,
        ur_moveit_launch,
    ])
