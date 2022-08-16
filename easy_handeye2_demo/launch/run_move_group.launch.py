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

    ur_type = "ur5"
    robot_ip = "xxx.yyy.zzz.www"
    use_fake_hardware = "true"
    fake_sensor_commands = "true"
    initial_joint_controller = "joint_trajectory_controller"
    activate_joint_controller = "true"
    use_sim_time = "true"

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

    # Static TF
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base"],
    )

    return LaunchDescription([
        ur_robot_driver_launch,
        ur_moveit_launch,
        static_tf,
    ])
