from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    arg_name = DeclareLaunchArgument('name')
    arg_tracking_base_frame = DeclareLaunchArgument('tracking_base_frame')
    arg_tracking_marker_frame_1 = DeclareLaunchArgument('tracking_marker_frame_1')
    arg_tracking_marker_frame_2 = DeclareLaunchArgument('tracking_marker_frame_2')
    arg_robot_base_frame = DeclareLaunchArgument('robot_base_frame')
    arg_robot_effector_frame = DeclareLaunchArgument('robot_effector_frame')

    node_dummy_tf_pub_rob_base_to_mkr_1 = Node(package='tf2_ros', executable='static_transform_publisher',
                                               name='dummy_tf_publisher_robot_base_to_marker_1',
                                               arguments=f'0 0 0.1 0 0 0 1'.split(' ') + [LaunchConfiguration('robot_base_frame'),
                                                                           LaunchConfiguration('tracking_marker_frame_1')])

    node_dummy_tf_pub_rob_eef_to_mkr_2 = Node(package='tf2_ros', executable='static_transform_publisher',
                                              name='dummy_tf_publisher_robot_eef_to_marker_2',
                                              arguments=f'0 0 0.1 0 0 0 1'.split(' ') + [LaunchConfiguration('robot_effector_frame'),
                                                                         LaunchConfiguration('tracking_marker_frame_2')])

    handeye_server = Node(package='easy_handeye2', executable='handeye_server_two_markers',
                          name='handeye_server_two_markers',
                          parameters=[{
            'name': LaunchConfiguration('name'),
            'tracking_base_frame': LaunchConfiguration('tracking_base_frame'),
            'tracking_marker_frame': LaunchConfiguration('tracking_marker_frame'),
            'robot_base_frame': LaunchConfiguration('robot_base_frame'),
            'robot_effector_frame': LaunchConfiguration('robot_effector_frame'),
        }])

    handeye_rqt_calibrator = Node(package='easy_handeye2', executable='rqt_calibrator.py',
                                  name='handeye_rqt_calibrator',
                                  # arguments=['--ros-args', '--log-level', 'debug'],
                                  parameters=[{
            'name': LaunchConfiguration('name'),
            'tracking_base_frame': LaunchConfiguration('tracking_base_frame'),
            'tracking_marker_frame': LaunchConfiguration('tracking_marker_frame'),
            'robot_base_frame': LaunchConfiguration('robot_base_frame'),
            'robot_effector_frame': LaunchConfiguration('robot_effector_frame'),
        }])

    return LaunchDescription([
        arg_name,
        arg_tracking_base_frame,
        arg_tracking_marker_frame_1,
        arg_tracking_marker_frame_2,
        arg_robot_base_frame,
        arg_robot_effector_frame,
        node_dummy_tf_pub_rob_base_to_mkr_1,
        node_dummy_tf_pub_rob_eef_to_mkr_2,
        handeye_server,
        handeye_rqt_calibrator,
    ])
