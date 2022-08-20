import os
import pathlib

import yaml
from easy_handeye2_msgs.msg import HandeyeCalibration, HandeyeCalibrationParameters
from rclpy.node import Node
from rosidl_runtime_py import set_message_fields, message_to_yaml

from . import CALIBRATIONS_DIRECTORY


def filepath_for_calibration(name) -> pathlib.Path:
    return CALIBRATIONS_DIRECTORY / f'{name}.calib'


class HandeyeCalibrationParametersProvider:
    def __init__(self, node: Node):
        self.node = node
        # declare and read parameters
        self.node.declare_parameter('name')
        self.node.declare_parameter('robot_base_frame')
        self.node.declare_parameter('robot_effector_frame')
        self.node.declare_parameter('tracking_base_frame')
        self.node.declare_parameter('tracking_marker_frame_1')
        self.node.declare_parameter('tracking_marker_frame_2')
        self.node.declare_parameter('freehand_robot_movement')

    def read(self):
        ret = HandeyeCalibrationParameters(
            name=self.node.get_parameter('name').get_parameter_value().string_value,
            robot_base_frame=self.node.get_parameter('robot_base_frame').get_parameter_value().string_value,
            robot_effector_frame=self.node.get_parameter('robot_effector_frame').get_parameter_value().string_value,
            tracking_base_frame=self.node.get_parameter('tracking_base_frame').get_parameter_value().string_value,
            tracking_marker_frame_1=self.node.get_parameter('tracking_marker_frame_1').get_parameter_value().string_value,
            tracking_marker_frame_2=self.node.get_parameter('tracking_marker_frame_2').get_parameter_value().string_value,
            freehand_robot_movement=self.node.get_parameter('freehand_robot_movement').get_parameter_value().bool_value,
        )
        return ret


def load_calibration(name) -> HandeyeCalibration:
    filepath = filepath_for_calibration(name)
    with open(filepath) as f:
        m = yaml.load(f.read())
    ret = HandeyeCalibration()
    set_message_fields(ret, m)
    return ret


def save_calibration(calibration: HandeyeCalibration) -> pathlib.Path:
    if not os.path.exists(CALIBRATIONS_DIRECTORY):
        os.makedirs(CALIBRATIONS_DIRECTORY)
    filepath = filepath_for_calibration(calibration.parameters.name)
    with open(filepath, 'w') as f:
        f.write(message_to_yaml(calibration))
    return filepath