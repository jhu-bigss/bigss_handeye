import os
import pathlib
from typing import Optional

import easy_handeye2_msgs.msg
import tf2_ros
import yaml
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from rosidl_runtime_py import message_to_yaml, set_message_fields
from easy_handeye2_msgs.msg import Sample, SampleList
import rclpy
from rclpy.time import Duration, Time

from easy_handeye2 import SAMPLES_DIRECTORY
from easy_handeye2.handeye_calibration import HandeyeCalibrationParameters

import easy_handeye2
import numpy as np
from geometry_msgs.msg import Quaternion, Transform, Vector3
class HandeyeSampler:
    """
    Manages the samples acquired from tf.
    """

    def __init__(self, node: rclpy.node.Node, handeye_parameters: HandeyeCalibrationParameters):
        self.node = node
        self.sample_num = None
        self.handeye_parameters = handeye_parameters

        # tf structures
        self.tfBuffer: tf2_ros.Buffer = Buffer()
        """
        used to get transforms to build each sample
        """
        self.tfListener: tf2_ros.TransformListener = TransformListener(self.tfBuffer, self.node, spin_thread=True)
        """
        used to get transforms to build each sample
        """
        self.tfBroadcaster: tf2_ros.TransformBroadcaster = TransformBroadcaster(self.node)
        """
        used to publish the calibration after saving it
        """

        # internal input data
        self.samples: easy_handeye2.msg.SampleList = SampleList()
        """
        list of acquired samples
        """

    def wait_for_tf_init(self) -> bool:
        """
        Waits until all needed frames are present in tf.
        """
        base_frame = self.handeye_parameters.robot_base_frame
        effector_frame = self.handeye_parameters.robot_effector_frame
        camera_frame = self.handeye_parameters.tracking_base_frame
        marker_frame = self.handeye_parameters.tracking_marker_frame
        self.node.get_logger().info('Checking that the expected transforms are available in tf')
        self.node.get_logger().info(f'Robot transform: {base_frame} -> {effector_frame}')
        self.node.get_logger().info(f'Tracking transform: {camera_frame} -> {marker_frame}')
        try:
            self.tfBuffer.lookup_transform(base_frame, effector_frame, Time(), Duration(seconds=10))
        except tf2_ros.TransformException as e:
            self.node.get_logger().error(
                'The specified tf frames for the robot base and hand do not seem to be connected')
            self.node.get_logger().error('Run the following command and check its output:')
            self.node.get_logger().error(f'ros2 run tf2_ros tf2_echo {base_frame} {effector_frame}')
            self.node.get_logger().error(
                f'You may need to correct the base_frame or effector_frame argument passed to the easy_handeye2 launch file')
            self.node.get_logger().error(f'Underlying tf exception: {e}')
            return False

        try:
            self.tfBuffer.lookup_transform(camera_frame, marker_frame, Time(), Duration(seconds=10))
        except tf2_ros.TransformException as e:
            self.node.get_logger().error(
                'The specified tf frames for the tracking system base/camera and marker do not seem to be connected')
            self.node.get_logger().error('Run the following command and check its output:')
            self.node.get_logger().error(f'ros2 run tf2_ros tf2_echo {camera_frame} {marker_frame}')
            self.node.get_logger().error(
                f'You may need to correct the base_frame or effector_frame argument passed to the easy_handeye2 launch file')
            self.node.get_logger().error(f'Underlying tf exception: {e}')
            return False

        self.node.get_logger().info('All expected transforms are available on tf; ready to take samples')
        return True

    def _get_transforms(self, time: Optional[rclpy.time.Time] = None) -> easy_handeye2_msgs.msg.Sample:
        """
        Samples the transforms at the given time.
        """
        if time is None:
            time = self.node.get_clock().now() - rclpy.time.Duration(nanoseconds=200000000)

        # here we trick the library (it is actually made for eye_in_hand only). Trust me, I'm an engineer
        if self.handeye_parameters.eye_in_hand:
            robot = self.tfBuffer.lookup_transform(self.handeye_parameters.robot_base_frame,
                                                   self.handeye_parameters.robot_effector_frame, time,
                                                   Duration(seconds=10))
        else:
            robot = self.tfBuffer.lookup_transform(self.handeye_parameters.robot_effector_frame,
                                                   self.handeye_parameters.robot_base_frame, time,
                                                   Duration(seconds=10))
        tracking = self.tfBuffer.lookup_transform(self.handeye_parameters.tracking_base_frame,
                                                  self.handeye_parameters.tracking_marker_frame, time,
                                                  Duration(seconds=10))
        ret = Sample()
        ret.robot = robot.transform
        ret.tracking = tracking.transform
        return ret

    def _quarternion_avg(self, quaternions:list):
        M = np.zeros((4,4))
        n = len(quaternions)
        wSum = 0
        for i in range(n):
            q = quaternions[i]
            q = np.asarray([q.x,q.y,q.z,q.w])[:,None]
            M += np.matmul(q,q.T)
            wSum += 1
        M = (1.0/wSum) * M
        w,v = np.linalg.eig(M)
        avg_q = v[np.argmax(w)]
        Q_avg = Quaternion(x = avg_q[0],y=avg_q[1],z=avg_q[2],w=avg_q[3])
        return Q_avg

    def _translation_avg(self, translations:list):
        translations = [[t.x, t.y, t.z] for t in translations]
        translations = np.asarray(translations)
        trans_avg = np.mean(translations,axis=0)
        vec_avg = Vector3(x = trans_avg[0],y = trans_avg[1],z=trans_avg[2])
        return vec_avg

    def mean_filter(self, transforms:list) -> Transform:
        translations = [s.translation for s in transforms]
        rotations = [s.rotation for s in transforms]
        transform_avg = Transform(translation=self._translation_avg(translations),\
                                rotation=self._quarternion_avg(rotations)    )
        return transform_avg
    

    def take_sample(self) -> bool:
        """
        Samples the transformations and appends the sample to the list.
        """
        try:
            self.node.get_logger().info("Taking a sample...")
            self.node.get_logger().info("all frames: " + self.tfBuffer.all_frames_as_string())
            if self.sample_num != None:
                tracking_transforms = []
                for i in range(self.sample_num):
                    sample = self._get_transforms()
                    tracking_transforms.append(sample.tracking)
                tracking_transform = self.mean_filter(tracking_transforms)
                sample.tracking = tracking_transform 
            else:
                sample = self._get_transforms()
            self.node.get_logger().info("Got a sample")
            new_samples = self.samples.samples
            new_samples.append(sample)
            self.samples.samples = new_samples
            return True
        except:
            return False

    def remove_sample(self, index: int) -> int:
        """
        Removes a sample from the list. Returns the updated number of samples
        """
        if 0 <= index < len(self.samples.samples):
            new_samples = self.samples.samples
            del new_samples[index]
            self.samples.samples = new_samples
        return len(self.samples.samples)

    def get_samples(self) -> easy_handeye2_msgs.msg.SampleList:
        """
        Returns the samples accumulated so far.
        """
        return self.samples

    @staticmethod
    def _filepath_for_samplelist(name) -> pathlib.Path:
        return SAMPLES_DIRECTORY / f'{name}.samples'

    def load_samples(self) -> bool:
        filepath = HandeyeSampler._filepath_for_samplelist(self.handeye_parameters.name)
        with open(filepath) as f:
            m = yaml.load(f.read())
            ret = SampleList()
            set_message_fields(ret, m)
            self.samples = ret
        return True

    def save_samples(self) -> bool:
        if not os.path.exists(SAMPLES_DIRECTORY):
            os.makedirs(SAMPLES_DIRECTORY)
        filepath = HandeyeSampler._filepath_for_samplelist(self.handeye_parameters.name)
        with open(filepath, 'w') as f:
            f.write(message_to_yaml(self.samples))
        return True
