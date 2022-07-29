# Handeye Calibration

Python-based Handeye Calibration Tool

The code has been tested on a KUKA LBR iiwa with RealSense cameras attached

## Requirements
- Python >= 3.8
- pyrealsense2
- RoboDK

## Robot Setup
See [iiwa_robodk](https://github.com/jhu-bigss/KUKA_LBR_iiwa_ROS2_Driver/tree/main/iiwa_robodk) to set up a hand guiding application using RoboDK.

## Instructions
1. run [rs_image_capture_rgb.py](/handeye_calibration_python/rs_image_capture_rgb.py) to perform camera calibration.
2. run [rs_image_capture_handeye_calibration_rdk.py](/handeye_calibration_python/rs_image_capture_handeye_calibration_rdk.py), which will read robot transformations from RoboDK and save to file. 
3. compute the hand-eye calibration result.

## Reference
- [Wiki: Hand eye calibration problem](https://en.wikipedia.org/wiki/Hand_eye_calibration_problem)