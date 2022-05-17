# https://github.com/SintefManufacturing/python-urx/issues/64  i have changed timeout and limit in usercom.py
import argparse
import configparser
import click
from pathlib import Path
import time
import datetime
import sys
import os
import logging

# sys.path.append("/home/wbk-ur/python-urx")
import urx
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import zivid
import glob
import os


rob = None
robot_ip = ''
zivid_acquisitions = []


def pose_from_datastring(datastring: str):
    """Extract pose from yaml file saved by openCV
    Args:
        datastring: String of text from .yaml file
    Returns:
        Robotic pose as zivid Pose class
    """

    string = datastring.split("data:")[-1].strip().strip("[").strip("]")
    pose_matrix = np.fromstring(string, dtype=np.float, count=16, sep=",").reshape((4, 4))
    return zivid.calibration.Pose(pose_matrix)


def _verify_good_capture(frame: zivid.Frame):
    """Verify that checkeroard featurepoints are detected in the frame
    Args:
        frame: Zivid frame containing point cloud
    Raises:
        RuntimeError: If no feature points are detected in frame
    """

    point_cloud = frame.point_cloud()
    valid = False
    try:
        detected_features = zivid.calibration.detect_feature_points(point_cloud)
        print('df :', detected_features)
        valid = detected_features.valid()

    except RuntimeError:
        print('ERROR: Failed to detect feature points from captured frame')
        return False

    return detected_features.valid()


def _camera_settings() -> zivid.Settings:
    """Set camera settings
    Returns:
        Zivid Settings
    """

    return zivid.Settings(
        acquisitions=[
            zivid.Settings.Acquisition(
                aperture=8.72,
                exposure_time=datetime.timedelta(microseconds=20000),
                brightness=0.6,
                gain=1.0,
            ),
            zivid.Settings.Acquisition(
                aperture=6.17,
                exposure_time=datetime.timedelta(microseconds=25000),
                brightness=0.8,
                gain=1.0,
            ),
        ],
        processing=zivid.Settings.Processing(
            filters=zivid.Settings.Processing.Filters(
                smoothing=zivid.Settings.Processing.Filters.Smoothing(
                    gaussian=zivid.Settings.Processing.Filters.Smoothing.Gaussian(enabled=True)
                )
            )
        ),
    )


def _save_hand_eye_results(save_dir: Path, transform: np.array, residuals: list):
    """Save transformation and residuals to folder
    Args:
        save_dir: Path to where data will be saved
        transform: 4x4 transformation matrix
        residuals: List of residuals
    """

    file_storage_transform = cv2.FileStorage(str(save_dir + "/" "transformation.yaml"), cv2.FILE_STORAGE_WRITE)
    file_storage_transform.write("PoseState", transform)
    file_storage_transform.release()
    print('done')
    file_storage_residuals = cv2.FileStorage(str(save_dir + "/" "residuals.yaml"), cv2.FILE_STORAGE_WRITE)
    residual_list = []
    for res in residuals:
        tmp = list([res.translation(), res.translation()])
        residual_list.append(tmp)

    file_storage_residuals.write(
        "Per pose residuals for rotation in deg and translation in mm",
        np.array(residual_list),
    )
    file_storage_residuals.release()


def _get_frame_and_transform_matrix(camera: zivid.Camera, settings: zivid.Settings):
    """Capture image with Zivid camera and read robot pose
    Args:
        con: Connection between computer and robot
        camera: Zivid camera
        settings: Zivid settings
    Returns:
        Zivid frame
        4x4 tranformation matrix
    """
    camera_in_position = False
    # Capturing new Frames till user is ok with the Frame
    while not camera_in_position:
        # Capture a Frame
        frame = camera.capture(settings)
        time.sleep(1)
        point_cloud = frame.point_cloud()
        xyz = point_cloud.copy_data("xyz")
        rgba = point_cloud.copy_data("rgba")

        # Show captured scan
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2RGB)
        cv2.imshow('image', rgba[:, :, 0:3])
        cv2.resizeWindow('image', 1500, 1000)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        camera_in_position = click.confirm("is the image in appropriate position?", default=True)

    # Get current Robot position
    global rob
    print("read robot's position")
    if rob is None: # connect Robot
        rob = urx.Robot(robot_ip)

    time.sleep(1)
    robot_pose = np.array(rob.getl())

    """
    for i in range(20):  # try 20 Times to connect to Robot
        try:
            rob = urx.Robot("129.13.234.156")
            break;
        except urx.ursecmon.TimeoutException as ex:
            print("ROBOT doesn't answer, retry", i)
    """


    print(robot_pose)

    translation = robot_pose[:3] * 1000
    # translation[2] -= 400
    rotation_vector = robot_pose[3:]

    return frame, translation, rotation_vector


def _capture_frames(
        camera: zivid.Camera,
        settings: zivid.Settings,
):
    """Capture 3D image and robot pose for a given robot posture,
    then signals robot to move to next posture
    Args:
        con: Connection between computer and robot
        camera: Zivid camera
        settings: Zivid settings
        save_dir: Path to where data will be saved
        input_data: Input package containing the specific input data registers
        image_num: Image number
        ready_to_capture: Boolean value to robot_state that camera is ready to capture images
    """

    with zivid.Application() as app:
        with app.connect_camera() as camera:
            settings = _camera_settings()
            frame, translation, rotation_vector = _get_frame_and_transform_matrix(camera, settings)
            time.sleep(0.1)
            if _verify_good_capture(frame):
                print('Frame is valid. saving Frame and rotation')
                rotation = Rotation.from_rotvec(rotation_vector)
                transform = np.eye(4)
                transform[:3, :3] = rotation.as_matrix()
                transform[:3, 3] = translation.T
                # print(translation)
                filename = (f'{translation[0]:.2f}') + '_' + (f'{translation[1]:.2f}') + '_' + (
                    f'{translation[2]:.2f}') + '_' + (f'{rotation_vector[0]:.2f}') + '_' + (
                               f'{rotation_vector[1]:.2f}') + '_' + (f'{rotation_vector[2]:.2f}')
                file_storage = cv2.FileStorage(path + '/' + filename + ".yaml", cv2.FILE_STORAGE_WRITE)
                file_storage.write("PoseState", transform)
                file_storage.release()
                frame.save(path + '/' + filename + ".zdf")
            else:
                print('No Features found - Frame rejected')



def perform_hand_eye_calibration(save_dir):
    """Perform had-eye calibration based on mode
    Args:
        mode: Calibration mode, eye-in-hand or eye-to-hand
        data_dir: Path to dataset
    Returns:
        4x4 transformation matrix
        List of residuals
    Raises:
        RuntimeError: If no feature points are detected
        ValueError: If calibration mode is invalid
    """
    # setup zivid
    app = zivid.Application()
    calibration_inputs = []

    files_yaml = glob.glob1(save_dir, "*.yaml")
    files_yaml.sort()
    files_zdf = glob.glob1(save_dir, "*.zdf")
    files_zdf.sort()

    for i in range(len(files_yaml)):
        if files_yaml[i][:len(files_yaml[i]) - 5] != files_zdf[i][:len(files_zdf[i]) - 4]:
            raise RuntimeError(".yaml and .zdf do not have the same robot coordinates")

        point_cloud = zivid.Frame(save_dir + '/' + files_zdf[i]).point_cloud()
        detection_result = zivid.calibration.detect_feature_points(point_cloud)
        with open(save_dir + '/' + files_yaml[i]) as file:
            pose = pose_from_datastring(file.read())
        calibration_inputs.append(zivid.calibration.HandEyeInput(pose, detection_result))

    calibration_result = zivid.calibration.calibrate_eye_in_hand(calibration_inputs)

    transform = calibration_result.transform()
    residuals = calibration_result.residuals()

    print("\n\nTransform: \n")
    np.set_printoptions(precision=5, suppress=True)
    print(transform)

    print("\n\nResiduals: \n")
    for res in residuals:
        print(f"Rotation: {res.rotation():.6f}   Translation: {res.translation():.6f}")
    _save_hand_eye_results(save_dir + '/result', transform, residuals)


if __name__ == '__main__':
    if os.path.isfile('calibration.config'):
        print("Config found:")
        config = configparser.ConfigParser()

        try:
            config.read('calibration.config')
            robot_ip = config['ROBOT'].get('IP')

            for i in range(0, 1):
                civit_acq = 'ZIVID_ACQ_' + str(i)
                acquisition = dict(
                    aperture=config[civit_acq].get('aperture'),
                    exposure_time=config[civit_acq].get('exposure_time'),
                    brightness=config[civit_acq].get('brightness'),
                    gain=config[civit_acq].get('gain'),
                )
                zivid_acquisitions.append(acquisition)
        except:
            err = sys.exc_info()
            print('ERR in calibration.config. Please check. - ERROR: ', err)
            sys.exit(0)

    else:
        print("No Config found, using default Parameters")



    print("Hand-in-Eye_Calibration:")
    if click.confirm('Capture new Frames?',default=True):
        # Creating new Folder for captures
        now = datetime.datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        path = 'calibration_data' + '/' + dt_string
        print('files will be stored in ', path)
        os.mkdir(path)

        for i in range(20):
            if i == 0:
                input("Set robot in position and press [Enter] to start scan")
            _capture_frames(zivid.Camera, zivid.Settings)
            if not click.confirm("capture another scan?", default=True):
                break

        print("scanning finished. Start calibration now")

    path = config['DEFAULT'].get('out_path')
    # Perform calibration
    os.mkdir(os.path.join(path, 'result'))
    perform_hand_eye_calibration(path)

    user_input = input(" Press: s to capture frames and robot's positions or c to calibrate  ")
    if user_input == 's':
        now = datetime.datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        path = 'calibration_data' + '/' + dt_string
        os.mkdir(path)
        print('files will be stored in ', path)
        _capture_frames(zivid.Camera, zivid.Settings)
    elif user_input == 'c':
        save_dir = input("Enter the folder path where the frames and the robot's positions are stored  ")
        try:
            os.mkdir(save_dir + '/result')
        except:
            pass
        perform_hand_eye_calibration(save_dir)
