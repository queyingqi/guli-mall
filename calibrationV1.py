#https://github.com/SintefManufacturing/python-urx/issues/64  i have changed timeout and limit in usercom.py 
import argparse 
from pathlib import Path
import time
import datetime
import sys
import os
import logging
#sys.path.append("/home/wbk-ur/python-urx")

import urx
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

#sys.path.append("~/zivid-python")
#sys.path.append("~/zivid-python/modules")
import zivid
import matplotlib.pyplot as plt
import glob


print(urx.__version__)



rob = None

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
    valid = False
    point_cloud = frame.point_cloud()

    detected_features = zivid.calibration.detect_feature_points(point_cloud)
    #print('df :',detected_features)

    if not detected_features.valid():
        #raise RuntimeError("Failed to detect feature points from captured frame.")
        print('no')
    else:
        print('Valid!')
        valid = True
    return valid

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
    """   
    return zivid.Settings(
        acquisitions=[
            zivid.Settings.Acquisition(
                aperture=8.55,
                exposure_time=datetime.timedelta(microseconds=20000),
                brightness=0.8,
                gain=1.0,
            )
        ],
        processing=zivid.Settings.Processing(
            filters=zivid.Settings.Processing.Filters(
                smoothing=zivid.Settings.Processing.Filters.Smoothing(
                    gaussian=zivid.Settings.Processing.Filters.Smoothing.Gaussian(enabled=True)
                )
            )
        ),
    )
    """
def _save_hand_eye_results(save_dir: Path, transform: np.array, residuals: list):
    """Save transformation and residuals to folder
    Args:
        save_dir: Path to where data will be saved
        transform: 4x4 transformation matrix
        residuals: List of residuals
    """

    file_storage_transform = cv2.FileStorage(str(save_dir +"/" "transformation.yaml"), cv2.FILE_STORAGE_WRITE)
    file_storage_transform.write("PoseState", transform)
    file_storage_transform.release()
    print('done')
    file_storage_residuals = cv2.FileStorage(str(save_dir +"/" "residuals.yaml"), cv2.FILE_STORAGE_WRITE)
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

    #time.sleep(1)

    
    camera_in_position = False
    while not camera_in_position:
        while True:
            user_input_r = input(" Move robot/camera to new position and press r. To start calibrating press c ")
            if user_input_r == 'r':
                 break
            elif user_input_r == 'c':
                os.mkdir(path+'/result')
                perform_hand_eye_calibration(path)


  
        frame = camera.capture(settings)
        time.sleep(1)
        point_cloud = frame.point_cloud()
        xyz = point_cloud.copy_data("xyz")

        """
        plt.figure()
        plt.imshow(
            xyz[:, :, 2],
            vmin=np.nanmin(xyz[:, :, 2]),
            vmax=np.nanmax(xyz[:, :, 2]),
            cmap="viridis",
        )
        plt.colorbar()
        plt.title("Depth map")
        plt.show(block=False)
        """
        rgba = point_cloud.copy_data("rgba")
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        rgba = cv2.cvtColor(rgba,cv2.COLOR_BGR2RGB)
        cv2.imshow('image', rgba[:, :, 0:3])
        cv2.resizeWindow('image', 1500,1000)

        cv2.waitKey(2000)

        cv2.destroyAllWindows()

        print("is the image in appropriate position?")
        
        user_input = input(" press 'y' or 'n'   ")
        if user_input == 'y':
            camera_in_position = True

    print("commumnicating")
    robot_ip = '172.22.132.6'
    rob = urx.Robot(robot_ip)
    time.sleep(1)

    robot_pose = np.array(rob.getl())

    print(robot_pose)

    translation = robot_pose[:3] * 1000
    #translation[2] -= 400
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
            while True:
                frame, translation, rotation_vector  = _get_frame_and_transform_matrix(camera, settings)
                time.sleep(0.1)
                valid = _verify_good_capture(frame)
                if valid:		
                    rotation = Rotation.from_rotvec(rotation_vector)				   
                    transform = np.eye(4)
                    transform[:3, :3] = rotation.as_matrix()
                    transform[:3, 3] = translation.T
                    print("fertig")
                    #print(translation)
                    filename = (f'{translation[0]:.2f}')+'_'+(f'{translation[1]:.2f}')+'_' +(f'{translation[2]:.2f}')+'_'+(f'{rotation_vector[0]:.2f}')+'_'+(f'{rotation_vector[1]:.2f}')+'_'+(f'{rotation_vector[2]:.2f}')
                    file_storage = cv2.FileStorage(path+'/'+filename+".yaml", cv2.FILE_STORAGE_WRITE)
                    file_storage.write("PoseState", transform)
                    file_storage.release()
                    frame.save(path+'/'+filename+".zdf")



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

    files_yaml = glob.glob1(save_dir,"*.yaml")
    files_yaml.sort()
    files_zdf = glob.glob1(save_dir,"*.zdf")
    files_zdf.sort()

    for i in range(len(files_yaml)):
        if files_yaml[i][:len(files_yaml[i])-5] != files_zdf[i][:len(files_zdf[i])-4]:
            raise RuntimeError(".yaml and .zdf do not have the same robot coordinates")
                
        point_cloud = zivid.Frame(save_dir+'/'+files_zdf[i]).point_cloud()
        detection_result = zivid.calibration.detect_feature_points(point_cloud)
        with open(save_dir+'/'+files_yaml[i]) as file:
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
    _save_hand_eye_results(save_dir+'/result', transform, residuals)


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
        os.mkdir(save_dir+'/result')
    except:
        pass
    perform_hand_eye_calibration(save_dir)    
  




