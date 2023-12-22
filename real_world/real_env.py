from typing import Optional
import sys
import os

import cv2
import time
import pickle
import numpy as np
import math

from multiprocessing.managers import SharedMemoryManager
from real_world.camera.multi_realsense import MultiRealsense, SingleRealsense

from real_world.xarm6 import XARM6
from real_world.utils import depth2fgpcd, rpy_to_rotation_matrix


class RealEnv:
    def __init__(self, 
            WH=[640, 480],
            capture_fps=30,
            obs_fps=15,
            n_obs_steps=2,
            enable_color=True,
            enable_depth=True,
            process_depth=False,
            use_robot=True,
            use_wrist_cam=True,
            num_fixed_cams=4,
            gripper_enable=False,
            verbose=False,
        ):
        self.WH = WH
        self.capture_fps = capture_fps
        self.obs_fps = obs_fps
        self.n_obs_steps = n_obs_steps
        self.vis_dir = 'vis'

        self.WRIST = '246322303938'  # device id of the wrist camera
        num_cams = num_fixed_cams + (1 if use_wrist_cam else 0)
        self.serial_numbers = SingleRealsense.get_connected_devices_serial()
        assert len(self.serial_numbers) == num_cams
        if use_wrist_cam:
            assert self.WRIST in self.serial_numbers
            self.serial_numbers.remove(self.WRIST)
            self.serial_numbers = self.serial_numbers + [self.WRIST]  # put the wrist camera at the end
        self.num_fixed_cams = num_fixed_cams

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        self.realsense =  MultiRealsense(
                serial_numbers=self.serial_numbers,
                shm_manager=self.shm_manager,
                resolution=(self.WH[0], self.WH[1]),
                capture_fps=self.capture_fps,
                enable_color=enable_color,
                enable_depth=enable_depth,
                process_depth=process_depth,
                verbose=verbose)
        self.realsense.set_exposure(exposure=100, gain=60)
        self.last_realsense_data = None
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.use_robot = use_robot

        if use_robot:
            self.robot = XARM6(gripper_enable=gripper_enable)
        self.gripper_enable = gripper_enable

        calibration_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        calibration_parameters =  cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(calibration_dictionary, calibration_parameters)
        self.calibration_board = cv2.aruco.GridBoard(
            (5, 7),
            markerLength=0.03439,
            markerSeparation=0.00382,
            dictionary=calibration_dictionary,
        )

        self.R_cam2world = None
        self.t_cam2world = None
        self.R_base2world = None
        self.t_base2world = None
        
        self.calibrate_result_dir = 'real_world/calibration_result'
        os.makedirs(self.calibrate_result_dir, exist_ok=True)

        # TODO may change according to each calibration & task & robot end-effector
        self.gripper_point_opened = np.array([
            [0.013, 0.043, 0.122],
            [-0.013, 0.043, 0.122],
            [0.013, -0.043, 0.122],
            [-0.013, -0.043, 0.122],
            [0.013, 0.043, 0.152],
            [-0.013, 0.043, 0.152],
            [0.013, -0.043, 0.152],
            [-0.013, -0.043, 0.152],
        ])
        self.gripper_point_opened += np.array([0, 0.01, 0])
        self.bbox = np.array([[-0.45, 0.1], [-0.3, 0.55], [-0.2, 0.1]])  # bounding box of the workspace

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and (self.robot.is_alive if self.use_robot else True)
    
    def start(self, wait=True, exposure_time=5):
        self.realsense.start(wait=False, put_start_time=time.time() + exposure_time)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
    
    def stop_wait(self):
        self.realsense.stop_wait()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self, get_color=True, get_depth=False) -> dict:
        assert self.is_ready

        # get data
        k = math.ceil(self.n_obs_steps * (self.capture_fps / self.obs_fps))
        self.last_realsense_data = self.realsense.get(
            k=k,
            out=self.last_realsense_data
        )

        robot_obs = dict()
        if self.use_robot:
            robot_obs['joint_angles'] = self.robot.get_current_joint()
            robot_obs['pose'] = self.robot.get_current_pose()
            if self.gripper_enable:
                robot_obs['gripper_position'] = self.robot.get_gripper_state()

        # align camera obs timestamps
        dt = 1 / self.obs_fps
        timestamp_list = [x['timestamp'][-1] for x in self.last_realsense_data.values()]
        last_timestamp = np.max(timestamp_list)
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)
        # the last timestamp is the latest one

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            if get_color:
                assert self.enable_color
                camera_obs[f'color_{camera_idx}'] = value['color'][this_idxs]
            if get_depth:
                assert self.enable_depth
                camera_obs[f'depth_{camera_idx}'] = value['depth'][this_idxs] / 1000.0
                # if process_depth:
                #     camera_obs[f'depth_{camera_idx}'] = depth_process(camera_obs[f'depth_{camera_idx}'])
                # else:
                # camera_obs[f'depth_{camera_idx}'] = camera_obs[f'depth_{camera_idx}'] / 1000.0
            # camera_obs[f'timestamp_{camera_idx}'] = value['timestamp'][this_idxs]

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data
    
    def get_intrinsics(self):
        return self.realsense.get_intrinsics()

    def get_extrinsics(self):
        return (
            [self.R_cam2world[i].copy() for i in self.serial_numbers[:self.num_fixed_cams]],
            [self.t_cam2world[i].copy() for i in self.serial_numbers[:self.num_fixed_cams]],
        )

    def get_bbox(self):
        return self.bbox.copy()

    def step(self, actions: np.ndarray):  
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        import pdb; pdb.set_trace()

    def get_robot_pose(self, raw=False):
        raw_pose = self.robot.get_current_pose()
        if raw:
            return raw_pose
        else:
            R_gripper2base = rpy_to_rotation_matrix(
                raw_pose[3], raw_pose[4], raw_pose[5]
            )
            t_gripper2base = np.array(raw_pose[:3]) / 1000
        return R_gripper2base, t_gripper2base

    def set_robot_pose(self, pose, wait=True):
        self.robot.move_to_pose(pose=pose, wait=wait, ignore_error=True)
    
    def reset_robot(self):
        self.robot.reset()
    
    def hand_eye_calibrate(self, visualize=True, save=True, return_results=True):
        assert self.use_robot
        self.reset_robot()
        time.sleep(1)

        poses = [
            [522.6,-1.6,279.5,179.2,0,0.3],
            [494.3,133,279.5,179.2,0,-24.3],
            [498.8,-127.3,314.9,179.3,0,31.1],
            [589.5,16.6,292.9,-175,17,1.2],
            [515.8,178.5,469.2,-164.3,17.5,-90.8],
            [507.9,-255.5,248.5,-174.6,-16.5,50.3],
            [507.9,258.2,248.5,-173.5,-8,-46.8],
            [569,-155.6,245.8,179.5,3.7,49.7],
            [570.8,-1.2,435,-178.5,52.3,-153.9],
            [474.3,12.5,165.3,179.3,-15,0.3],
        ]
        R_gripper2base = []
        t_gripper2base = []
        R_board2cam = []
        t_board2cam = []

        if visualize:
            os.makedirs(f'{self.vis_dir}', exist_ok=True)
        
        for pose in poses:
            # Move to the pose and wait for 5s to make it stable
            self.set_robot_pose(pose)
            time.sleep(5)

            # Calculate the markers
            obs = self.get_obs()

            pose_real = obs['pose']
            calibration_img = obs[f'color_{len(self.serial_numbers) - 1}'][-1]

            intr = self.get_intrinsics()[-1]
            dist_coef = np.zeros(5)

            if visualize:
                cv2.imwrite(f'{self.vis_dir}/calibration_handeye_img_{pose}.jpg', calibration_img)

            calibration_img = cv2.cvtColor(calibration_img, cv2.COLOR_BGR2GRAY)

            # calibrate
            corners, ids, rejected_img_points = self.aruco_detector.detectMarkers(calibration_img)
            detected_corners, detected_ids, rejected_corners, recovered_ids = self.aruco_detector.refineDetectedMarkers(
                detectedCorners=corners, 
                detectedIds=ids,
                rejectedCorners=rejected_img_points,
                image=calibration_img,
                board=self.calibration_board,
                cameraMatrix=intr,
                distCoeffs=dist_coef,
            )

            if visualize:
                calibration_img_vis = cv2.aruco.drawDetectedMarkers(calibration_img.copy(), detected_corners, detected_ids)
                cv2.imwrite(f'{self.vis_dir}/calibration_marker_handeye_{pose}.jpg', calibration_img_vis)

            retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                corners=detected_corners,
                ids=detected_ids,
                board=self.calibration_board,
                cameraMatrix=intr,
                distCoeffs=dist_coef,
                rvec=None,
                tvec=None,
            )

            if visualize:
                calibration_img_vis = calibration_img.copy()[:, :, np.newaxis].repeat(3, axis=2)
                cv2.drawFrameAxes(calibration_img_vis, intr, dist_coef ,rvec, tvec, 0.1)
                cv2.imwrite(f"{self.vis_dir}/calibration_result_handeye_{pose}.jpg", calibration_img_vis)

            if not retval:
                raise ValueError("pose estimation failed")

            # Save the transformation of board2cam
            R_board2cam.append(cv2.Rodrigues(rvec)[0])
            t_board2cam.append(tvec[:, 0])

            # Save the transformation of the gripper2base
            print("Current pose: ", pose_real)

            R_gripper2base.append(
                rpy_to_rotation_matrix(
                    pose_real[3], pose_real[4], pose_real[5]
                )
            )
            t_gripper2base.append(np.array(pose_real[:3]) / 1000)
        
        self.reset_robot()

        R_base2gripper = []
        t_base2gripper = []
        for i in range(len(R_gripper2base)):
            R_base2gripper.append(R_gripper2base[i].T)
            t_base2gripper.append(-R_gripper2base[i].T @ t_gripper2base[i])

        # Do the robot-world hand-eye calibration
        R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
            R_world2cam=R_board2cam,
            t_world2cam=t_board2cam,
            R_base2gripper=R_base2gripper,
            t_base2gripper=t_base2gripper,
            R_base2world=None,
            t_base2world=None,
            R_gripper2cam=None,
            t_gripper2cam=None,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        # t_cam2gripper = t_cam2gripper[:, 0]  # (3, 1) -> (3,)
        t_gripper2cam = t_gripper2cam[:, 0]  # (3, 1) -> (3,)
        t_base2world = t_base2world[:, 0]  # (3, 1) -> (3,)

        results = {}
        results["R_gripper2cam"] = R_gripper2cam
        results["t_gripper2cam"] = t_gripper2cam
        results["R_base2world"] = R_base2world
        results["t_base2world"] = t_base2world
        # results["R_cam2gripper"] = R_cam2gripper
        # results["t_cam2gripper"] = t_cam2gripper
        # results["R_gripper2base"] = R_gripper2base
        # results["t_gripper2base"] = t_gripper2base
        # results["R_board2cam"] = R_board2cam
        # results["t_board2cam"] = t_board2cam

        print("R_gripper2cam", R_gripper2cam)
        print("t_gripper2cam", t_gripper2cam)
        if save:
            with open(f"{self.calibrate_result_dir}/calibration_handeye_result.pkl", "wb") as f:
                pickle.dump(results, f)
        if return_results:
            return results

    def fixed_camera_calibrate(self, visualize=True, save=True, return_results=True):
        if visualize:
            os.makedirs(f'{self.vis_dir}', exist_ok=True)
        
        rvecs = {}
        tvecs = {}

        # Calculate the markers
        obs = self.get_obs()
        intrs = self.get_intrinsics()
        dist_coef = np.zeros(5)

        for i in range(self.num_fixed_cams):  # ignore the wrist camera
            device = self.serial_numbers[i]
            intr = intrs[i]
            calibration_img = obs[f'color_{i}'][-1].copy()
            if visualize:
                cv2.imwrite(f'{self.vis_dir}/calibration_img_{device}.jpg', calibration_img)
            
            calibration_img = cv2.cvtColor(calibration_img, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected_img_points = self.aruco_detector.detectMarkers(calibration_img)
            detected_corners, detected_ids, rejected_corners, recovered_ids = self.aruco_detector.refineDetectedMarkers(
                detectedCorners=corners, 
                detectedIds=ids,
                rejectedCorners=rejected_img_points,
                image=calibration_img,
                board=self.calibration_board,
                cameraMatrix=intr,
                distCoeffs=dist_coef,
            )

            if visualize:
                calibration_img_vis = cv2.aruco.drawDetectedMarkers(calibration_img.copy(), detected_corners, detected_ids)
                cv2.imwrite(f'{self.vis_dir}/calibration_detected_marker_{device}.jpg', calibration_img_vis)


            retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                corners=detected_corners,
                ids=detected_ids,
                board=self.calibration_board,
                cameraMatrix=intr,
                distCoeffs=dist_coef,
                rvec=None,
                tvec=None,
            )

            if not retval:
                print("pose estimation failed")
                import pdb; pdb.set_trace()

            if visualize:
                calibration_img_vis = calibration_img.copy()[:, :, np.newaxis].repeat(3, axis=2)
                cv2.drawFrameAxes(calibration_img_vis, intr, dist_coef, rvec, tvec, 0.1)
                cv2.imwrite(f"{self.vis_dir}/calibration_result_{device}.jpg", calibration_img_vis)

            rvecs[device] = rvec
            tvecs[device] = tvec
        
        if save:
            # save rvecs, tvecs
            with open(f'{self.calibrate_result_dir}/rvecs.pkl', 'wb') as f:
                pickle.dump(rvecs, f)
            with open(f'{self.calibrate_result_dir}/tvecs.pkl', 'wb') as f:
                pickle.dump(tvecs, f)
        if return_results:
            return rvecs, tvecs

    def calibrate(self, re_calibrate=False):
        if re_calibrate:
            if self.use_robot:
                calibration_handeye_result = self.hand_eye_calibrate()
                # R_gripper2wristcam = calibration_handeye_result['R_gripper2cam']
                # t_gripper2wristcam = calibration_handeye_result['t_gripper2cam']
                R_base2board = calibration_handeye_result['R_base2world']
                t_base2board = calibration_handeye_result['t_base2world']
            else:
                R_base2board = None
                t_base2board = None
            rvecs, tvecs = self.fixed_camera_calibrate()
        else:
            with open(f'{self.calibrate_result_dir}/calibration_handeye_result.pkl', 'rb') as f:
                calibration_handeye_result = pickle.load(f)
            with open(f'{self.calibrate_result_dir}/rvecs.pkl', 'rb') as f:
                rvecs = pickle.load(f)
            with open(f'{self.calibrate_result_dir}/tvecs.pkl', 'rb') as f:
                tvecs = pickle.load(f)
            # R_gripper2wristcam = calibration_handeye_result['R_gripper2cam']
            # t_gripper2wristcam = calibration_handeye_result['t_gripper2cam']
            R_base2board = calibration_handeye_result['R_base2world']
            t_base2board = calibration_handeye_result['t_base2world']
        
        self.R_cam2world = {}
        self.t_cam2world = {}
        self.R_base2world = R_base2board
        self.t_base2world = t_base2board

        for i in range(self.num_fixed_cams):
            device = self.serial_numbers[i]
            R_world2cam = cv2.Rodrigues(rvecs[device])[0]
            t_world2cam = tvecs[device][:, 0]
            self.R_cam2world[device] = R_world2cam.T
            self.t_cam2world[device] = -R_world2cam.T @ t_world2cam

    def get_gripper_points(self):
        assert self.R_base2world is not None
        assert self.t_base2world is not None
        R_gripper2base, t_gripper2base = self.get_robot_pose()
        R_gripper2world = self.R_base2world @ R_gripper2base
        t_gripper2world = self.R_base2world @ t_gripper2base + self.t_base2world
        gripper_points_in_world = R_gripper2world @ self.gripper_point_opened.T + t_gripper2world[:, np.newaxis]
        gripper_points_in_world = gripper_points_in_world.T
        return gripper_points_in_world


if __name__ == "__main__":

    use_robot = True
    exposure_time = 5
    env = RealEnv(
        WH=[1280, 720],
        obs_fps=15,
        n_obs_steps=2,
        use_robot=use_robot,
    )

    try:
        env.start(exposure_time=exposure_time)
        if use_robot:
            env.reset_robot()
        print('env started')
        time.sleep(exposure_time)
        print('start recording')

        env.calibrate(re_calibrate=False)

        obs = env.get_obs()
        
        pose = [569,-155.6,245.8,179.5,3.7,49.7]
        env.set_robot_pose(pose, wait=False)
        while True:
            gripper_position = env.get_gripper_position()
            print(f'\r{gripper_position.mean(0)}', end='')
            sys.stdout.flush()

    finally:
        env.stop()
        print('env stopped')
