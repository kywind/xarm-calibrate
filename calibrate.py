import cv2
import time
import numpy as np
import open3d as o3d

from real_world.real_env import RealEnv
from real_world.utils import depth2fgpcd, rpy_to_rotation_matrix, visualize_o3d


def get_tabletop_points(rgb_list, depth_list, R_list, t_list, intr_list, bbox, depth_threshold=[0, 2]):

    pcd_all = o3d.geometry.PointCloud()
    point_colors = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1]]

    for i in range(len(rgb_list)):
        intr = intr_list[i]
        R_cam2board = R_list[i]
        t_cam2board = t_list[i]

        depth = depth_list[i].copy().astype(np.float32)

        points = depth2fgpcd(depth, intr)
        points = points.reshape(depth.shape[0], depth.shape[1], 3)
        points = points[::4, ::4, :].reshape(-1, 3)

        mask = np.logical_and(
            (depth > depth_threshold[0]), (depth < depth_threshold[1])
        )  # (H, W)
        mask = mask[::4, ::4].reshape(-1)

        img = rgb_list[i].copy()
        points = points[mask].reshape(-1, 3)

        points = R_cam2board @ points.T + t_cam2board[:, None]
        points = points.T  # (N, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(np.array(point_colors[i]) * np.ones((points.shape[0], 3)))

        # cv2.imwrite(f'vis/{i}re_calibrate_img.png', img)

        colors = img[::4, ::4, :].reshape(-1, 3).astype(np.float64)
        colors = colors[mask].reshape(-1, 3)
        colors = colors[:, ::-1].copy()
        pcd.colors = o3d.utility.Vector3dVector(colors / 255)

        pcd_all += pcd
    
    # crop axis-aligned bounding box
    pcd_all = pcd_all.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[:, 0], max_bound=bbox[:, 1]))

    return pcd_all


def main():
    use_robot = True  # whether the robot is connected. Set to False for cam-only env
    calibrate = True  # whether to calibrate the camera. If false, use the pre-calibrated camera parameters
    use_wrist_cam = True  # whether the wrist camera is connected. When doing hand-eye calibration, must be True
    gripper_enable = True  # whether to use the gripper. If changed to other end effectors, should be False
    num_fixed_cams = 4  # the number of fixed cameras connected to the computer

    env = RealEnv(
        WH=[1280, 720],
        obs_fps=15,
        n_obs_steps=2,
        use_robot=use_robot,
        use_wrist_cam=use_wrist_cam,
        num_fixed_cams=num_fixed_cams,
        gripper_enable=gripper_enable,
    )
    exposure_time = 5

    try:
        env.start(exposure_time=exposure_time)
        if use_robot:
            env.reset_robot()
        print('env started')
        time.sleep(exposure_time)
        print('start recording')

        env.calibrate(re_calibrate=calibrate)

        obs = env.get_obs(get_color=True, get_depth=True)
        intr_list = env.get_intrinsics()
        R_list, t_list = env.get_extrinsics()
        bbox = env.get_bbox()

        rgb_list = []
        depth_list = []
        for i in range(num_fixed_cams):
            rgb = obs[f'color_{i}'][-1]
            depth = obs[f'depth_{i}'][-1]
            rgb_list.append(rgb)
            depth_list.append(depth)

        pcd = get_tabletop_points(rgb_list, depth_list, R_list, t_list, intr_list, bbox)

        # env.set_robot_pose([443.7, -88.9, 119.7, 179.2, 0., 0.3])

        pcd_eef = o3d.geometry.PointCloud()
        if use_robot and gripper_enable:
            eef_points = env.get_gripper_points()
            pcd_eef.points = o3d.utility.Vector3dVector(eef_points)
            pcd_eef.paint_uniform_color([1, 0, 0])

        visualize_o3d([pcd, pcd_eef])

    finally:
        env.stop()
        print('env stopped')


if __name__ == "__main__":
    main()
