import numpy as np
import open3d as o3d


def rpy_to_rotation_matrix(roll, pitch, yaw):
    # Assume the input in in degree
    roll = roll / 180 * np.pi
    pitch = pitch / 180 * np.pi
    yaw = yaw / 180 * np.pi
    # Define the rotation matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    # Combine the rotations
    R = Rz @ Ry @ Rx
    return R


def depth2fgpcd(depth, intrinsic_matrix):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth = depth.reshape(-1)
    points = np.stack([x, y, np.ones_like(x)], axis=1)
    points = points * depth[:, None]
    points = points @ np.linalg.inv(intrinsic_matrix).T
    return points


def visualize_o3d(geometries):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries.append(frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector()
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in geometries:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    # opt.show_coordinate_frame = True
    opt.background_color = np.asarray([1., 1., 1.])
    viewer.run()
    # viewer.destroy_window()