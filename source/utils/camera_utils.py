import pdb

import cv2
import numpy as np
import scipy.optimize as opt
import scipy
import torch
from scipy.spatial.distance import cdist
import open3d as o3d

def build_depth_from_pointcloud(pointcloud, matrix_world_to_camera, imsize):
    height, width = imsize
    pointcloud = np.concatenate([pointcloud, np.ones((len(pointcloud), 1))], axis=1)  # n x 4
    camera_coordinate = matrix_world_to_camera @ pointcloud.T  # 3 x n
    camera_coordinate = camera_coordinate.T  # n x 3
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
    u = np.rint((x * fx / depth + u0).astype("int"))
    v = np.rint((y * fy / depth + v0).astype("int"))

    us = u.flatten()
    vs = v.flatten()
    depth = depth.flatten()

    depth_map = dict()
    for u, v, d in zip(us, vs, depth):
        if depth_map.get((u, v)) is None:
            depth_map[(u, v)] = []
            depth_map[(u, v)].append(d)
        else:
            depth_map[(u, v)].append(d)

    depth_2d = np.zeros((height, width))
    for u in range(width):
        for v in range(height):
            if (u, v) in depth_map.keys():
                depth_2d[v][u] = np.min(depth_map[(u, v)])

    return depth_2d


def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    fov is fovy, which corresponds to the vertical fov
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    vfov = np.deg2rad(fov)
    fy = py / np.tan(vfov / 2.)

    hfov = 2. * np.arctan(np.tan(vfov / 2) * width / height)
    fx = px / np.tan(hfov / 2.)

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])


def get_rotation_matrix(angle, axis):
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle)
    c = np.cos(angle)

    m = np.zeros((4, 4))

    m[0][0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c
    m[0][1] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
    m[0][2] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
    m[0][3] = 0.0

    m[1][0] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
    m[1][1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c
    m[1][2] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
    m[1][3] = 0.0

    m[2][0] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
    m[2][1] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
    m[2][2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c
    m[2][3] = 0.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 0.0
    m[3][3] = 1.0

    return m


def get_intrinsics_mujoco(renderer, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    fov is fovy, which corresponds to the vertical fov
    Returns:
        K:      [4, 4]
    """
    focal_scaling = (1. / np.tan(np.deg2rad(fov) / 2)) * renderer.height / 2.0
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (renderer.width - 1) / 2.0
    image[1, 2] = (renderer.height - 1) / 2.0
    return image @ focal


def compute_camera_matrix_mujoco(renderer, model):
    """Returns the 3x4 camera matrix."""
    # If the camera is a 'free' camera, we get its position and orientation
    # from the scene data structure. It is a stereo camera, so we average over
    # the left and right channels. Note: we call `self.update()` in order to
    # ensure that the contents of `scene.camera` are correct.
    # renderer.update_scene(data)
    pos = np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
    z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
    y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
    rot = np.vstack((np.cross(y, z), y, z))
    fov = model.vis.global_.fovy
    # Translation matrix (4x4).
    translation = np.eye(4)
    translation[0:3, 3] = -pos

    # Rotation matrix (4x4).
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot
    # print(translation)
    # print(rotation)

    # Focal transformation matrix (3x4).
    focal_scaling = (1. / np.tan(np.deg2rad(fov) / 2)) * renderer.height / 2.0
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (renderer.width - 1) / 2.0
    image[1, 2] = (renderer.height - 1) / 2.0
    # print("ext ", rotation@translation)
    return image @ focal @ rotation @ translation


def project_to_image_mujoco(renderer, model, points):
    """Projects a set of 3D points to the image plane."""
    camera_matrix = compute_camera_matrix_mujoco(renderer, model)
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = points @ camera_matrix.T
    points = points[:, 0:2] / points[:, 2:3]
    return points


def get_world_coords(rgb, depth, env, particle_pos=None):
    height, width, _ = rgb.shape
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

    # Apply back-projection: K_inv @ pixels * depth
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x = np.linspace(0, width - 1, width).astype(np.float)
    y = np.linspace(0, height - 1, height).astype(np.float)
    u, v = np.meshgrid(x, y)
    one = np.ones((height, width, 1))
    x = (u - u0) * depth / fx
    y = (v - v0) * depth / fy
    z = depth
    cam_coords = np.dstack([x, y, z, one])

    matrix_world_to_camera = get_matrix_world_to_camera(
        env.camera_params[env.camera_name]['pos'], env.camera_params[env.camera_name]['angle'])

    # convert the camera coordinate back to the world coordinate using the rotation and translation matrix
    cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)
    world_coords = np.linalg.inv(matrix_world_to_camera) @ cam_coords  # 4 x (height x width)
    world_coords = world_coords.transpose().reshape((height, width, 4))

    return world_coords


def get_observable_particle_index(world_coords, particle_pos, rgb, depth):
    height, width, _ = rgb.shape
    # perform the matching of pixel particle to real particle
    particle_pos = particle_pos[:, :3]

    estimated_world_coords = np.array(world_coords)[np.where(depth > 0)][:, :3]

    distance = scipy.spatial.distance.cdist(estimated_world_coords, particle_pos)
    # Each point in the point cloud will cover at most two particles. Particles not covered will be deemed occluded
    estimated_particle_idx = np.argpartition(distance, 2)[:, :2].flatten()
    estimated_particle_idx = np.unique(estimated_particle_idx)

    return np.array(estimated_particle_idx, dtype=np.int32)


def get_observable_particle_index_old(world_coords, particle_pos, rgb, depth):
    height, width, _ = rgb.shape
    # perform the matching of pixel particle to real particle
    particle_pos = particle_pos[:, :3]

    estimated_world_coords = np.array(world_coords)[np.where(depth > 0)][:, :3]

    distance = scipy.spatial.distance.cdist(estimated_world_coords, particle_pos)
    estimated_particle_idx = np.argmin(distance, axis=1)
    estimated_particle_idx = np.unique(estimated_particle_idx)

    return np.array(estimated_particle_idx, dtype=np.int32)


def get_observable_particle_index_3(pointcloud, mesh, threshold=0.0216):
    ### bi-partite graph matching
    distance = scipy.spatial.distance.cdist(pointcloud, mesh)
    distance[distance > threshold] = 1e10
    row_idx, column_idx = opt.linear_sum_assignment(distance)

    distance_mapped = distance[np.arange(len(pointcloud)), column_idx]
    bad_mapping = distance_mapped > threshold
    if np.sum(bad_mapping) > 0:
        column_idx[bad_mapping] = np.argmin(distance[bad_mapping], axis=1)

    return pointcloud, column_idx


def get_mapping_from_pointcloud_to_partile_nearest_neighbor(pointcloud, particle):
    distance = scipy.spatial.distance.cdist(pointcloud, particle)
    nearest_idx = np.argmin(distance, axis=1)
    return nearest_idx


def get_observable_particle_index_4(pointcloud, mesh, threshold=0.0216):
    # perform the matching of pixel particle to real particle
    estimated_world_coords = pointcloud

    distance = scipy.spatial.distance.cdist(estimated_world_coords, mesh)
    estimated_particle_idx = np.argmin(distance, axis=1)

    return pointcloud, np.array(estimated_particle_idx, dtype=np.int32)


def get_observable_particle_pos(world_coords, particle_pos):
    # perform the matching of pixel particle to real particle
    particle_pos = particle_pos[:, :3]
    distance = scipy.spatial.distance.cdist(world_coords, particle_pos)
    estimated_particle_idx = np.argmin(distance, axis=1)
    observable_particle_pos = particle_pos[estimated_particle_idx]

    return observable_particle_pos


def get_matrix_world_to_camera(cam_pos=[-0.0, 0.82, 0.82], cam_angle=[0, -45 / 180. * np.pi, 0.]):
    cam_x, cam_y, cam_z = cam_pos[0], cam_pos[1], \
        cam_pos[2]
    cam_x_angle, cam_y_angle, cam_z_angle = cam_angle[0], cam_angle[1], \
        cam_angle[2]

    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0])
    matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # get translation matrix: from world to camera
    translation_matrix = np.zeros((4, 4))
    translation_matrix[0][0] = 1
    translation_matrix[1][1] = 1
    translation_matrix[2][2] = 1
    translation_matrix[3][3] = 1
    translation_matrix[0][3] = - cam_x
    translation_matrix[1][3] = - cam_y
    translation_matrix[2][3] = - cam_z

    return rotation_matrix @ translation_matrix


def get_matrix_world_to_camera_mujoco(renderer):
    # print(f'{len(renderer.scene.camera)} cameras in the renderer')
    pos = np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
    z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
    y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
    rot = np.vstack((np.cross(y, z), y, z))

    translation = np.eye(4)
    translation[0:3, 3] = -pos

    # Rotation matrix (4x4).
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot
    return rotation @ translation


def is_in_frame(pts, matrix_world_to_camera, image_size):
    u, v = project_to_image(matrix_world_to_camera, pts, image_size[0], image_size[1], is_mujoco=True)
    if 0 < u < image_size[0] and 0 < v < image_size[1]:
        return True
    else:
        return False


def project_to_image(matrix_world_to_camera, world_coordinate, height=360, width=360, fovy=45,
                     is_mujoco=False):
    world_coordinate = np.concatenate([world_coordinate, np.ones((len(world_coordinate), 1))],
                                      axis=1)  # n x 4

    camera_coordinate = matrix_world_to_camera @ world_coordinate.T  # 3 x n
    camera_coordinate = camera_coordinate.T  # n x 3
    # print("matrix_world_to_camera ", matrix_world_to_camera)
    # print("world_coordinate ", world_coordinate.T)
    # print("cam coord1", camera_coordinate)

    K = intrinsic_from_fov(height, width, fovy)  # the fov is 90 degrees
    # print("int 2", K)
    # print("Ext 2", matrix_world_to_camera)
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
    if is_mujoco:
        fx = -fx
    u = (x * fx / depth + u0).astype("int")
    v = (y * fy / depth + v0).astype("int")
    # print(f"{u} = {x} * {fx} / {depth} + {u0}")
    return u, v


def uv_to_world_pos(u, v, z, K, matrix_world_to_camera, is_mujoco=False):
    x0 = K[0, 2]
    y0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    one = np.ones(u.shape, np.float32)
    if is_mujoco:
        fx = -fx
        z = - z

    x = (v - x0) * z / fx
    # print(f"{x} = {v} - {x0} * {z} / {fx}")
    y = (u - y0) * z / fy
    cam_coords = np.stack([x, y, z, one], axis=1)
    # print("cam coords2 ", cam_coords, cam_coords.shape)
    cam2world = np.linalg.inv(matrix_world_to_camera).T
    # print("cam2world ", cam2world)
    world_coords = cam_coords @ cam2world
    return world_coords[:, :3].astype(np.float32)


def get_pointcloud(depth, matrix_world_to_camera, get_img=False, is_mujoco=False,
                   mask=None, nb_points=5, nb_radius=0.03):
    height, width = depth.shape
    if mask is not None:
        depth = depth * mask
    u, v = depth.nonzero()
    z = depth[u, v]
    K = intrinsic_from_fov(height, width, 45).astype(np.float32)  # the fov is 90 degrees
    pc = uv_to_world_pos(u, v, z, K, matrix_world_to_camera, is_mujoco=is_mujoco)
    if len(pc)>0:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
        pc, _ = pc.remove_radius_outlier(nb_points=nb_points, radius=nb_radius)
        pc = np.array(pc.points)
    if get_img:
        img_pc = np.zeros((height, width, 3), dtype=np.float32)
        img_pc[u, v] = pc
        return pc, img_pc
    return pc


def get_mask(rgb, low=None, high=None, k_c=2, k_o=2, k_e=0, valid_mask=None):
    if high is None:
        high = [240., 251., 255.]
    if low is None:
        low = [34., 53., 0.]
    if not isinstance(high, np.ndarray):
        high = np.array(high)
    if not isinstance(low, np.ndarray):
        low = np.array(low)
    high = high.copy()
    low = low.copy()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, low, high)  # result is from 0-255

    mask = np.array(mask > 0).astype(np.uint8)

    if k_c > 0:
        kernel_close = np.ones((k_c, k_c), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    if k_o > 0:
        kernel_open = np.ones((k_o, k_o), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    if k_e > 0:
        kernel_erode = np.ones((k_e, k_e), np.uint8)
        mask = cv2.erode(mask, kernel_erode, iterations=1)
    if valid_mask is not None:
        mask = valid_mask * mask
    return mask


def _get_depth(matrix, vec, height):
    """ Get the depth such that the back-projected point has a fixed height"""
    return (height - matrix[1, 3]) / (vec[0] * matrix[1, 0] + vec[1] * matrix[1, 1] + matrix[1, 2])


def get_world_coor_from_image(u, v, image_size, matrix_world_to_camera, all_depth):
    height, width = image_size
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

    matrix = np.linalg.inv(matrix_world_to_camera)

    u0, v0, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]

    depth = all_depth[v][u]
    if depth == 0:
        vec = ((u - u0) / fx, (v - v0) / fy)
        depth = _get_depth(matrix, vec, 0.00625)  # Height to be the particle radius

    x = (u - u0) * depth / fx
    y = (v - v0) * depth / fy
    z = depth
    cam_coords = np.array([x, y, z, 1])
    cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)

    world_coord = matrix @ cam_coords  # 4 x (height x width)
    world_coord = world_coord.reshape(4)
    return world_coord[:3]


def get_target_pos(pos, u, v, image_size, matrix_world_to_camera, depth):
    coor = get_world_coor_from_image(u, v, image_size, matrix_world_to_camera, depth)
    dists = cdist(coor[None], pos)[0]
    idx = np.argmin(dists)
    return pos[idx] + np.array([0, 0.01, 0])


def check_if_oob(pts, mat_world2cam, height, width, fovy, is_mujoco=True):
    pts = pts.copy().T
    corners_homogeneous = np.ones((4, pts.shape[1]), dtype=float)
    corners_homogeneous[:3, :] = pts
    u, v = project_to_image(mat_world2cam, corners_homogeneous[:3, :].T,
                            height=height, width=width,
                            fovy=fovy, is_mujoco=is_mujoco)
    # return u,v
    if np.any(u < 0) or np.any(u > height) or np.any(v < 0) or np.any(v > width):
        return True
    return False
