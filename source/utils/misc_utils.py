import numpy as np

import os
from chester import logger
import random
# import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pytorch3d.structures import Pointclouds
import torch


def configure_seed(seed):
    # Configure seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


############### for planning ###############################


def set_resource():
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def find_3D_rigid_trans(pts1, pts2, return_quat=False):
    """
    Find the optimal rigid transformation between two point clouds
    pts1, pts2: N x 3
    """
    centroid_A = pts1.mean(0, keepdims=True)
    centroid_B = pts2.mean(0, keepdims=True)
    # centre the points
    AA = pts1 - centroid_A
    BB = pts2 - centroid_B
    # dot is matrix multiplication for array
    H = np.dot(np.transpose(AA), BB)
    U, S, Vt = np.linalg.svd(H)

    # special reflection case
    d = np.linalg.det(Vt.T @ U.T)
    m = np.eye(3)
    m[2, 2] = d
    R = Vt.T @ m @ U.T
    t = -np.dot(R, centroid_A.T) + centroid_B.T
    return R, t


def icp_rot_axis(pts1, pts2, axis_line, return_rot=False):
    u = axis_line[0] - axis_line[1]
    u = u.reshape(3, 1) / np.linalg.norm(u)
    # plot_pointclouds([pts1, pts2]).show()
    pts1 = pts1 - axis_line[1]
    pts2 = pts2 - axis_line[1]
    up = np.sum(np.cross(pts2, pts1) @ u)
    # up = np.cross(np.sum(pts2, 0), np.sum(pts1, 0)) @ u
    # pdb.set_trace()
    bottom = np.sum(pts1 * pts2) - np.sum((pts2 @ u) * (pts1 @ u))
    radian = - np.arctan2(up, bottom)
    if return_rot:
        return radian
    # print(radian, u)
    rotation = R.from_rotvec(u.flatten() * radian)
    # print(rotation.as_matrix())
    # print(axis_line)
    page = rotation.apply(pts1) + axis_line[1]
    loss = np.mean(np.linalg.norm(page - pts2, axis=1))
    return page, loss
