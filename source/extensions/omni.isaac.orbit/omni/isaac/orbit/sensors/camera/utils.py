# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions to project between pointcloud and depth images."""


import numpy as np
import scipy.spatial.transform as tf
import torch
from typing import Sequence, Tuple, Union

from omni.isaac.orbit.utils.math import convert_quat


def transform_pointcloud(
    points: Union[np.ndarray, torch.Tensor], position: Sequence[float] = None, orientation: Sequence[float] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Transform input points in a given frame to a target frame.

    Args:
        points (Union[np.ndarray, torch.Tensor]): An array of shape (N, 3) comprising of 3D points in source frame.
        position (Sequence[float], optional): The position of source frame in target frame. Defaults to None.
        orientation (Sequence[float], optional): The orientation `(w, x, y, z)` of source frame in target frame.
            Defaults to None.

    Returns:
        Union[np.ndarray, torch.Tensor]: An array of shape (N, 3) comprising of 3D points in target frame.
    """
    # device for carrying out conversion
    device = "cpu"
    is_torch = True
    # convert to torch
    if not isinstance(points, torch.Tensor):
        is_torch = False
        points = torch.from_numpy(points).to(device)
    # rotate points
    if orientation is not None:
        # convert to numpy (sanity)
        orientation = np.asarray(orientation)
        # convert using scipy to simplify life
        rot = tf.Rotation.from_quat(convert_quat(orientation, "xyzw"))
        rot_matrix = torch.from_numpy(rot.as_matrix()).to(device)
        # apply rotation
        points = torch.matmul(rot_matrix, points.T).T
    # translate points
    if position is not None:
        # convert to torch  to simplify life
        position = torch.from_numpy(position).to(device)
        # apply translation
        points += position
    # return results
    if is_torch:
        return points
    else:
        return points.detach().cpu().numpy()


def create_pointcloud_from_depth(
    intrinsic_matrix: np.ndarray,
    depth: np.ndarray,
    keep_invalid: bool = False,
    position: Sequence[float] = None,
    orientation: Sequence[float] = None,
) -> np.ndarray:
    """Creates pointcloud from input depth image and camera intrinsic matrix.

    If the inputs `camera_position` and `camera_orientation` are provided, the pointcloud is transformed
    from the camera frame to the target frame.

    We use PyTorch here for matrix multiplication since it is compiled with Intel MKL while numpy
    by default uses OpenBLAS. With PyTorch (CPU), we could process a depth image of size (480, 640)
    in 0.0106 secs, while with numpy it took 0.0292 secs.

    Args:
        intrinsic_matrix (np.ndarray): A (3, 3) numpy array providing camera's calibration matrix.
        depth (np.ndarray): An array of shape (H, W) with values encoding the depth measurement.
        keep_invalid (bool, optional): Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position (Sequence[float], optional): The position of the camera in a target frame. Defaults to None.
        orientation (Sequence[float], optional): The orientation `(w, x, y, z)` of the camera in a target frame.
            Defaults to None.

    Raises:
        ValueError: When intrinsic matrix is not of shape (3, 3).
        ValueError: When depth image is not of shape (H, W) or (H, W, 1).

    Returns:
        np.ndarray: An array of shape (N, 3) comprising of 3D coordinates of points.

    """
    # device for carrying out conversion
    device = "cpu"
    # convert to numpy matrix
    intrinsic_matrix = np.asarray(intrinsic_matrix)
    depth = np.asarray(depth).copy()
    # squeeze out excess dimension
    if len(depth.shape) == 3:
        depth = depth.squeeze(axis=2)
    # check shape of inputs
    if intrinsic_matrix.shape != (3, 3):
        raise ValueError(f"Input intrinsic matrix of invalid shape: {intrinsic_matrix.shape} != (3, 3).")
    if len(depth.shape) != 2:
        raise ValueError(f"Input depth image not two-dimensional. Received shape: {depth.shape}.")
    # convert inputs to numpy arrays
    intrinsic_matrix = torch.from_numpy(intrinsic_matrix).to(device)
    depth = torch.from_numpy(depth).to(device)
    # get image height and width
    im_height, im_width = depth.shape
    # convert image points into list of shape (3, H x W)
    img_indices = np.indices((im_width, im_height)).reshape(2, -1)
    pixels = np.pad(img_indices, [(0, 1), (0, 0)], mode="constant", constant_values=1.0)
    pixels = torch.tensor(pixels, dtype=torch.double, device=device)
    # convert into 3D points
    points = torch.matmul(torch.inverse(intrinsic_matrix), pixels)
    points = points / points[-1, :]
    points_xyz = points * depth.T.reshape(-1)
    # convert it to (H x W , 3)
    points_xyz = torch.transpose(points_xyz, dim0=0, dim1=1)
    # convert 3D points to world frame
    if position is not None or orientation is not None:
        points_xyz = transform_pointcloud(points_xyz, position, orientation)
    # convert to numpy
    points_xyz = points_xyz.detach().cpu().numpy()
    # remove points that have invalid depth
    if not keep_invalid:
        invalid_points_idx = np.where(np.logical_or(np.isnan(points_xyz), np.isinf(points_xyz)))
        points_xyz = np.delete(points_xyz, invalid_points_idx, axis=0)

    return points_xyz


def create_pointcloud_from_rgbd(
    intrinsic_matrix: np.ndarray,
    depth: np.ndarray,
    rgb: Union[np.ndarray, Tuple[float, float, float]] = None,
    normalize_rgb: bool = False,
    position: Sequence[float] = None,
    orientation: Sequence[float] = None,
    num_channels: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates pointcloud from input depth image and camera transformation matrix.

    The `rgb` attribute is used to resolve the corresponding point's color:

        - If a numpy array of shape (H, W, 3) then the corresponding channels encode RGB values.
        - If a tuple then the point cloud has a single color specified by the values (r, g, b).
        - If None, then default color is white, i.e. (0, 0, 0).

    If the inputs `camera_position` and `camera_orientation` are provided, the pointcloud is transformed
    from the camera frame to the target frame.

    Args:
        intrinsic_matrix (np.ndarray): A (3, 3) numpy array providing camera's calibration matrix.
        depth (np.ndarray): An array of shape (H, W) with values encoding the depth measurement.
        rgb (Union[np.ndarray, Tuple[float, float, float]], optional): Color for generated point cloud.
            Defaults to None.
        normalize_rgb (bool, optional): Whether to normalize input rgb. Defaults to False.
        position (Sequence[float], optional): The position of the camera in a target frame. Defaults to None.
        orientation (Sequence[float], optional): The orientation `(w, x, y, z)` of the camera in a target frame.
            Defaults to None.
        num_channels (int, optional): Number of channels in RGB pointcloud. Defaults to 3.

    Raises:
        ValueError:  When rgb image is a numpy array but not of shape (H, W, 3) or (H, W, 4).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of (N, 3) numpy arrays containing 3D coordinates of
            points and their RGB color respectively.
    """
    # check valid inputs
    if rgb is not None and not isinstance(rgb, tuple):
        if len(rgb.shape) == 3:
            if rgb.shape[2] not in [3, 4]:
                raise ValueError(f"Input rgb image of invalid shape: {rgb.shape} != (H, W, 3) or (H, W, 4).")
        else:
            raise ValueError(f"Input rgb image not three-dimensional. Received shape: {rgb.shape}.")
    if num_channels not in [3, 4]:
        raise ValueError(f"Invalid number of channels: {num_channels} != 3 or 4.")
    # retrieve XYZ pointcloud
    points_xyz = create_pointcloud_from_depth(
        intrinsic_matrix, depth, position=position, orientation=orientation, keep_invalid=True
    )

    # get image height and width
    im_height, im_width = depth.shape[:2]
    # total number of points
    num_points = im_height * im_width
    # extract color value
    if rgb is not None:
        if isinstance(rgb, np.ndarray):
            # copy numpy array to preserve
            rgb = np.asarray(rgb, dtype="float").copy()
            rgb = rgb[:, :, :3]
            # convert the matrix to (W, H, 3) since depth processing
            # is done in the order (u, v) where u: 0 - W-1 and v: 0 - H-1
            points_rgb = np.reshape(rgb.transpose(1, 0, 2), (-1, 3))
        elif isinstance(rgb, Tuple):
            points_rgb = np.asarray((rgb,) * num_points, dtype=np.uint8)
        else:
            points_rgb = np.asarray(((0, 0, 0),) * num_points, dtype=np.uint8)
    else:
        points_rgb = np.asarray(((0, 0, 0),) * num_points, dtype=np.uint8)
    # normalize color values
    if normalize_rgb:
        points_rgb = np.asarray(points_rgb, dtype="float") / 255

    # remove invalid points
    invalid_points_idx = np.where(np.logical_or(np.isnan(points_xyz), np.isinf(points_xyz)))
    points_xyz = np.delete(points_xyz, invalid_points_idx, axis=0)
    points_rgb = np.delete(points_rgb, invalid_points_idx, axis=0)
    # add additional channels if required
    if num_channels == 4:
        points_rgb = np.pad(points_rgb, [(0, 0), (0, 1)], mode="constant", constant_values=1.0)

    return points_xyz, points_rgb
