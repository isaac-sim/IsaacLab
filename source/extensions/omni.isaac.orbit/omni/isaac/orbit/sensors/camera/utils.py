# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Helper functions to project between pointcloud and depth images."""


import numpy as np
import torch
from typing import Optional, Sequence, Tuple, Union

import warp as wp

from omni.isaac.orbit.utils.array import convert_to_torch
from omni.isaac.orbit.utils.math import matrix_from_quat

__all__ = ["transform_points", "create_pointcloud_from_depth", "create_pointcloud_from_rgbd"]


"""
Depth <-> Pointcloud conversions.
"""


def transform_points(
    points: Union[np.ndarray, torch.Tensor, wp.array],
    position: Optional[Sequence[float]] = None,
    orientation: Optional[Sequence[float]] = None,
    device: Union[torch.device, str, None] = None,
) -> Union[np.ndarray, torch.Tensor]:
    r"""Transform input points in a given frame to a target frame.

    This function uses torch operations to transform points from a source frame to a target frame. The
    transformation is defined by the position ``t`` and orientation ``R`` of the target frame in the source frame.

    .. math::
        p_{target} = R_{target} \times p_{source} + t_{target}

    If either the inputs `position` and `orientation` are :obj:`None`, the corresponding transformation is not applied.

    Args:
        points (Union[np.ndarray, torch.Tensor, wp.array]): An array of shape (N, 3) comprising of 3D points in source frame.
        position (Optional[Sequence[float]], optional): The position of source frame in target frame. Defaults to None.
        orientation (Optional[Sequence[float]], optional): The orientation ``(w, x, y, z)`` of source frame in target frame.
            Defaults to None.
        device (Optional[Union[torch.device, str]], optional): The device for torch where the computation
            should be executed. Defaults to None, i.e. takes the device that matches the depth image.

    Returns:
        Union[np.ndarray, torch.Tensor]:
          A tensor of shape (N, 3) comprising of 3D points in target frame.
          If the input is a numpy array, the output is a numpy array. Otherwise, it is a torch tensor.
    """
    # check if numpy
    is_numpy = isinstance(points, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert to torch
    points = convert_to_torch(points, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = points.device
    # apply rotation
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # apply translation
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    # apply transformation
    points = _transform_points_jit(points, position, orientation)

    # return everything according to input type
    if is_numpy:
        return points.detach().cpu().numpy()
    else:
        return points


def create_pointcloud_from_depth(
    intrinsic_matrix: Union[np.ndarray, torch.Tensor, wp.array],
    depth: Union[np.ndarray, torch.Tensor, wp.array],
    keep_invalid: bool = False,
    position: Optional[Sequence[float]] = None,
    orientation: Optional[Sequence[float]] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    r"""Creates pointcloud from input depth image and camera intrinsic matrix.

    This function creates a pointcloud from a depth image and camera intrinsic matrix. The pointcloud is
    computed using the following equation:

    .. math::
        p_{camera} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`K` is the camera intrinsic matrix, :math:`u` and :math:`v` are the pixel coordinates and
    :math:`d` is the depth value at the pixel.

    Additionally, the pointcloud can be transformed from the camera frame to a target frame by providing
    the position ``t`` and orientation ``R`` of the camera in the target frame:

    .. math::
        p_{target} = R_{target} \times p_{camera} + t_{target}

    Args:
        intrinsic_matrix (Union[np.ndarray, torch.Tensor, wp.array]): A (3, 3) array providing camera's calibration
            matrix.
        depth (Union[np.ndarray, torch.Tensor, wp.array]): An array of shape (H, W) with values encoding the depth
            measurement.
        keep_invalid (bool, optional): Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position (Optional[Sequence[float]], optional): The position of the camera in a target frame.
            Defaults to None.
        orientation (Optional[Sequence[float]], optional): The orientation ``(w, x, y, z)`` of the
            camera in a target frame. Defaults to None.
        device (Optional[Union[torch.device, str]], optional): The device for torch where the computation
            should be executed. Defaults to None, i.e. takes the device that matches the depth image.

    Raises:
        ValueError: When intrinsic matrix is not of shape (3, 3).
        ValueError: When depth image is not of shape (H, W) or (H, W, 1).

    Returns:
        Union[np.ndarray, torch.Tensor]:
          An array/tensor of shape (N, 3) comprising of 3D coordinates of points.
          The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
          is returned.
    """
    # We use PyTorch here for matrix multiplication since it is compiled with Intel MKL while numpy
    # by default uses OpenBLAS. With PyTorch (CPU), we could process a depth image of size (480, 640)
    # in 0.0051 secs, while with numpy it took 0.0292 secs.

    # convert to numpy matrix
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    depth = convert_to_torch(depth, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = depth.device
    # convert inputs to torch tensors
    intrinsic_matrix = convert_to_torch(intrinsic_matrix, dtype=torch.float32, device=device)
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # compute pointcloud
    depth_cloud = _create_pointcloud_from_depth_jit(intrinsic_matrix, depth, keep_invalid, position, orientation)

    # return everything according to input type
    if is_numpy:
        return depth_cloud.detach().cpu().numpy()
    else:
        return depth_cloud


def create_pointcloud_from_rgbd(
    intrinsic_matrix: Union[torch.Tensor, np.ndarray, wp.array],
    depth: Union[torch.Tensor, np.ndarray, wp.array],
    rgb: Union[torch.Tensor, wp.array, np.ndarray, Tuple[float, float, float]] = None,
    normalize_rgb: bool = False,
    position: Optional[Sequence[float]] = None,
    orientation: Optional[Sequence[float]] = None,
    device: Optional[Union[torch.device, str]] = None,
    num_channels: int = 3,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
    """Creates pointcloud from input depth image and camera transformation matrix.

    This function provides the same functionality as :meth:`create_pointcloud_from_depth` but also allows
    to provide the RGB values for each point.

    The ``rgb`` attribute is used to resolve the corresponding point's color:

    - If a ``np.array``/``wp.arrray``/``torch.tensor`` of shape (H, W, 3), then the corresponding channels encode RGB values.
    - If a tuple, then the point cloud has a single color specified by the values (r, g, b).
    - If :obj:`None`, then default color is white, i.e. (0, 0, 0).

    If the input ``normalize_rgb`` is set to :obj:`True`, then the RGB values are normalized to be in the range [0, 1].

    Args:
        intrinsic_matrix (Union[torch.Tensor, np.ndarray, wp.array]): A (3, 3) array/tensor providing camera's
            calibration matrix.
        depth (Union[torch.Tensor, np.ndarray, wp.array]): An array/tensor of shape (H, W) with values encoding
            the depth measurement.
        rgb (Union[np.ndarray, Tuple[float, float, float]], optional): Color for generated point cloud.
            Defaults to None.
        normalize_rgb (bool, optional): Whether to normalize input rgb. Defaults to False.
        position (Optional[Sequence[float]], optional): The position of the camera in a target frame.
            Defaults to None.
        orientation (Optional[Sequence[float]], optional): The orientation `(w, x, y, z)` of the
            camera in a target frame. Defaults to None.
        device (Optional[Union[torch.device, str]], optional): The device for torch where the computation
            should be executed. Defaults to None, i.e. takes the device that matches the depth image.
        num_channels (int, optional): Number of channels in RGB pointcloud. Defaults to 3.

    Raises:
        ValueError:  When rgb image is a numpy array but not of shape (H, W, 3) or (H, W, 4).

    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
          A tuple of (N, 3) arrays or tensors containing the 3D coordinates of points and their RGB color respectively.
          The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
          is returned.
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

    # check if input depth is numpy array
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    if is_numpy:
        depth = torch.from_numpy(depth).to(device=device)
    # retrieve XYZ pointcloud
    points_xyz = create_pointcloud_from_depth(intrinsic_matrix, depth, True, position, orientation, device=device)

    # get image height and width
    im_height, im_width = depth.shape[:2]
    # total number of points
    num_points = im_height * im_width
    # extract color value
    if rgb is not None:
        if isinstance(rgb, (np.ndarray, torch.Tensor, wp.array)):
            # copy numpy array to preserve
            rgb = convert_to_torch(rgb, device=device, dtype=torch.float32)
            rgb = rgb[:, :, :3]
            # convert the matrix to (W, H, 3) from (H, W, 3) since depth processing
            # is done in the order (u, v) where u: (0, W-1) and v: (0 - H-1)
            points_rgb = rgb.permute(1, 0, 2).reshape(-1, 3)
        elif isinstance(rgb, Tuple):
            # same color for all points
            points_rgb = torch.Tensor((rgb,) * num_points, device=device, dtype=torch.uint8)
        else:
            # default color is white
            points_rgb = torch.Tensor(((0, 0, 0),) * num_points, device=device, dtype=torch.uint8)
    else:
        points_rgb = torch.Tensor(((0, 0, 0),) * num_points, device=device, dtype=torch.uint8)
    # normalize color values
    if normalize_rgb:
        points_rgb = points_rgb.float() / 255

    # remove invalid points
    pts_idx_to_keep = torch.all(torch.logical_and(~torch.isnan(points_xyz), ~torch.isinf(points_xyz)), dim=1)
    points_rgb = points_rgb[pts_idx_to_keep, ...]
    points_xyz = points_xyz[pts_idx_to_keep, ...]

    # add additional channels if required
    if num_channels == 4:
        points_rgb = torch.nn.functional.pad(points_rgb, (0, 1), mode="constant", value=1.0)

    # return everything according to input type
    if is_numpy:
        return points_xyz.cpu().numpy(), points_rgb.cpu().numpy()
    else:
        return points_xyz, points_rgb


"""
Helper functions -- Internal
"""


@torch.jit.script
def _transform_points_jit(
    points: torch.Tensor,
    position: Optional[torch.Tensor] = None,
    orientation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transform input points in a given frame to a target frame.

    Args:
        points (torch.Tensor): An array of shape (N, 3) comprising of 3D points in source frame.
        position (Optional[torch.Tensor], optional): The position of source frame in target frame. Defaults to None.
        orientation (Optional[torch.Tensor], optional): The orientation ``(w, x, y, z)`` of source frame in target frame.
            Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape (N, 3) comprising of 3D points in target frame.
    """
    # -- apply rotation
    if orientation is not None:
        points = torch.matmul(matrix_from_quat(orientation), points.T).T
    # -- apply translation
    if position is not None:
        points += position

    return points


@torch.jit.script
def _create_pointcloud_from_depth_jit(
    intrinsic_matrix: torch.Tensor,
    depth: torch.Tensor,
    keep_invalid: bool = False,
    position: Optional[torch.Tensor] = None,
    orientation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Creates pointcloud from input depth image and camera intrinsic matrix.

    Args:
        intrinsic_matrix (torch.Tensor): A (3, 3) python tensor providing camera's calibration matrix.
        depth (torch.tensor): An tensor of shape (H, W) with values encoding the depth measurement.
        keep_invalid (bool, optional): Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position (torch.Tensor, optional): The position of the camera in a target frame. Defaults to None.
        orientation (torch.Tensor, optional): The orientation ``(w, x, y, z)`` of the camera in a target frame.
            Defaults to None.

    Raises:
        ValueError: When intrinsic matrix is not of shape (3, 3).
        ValueError: When depth image is not of shape (H, W) or (H, W, 1).

    Returns:
        torch.Tensor: A tensor of shape (N, 3) comprising of 3D coordinates of points.

    """
    # squeeze out excess dimension
    if len(depth.shape) == 3:
        depth = depth.squeeze(dim=2)
    # check shape of inputs
    if intrinsic_matrix.shape != (3, 3):
        raise ValueError(f"Input intrinsic matrix of invalid shape: {intrinsic_matrix.shape} != (3, 3).")
    if len(depth.shape) != 2:
        raise ValueError(f"Input depth image not two-dimensional. Received shape: {depth.shape}.")
    # get image height and width
    im_height, im_width = depth.shape

    # convert image points into list of shape (3, H x W)
    indices_u = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    indices_v = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    img_indices = torch.stack(torch.meshgrid([indices_u, indices_v], indexing="ij"), dim=0).reshape(2, -1)
    pixels = torch.nn.functional.pad(img_indices, (0, 0, 0, 1), mode="constant", value=1.0)

    # convert into 3D points
    points = torch.matmul(torch.inverse(intrinsic_matrix), pixels)
    points = points / points[-1, :]
    points_xyz = points * depth.T.reshape(-1)
    # convert it to (H x W , 3)
    points_xyz = torch.transpose(points_xyz, dim0=0, dim1=1)
    # convert 3D points to world frame
    points_xyz = _transform_points_jit(points_xyz, position, orientation)

    # remove points that have invalid depth
    if not keep_invalid:
        pts_idx_to_keep = torch.all(torch.logical_and(~torch.isnan(points_xyz), ~torch.isinf(points_xyz)), dim=1)
        points_xyz = points_xyz[pts_idx_to_keep, ...]

    return points_xyz  # noqa: D504
