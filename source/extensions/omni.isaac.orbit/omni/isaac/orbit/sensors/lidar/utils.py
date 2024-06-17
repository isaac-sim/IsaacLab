# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Helper functions to project between pointcloud and depth images.
Note: These functions are taken from camera sensor utils.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import Literal
import omni.isaac.core.utils.stage as stage_utils
from pxr import UsdGeom
import omni.isaac.orbit.utils.math as math_utils



def convert_orientation_convention(
    orientation: torch.Tensor,
    origin: Literal["opengl", "ros", "world"] = "opengl",
    target: Literal["opengl", "ros", "world"] = "ros",
) -> torch.Tensor:
    r"""Converts a quaternion representing a rotation from one convention to another.

    In USD, the camera follows the ``"opengl"`` convention. Thus, it is always in **Y up** convention.
    This means that the camera is looking down the -Z axis with the +Y axis pointing up , and +X axis pointing right.
    However, in ROS, the camera is looking down the +Z axis with the +Y axis pointing down, and +X axis pointing right.
    Thus, the camera needs to be rotated by :math:`180^{\circ}` around the X axis to follow the ROS convention.

    .. math::

        T_{ROS} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} T_{USD}

    On the other hand, the typical world coordinate system is with +X pointing forward, +Y pointing left,
    and +Z pointing up. The camera can also be set in this convention by rotating the camera by :math:`90^{\circ}`
    around the X axis and :math:`-90^{\circ}` around the Y axis.

    .. math::

        T_{WORLD} = \begin{bmatrix} 0 & 0 & -1 & 0 \\ -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} T_{USD}

    Thus, based on their application, cameras follow different conventions for their orientation. This function
    converts a quaternion from one convention to another.

    Possible conventions are:

    - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
    - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
    - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

    Args:
        orientation: Quaternion of form `(w, x, y, z)` with shape (..., 4) in source convention
        origin: Convention to convert to. Defaults to "ros".
        target: Convention to convert from. Defaults to "opengl".

    Returns:
        Quaternion of form `(w, x, y, z)` with shape (..., 4) in target convention
    """
    if target == origin:
        return orientation.clone()

    # -- unify input type
    if origin == "ros":
        # convert from ros to opengl convention
        rotm = math_utils.matrix_from_quat(orientation)
        rotm[:, :, 2] = -rotm[:, :, 2]
        rotm[:, :, 1] = -rotm[:, :, 1]
        # convert to opengl convention
        quat_gl = math_utils.quat_from_matrix(rotm)
    elif origin == "world":
        # convert from world (x forward and z up) to opengl convention
        rotm = math_utils.matrix_from_quat(orientation)
        rotm = torch.matmul(
            rotm,
            math_utils.matrix_from_euler(
                torch.tensor([math.pi / 2, -math.pi / 2, 0], device=orientation.device), "XYZ"
            ),
        )
        # convert to isaac-sim convention
        quat_gl = math_utils.quat_from_matrix(rotm)
    else:
        quat_gl = orientation

    # -- convert to target convention
    if target == "ros":
        # convert from opengl to ros convention
        rotm = math_utils.matrix_from_quat(quat_gl)
        rotm[:, :, 2] = -rotm[:, :, 2]
        rotm[:, :, 1] = -rotm[:, :, 1]
        return math_utils.quat_from_matrix(rotm)
    elif target == "world":
        # convert from opengl to world (x forward and z up) convention
        rotm = math_utils.matrix_from_quat(quat_gl)
        rotm = torch.matmul(
            rotm,
            math_utils.matrix_from_euler(
                torch.tensor([math.pi / 2, -math.pi / 2, 0], device=orientation.device), "XYZ"
            ).T,
        )
        return math_utils.quat_from_matrix(rotm)
    else:
        return quat_gl.clone()


# @torch.jit.script
def create_rotation_matrix_from_view(
    eyes: torch.Tensor,
    targets: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    This function takes a vector ''eyes'' which specifies the location
    of the camera in world coordinates and the vector ''targets'' which
    indicate the position of the object.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

        The inputs camera_position and targets can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    Args:
        eyes: position of the camera in world coordinates
        targets: position of the object in world coordinates

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices

    Reference:
    Based on PyTorch3D (https://github.com/facebookresearch/pytorch3d/blob/eaf0709d6af0025fe94d1ee7cec454bc3054826a/pytorch3d/renderer/cameras.py#L1635-L1685)
    """
    up_axis_token = stage_utils.get_stage_up_axis()
    if up_axis_token == UsdGeom.Tokens.y:
        up_axis = torch.tensor((0, 1, 0), device=device, dtype=torch.float32).repeat(eyes.shape[0], 1)
    elif up_axis_token == UsdGeom.Tokens.z:
        up_axis = torch.tensor((0, 0, 1), device=device, dtype=torch.float32).repeat(eyes.shape[0], 1)
    else:
        raise ValueError(f"Invalid up axis: {up_axis_token}")

    # get rotation matrix in opengl format (-Z forward, +Y up)
    z_axis = -F.normalize(targets - eyes, eps=1e-5)
    x_axis = F.normalize(torch.cross(up_axis, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)
