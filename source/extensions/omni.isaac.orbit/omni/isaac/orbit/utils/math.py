# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Provides utilities for math operations.

Some of these are imported from the module `omni.isaac.core.utils.torch` for convenience.
"""


import numpy as np
import torch
import torch.nn.functional
from typing import Optional, Sequence, Tuple, Union

from omni.isaac.core.utils.torch.maths import normalize, scale_transform, unscale_transform
from omni.isaac.core.utils.torch.rotations import (
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)

__all__ = [
    # General
    "wrap_to_pi",
    "saturate",
    "copysign",
    # General-Isaac Sim
    "normalize",
    "scale_transform",
    "unscale_transform",
    # Rotation
    "matrix_from_quat",
    "quat_inv",
    "quat_from_euler_xyz",
    "quat_apply_yaw",
    "quat_box_minus",
    "euler_xyz_from_quat",
    "axis_angle_from_quat",
    # Rotation-Isaac Sim
    "quat_apply",
    "quat_from_angle_axis",
    "quat_mul",
    "quat_conjugate",
    "quat_rotate",
    "quat_rotate_inverse",
    # Transformations
    "combine_frame_transforms",
    "subtract_frame_transforms",
    "compute_pose_error",
    "apply_delta_pose",
    # Sampling
    "default_orientation",
    "random_orientation",
    "random_yaw_orientation",
    "sample_triangle",
    "sample_uniform",
]

"""
General
"""


@torch.jit.script
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wraps input angles (in radians) to the range [-pi, pi].

    Args:
        angles (torch.Tensor): Input angles.

    Returns:
        torch.Tensor: Angles in the range [-pi, pi].
    """
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Clamps a given input tensor to (lower, upper).

    It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Clamped transform of the tensor. Shape (N, dims)
    """
    return torch.max(torch.min(x, upper), lower)


@torch.jit.script
def copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    """Create a new floating-point tensor with the magnitude of input and the sign of other, element-wise.

    Note:
        The implementation follows from `torch.copysign`. The function allows a scalar magnitude.

    Args:
        mag (float): The magnitude scalar.
        other (torch.Tensor): The tensor containing values whose signbits are applied to magnitude.

    Returns:
        torch.Tensor: The output tensor.
    """
    mag = torch.tensor(mag, device=other.device, dtype=torch.float).repeat(other.shape[0])
    return torch.abs(mag) * torch.sign(other)


"""
Rotation
"""


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    Reference:
        Based on PyTorch3D (https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70)
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def convert_quat(
    quat: Union[torch.tensor, Sequence[float]], to: Optional[str] = "xyzw"
) -> Union[torch.tensor, np.ndarray]:
    """Converts quaternion from one convention to another.

    The convention to convert TO is specified as an optional argument. If to == 'xyzw',
    then the input is in 'wxyz' format, and vice-versa.

    Args:
        quat (Union[torch.tensor, Sequence[float]]): Input quaternion of shape (..., 4).
        to (Optional[str], optional): Convention to convert quaternion to.. Defaults to "xyzw".

    Raises:
        ValueError: Invalid input argument `to`, i.e. not "xyzw" or "wxyz".
        ValueError: Invalid shape of input `quat`, i.e. not (..., 4,).

    Returns:
        Union[torch.tensor, np.ndarray]: The converted quaternion in specified convention.
    """
    # convert to numpy (sanity check)
    if not isinstance(quat, torch.Tensor):
        quat = np.asarray(quat)
    # check input is correct
    if quat.shape[-1] != 4:
        msg = f"convert_quat(): Expected input quaternion shape mismatch: {quat.shape} != (..., 4)."
        raise ValueError(msg)
    # convert to specified quaternion type
    if to == "xyzw":
        return quat[..., [1, 2, 3, 0]]
    elif to == "wxyz":
        return quat[..., [3, 0, 1, 2]]
    else:
        raise ValueError("convert_quat(): Choose a valid `to` argument (xyzw or wxyz).")


@torch.jit.script
def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a quaternion.

    Args:
        q (torch.Tensor): The input quaternion (w, x, y, z).

    Returns:
        torch.Tensor: The inverse quaternion (w, x, y, z).
    """
    return normalize(quat_conjugate(q))


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape: [N,]
        pitch: Rotation around y-axis (in radians). Shape: [N,]
        yaw: Rotation around z-axis (in radians). Shape: [N,]

    Returns:
        torch.Tensor: Quaternion with real part in the start. Shape: [N, 4,]
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qw, qx, qy, qz], dim=-1)


@torch.jit.script
def euler_xyz_from_quat(quat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert rotations given as quaternions to Euler angles in radians.

    Note:
        The euler angles are assumed in XYZ convention.

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Args:
        quat: Quaternion with real part in the start. Shape: [N, 4,]

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing roll-pitch-yaw.
    """
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(torch.abs(sin_pitch) >= 1, copysign(np.pi / 2.0, sin_pitch), torch.asin(sin_pitch))

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


@torch.jit.script
def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector only around the yaw-direction.

    Args:
        quat (torch.Tensor): Input orientation to extract yaw from.
        vec (torch.Tensor): Input vector.

    Returns:
        torch.Tensor: Rotated vector.
    """
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.0  # set x, y components as zero
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Implements box-minus operator (quaternion difference).

    Args:
        q1 (torch.Tensor): A (N, 4) tensor for quaternion (x, y, z, w)
        q2 (torch.Tensor): A (N, 4) tensor for quaternion (x, y, z, w)

    Returns:
        torch.Tensor: q1 box-minus q2

    Reference:
        https://docs.leggedrobotics.com/kindr/cheatsheet_latest.pdf
    """
    quat_diff = quat_mul(q1, quat_conjugate(q2))  # q1 * q2^-1
    re = quat_diff[:, 0]  # real part, q = [w, x, y, z] = [re, im]
    im = quat_diff[:, 1:]  # imaginary part
    norm_im = torch.norm(im, dim=1)
    scale = 2.0 * torch.where(norm_im > 1.0e-7, torch.atan(norm_im / re) / norm_im, torch.sign(re))
    return scale.unsqueeze(-1) * im


@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat (torch.Tensor): quaternions with real part first, as tensor of shape (..., 4).
        eps (float): The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        torch.Tensor: Rotations given as a vector in axis angle form, as a tensor
                of shape (..., 3), where the magnitude is the angle turned
                anti-clockwise in radians around the vector's direction.

    Reference:
        Based on PyTorch3D (https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554)
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        torch.abs(angle.abs()) > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


"""
Transformations
"""


@torch.jit.script
def combine_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t12: torch.Tensor = None, q12: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Combine transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{02} = T_{01} \times T_{12}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01 (torch.Tensor): Position of frame 1 w.r.t. frame 0.
        q01 (torch.Tensor): Quaternion orientation of frame 1 w.r.t. frame 0.
        t12 (torch.Tensor): Position of frame 2 w.r.t. frame 1.
        q12 (torch.Tensor): Quaternion orientation of frame 2 w.r.t. frame 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the position and orientation of
            frame 2 w.r.t. frame 0.
    """
    # compute orientation
    if q12 is not None:
        q02 = quat_mul(q01, q12)
    else:
        q02 = q01
    # compute translation
    if t12 is not None:
        t02 = t01 + quat_apply(q01, t12)
    else:
        t02 = t01

    return t02, q02


@torch.jit.script
def subtract_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor, q02: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Subtract transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{12} = T_{01}^{-1} \times T_{02}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01 (torch.Tensor): Position of frame 1 w.r.t. frame 0.
        q01 (torch.Tensor): Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z).
        t02 (torch.Tensor): Position of frame 2 w.r.t. frame 0.
        q02 (torch.Tensor): Quaternion orientation of frame 2 w.r.t. frame 0 in (w, x, y, z).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the position and orientation of
            frame 2 w.r.t. frame 1.
    """
    # compute orientation
    q10 = quat_inv(q01)
    q12 = quat_mul(q10, q02)
    # compute translation
    t12 = quat_apply(q10, t02 - t01)

    return t12, q12


@torch.jit.script
def compute_pose_error(t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor, q02: torch.Tensor, rot_error_type: str):
    """Compute the position and orientation error between source and target frames.

    Args:
        t01 (torch.Tensor): Position of source frame.
        q01 (torch.Tensor): Quaternion orientation of source frame.
        t02 (torch.Tensor): Position of target frame.
        q02 (torch.Tensor): Quaternion orientation of target frame.
        rot_error_type (str): The rotation error type to return: "quat", "axis_angle".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing position and orientation error.
    """
    # Compute quaternion error (i.e., difference quaternion)
    # Reference: https://personal.utdallas.edu/~sxb027100/dock/quaternion.html
    # q_current_norm = q_current * q_current_conj
    source_quat_norm = quat_mul(q01, quat_conjugate(q01))[:, 0]
    # q_current_inv = q_current_conj / q_current_norm
    source_quat_inv = quat_conjugate(q01) / source_quat_norm.unsqueeze(-1)
    # q_error = q_target * q_current_inv
    quat_error = quat_mul(q02, source_quat_inv)

    # Compute position error
    pos_error = t02 - t01

    # return error based on specified type
    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)
        return pos_error, axis_angle_error
    else:
        raise ValueError(f"Unsupported orientation error type: {rot_error_type}. Valid: 'quat', 'axis_angle'.")


@torch.jit.script
def apply_delta_pose(
    source_pos: torch.Tensor, source_rot, delta_pose: torch.Tensor, eps: float = 1.0e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies delta pose transformation on source pose.

    The first three elements of `delta_pose` are interpreted as cartesian position displacement.
    The remaining three elements of `delta_pose` are interpreted as orientation displacement
    in the angle-axis format.

    Args:
        frame_pos (torch.Tensor): Position of source frame. Shape: [N, 3]
        frame_rot (torch.Tensor): Quaternion orientation of source frame in (w, x, y,z).
        delta_pose (torch.Tensor): Position and orientation displacements. Shape [N, 6].
        eps (float): The tolerance to consider orientation displacement as zero.

    Returns:
        torch.Tensor: A tuple containing the displaced position and orientation frames. Shape: ([N, 3], [N, 4])
    """
    # number of poses given
    num_poses = source_pos.shape[0]
    device = source_pos.device

    # interpret delta_pose[:, 0:3] as target position displacements
    target_pos = source_pos + delta_pose[:, 0:3]
    # interpret delta_pose[:, 3:6] as target rotation displacements
    rot_actions = delta_pose[:, 3:6]
    angle = torch.linalg.vector_norm(rot_actions, dim=1)
    axis = rot_actions / angle.unsqueeze(-1)
    # change from axis-angle to quat convention
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_poses, 1)
    rot_delta_quat = torch.where(
        angle.unsqueeze(-1).repeat(1, 4) > eps, quat_from_angle_axis(angle, axis), identity_quat
    )
    # TODO: Check if this is the correct order for this multiplication.
    target_rot = quat_mul(rot_delta_quat, source_rot)

    return target_pos, target_rot


"""
Sampling
"""


@torch.jit.script
def default_orientation(num: int, device: str) -> torch.Tensor:
    """Returns identity rotation transform.

    Args:
        num (int): The number of rotations to sample.
        device (str): Device to create tensor on.

    Returns:
        torch.Tensor: Identity quaternion (w, x, y, z).
    """
    quat = torch.zeros((num, 4), dtype=torch.float, device=device)
    quat[..., 0] = 1.0

    return quat


@torch.jit.script
def random_orientation(num: int, device: str) -> torch.Tensor:
    """Returns sampled rotation in 3D as quaternion.

    Reference:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html

    Args:
        num (int): The number of rotations to sample.
        device (str): Device to create tensor on.

    Returns:
        torch.Tensor: Sampled quaternion (w, x, y, z).
    """
    # sample random orientation from normal distribution
    quat = torch.randn((num, 4), dtype=torch.float, device=device)
    # normalize the quaternion
    return torch.nn.functional.normalize(quat, p=2.0, dim=-1, eps=1e-12)


@torch.jit.script
def random_yaw_orientation(num: int, device: str) -> torch.Tensor:
    """Returns sampled rotation around z-axis.

    Args:
        num (int): The number of rotations to sample.
        device (str): Device to create tensor on.

    Returns:
        torch.Tensor: Sampled quaternion (w, x, y, z).
    """
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)


def sample_triangle(lower: float, upper: float, size: Union[int, Tuple[int, ...]], device: str) -> torch.Tensor:
    """Randomly samples tensor from a triangular distribution.

    Args:
        lower (float): The lower range of the sampled tensor.
        upper (float): The upper range of the sampled tensor.
        size (Union[int, Tuple[int, ...]]): The shape of the tensor.
        device (str): Device to create tensor on.

    Returns:
        torch.Tensor: Sampled tensor of shape :obj:`size`.
    """
    # convert to tuple
    if isinstance(size, int):
        size = (size,)
    # create random tensor in the range [-1, 1]
    r = 2 * torch.rand(*size, device=device) - 1
    # convert to triangular distribution
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    # rescale back to [0, 1]
    r = (r + 1.0) / 2.0
    # rescale to range [lower, upper]
    return (upper - lower) * r + lower


def sample_uniform(
    lower: Union[torch.Tensor, float], upper: Union[torch.Tensor, float], size: Union[int, Tuple[int, ...]], device: str
) -> torch.Tensor:
    """Sample uniformly within a range.

    Args:
        lower (Union[torch.Tensor, float]): Lower bound of uniform range.
        upper (Union[torch.Tensor, float]): Upper bound of uniform range.
        size (Union[int, Tuple[int, ...]]): The shape of the tensor.
        device (str): Device to create tensor on.

    Returns:
        torch.Tensor: Sampled tensor of shape :obj:`size`.
    """
    # convert to tuple
    if isinstance(size, int):
        size = (size,)
    # return tensor
    return torch.rand(*size, device=device) * (upper - lower) + lower


def sample_cylinder(
    radius: float, h_range: Tuple[float, float], size: Union[int, Tuple[int, ...]], device: str
) -> torch.Tensor:
    """Sample 3D points uniformly on a cylinder's surface.

    The cylinder is centered at the origin and aligned with the z-axis. The height of the cylinder is
    sampled uniformly from the range :obj:`h_range`, while the radius is fixed to :obj:`radius`.

    The sampled points are returned as a tensor of shape :obj:`(*size, 3)`, i.e. the last dimension
    contains the x, y, and z coordinates of the sampled points.

    Args:
        radius (float): The radius of the cylinder.
        h_range (Tuple[float, float]): The minimum and maximum height of the cylinder.
        size (Union[int, Tuple[int, ...]]): The shape of the tensor.
        device (str): Device to create tensor on.

    Returns:
        torch.Tensor: Sampled tensor of shape :obj:`(*size, 3)`.
    """
    # sample angles
    angles = (torch.rand(size, device=device) * 2 - 1) * np.pi
    h_min, h_max = h_range
    # add shape
    if isinstance(size, int):
        size = (size, 3)
    else:
        size += (3,)
    # allocate a tensor
    xyz = torch.zeros(size, device=device)
    xyz[..., 0] = radius * torch.cos(angles)
    xyz[..., 1] = radius * torch.sin(angles)
    xyz[..., 2].uniform_(h_min, h_max)
    # return positions
    return xyz
