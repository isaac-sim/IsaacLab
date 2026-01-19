# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for various math operations."""

# needed to import for allowing type-hinting: torch.Tensor | np.ndarray
from __future__ import annotations

import logging
import math
import numpy as np
import torch
import torch.nn.functional
from typing import Literal

# import logger
logger = logging.getLogger(__name__)


"""
General
"""


@torch.jit.script
def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Normalizes a given input tensor to a range of [-1, 1].

    .. note::
        It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape is (N, dims) or (dims,).
        upper: The maximum value of the tensor. Shape is (N, dims) or (dims,).

    Returns:
        Normalized transform of the tensor. Shape is (N, dims).
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return 2 * (x - offset) / (upper - lower)


@torch.jit.script
def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """De-normalizes a given input tensor from range of [-1, 1] to (lower, upper).

    .. note::
        It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape is (N, dims) or (dims,).
        upper: The maximum value of the tensor. Shape is (N, dims) or (dims,).

    Returns:
        De-normalized transform of the tensor. Shape is (N, dims).
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return x * (upper - lower) * 0.5 + offset


@torch.jit.script
def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Clamps a given input tensor to (lower, upper).

    It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape is (N, dims) or (dims,).
        upper: The maximum value of the tensor. Shape is (N, dims) or (dims,).

    Returns:
        Clamped transform of the tensor. Shape is (N, dims).
    """
    return torch.max(torch.min(x, upper), lower)


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    r"""Wraps input angles (in radians) to the range :math:`[-\pi, \pi]`.

    This function wraps angles in radians to the range :math:`[-\pi, \pi]`, such that
    :math:`\pi` maps to :math:`\pi`, and :math:`-\pi` maps to :math:`-\pi`. In general,
    odd positive multiples of :math:`\pi` are mapped to :math:`\pi`, and odd negative
    multiples of :math:`\pi` are mapped to :math:`-\pi`.

    The function behaves similar to MATLAB's `wrapToPi <https://www.mathworks.com/help/map/ref/wraptopi.html>`_
    function.

    Args:
        angles: Input angles of any shape.

    Returns:
        Angles in the range :math:`[-\pi, \pi]`.
    """
    # wrap to [0, 2*pi)
    wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
    # map to [-pi, pi]
    # we check for zero in wrapped angle to make it go to pi when input angle is odd multiple of pi
    return torch.where((wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi)


@torch.jit.script
def copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    """Create a new floating-point tensor with the magnitude of input and the sign of other, element-wise.

    Note:
        The implementation follows from `torch.copysign`. The function allows a scalar magnitude.

    Args:
        mag: The magnitude scalar.
        other: The tensor containing values whose signbits are applied to magnitude.

    Returns:
        The output tensor.
    """
    mag_torch = abs(mag) * torch.ones_like(other)
    return torch.copysign(mag_torch, other)


"""
Rotation
"""


@torch.jit.script
def quat_unique(q: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion to a standard form where the real part is non-negative.

    Quaternion representations have a singularity since ``q`` and ``-q`` represent the same
    rotation. This function ensures the real part of the quaternion is non-negative.

    Args:
        q: The quaternion orientation in (x, y, z, w). Shape is (..., 4).

    Returns:
        Standardized quaternions. Shape is (..., 4).
    """
    return torch.where(q[..., 3:4] < 0, -q, q)


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (x, y, z, w). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    i, j, k, r = torch.unbind(quaternions, -1)
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


def convert_quat(quat: torch.Tensor | np.ndarray, to: Literal["xyzw", "wxyz"] = "xyzw") -> torch.Tensor | np.ndarray:
    """Converts quaternion from one convention to another.

    The convention to convert TO is specified as an optional argument. If to == 'xyzw',
    then the input is in 'wxyz' format, and vice-versa.

    Args:
        quat: The quaternion of shape (..., 4).
        to: Convention to convert quaternion to.. Defaults to "xyzw".

    Returns:
        The converted quaternion in specified convention.

    Raises:
        ValueError: Invalid input argument `to`, i.e. not "xyzw" or "wxyz".
        ValueError: Invalid shape of input `quat`, i.e. not (..., 4,).
    """
    # check input is correct
    if quat.shape[-1] != 4:
        msg = f"Expected input quaternion shape mismatch: {quat.shape} != (..., 4)."
        raise ValueError(msg)
    if to not in ["xyzw", "wxyz"]:
        msg = f"Expected input argument `to` to be 'xyzw' or 'wxyz'. Received: {to}."
        raise ValueError(msg)
    # check if input is numpy array (we support this backend since some classes use numpy)
    if isinstance(quat, np.ndarray):
        # use numpy functions
        if to == "xyzw":
            # wxyz -> xyzw
            return np.roll(quat, -1, axis=-1)
        else:
            # xyzw -> wxyz
            return np.roll(quat, 1, axis=-1)
    else:
        # convert to torch (sanity check)
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=float)
        # convert to specified quaternion type
        if to == "xyzw":
            # wxyz -> xyzw
            return quat.roll(-1, dims=-1)
        else:
            # xyzw -> wxyz
            return quat.roll(1, dims=-1)


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (x, y, z, w). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (x, y, z, w). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1).view(shape)


@torch.jit.script
def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Computes the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (x, y, z, w). Shape is (N, 4).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        The inverse quaternion in (x, y, z, w). Shape is (N, 4).
    """
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (x, y, z, w). Shape is (N, 4).
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

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) but with a zero sub-gradient where x is 0.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L91-L99
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: The rotation matrices. Shape is (..., 3, 3).

    Returns:
        The quaternion in (x, y, z, w). Shape is (..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L102-L161
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    wxyz = quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )

    out_xyzw = torch.stack(
        (wxyz[..., 1], wxyz[..., 2], wxyz[..., 3], wxyz[..., 0]), dim=-1
    )  # this is not optimum but works
    return out_xyzw


def _axis_angle_rotation(axis: Literal["X", "Y", "Z"], angle: torch.Tensor) -> torch.Tensor:
    """Return the rotation matrices for one of the rotations about an axis of which Euler angles describe,
    for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: Euler angles in radians of any shape.

    Returns:
        Rotation matrices. Shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L164-L191
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def matrix_from_euler(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles (intrinsic) in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians. Shape is (..., 3).
        convention: Convention string of three uppercase letters from {"X", "Y", and "Z"}.
            For example, "XYZ" means that the rotations should be applied first about x,
            then y, then z.

    Returns:
        Rotation matrices. Shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L194-L220
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler_angles, -1))]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


@torch.jit.script
def euler_xyz_from_quat(
    quat: torch.Tensor, wrap_to_2pi: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert rotations given as quaternions to Euler angles in radians.

    Note:
        The euler angles are assumed in XYZ extrinsic convention.

    Args:
        quat: The quaternion orientation in (x, y, z, w). Shape is (N, 4).
        wrap_to_2pi (bool): Whether to wrap output Euler angles into [0, 2π). If
            False, angles are returned in the default range (−π, π]. Defaults to
            False.

    Returns:
        A tuple containing roll-pitch-yaw. Each element is a tensor of shape (N,).

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    q_x, q_y, q_z, q_w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(torch.abs(sin_pitch) >= 1, copysign(torch.pi / 2.0, sin_pitch), torch.asin(sin_pitch))

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    if wrap_to_2pi:
        return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)
    return roll, pitch, yaw


@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (x, y, z, w). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_x, q_y, q_z, q_w] (xyzw format)
    # Quaternion is [q_x, q_y, q_z, q_w] = [n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2), cos(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 3:] < 0.0))
    mag = torch.linalg.norm(quat[..., :3], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 3])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., :3] / sin_half_angles_over_angles.unsqueeze(-1)


@torch.jit.script
def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as angle-axis to quaternions.

    Args:
        angle: The angle turned anti-clockwise in radians around the vector's direction. Shape is (N,).
        axis: The axis of rotation. Shape is (N, 3).

    Returns:
        The quaternion in (x, y, z, w). Shape is (N, 4).
    """
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (x, y, z, w). Shape is (..., 4).
        q2: The second quaternion in (x, y, z, w). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (x, y, z, w). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    w = qq - ww + (z1 - y1) * (y2 - z2)

    return torch.stack([x, y, z, w], dim=-1).view(shape)


@torch.jit.script
def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (x, y, z, w). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = quat.view(-1, 4)
    qx = quat_yaw[:, 0]
    qy = quat_yaw[:, 1]
    qz = quat_yaw[:, 2]
    qw = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = torch.zeros_like(quat_yaw)
    # For xyzw format: z = sin(yaw/2), w = cos(yaw/2)
    quat_yaw[:, 2] = torch.sin(yaw / 2)
    quat_yaw[:, 3] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)


@torch.jit.script
def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """The box-minus operator (quaternion difference) between two quaternions.

    Args:
        q1: The first quaternion in (x, y, z, w). Shape is (N, 4).
        q2: The second quaternion in (x, y, z, w). Shape is (N, 4).

    Returns:
        The difference between the two quaternions. Shape is (N, 3).

    Reference:
        https://github.com/ANYbotics/kindr/blob/master/doc/cheatsheet/cheatsheet_latest.pdf
    """
    quat_diff = quat_mul(q1, quat_conjugate(q2))  # q1 * q2^-1
    return axis_angle_from_quat(quat_diff)  # log(qd)


@torch.jit.script
def quat_box_plus(q: torch.Tensor, delta: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """The box-plus operator (quaternion update) to apply an increment to a quaternion.

    Args:
        q: The initial quaternion in (x, y, z, w). Shape is (N, 4).
        delta: The axis-angle perturbation. Shape is (N, 3).
            eps: A small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        The updated quaternion after applying the perturbation. Shape is (N, 4).

    Reference:
        https://github.com/ANYbotics/kindr/blob/master/doc/cheatsheet/cheatsheet_latest.pdf
    """
    delta_norm = torch.clamp_min(torch.linalg.norm(delta, dim=-1, keepdim=True), min=eps)
    delta_quat = quat_from_angle_axis(delta_norm.squeeze(-1), delta / delta_norm)  # exp(dq)
    new_quat = quat_mul(delta_quat, q)  # Apply perturbation
    return quat_unique(new_quat)


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (x, y, z, w). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, :3]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (x, y, z, w). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, :3]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector only around the yaw-direction.

    Args:
        quat: The orientation in (x, y, z, w). Shape is (N, 4).
        vec: The vector in (x, y, z). Shape is (N, 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (N, 3).
    """
    quat_yaw = yaw_quat(quat)
    return quat_apply(quat_yaw, vec)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion along the last dimension of q and v.

    .. deprecated:: v2.1.0
        This function will be removed in a future release in favor of the faster implementation :meth:`quat_apply`.

    Args:
        q: The quaternion in (x, y, z, w). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # deprecation
    logger.warning(
        "The function 'quat_rotate' will be deprecated in favor of the faster method 'quat_apply'."
        " Please use 'quat_apply' instead...."
    )
    return quat_apply(q, v)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v.

    .. deprecated:: v2.1.0
        This function will be removed in a future release in favor of the faster implementation :meth:`quat_apply_inverse`.

    Args:
        q: The quaternion in (x, y, z, w). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    logger.warning(
        "The function 'quat_rotate_inverse' will be deprecated in favor of the faster method 'quat_apply_inverse'."
        " Please use 'quat_apply_inverse' instead...."
    )
    return quat_apply_inverse(q, v)


@torch.jit.script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Computes the rotation difference between two quaternions.

    Args:
        q1: The first quaternion in (x, y, z, w). Shape is (..., 4).
        q2: The second quaternion in (x, y, z, w). Shape is (..., 4).

    Returns:
        Angular error between input quaternions in radians.
    """
    axis_angle_error = quat_box_minus(q1, q2)
    return torch.norm(axis_angle_error, dim=-1)


@torch.jit.script
def skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
    """Computes the skew-symmetric matrix of a vector.

    Args:
        vec: The input vector. Shape is (3,) or (N, 3).

    Returns:
        The skew-symmetric matrix. Shape is (1, 3, 3) or (N, 3, 3).

    Raises:
        ValueError: If input tensor is not of shape (..., 3).
    """
    # check input is correct
    if vec.shape[-1] != 3:
        raise ValueError(f"Expected input vector shape mismatch: {vec.shape} != (..., 3).")
    # unsqueeze the last dimension
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    # create a skew-symmetric matrix
    skew_sym_mat = torch.zeros(vec.shape[0], 3, 3, device=vec.device, dtype=vec.dtype)
    skew_sym_mat[:, 0, 1] = -vec[:, 2]
    skew_sym_mat[:, 0, 2] = vec[:, 1]
    skew_sym_mat[:, 1, 2] = -vec[:, 0]
    skew_sym_mat[:, 1, 0] = vec[:, 2]
    skew_sym_mat[:, 2, 0] = -vec[:, 1]
    skew_sym_mat[:, 2, 1] = vec[:, 0]

    return skew_sym_mat


"""
Transformations
"""


def is_identity_pose(pos: torch.tensor, rot: torch.tensor) -> bool:
    """Checks if input poses are identity transforms.

    The function checks if the input position and orientation are close to zero and
    identity respectively using L2-norm. It does NOT check the error in the orientation.

    Args:
        pos: The cartesian position. Shape is (N, 3).
        rot: The quaternion in (x, y, z, w). Shape is (N, 4).

    Returns:
        True if all the input poses result in identity transform. Otherwise, False.
    """
    # create identity transformations
    pos_identity = torch.zeros_like(pos)
    rot_identity = torch.zeros_like(rot)
    rot_identity[..., 3] = 1
    # compare input to identity
    return torch.allclose(pos, pos_identity) and torch.allclose(rot, rot_identity)


@torch.jit.script
def combine_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t12: torch.Tensor | None = None, q12: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Combine transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{02} = T_{01} \times T_{12}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (x, y, z, w). Shape is (N, 4).
        t12: Position of frame 2 w.r.t. frame 1. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q12: Quaternion orientation of frame 2 w.r.t. frame 1 in (x, y, z, w). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 0.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
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


def rigid_body_twist_transform(
    v0: torch.Tensor, w0: torch.Tensor, t01: torch.Tensor, q01: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Transform the linear and angular velocity of a rigid body between reference frames.

    Given the twist of 0 relative to frame 0, this function computes the twist of 1 relative to frame 1
    from the position and orientation of frame 1 relative to frame 0. The transformation follows the
    equations:

    .. math::

        w_11 = R_{10} w_00 = R_{01}^{-1} w_00
        v_11 = R_{10} v_00 + R_{10} (w_00 \times t_01) = R_{01}^{-1} (v_00 + (w_00 \times t_01))

    where

        - :math:`R_{01}` is the rotation matrix from frame 0 to frame 1 derived from quaternion :math:`q_{01}`.
        - :math:`t_{01}` is the position of frame 1 relative to frame 0 expressed in frame 0
        - :math:`w_0` is the angular velocity of 0 in frame 0
        - :math:`v_0` is the linear velocity of 0 in frame 0

    Args:
        v0: Linear velocity of 0 in frame 0. Shape is (N, 3).
        w0: Angular velocity of 0 in frame 0. Shape is (N, 3).
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (x, y, z, w). Shape is (N, 4).

    Returns:
        A tuple containing:
        - The transformed linear velocity in frame 1. Shape is (N, 3).
        - The transformed angular velocity in frame 1. Shape is (N, 3).
    """
    w1 = quat_rotate_inverse(q01, w0)
    v1 = quat_rotate_inverse(q01, v0 + torch.cross(w0, t01, dim=-1))
    return v1, w1


# @torch.jit.script
def subtract_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor | None = None, q02: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Subtract transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{12} = T_{01}^{-1} \times T_{02}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (x, y, z, w). Shape is (N, 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q02: Quaternion orientation of frame 2 w.r.t. frame 0 in (x, y, z, w). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 1.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    # compute orientation
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    # compute translation
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12


# @torch.jit.script
def compute_pose_error(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor,
    q02: torch.Tensor,
    rot_error_type: Literal["quat", "axis_angle"] = "axis_angle",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the position and orientation error between source and target frames.

    Args:
        t01: Position of source frame. Shape is (N, 3).
        q01: Quaternion orientation of source frame in (x, y, z, w). Shape is (N, 4).
        t02: Position of target frame. Shape is (N, 3).
        q02: Quaternion orientation of target frame in (x, y, z, w). Shape is (N, 4).
        rot_error_type: The rotation error type to return: "quat", "axis_angle".
            Defaults to "axis_angle".

    Returns:
        A tuple containing position and orientation error. Shape of position error is (N, 3).
        Shape of orientation error depends on the value of :attr:`rot_error_type`:

        - If :attr:`rot_error_type` is "quat", the orientation error is returned
          as a quaternion. Shape is (N, 4).
        - If :attr:`rot_error_type` is "axis_angle", the orientation error is
          returned as an axis-angle vector. Shape is (N, 3).

    Raises:
        ValueError: Invalid rotation error type.
    """
    # Compute quaternion error (i.e., difference quaternion)
    # Reference: https://personal.utdallas.edu/~sxb027100/dock/quaternion.html
    # q_current_norm = q_current * q_current_conj
    source_quat_norm = quat_mul(q01, quat_conjugate(q01))[:, 3]
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
    source_pos: torch.Tensor, source_rot: torch.Tensor, delta_pose: torch.Tensor, eps: float = 1.0e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies delta pose transformation on source pose.

    The first three elements of `delta_pose` are interpreted as cartesian position displacement.
    The remaining three elements of `delta_pose` are interpreted as orientation displacement
    in the angle-axis format.

    Args:
        source_pos: Position of source frame. Shape is (N, 3).
        source_rot: Quaternion orientation of source frame in (x, y, z, w). Shape is (N, 4)..
        delta_pose: Position and orientation displacements. Shape is (N, 6).
        eps: The tolerance to consider orientation displacement as zero. Defaults to 1.0e-6.

    Returns:
        A tuple containing the displaced position and orientation frames.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
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
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(num_poses, 1)
    rot_delta_quat = torch.where(
        angle.unsqueeze(-1).repeat(1, 4) > eps, quat_from_angle_axis(angle, axis), identity_quat
    )
    # TODO: Check if this is the correct order for this multiplication.
    target_rot = quat_mul(rot_delta_quat, source_rot)

    return target_pos, target_rot


# @torch.jit.script
def transform_points(
    points: torch.Tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Transform input points in a given frame to a target frame.

    This function transform points from a source frame to a target frame. The transformation is defined by the
    position :math:`t` and orientation :math:`R` of the target frame in the source frame.

    .. math::
        p_{target} = R_{target} \times p_{source} + t_{target}

    If the input `points` is a batch of points, the inputs `pos` and `quat` must be either a batch of
    positions and quaternions or a single position and quaternion. If the inputs `pos` and `quat` are
    a single position and quaternion, the same transformation is applied to all points in the batch.

    If either the inputs :attr:`pos` and :attr:`quat` are None, the corresponding transformation is not applied.

    Args:
        points: Points to transform. Shape is (N, P, 3) or (P, 3).
        pos: Position of the target frame. Shape is (N, 3) or (3,).
            Defaults to None, in which case the position is assumed to be zero.
        quat: Quaternion orientation of the target frame in (x, y, z, w). Shape is (N, 4) or (4,).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        Transformed points in the target frame. Shape is (N, P, 3) or (P, 3).

    Raises:
        ValueError: If the inputs `points` is not of shape (N, P, 3) or (P, 3).
        ValueError: If the inputs `pos` is not of shape (N, 3) or (3,).
        ValueError: If the inputs `quat` is not of shape (N, 4) or (4,).
    """
    points_batch = points.clone()
    # check if inputs are batched
    is_batched = points_batch.dim() == 3
    # -- check inputs
    if points_batch.dim() == 2:
        points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
    if points_batch.dim() != 3:
        raise ValueError(f"Expected points to have dim = 2 or dim = 3: got shape {points.shape}")
    if not (pos is None or pos.dim() == 1 or pos.dim() == 2):
        raise ValueError(f"Expected pos to have dim = 1 or dim = 2: got shape {pos.shape}")
    if not (quat is None or quat.dim() == 1 or quat.dim() == 2):
        raise ValueError(f"Expected quat to have dim = 1 or dim = 2: got shape {quat.shape}")
    # -- rotation
    if quat is not None:
        # convert to batched rotation matrix
        rot_mat = matrix_from_quat(quat)
        if rot_mat.dim() == 2:
            rot_mat = rot_mat[None]  # (3, 3) -> (1, 3, 3)
        # convert points to matching batch size (N, P, 3) -> (N, 3, P)
        # and apply rotation
        points_batch = torch.matmul(rot_mat, points_batch.transpose_(1, 2))
        # (N, 3, P) -> (N, P, 3)
        points_batch = points_batch.transpose_(1, 2)
    # -- translation
    if pos is not None:
        # convert to batched translation vector
        if pos.dim() == 1:
            pos = pos[None, None, :]  # (3,) -> (1, 1, 3)
        else:
            pos = pos[:, None, :]  # (N, 3) -> (N, 1, 3)
        # apply translation
        points_batch += pos
    # -- return points in same shape as input
    if not is_batched:
        points_batch = points_batch.squeeze(0)  # (1, P, 3) -> (P, 3)

    return points_batch


"""
Projection operations.
"""


@torch.jit.script
def orthogonalize_perspective_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Converts perspective depth image to orthogonal depth image.

    Perspective depth images contain distances measured from the camera's optical center.
    Meanwhile, orthogonal depth images provide the distance from the camera's image plane.
    This method uses the camera geometry to convert perspective depth to orthogonal depth image.

    The function assumes that the width and height are both greater than 1.

    Args:
        depth: The perspective depth images. Shape is (H, W) or or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        intrinsics: The camera's calibration matrix. If a single matrix is provided, the same
            calibration matrix is used across all the depth images in the batch.
            Shape is (3, 3) or (N, 3, 3).

    Returns:
        The orthogonal depth images. Shape matches the input shape of depth images.

    Raises:
        ValueError: When depth is not of shape (H, W) or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        ValueError: When intrinsics is not of shape (3, 3) or (N, 3, 3).
    """
    # Clone inputs to avoid in-place modifications
    perspective_depth_batch = depth.clone()
    intrinsics_batch = intrinsics.clone()

    # Check if inputs are batched
    is_batched = perspective_depth_batch.dim() == 4 or (
        perspective_depth_batch.dim() == 3 and perspective_depth_batch.shape[-1] != 1
    )

    # Track whether the last dimension was singleton
    add_last_dim = False
    if perspective_depth_batch.dim() == 4 and perspective_depth_batch.shape[-1] == 1:
        add_last_dim = True
        perspective_depth_batch = perspective_depth_batch.squeeze(dim=3)  # (N, H, W, 1) -> (N, H, W)
    if perspective_depth_batch.dim() == 3 and perspective_depth_batch.shape[-1] == 1:
        add_last_dim = True
        perspective_depth_batch = perspective_depth_batch.squeeze(dim=2)  # (H, W, 1) -> (H, W)

    if perspective_depth_batch.dim() == 2:
        perspective_depth_batch = perspective_depth_batch[None]  # (H, W) -> (1, H, W)

    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]  # (3, 3) -> (1, 3, 3)

    if is_batched and intrinsics_batch.shape[0] == 1:
        intrinsics_batch = intrinsics_batch.expand(perspective_depth_batch.shape[0], -1, -1)  # (1, 3, 3) -> (N, 3, 3)

    # Validate input shapes
    if perspective_depth_batch.dim() != 3:
        raise ValueError(f"Expected depth images to have 2, 3, or 4 dimensions; got {depth.shape}.")
    if intrinsics_batch.dim() != 3:
        raise ValueError(f"Expected intrinsics to have shape (3, 3) or (N, 3, 3); got {intrinsics.shape}.")

    # Image dimensions
    im_height, im_width = perspective_depth_batch.shape[1:]

    # Get the intrinsics parameters
    fx = intrinsics_batch[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics_batch[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics_batch[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics_batch[:, 1, 2].view(-1, 1, 1)

    # Create meshgrid of pixel coordinates
    u_grid = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    v_grid = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    u_grid, v_grid = torch.meshgrid(u_grid, v_grid, indexing="xy")

    # Expand the grids for batch processing
    u_grid = u_grid.unsqueeze(0).expand(perspective_depth_batch.shape[0], -1, -1)
    v_grid = v_grid.unsqueeze(0).expand(perspective_depth_batch.shape[0], -1, -1)

    # Compute the squared terms for efficiency
    x_term = ((u_grid - cx) / fx) ** 2
    y_term = ((v_grid - cy) / fy) ** 2

    # Calculate the orthogonal (normal) depth
    orthogonal_depth = perspective_depth_batch / torch.sqrt(1 + x_term + y_term)

    # Restore the last dimension if it was present in the input
    if add_last_dim:
        orthogonal_depth = orthogonal_depth.unsqueeze(-1)

    # Return to original shape if input was not batched
    if not is_batched:
        orthogonal_depth = orthogonal_depth.squeeze(0)

    return orthogonal_depth


@torch.jit.script
def unproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor, is_ortho: bool = True) -> torch.Tensor:
    r"""Un-project depth image into a pointcloud.

    This function converts orthogonal or perspective depth images into points given the calibration matrix
    of the camera. It uses the following transformation based on camera geometry:

    .. math::
        p_{3D} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`p_{3D}` is the 3D point, :math:`d` is the depth value (measured from the image plane),
    :math:`u` and :math:`v` are the pixel coordinates and :math:`K` is the intrinsic matrix.

    The function assumes that the width and height are both greater than 1. This makes the function
    deal with many possible shapes of depth images and intrinsics matrices.

    .. note::
        If :attr:`is_ortho` is False, the input depth images are transformed to orthogonal depth images
        by using the :meth:`orthogonalize_perspective_depth` method.

    Args:
        depth: The depth measurement. Shape is (H, W) or or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        intrinsics: The camera's calibration matrix. If a single matrix is provided, the same
            calibration matrix is used across all the depth images in the batch.
            Shape is (3, 3) or (N, 3, 3).
        is_ortho: Whether the input depth image is orthogonal or perspective depth image. If True, the input
            depth image is considered as the *orthogonal* type, where the measurements are from the camera's
            image plane. If False, the depth image is considered as the *perspective* type, where the
            measurements are from the camera's optical center. Defaults to True.

    Returns:
        The 3D coordinates of points. Shape is (P, 3) or (N, P, 3).

    Raises:
        ValueError: When depth is not of shape (H, W) or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        ValueError: When intrinsics is not of shape (3, 3) or (N, 3, 3).
    """
    # clone inputs to avoid in-place modifications
    intrinsics_batch = intrinsics.clone()
    # convert depth image to orthogonal if needed
    if not is_ortho:
        depth_batch = orthogonalize_perspective_depth(depth, intrinsics)
    else:
        depth_batch = depth.clone()

    # check if inputs are batched
    is_batched = depth_batch.dim() == 4 or (depth_batch.dim() == 3 and depth_batch.shape[-1] != 1)
    # make sure inputs are batched
    if depth_batch.dim() == 3 and depth_batch.shape[-1] == 1:
        depth_batch = depth_batch.squeeze(dim=2)  # (H, W, 1) -> (H, W)
    if depth_batch.dim() == 2:
        depth_batch = depth_batch[None]  # (H, W) -> (1, H, W)
    if depth_batch.dim() == 4 and depth_batch.shape[-1] == 1:
        depth_batch = depth_batch.squeeze(dim=3)  # (N, H, W, 1) -> (N, H, W)
    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]  # (3, 3) -> (1, 3, 3)
    # check shape of inputs
    if depth_batch.dim() != 3:
        raise ValueError(f"Expected depth images to have dim = 2 or 3 or 4: got shape {depth.shape}")
    if intrinsics_batch.dim() != 3:
        raise ValueError(f"Expected intrinsics to have shape (3, 3) or (N, 3, 3): got shape {intrinsics.shape}")

    # get image height and width
    im_height, im_width = depth_batch.shape[1:]
    # create image points in homogeneous coordinates (3, H x W)
    indices_u = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    indices_v = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    img_indices = torch.stack(torch.meshgrid([indices_u, indices_v], indexing="ij"), dim=0).reshape(2, -1)
    pixels = torch.nn.functional.pad(img_indices, (0, 0, 0, 1), mode="constant", value=1.0)
    pixels = pixels.unsqueeze(0)  # (3, H x W) -> (1, 3, H x W)

    # unproject points into 3D space
    points = torch.matmul(torch.inverse(intrinsics_batch), pixels)  # (N, 3, H x W)
    points = points / points[:, -1, :].unsqueeze(1)  # normalize by last coordinate
    # flatten depth image (N, H, W) -> (N, H x W)
    depth_batch = depth_batch.transpose_(1, 2).reshape(depth_batch.shape[0], -1).unsqueeze(2)
    depth_batch = depth_batch.expand(-1, -1, 3)
    # scale points by depth
    points_xyz = points.transpose_(1, 2) * depth_batch  # (N, H x W, 3)

    # return points in same shape as input
    if not is_batched:
        points_xyz = points_xyz.squeeze(0)

    return points_xyz


@torch.jit.script
def project_points(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    r"""Projects 3D points into 2D image plane.

    This project 3D points into a 2D image plane. The transformation is defined by the intrinsic
    matrix of the camera.

    .. math::

        \begin{align}
            p &= K \times p_{3D}  = \\
            p_{2D} &= \begin{pmatrix} u \\ v \\  d \end{pmatrix}
                    = \begin{pmatrix} p[0] / p[2] \\  p[1] / p[2] \\ Z \end{pmatrix}
        \end{align}

    where :math:`p_{2D} = (u, v, d)` is the projected 3D point, :math:`p_{3D} = (X, Y, Z)` is the
    3D point and :math:`K \in \mathbb{R}^{3 \times 3}` is the intrinsic matrix.

    If `points` is a batch of 3D points and `intrinsics` is a single intrinsic matrix, the same
    calibration matrix is applied to all points in the batch.

    Args:
        points: The 3D coordinates of points. Shape is (P, 3) or (N, P, 3).
        intrinsics: Camera's calibration matrix. Shape is (3, 3) or (N, 3, 3).

    Returns:
        Projected 3D coordinates of points. Shape is (P, 3) or (N, P, 3).
    """
    # clone the inputs to avoid in-place operations modifying the original data
    points_batch = points.clone()
    intrinsics_batch = intrinsics.clone()

    # check if inputs are batched
    is_batched = points_batch.dim() == 2
    # make sure inputs are batched
    if points_batch.dim() == 2:
        points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]  # (3, 3) -> (1, 3, 3)
    # check shape of inputs
    if points_batch.dim() != 3:
        raise ValueError(f"Expected points to have dim = 3: got shape {points.shape}.")
    if intrinsics_batch.dim() != 3:
        raise ValueError(f"Expected intrinsics to have shape (3, 3) or (N, 3, 3): got shape {intrinsics.shape}.")

    # project points into 2D image plane
    points_2d = torch.matmul(intrinsics_batch, points_batch.transpose(1, 2))
    points_2d = points_2d / points_2d[:, -1, :].unsqueeze(1)  # normalize by last coordinate
    points_2d = points_2d.transpose_(1, 2)  # (N, 3, P) -> (N, P, 3)
    # replace last coordinate with depth
    points_2d[:, :, -1] = points_batch[:, :, -1]

    # return points in same shape as input
    if not is_batched:
        points_2d = points_2d.squeeze(0)  # (1, 3, P) -> (3, P)

    return points_2d


"""
Sampling
"""


@torch.jit.script
def default_orientation(num: int, device: str) -> torch.Tensor:
    """Returns identity rotation transform.

    Args:
        num: The number of rotations to sample.
        device: Device to create tensor on.

    Returns:
        Identity quaternion in (x, y, z, w). Shape is (num, 4).
    """
    quat = torch.zeros((num, 4), dtype=torch.float, device=device)
    quat[..., 3] = 1.0

    return quat


@torch.jit.script
def random_orientation(num: int, device: str) -> torch.Tensor:
    """Returns sampled rotation in 3D as quaternion.

    Args:
        num: The number of rotations to sample.
        device: Device to create tensor on.

    Returns:
        Sampled quaternion in (x, y, z, w). Shape is (num, 4).

    Reference:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html
    """
    # sample random orientation from normal distribution
    quat = torch.randn((num, 4), dtype=torch.float, device=device)
    # normalize the quaternion
    return torch.nn.functional.normalize(quat, p=2.0, dim=-1, eps=1e-12)


@torch.jit.script
def random_yaw_orientation(num: int, device: str) -> torch.Tensor:
    """Returns sampled rotation around z-axis.

    Args:
        num: The number of rotations to sample.
        device: Device to create tensor on.

    Returns:
        Sampled quaternion in (x, y, z, w). Shape is (num, 4).
    """
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = 2 * torch.pi * torch.rand(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)


def sample_triangle(lower: float, upper: float, size: int | tuple[int, ...], device: str) -> torch.Tensor:
    """Randomly samples tensor from a triangular distribution.

    Args:
        lower: The lower range of the sampled tensor.
        upper: The upper range of the sampled tensor.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor. Shape is based on :attr:`size`.
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
    lower: torch.Tensor | float, upper: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    """Sample uniformly within a range.

    Args:
        lower: Lower bound of uniform range.
        upper: Upper bound of uniform range.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor. Shape is based on :attr:`size`.
    """
    # convert to tuple
    if isinstance(size, int):
        size = (size,)
    # return tensor
    return torch.rand(*size, device=device) * (upper - lower) + lower


def sample_log_uniform(
    lower: torch.Tensor | float, upper: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    r"""Sample using log-uniform distribution within a range.

    The log-uniform distribution is defined as a uniform distribution in the log-space. It
    is useful for sampling values that span several orders of magnitude. The sampled values
    are uniformly distributed in the log-space and then exponentiated to get the final values.

    .. math::

        x = \exp(\text{uniform}(\log(\text{lower}), \log(\text{upper})))

    Args:
        lower: Lower bound of uniform range.
        upper: Upper bound of uniform range.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor. Shape is based on :attr:`size`.
    """
    # cast to tensor if not already
    if not isinstance(lower, torch.Tensor):
        lower = torch.tensor(lower, dtype=torch.float, device=device)
    if not isinstance(upper, torch.Tensor):
        upper = torch.tensor(upper, dtype=torch.float, device=device)
    # sample in log-space and exponentiate
    return torch.exp(sample_uniform(torch.log(lower), torch.log(upper), size, device))


def sample_gaussian(
    mean: torch.Tensor | float, std: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    """Sample using gaussian distribution.

    Args:
        mean: Mean of the gaussian.
        std: Std of the gaussian.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor.
    """
    if isinstance(mean, float):
        if isinstance(size, int):
            size = (size,)
        return torch.normal(mean=mean, std=std, size=size).to(device=device)
    else:
        return torch.normal(mean=mean, std=std).to(device=device)


def sample_cylinder(
    radius: float, h_range: tuple[float, float], size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    """Sample 3D points uniformly on a cylinder's surface.

    The cylinder is centered at the origin and aligned with the z-axis. The height of the cylinder is
    sampled uniformly from the range :obj:`h_range`, while the radius is fixed to :obj:`radius`.

    The sampled points are returned as a tensor of shape :obj:`(*size, 3)`, i.e. the last dimension
    contains the x, y, and z coordinates of the sampled points.

    Args:
        radius: The radius of the cylinder.
        h_range: The minimum and maximum height of the cylinder.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor. Shape is :obj:`(*size, 3)`.
    """
    # sample angles
    angles = (torch.rand(size, device=device) * 2 - 1) * torch.pi
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


"""
Orientation Conversions
"""


def convert_camera_frame_orientation_convention(
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
        orientation: Quaternion of form `(x, y, z, w)` with shape (..., 4) in source convention.
        origin: Convention to convert from. Defaults to "opengl".
        target: Convention to convert to. Defaults to "ros".

    Returns:
        Quaternion of form `(x, y, z, w)` with shape (..., 4) in target convention
    """
    if target == origin:
        return orientation.clone()

    # -- unify input type
    if origin == "ros":
        # convert from ros to opengl convention
        rotm = matrix_from_quat(orientation)
        rotm[:, :, 2] = -rotm[:, :, 2]
        rotm[:, :, 1] = -rotm[:, :, 1]
        # convert to opengl convention
        quat_gl = quat_from_matrix(rotm)
    elif origin == "world":
        # convert from world (x forward and z up) to opengl convention
        rotm = matrix_from_quat(orientation)
        rotm = torch.matmul(
            rotm,
            matrix_from_euler(torch.tensor([math.pi / 2, -math.pi / 2, 0], device=orientation.device), "XYZ"),
        )
        # convert to isaac-sim convention
        quat_gl = quat_from_matrix(rotm)
    else:
        quat_gl = orientation

    # -- convert to target convention
    if target == "ros":
        # convert from opengl to ros convention
        rotm = matrix_from_quat(quat_gl)
        rotm[:, :, 2] = -rotm[:, :, 2]
        rotm[:, :, 1] = -rotm[:, :, 1]
        return quat_from_matrix(rotm)
    elif target == "world":
        # convert from opengl to world (x forward and z up) convention
        rotm = matrix_from_quat(quat_gl)
        rotm = torch.matmul(
            rotm,
            matrix_from_euler(torch.tensor([math.pi / 2, -math.pi / 2, 0], device=orientation.device), "XYZ").T,
        )
        return quat_from_matrix(rotm)
    else:
        return quat_gl.clone()


def create_rotation_matrix_from_view(
    eyes: torch.Tensor,
    targets: torch.Tensor,
    up_axis: Literal["Y", "Z"] = "Z",
    device: str = "cpu",
) -> torch.Tensor:
    """Compute the rotation matrix from world to view coordinates.

    This function takes a vector ''eyes'' which specifies the location
    of the camera in world coordinates and the vector ''targets'' which
    indicate the position of the object.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

        The inputs eyes and targets can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    Args:
        eyes: Position of the camera in world coordinates.
        targets: Position of the object in world coordinates.
        up_axis: The up axis of the camera. Defaults to "Z".
        device: The device to create torch tensors on. Defaults to "cpu".

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices

    Reference:
    Based on PyTorch3D (https://github.com/facebookresearch/pytorch3d/blob/eaf0709d6af0025fe94d1ee7cec454bc3054826a/pytorch3d/renderer/cameras.py#L1635-L1685)
    """
    if up_axis == "Y":
        up_axis_vec = torch.tensor((0, 1, 0), device=device, dtype=torch.float32).repeat(eyes.shape[0], 1)
    elif up_axis == "Z":
        up_axis_vec = torch.tensor((0, 0, 1), device=device, dtype=torch.float32).repeat(eyes.shape[0], 1)
    else:
        raise ValueError(f"Invalid up axis: {up_axis}. Valid options are 'Y' and 'Z'.")

    # get rotation matrix in opengl format (-Z forward, +Y up)
    z_axis = -torch.nn.functional.normalize(targets - eyes, eps=1e-5)
    x_axis = torch.nn.functional.normalize(torch.cross(up_axis_vec, z_axis, dim=1), eps=1e-5)
    y_axis = torch.nn.functional.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = torch.nn.functional.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)


def make_pose(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Creates transformation matrices from positions and rotation matrices.

    Args:
        pos: Batch of position vectors with last dimension of 3.
        rot: Batch of rotation matrices with last 2 dimensions of (3, 3).

    Returns:
        Batch of pose matrices with last 2 dimensions of (4, 4).
    """
    assert isinstance(pos, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(rot, torch.Tensor), "Input must be a torch tensor"
    assert pos.shape[:-1] == rot.shape[:-2]
    assert pos.shape[-1] == rot.shape[-2] == rot.shape[-1] == 3
    pose = torch.zeros(pos.shape[:-1] + (4, 4), dtype=pos.dtype, device=pos.device)
    pose[..., :3, :3] = rot
    pose[..., :3, 3] = pos
    pose[..., 3, 3] = 1.0
    return pose


def unmake_pose(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits transformation matrices into positions and rotation matrices.

    Args:
        pose: Batch of pose matrices with last 2 dimensions of (4, 4).

    Returns:
        Tuple containing:
            - Batch of position vectors with last dimension of 3.
            - Batch of rotation matrices with last 2 dimensions of (3, 3).
    """
    assert isinstance(pose, torch.Tensor), "Input must be a torch tensor"
    return pose[..., :3, 3], pose[..., :3, :3]


def pose_inv(pose: torch.Tensor) -> torch.Tensor:
    """Computes the inverse of transformation matrices.

    The inverse of a pose matrix [R t; 0 1] is [R.T -R.T*t; 0 1].

    Args:
        pose: Batch of pose matrices with last 2 dimensions of (4, 4).

    Returns:
        Batch of inverse pose matrices with last 2 dimensions of (4, 4).
    """
    assert isinstance(pose, torch.Tensor), "Input must be a torch tensor"
    num_axes = len(pose.shape)
    assert num_axes >= 2

    inv_pose = torch.zeros_like(pose)

    # Take transpose of last 2 dimensions
    inv_pose[..., :3, :3] = pose[..., :3, :3].transpose(-1, -2)

    # note: PyTorch matmul wants shapes [..., 3, 3] x [..., 3, 1] -> [..., 3, 1] so we add a dimension and take it away after
    inv_pose[..., :3, 3] = torch.matmul(-inv_pose[..., :3, :3], pose[..., :3, 3:4])[..., 0]
    inv_pose[..., 3, 3] = 1.0
    return inv_pose


def pose_in_A_to_pose_in_B(pose_in_A: torch.Tensor, pose_A_in_B: torch.Tensor) -> torch.Tensor:
    """Converts poses from one coordinate frame to another.

    Transforms matrices representing point C in frame A
    to matrices representing the same point C in frame B.

    Example usage:

    frame_C_in_B = pose_in_A_to_pose_in_B(frame_C_in_A, frame_A_in_B)

    Args:
        pose_in_A: Batch of transformation matrices of point C in frame A.
        pose_A_in_B: Batch of transformation matrices of frame A in frame B.

    Returns:
        Batch of transformation matrices of point C in frame B.
    """
    assert isinstance(pose_in_A, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(pose_A_in_B, torch.Tensor), "Input must be a torch tensor"
    return torch.matmul(pose_A_in_B, pose_in_A)


def quat_slerp(q1: torch.Tensor, q2: torch.Tensor, tau: float) -> torch.Tensor:
    """Performs spherical linear interpolation (SLERP) between two quaternions.

    This function does not support batch processing.

    Args:
        q1: First quaternion in (x, y, z, w) format.
        q2: Second quaternion in (x, y, z, w) format.
        tau: Interpolation coefficient between 0 (q1) and 1 (q2).

    Returns:
        Interpolated quaternion in (x, y, z, w) format.
    """
    assert isinstance(q1, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(q2, torch.Tensor), "Input must be a torch tensor"
    if tau == 0.0:
        return q1
    elif tau == 1.0:
        return q2
    d = torch.dot(q1, q2)
    if abs(abs(d) - 1.0) < torch.finfo(q1.dtype).eps * 4.0:
        return q1
    if d < 0.0:
        # Invert rotation
        d = -d
        q2 *= -1.0
    angle = torch.acos(torch.clamp(d, -1, 1))
    if abs(angle) < torch.finfo(q1.dtype).eps * 4.0:
        return q1
    isin = 1.0 / torch.sin(angle)
    q1 = q1 * torch.sin((1.0 - tau) * angle) * isin
    q2 = q2 * torch.sin(tau * angle) * isin
    q1 = q1 + q2
    return q1


def interpolate_rotations(R1: torch.Tensor, R2: torch.Tensor, num_steps: int, axis_angle: bool = True) -> torch.Tensor:
    """Interpolates between two rotation matrices.

    Args:
        R1: First rotation matrix. (4x4).
        R2: Second rotation matrix. (4x4).
        num_steps: Number of desired interpolated rotations (excluding start and end).
        axis_angle: If True, interpolate in axis-angle representation;
                   otherwise use slerp. Defaults to True.

    Returns:
        Stack of interpolated rotation matrices of shape (num_steps + 1, 4, 4),
        including the start and end rotations.
    """
    assert isinstance(R1, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(R2, torch.Tensor), "Input must be a torch tensor"
    if axis_angle:
        # Delta rotation expressed as axis-angle
        delta_rot_mat = torch.matmul(R2, R1.transpose(-1, -2))
        delta_quat = quat_from_matrix(delta_rot_mat)
        delta_axis_angle = axis_angle_from_quat(delta_quat)

        # Grab angle
        delta_angle = torch.linalg.norm(delta_axis_angle)

        # Fix the axis, and chunk the angle up into steps
        rot_step_size = delta_angle / num_steps

        # Convert into delta rotation matrices, and then convert to absolute rotations
        if delta_angle < 0.05:
            # Small angle - don't bother with interpolation
            rot_steps = torch.stack([R2 for _ in range(num_steps)])
        else:
            # Make sure that axis is a unit vector
            delta_axis = delta_axis_angle / delta_angle
            delta_rot_steps = [
                matrix_from_quat(quat_from_angle_axis(i * rot_step_size, delta_axis)) for i in range(num_steps)
            ]
            rot_steps = torch.stack([torch.matmul(delta_rot_steps[i], R1) for i in range(num_steps)])
    else:
        q1 = quat_from_matrix(R1)
        q2 = quat_from_matrix(R2)
        rot_steps = torch.stack(
            [matrix_from_quat(quat_slerp(q1, q2, tau=float(i) / num_steps)) for i in range(num_steps)]
        )

    # Add in endpoint
    rot_steps = torch.cat([rot_steps, R2[None]], dim=0)

    return rot_steps


def interpolate_poses(
    pose_1: torch.Tensor,
    pose_2: torch.Tensor,
    num_steps: int = None,
    step_size: float = None,
    perturb: bool = False,
) -> tuple[torch.Tensor, int]:
    """Performs linear interpolation between two poses.

    Args:
        pose_1: 4x4 start pose.
        pose_2: 4x4 end pose.
        num_steps: If provided, specifies the number of desired interpolated points.
                  Passing 0 corresponds to no interpolation. If None, step_size must be provided.
        step_size: If provided, determines number of steps based on distance between poses.
        perturb: If True, randomly perturbs interpolated position points.

    Returns:
        Tuple containing:
            - Array of shape (N + 2, 4, 4) corresponding to the interpolated pose path.
            - Number of interpolated points (N) in the path.
    """
    assert isinstance(pose_1, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(pose_2, torch.Tensor), "Input must be a torch tensor"
    assert step_size is None or num_steps is None

    pos1, rot1 = unmake_pose(pose_1)
    pos2, rot2 = unmake_pose(pose_2)

    if num_steps == 0:
        # Skip interpolation
        return (
            torch.cat([pos1[None], pos2[None]], dim=0),
            torch.cat([rot1[None], rot2[None]], dim=0),
            num_steps,
        )

    delta_pos = pos2 - pos1
    if num_steps is None:
        assert torch.norm(delta_pos) > 0
        num_steps = math.ceil(torch.norm(delta_pos) / step_size)

    num_steps += 1  # Include starting pose
    assert num_steps >= 2

    # Linear interpolation of positions
    pos_step_size = delta_pos / num_steps
    grid = torch.arange(num_steps, dtype=torch.float32)
    if perturb:
        # Move interpolation grid points by up to half-size forward or backward
        perturbations = torch.rand(num_steps - 2) - 0.5
        grid[1:-1] += perturbations
    pos_steps = torch.stack([pos1 + grid[i] * pos_step_size for i in range(num_steps)])

    # Add in endpoint
    pos_steps = torch.cat([pos_steps, pos2[None]], dim=0)

    # Interpolate rotations
    rot_steps = interpolate_rotations(R1=rot1, R2=rot2, num_steps=num_steps, axis_angle=True)

    pose_steps = make_pose(pos_steps, rot_steps)
    return pose_steps, num_steps - 1


def transform_poses_from_frame_A_to_frame_B(
    src_poses: torch.Tensor, frame_A: torch.Tensor, frame_B: torch.Tensor
) -> torch.Tensor:
    """Transforms poses from one coordinate frame to another preserving relative poses.

    Args:
        src_poses: Input pose sequence (shape [T, 4, 4]) from source demonstration.
        frame_A: 4x4 frame A pose.
        frame_B: 4x4 frame B pose.

    Returns:
        Transformed pose sequence of shape [T, 4, 4].
    """
    # Transform source end effector poses to be relative to source object frame
    src_poses_rel_frame_B = pose_in_A_to_pose_in_B(
        pose_in_A=src_poses,
        pose_A_in_B=pose_inv(frame_B[None]),
    )

    # Apply relative poses to current object frame to obtain new target eef poses
    transformed_poses = pose_in_A_to_pose_in_B(
        pose_in_A=src_poses_rel_frame_B,
        pose_A_in_B=frame_A[None],
    )
    return transformed_poses


def generate_random_rotation(rot_boundary: float = (2 * math.pi)) -> torch.Tensor:
    """Generates a random rotation matrix using Euler angles.

    Args:
        rot_boundary: Range for random rotation angles around each axis (x, y, z).

    Returns:
        3x3 rotation matrix.
    """
    angles = torch.rand(3) * rot_boundary
    Rx = torch.tensor(
        [[1, 0, 0], [0, torch.cos(angles[0]), -torch.sin(angles[0])], [0, torch.sin(angles[0]), torch.cos(angles[0])]]
    )

    Ry = torch.tensor(
        [[torch.cos(angles[1]), 0, torch.sin(angles[1])], [0, 1, 0], [-torch.sin(angles[1]), 0, torch.cos(angles[1])]]
    )

    Rz = torch.tensor(
        [[torch.cos(angles[2]), -torch.sin(angles[2]), 0], [torch.sin(angles[2]), torch.cos(angles[2]), 0], [0, 0, 1]]
    )

    # Combined rotation matrix
    R = torch.matmul(torch.matmul(Rz, Ry), Rx)
    return R


def generate_random_translation(pos_boundary: float = 1) -> torch.Tensor:
    """Generates a random translation vector.

    Args:
        pos_boundary: Range for random translation values in 3D space.

    Returns:
        3-element translation vector.
    """
    return torch.rand(3) * 2 * pos_boundary - pos_boundary  # Random translation in 3D space


def generate_random_transformation_matrix(pos_boundary: float = 1, rot_boundary: float = (2 * math.pi)) -> torch.Tensor:
    """Generates a random transformation matrix combining rotation and translation.

    Args:
        pos_boundary: Range for random translation values.
        rot_boundary: Range for random rotation angles.

    Returns:
        4x4 transformation matrix.
    """
    R = generate_random_rotation(rot_boundary)
    translation = generate_random_translation(pos_boundary)

    # Create the transformation matrix
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    return T
