# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared utilities for Lee-style geometric controllers."""

import torch

import isaaclab.utils.math as math_utils


def compute_desired_orientation(
    forces_w: torch.Tensor, yaw_setpoint: torch.Tensor, rotation_matrix_buffer: torch.Tensor
) -> torch.Tensor:
    """Compute desired orientation from force direction and yaw setpoint.

    Args:
        forces_w: (num_envs, 3) desired force vector in world frame.
        yaw_setpoint: (num_envs,) desired yaw angle [rad].
        rotation_matrix_buffer: (num_envs, 3, 3) pre-allocated buffer for rotation matrix.

    Returns:
        (num_envs, 4) desired orientation quaternion (wxyz).
    """
    # Desired z-axis (thrust direction)
    b3_c = forces_w / torch.norm(forces_w, dim=1, keepdim=True)

    # Intermediate direction for yaw
    temp_dir = torch.zeros_like(forces_w)
    temp_dir[:, 0] = torch.cos(yaw_setpoint)
    temp_dir[:, 1] = torch.sin(yaw_setpoint)

    # Desired y-axis (orthogonal to thrust and yaw direction)
    b2_c = torch.cross(b3_c, temp_dir, dim=1)
    b2_c = b2_c / torch.norm(b2_c, dim=1, keepdim=True)

    # Desired x-axis (complete right-handed frame)
    b1_c = torch.cross(b2_c, b3_c, dim=1)

    # Build rotation matrix
    rotation_matrix_buffer[:, :, 0] = b1_c
    rotation_matrix_buffer[:, :, 1] = b2_c
    rotation_matrix_buffer[:, :, 2] = b3_c

    # Convert to quaternion
    return math_utils.quat_from_matrix(rotation_matrix_buffer)


def compute_body_torque(
    setpoint_orientation: torch.Tensor,
    setpoint_angvel_b: torch.Tensor,
    current_quat: torch.Tensor,
    current_angvel_b: torch.Tensor,
    robot_inertia: torch.Tensor,
    K_rot: torch.Tensor,
    K_angvel: torch.Tensor,
    max_yaw_rate: float,
) -> torch.Tensor:
    """PD attitude control in body frame with feedforward Coriolis term.

    Args:
        setpoint_orientation: (num_envs, 4) desired orientation quaternion (wxyz) in world frame.
        setpoint_angvel_b: (num_envs, 3) desired angular velocity in body frame [rad/s].
        current_quat: (num_envs, 4) current orientation quaternion (wxyz).
        current_angvel_b: (num_envs, 3) current angular velocity in body frame [rad/s].
        robot_inertia: (num_envs, 3, 3) robot inertia matrix.
        K_rot: (num_envs, 3) rotation gain.
        K_angvel: (num_envs, 3) angular velocity gain.
        max_yaw_rate: Maximum yaw rate [rad/s].

    Returns:
        (num_envs, 3) body torque command [N·m].
    """
    # Clamp yaw rate
    setpoint_angvel_b[:, 2] = torch.clamp(setpoint_angvel_b[:, 2], -max_yaw_rate, max_yaw_rate)

    # Compute orientation error (R^T @ R_d)
    RT_Rd_quat = math_utils.quat_mul(math_utils.quat_inv(current_quat), setpoint_orientation)
    R_err = math_utils.matrix_from_quat(RT_Rd_quat)

    # Extract rotation error vector from skew-symmetric part
    skew_matrix = R_err.transpose(-1, -2) - R_err
    rotation_error = 0.5 * torch.stack([-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1)

    # Angular velocity error
    angvel_error = current_angvel_b - setpoint_angvel_b

    # Coriolis feedforward term: ω × (I·ω)
    inertia_angvel = torch.bmm(robot_inertia, current_angvel_b.unsqueeze(2)).squeeze(2)
    coriolis_term = torch.cross(current_angvel_b, inertia_angvel, dim=1)

    # PD + feedforward
    torque = -K_rot * rotation_error - K_angvel * angvel_error + coriolis_term
    return torque


def yaw_rate_to_body_angvel(
    yaw_rate: torch.Tensor, roll: torch.Tensor, pitch: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Convert yaw rate command to body angular velocity.

    Transformation: ω_body = T(roll, pitch) @ [0, 0, yaw_rate]^T
    where T is the euler-to-body rate transformation matrix.

    Args:
        yaw_rate: (num_envs,) desired yaw rate [rad/s].
        roll: (num_envs,) current roll angle [rad].
        pitch: (num_envs,) current pitch angle [rad].
        device: Device to allocate tensors on.

    Returns:
        (num_envs, 3) desired angular velocity in body frame [rad/s].
    """
    s_pitch = torch.sin(pitch)
    c_pitch = torch.cos(pitch)
    s_roll = torch.sin(roll)
    c_roll = torch.cos(roll)

    # Only yaw rate is non-zero, so only the third column matters
    # ω_body = [−sin(pitch), sin(roll)*cos(pitch), cos(roll)*cos(pitch)]^T * yaw_rate
    angvel_b = torch.zeros((yaw_rate.shape[0], 3), device=device)
    angvel_b[:, 0] = -s_pitch * yaw_rate
    angvel_b[:, 1] = s_roll * c_pitch * yaw_rate
    angvel_b[:, 2] = c_roll * c_pitch * yaw_rate

    return angvel_b
