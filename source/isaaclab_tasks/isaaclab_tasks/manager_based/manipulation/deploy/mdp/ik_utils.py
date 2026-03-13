# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK utilities for gear assembly task on Newton.

This module replaces the dependency on ``factory_control.py`` (which uses
``isaacsim.core.utils.torch`` and ``root_physx_view.get_jacobians()``).

All quaternion operations use the XYZW convention native to isaaclab.utils.math.
"""

from __future__ import annotations

import torch

import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation


def get_pose_error(
    fingertip_midpoint_pos: torch.Tensor,
    fingertip_midpoint_quat: torch.Tensor,
    ctrl_target_fingertip_midpoint_pos: torch.Tensor,
    ctrl_target_fingertip_midpoint_quat: torch.Tensor,
    jacobian_type: str = "geometric",
    rot_error_type: str = "axis_angle",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pose error between current and target end-effector poses.

    Replaces ``factory_control.get_pose_error()``. Uses ``isaaclab.utils.math``
    functions which operate in XYZW quaternion convention.

    Args:
        fingertip_midpoint_pos: Current end-effector position. Shape (N, 3).
        fingertip_midpoint_quat: Current end-effector orientation in XYZW. Shape (N, 4).
        ctrl_target_fingertip_midpoint_pos: Target end-effector position. Shape (N, 3).
        ctrl_target_fingertip_midpoint_quat: Target end-effector orientation in XYZW. Shape (N, 4).
        jacobian_type: Type of Jacobian ("geometric" supported).
        rot_error_type: Type of rotation error ("axis_angle" supported).

    Returns:
        A tuple of (pos_error, rot_error) each of shape (N, 3).
    """
    # Position error
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    # Rotation error as axis-angle
    # q_error = q_target * q_current^{-1}
    fingertip_quat_inv = math_utils.quat_conjugate(fingertip_midpoint_quat)
    quat_error = math_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_quat_inv)

    # Ensure shortest path: if w < 0, negate (w is at index 3 in XYZW)
    quat_error = quat_error * (1.0 - 2.0 * (quat_error[:, 3:4] < 0.0).float())

    # Convert quaternion error to axis-angle
    rot_error = math_utils.axis_angle_from_quat(quat_error)

    return pos_error, rot_error


def compute_numerical_jacobian(
    robot: Articulation,
    arm_joint_ids: list[int],
    eef_body_idx: int,
    env_ids: torch.Tensor,
    delta: float = 1e-6,
) -> torch.Tensor:
    """Compute 6xN geometric Jacobian via finite differences.

    For each of N arm joints, perturbs the joint by +delta and computes the
    resulting end-effector pose change. Uses the robot's body transforms
    (Warp arrays converted to torch).

    Args:
        robot: The robot articulation asset.
        arm_joint_ids: List of arm joint indices.
        eef_body_idx: Body index of the end-effector.
        env_ids: Environment IDs to compute Jacobian for.
        delta: Perturbation size for finite differences.

    Returns:
        Jacobian tensor of shape (num_envs, 6, num_arm_joints).
    """
    num_envs = len(env_ids)
    num_joints = len(arm_joint_ids)
    device = robot.device

    jacobian = torch.zeros(num_envs, 6, num_joints, device=device, dtype=torch.float32)

    # Get current joint positions
    current_joint_pos = wp.to_torch(robot.data.joint_pos)[env_ids].clone()

    # Get current end-effector pose
    eef_pos_current = wp.to_torch(robot.data.body_link_pos_w)[env_ids, eef_body_idx].clone()
    eef_quat_current = wp.to_torch(robot.data.body_link_quat_w)[env_ids, eef_body_idx].clone()

    for i, joint_id in enumerate(arm_joint_ids):
        # Perturb joint i by +delta
        perturbed_joint_pos = current_joint_pos.clone()
        perturbed_joint_pos[:, joint_id] += delta

        # Write perturbed state to sim and step forward kinematics
        joint_vel = torch.zeros_like(perturbed_joint_pos)
        robot.write_joint_state_to_sim(perturbed_joint_pos, joint_vel, env_ids=env_ids)

        # Get perturbed end-effector pose
        eef_pos_perturbed = wp.to_torch(robot.data.body_link_pos_w)[env_ids, eef_body_idx]
        eef_quat_perturbed = wp.to_torch(robot.data.body_link_quat_w)[env_ids, eef_body_idx]

        # Linear velocity component (position change / delta)
        jacobian[:, :3, i] = (eef_pos_perturbed - eef_pos_current) / delta

        # Angular velocity component (orientation change / delta)
        # q_delta = q_perturbed * q_current^{-1}
        quat_current_inv = math_utils.quat_conjugate(eef_quat_current)
        quat_delta = math_utils.quat_mul(eef_quat_perturbed, quat_current_inv)
        axis_angle_delta = math_utils.axis_angle_from_quat(quat_delta)
        jacobian[:, 3:, i] = axis_angle_delta / delta

    # Restore original joint positions
    joint_vel = torch.zeros_like(current_joint_pos)
    robot.write_joint_state_to_sim(current_joint_pos, joint_vel, env_ids=env_ids)

    return jacobian


def solve_ik_dls(
    jacobian: torch.Tensor,
    delta_pose: torch.Tensor,
    lambda_val: float = 0.1,
) -> torch.Tensor:
    """Damped least squares IK solver.

    Computes: delta_q = J^T @ inv(J @ J^T + lambda^2 * I) @ delta_pose

    Direct port of ``factory_control._get_delta_dof_pos(ik_method='dls')``.

    Args:
        jacobian: Jacobian matrix of shape (N, 6, num_joints).
        delta_pose: Desired pose change of shape (N, 6).
        lambda_val: Damping factor. Defaults to 0.1.

    Returns:
        Joint position deltas of shape (N, num_joints).
    """
    # J^T: (N, num_joints, 6)
    j_transpose = torch.transpose(jacobian, 1, 2)

    # J @ J^T: (N, 6, 6)
    jjt = torch.bmm(jacobian, j_transpose)

    # Add damping: J @ J^T + lambda^2 * I
    eye = torch.eye(6, device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0)
    jjt_damped = jjt + (lambda_val**2) * eye

    # inv(J @ J^T + lambda^2 * I) @ delta_pose
    # delta_pose: (N, 6) -> (N, 6, 1)
    delta_pose_unsqueezed = delta_pose.unsqueeze(-1)
    solved = torch.linalg.solve(jjt_damped, delta_pose_unsqueezed)

    # J^T @ solved: (N, num_joints, 1) -> (N, num_joints)
    delta_dof_pos = torch.bmm(j_transpose, solved).squeeze(-1)

    return delta_dof_pos
