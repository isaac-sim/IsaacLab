# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 the SuperMimic authors.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Gaussian-kernel tracking reward on world-frame bodies.

Each tracking term is
``weight * exp(-err_sq / std**2)``, plus the action-rate and joint-limit
penalties; body pos/ori targets are re-anchored
to the predicted root xy + yaw, matching
``MotionCommand.update_relative_body_poses``.
"""

from __future__ import annotations

import torch

from isaaclab_rl.model_based.kinematics import (
    matrix_from_quat,
    quat_inv,
    quat_mul,
    rotation_error_magnitude,
    small_matmul,
    yaw_quat,
)


def _reanchor_targets(
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    pred_root_pos: torch.Tensor,
    pred_root_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-anchor target body poses to predicted root xy + yaw.

    Target bodies are expressed in a frame whose xy + yaw come from the
    predicted root and whose z + roll/pitch come from the target root.
    """
    anchor_pos = target_pos[..., 0:1, :]

    delta_pos = torch.cat([pred_root_pos[..., :2], target_pos[..., 0, 2:3]], dim=-1)
    delta_mat = matrix_from_quat(yaw_quat(quat_mul(pred_root_quat, quat_inv(target_quat[..., 0, :])))).unsqueeze(-3)

    rel_pos = target_pos - anchor_pos
    tgt_pos = delta_pos.unsqueeze(-2) + small_matmul(delta_mat, rel_pos.unsqueeze(-1))[..., 0]
    tgt_mat = small_matmul(delta_mat, matrix_from_quat(target_quat))
    return tgt_pos, tgt_mat


def tracking_reward(
    aux_state: torch.Tensor,
    pred_pos: torch.Tensor,
    pred_rot_mat: torch.Tensor,
    pred_lin_vel: torch.Tensor,
    pred_ang_vel: torch.Tensor,
    joint_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    target_lin_vel: torch.Tensor,
    target_ang_vel: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Gaussian-kernel reward matching the motion-tracking env.

    ``pred_*`` are the predicted bodies in the world frame and ``aux_state`` the
    predicted root pose the targets are re-anchored to.
    """
    tgt_pos, tgt_mat = _reanchor_targets(target_pos, target_quat, aux_state[..., :3], aux_state[..., 3:7])

    terms: list[tuple[torch.Tensor, float, float]] = [
        (torch.sum((pred_pos[..., 0, :] - target_pos[..., 0, :]) ** 2, dim=-1), 0.5, 0.3),
        (
            rotation_error_magnitude(pred_rot_mat[..., 0, :, :], matrix_from_quat(target_quat[..., 0, :])) ** 2,
            0.5,
            0.4,
        ),
        (torch.sum((pred_pos - tgt_pos) ** 2, dim=-1).mean(-1), 1.0, 0.3),
        ((rotation_error_magnitude(pred_rot_mat, tgt_mat) ** 2).mean(-1), 1.0, 0.4),
        (torch.sum((pred_lin_vel - target_lin_vel) ** 2, dim=-1).mean(-1), 1.0, 1.0),
        (torch.sum((pred_ang_vel - target_ang_vel) ** 2, dim=-1).mean(-1), 1.0, 3.14),
    ]
    reward = sum(w * torch.exp(-err / std**2) for err, w, std in terms)
    action_rate = torch.sum((action - last_action) ** 2, dim=-1)
    joint_pos, _ = joint_state.chunk(2, dim=-1)
    lo, hi = soft_joint_pos_limits.unbind(-1)
    joint_limit = ((lo - joint_pos).clamp(min=0) + (joint_pos - hi).clamp(min=0)).sum(-1)
    return dt * (reward - 0.1 * action_rate - 2.0 * joint_limit)
