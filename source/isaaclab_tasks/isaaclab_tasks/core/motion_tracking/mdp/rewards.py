# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from HybridRobotics/whole_body_tracking (MIT).
# See docs/licenses/dependencies/motion-tracking-license.txt.

"""Whole-body motion tracking rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward anchor position tracking in the world frame.

    ``reward = exp(-(dx**2 + dy**2 + dz**2) / std**2)``.

    All tracking rewards use reference-minus-actual deltas: ``dx/dy/dz`` for position,
    ``dvx/dvy/dvz`` for linear velocity and ``dwx/dwy/dwz`` for angular velocity.
    ``angle_error`` is the smallest rotation angle; ``mean`` averages over tracked bodies.
    ``std`` has the same units as the unsquared error (m, rad, m/s or rad/s);
    larger values tolerate larger errors. Rewards are 1 for a perfect match and approach 0 as errors grow.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    anchor_pos_w = command.anchor_pos_w
    robot_anchor_pos_w = command.robot_anchor_pos_w

    error = torch.sum(torch.square(anchor_pos_w - robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward anchor orientation tracking in the world frame.

    ``reward = exp(-angle_error**2 / std**2)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    anchor_quat_w = command.anchor_quat_w
    robot_anchor_quat_w = command.robot_anchor_quat_w

    error = quat_error_magnitude(anchor_quat_w, robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward body position tracking after horizontal position and yaw alignment.

    ``reward = exp(-mean(dx**2 + dy**2 + dz**2) / std**2)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_pos_w = command.body_pos_relative_w
    robot_body_pos_w = command.robot_body_pos_w

    error = torch.sum(torch.square(body_pos_w - robot_body_pos_w), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward body orientation tracking after reference yaw alignment.

    ``reward = exp(-mean(angle_error**2) / std**2)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_quat_w = command.body_quat_relative_w
    robot_body_quat_w = command.robot_body_quat_w

    error = quat_error_magnitude(body_quat_w, robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward body linear velocity tracking in the world frame.

    ``reward = exp(-mean(dvx**2 + dvy**2 + dvz**2) / std**2)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    error = torch.sum(torch.square(command.body_lin_vel_w - command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward body angular velocity tracking in the world frame.

    ``reward = exp(-mean(dwx**2 + dwy**2 + dwz**2) / std**2)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    error = torch.sum(torch.square(command.body_ang_vel_w - command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)
