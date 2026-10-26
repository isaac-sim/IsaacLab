# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms of the MicroDuck on a ball."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def ball_pos_b(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ball center relative to the middle of the feet, in the robot heading frame [m], shape (num_envs, 2)."""
    robot, ball = env.scene[robot_cfg.name], env.scene[ball_cfg.name]
    feet_center = robot.data.body_pos_w.torch[:, robot_cfg.body_ids].mean(dim=1)
    delta = ball.data.root_pos_w.torch - feet_center
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w.torch), delta)[:, :2]


def ball_vel_b(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ball velocity relative to the robot base, in the robot heading frame [m/s], shape (num_envs, 2)."""
    robot, ball = env.scene[robot_cfg.name], env.scene[ball_cfg.name]
    delta = ball.data.root_lin_vel_w.torch - robot.data.root_lin_vel_w.torch
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w.torch), delta)[:, :2]


def ball_ang_vel_w(env: ManagerBasedEnv, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ball angular velocity in the world frame [rad/s], shape (num_envs, 3)."""
    return env.scene[ball_cfg.name].data.root_ang_vel_w.torch


def gait_clock(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Sine and cosine of the stepping clock of the velocity command, shape (num_envs, 2)."""
    return env.command_manager.get_term(command_name).gait_clock
