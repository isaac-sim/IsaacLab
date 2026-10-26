# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms of the MicroDuck on a ball."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_duck_on_ball(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    standing_height: float,
    xy_offset: float,
    tilt: float,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
):
    """Reset the ball to rest on the ground at the environment origin and the robot standing on top of it.

    Args:
        env: The environment.
        env_ids: Environments to reset.
        standing_height: Height of the robot base above the top of the ball [m].
        xy_offset: Maximum horizontal offset of the robot from the top of the ball [m].
        tilt: Maximum roll and pitch of the robot [rad]. Its yaw is uniform in (-pi, pi).
        robot_cfg: The robot.
        ball_cfg: The ball.
    """
    robot, ball = env.scene[robot_cfg.name], env.scene[ball_cfg.name]
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    count = len(env_ids)
    origins = env.scene.env_origins[env_ids]
    radius = ball.cfg.spawn.radius
    zero_velocity = torch.zeros(count, 6, device=env.device)

    ball_position = origins + torch.tensor([0.0, 0.0, radius], device=env.device)
    ball_orientation = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).repeat(count, 1)
    ball.write_root_pose_to_sim_index(root_pose=torch.cat([ball_position, ball_orientation], dim=-1), env_ids=env_ids)
    ball.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)

    roll_pitch = math_utils.sample_uniform(-tilt, tilt, (count, 2), env.device)
    yaw = math_utils.sample_uniform(-torch.pi, torch.pi, (count,), env.device)
    robot_orientation = math_utils.quat_from_euler_xyz(roll_pitch[:, 0], roll_pitch[:, 1], yaw)
    offset = math_utils.sample_uniform(-xy_offset, xy_offset, (count, 2), env.device)
    height = torch.full((count, 1), 2 * radius + standing_height, device=env.device)
    robot_position = origins + torch.cat([offset, height], dim=-1)
    robot.write_root_pose_to_sim_index(
        root_pose=torch.cat([robot_position, robot_orientation], dim=-1), env_ids=env_ids
    )
    robot.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)
