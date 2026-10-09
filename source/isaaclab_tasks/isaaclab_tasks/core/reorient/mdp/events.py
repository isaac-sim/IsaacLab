# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the in-hand reorientation environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from ..utils import sample_joint_positions_within_limits

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def reset_reorient_hand(
    env: ManagerBasedRLEnv,
    env_mask: torch.Tensor,
    joint_position_noise: float,
    joint_velocity_noise: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the hand's joints and the position targets tracking them.

    Task-local rather than :func:`~isaaclab.envs.mdp.reset_joints_by_offset`: the
    hand's PD targets must be re-seeded alongside the joint state, and the framework
    terms write joint state only.

    Args:
        env: Environment containing the robot.
        env_mask: Boolean mask of the environments to reset. Shape is (num_envs,).
        joint_position_noise: Scale applied to sampled joint-position deltas.
        joint_velocity_noise: Joint-velocity noise half-width [rad/s].
        robot_cfg: Robot scene entity.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # sample for every environment and write only the selected ones
    default_position = robot.data.default_joint_pos.torch
    limits = robot.data.joint_limits.torch
    joint_position = sample_joint_positions_within_limits(default_position, limits, joint_position_noise)
    velocity_sample = math_utils.sample_uniform(-1.0, 1.0, default_position.shape, device=env.device)
    joint_velocity = robot.data.default_joint_vel.torch + joint_velocity_noise * velocity_sample
    robot.actuators.target_command.set_position_mask(
        value=joint_position, env_mask=wp.from_torch(env_mask, dtype=wp.bool)
    )
    robot.write_joint_position_to_sim_mask(position=joint_position, env_mask=env_mask)
    robot.write_joint_velocity_to_sim_mask(velocity=joint_velocity, env_mask=env_mask)
