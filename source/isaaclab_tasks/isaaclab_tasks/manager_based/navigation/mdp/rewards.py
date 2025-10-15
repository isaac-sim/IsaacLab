# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()

"""
Drone-navigation rewards.
"""

def distance_to_goal_exp(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        std: float = 1.0,
        command_name: str = "target_pose"
    ) -> torch.Tensor:
    """Reward the distance to a goal position using an exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins
    # weight based on the current curriculum level
    weight = 1.0 + env.scene.terrain.terrain_levels.float() / float(env.scene.terrain.max_terrain_level)

    # compute the error
    position_error_square = torch.sum(torch.square(target_position_w - current_position), dim=1)
    return weight * torch.exp(-position_error_square / std**2)

def velocity_to_goal_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "target_pose"
) -> torch.Tensor:
    """Reward the velocity towards a goal position using a dot product between the velocity and the direction to the goal."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # get the center of the environment
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins
    direction_to_goal = target_position_w - current_position
    direction_to_goal = direction_to_goal / (torch.norm(direction_to_goal, dim=1, keepdim=True) + 1e-8)
    # compute the reward as the dot product between the velocity and the direction to the goal
    velocity_towards_goal = torch.sum(asset.data.root_lin_vel_w * direction_to_goal, dim=1)
    weight = 1.0 + env.scene.terrain.terrain_levels.float() / float(env.scene.terrain.max_terrain_level)
    return weight * velocity_towards_goal