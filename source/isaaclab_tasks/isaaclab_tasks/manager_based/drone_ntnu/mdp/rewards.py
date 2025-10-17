# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

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

    # compute the error
    position_error_square = torch.sum(torch.square(target_position_w - current_position), dim=1)
    return torch.exp(-position_error_square / std**2)

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
    return velocity_towards_goal

def upright_posture_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # extract euler angles (in world frame)
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    
    # normalize angles to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    
    # compute deviation from upright (roll=0, pitch=0)
    orientation_error_square = roll**2 + pitch**2
    
    upright_reward = torch.exp(-orientation_error_square / std**2)
    return upright_reward

def ang_vel_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0
) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # compute squared magnitude of angular velocity (all axes)
    ang_vel_squared = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    
    angular_vel_reward = torch.exp(-ang_vel_squared / std**2)
    return angular_vel_reward

def yaw_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # extract yaw from current orientation
    _, _, current_yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    
    # normalize yaw to [-pi, pi] (target is 0)
    yaw_error = torch.atan2(torch.sin(current_yaw), torch.cos(current_yaw))
    
    # return exponential reward (1 when yaw=0, approaching 0 when rotated)
    return torch.exp(-yaw_error**2 / std**2)
