# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel with improvements."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    
    # IMPROVEMENT 1: Add adaptive scaling based on episode progress
    episode_progress = env.episode_length_buf.float() / env.max_episode_length
    std_adaptive = std * (1.0 + 0.1 * episode_progress)
    
    # IMPROVEMENT 2: Calculate reward with clipping to prevent extreme values
    reward = 1 - torch.tanh(object_ee_distance / std_adaptive)
    reward = torch.clamp(reward, 0.0, 1.0)
    
    return reward


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel with improvements."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    
    # IMPROVEMENT 1: Check if object is lifted
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height
    
    # IMPROVEMENT 2: Add velocity stability bonus
    velocity = torch.norm(object.data.root_lin_vel_w, dim=1)
    velocity_bonus = torch.exp(-2.0 * velocity)  # Reward stability
    
    # IMPROVEMENT 3: Combined reward with clipping
    distance_reward = 1 - torch.tanh(distance / std)
    combined_reward = is_lifted.float() * distance_reward * velocity_bonus
    combined_reward = torch.clamp(combined_reward, 0.0, 1.0)
    
    return combined_reward


def action_smoothness_penalty(
    env: ManagerBasedRLEnv,
    penalty_scale: float = 0.01,
) -> torch.Tensor:
    """NEW: Penalize large action changes to encourage smooth movements."""
    if not hasattr(env, '_prev_actions'):
        env._prev_actions = torch.zeros_like(env.action_manager.action)
        return torch.zeros(env.num_envs, device=env.device)
    
    action_diff = torch.norm(env.action_manager.action - env._prev_actions, dim=1)
    env._prev_actions = env.action_manager.action.clone()
    
    penalty = -penalty_scale * action_diff
    return penalty


def grasp_success_bonus(
    env: ManagerBasedRLEnv,
    bonus_value: float = 2.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """NEW: Provide large bonus when object is successfully grasped and stable."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Check if object is close to gripper
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)
    
    # Check if object velocity is low (stable grasp)
    velocity = torch.norm(object.data.root_lin_vel_w, dim=1)
    
    # Grasp is successful if distance < 0.05m and velocity < 0.1 m/s
    successful_grasp = (distance < 0.05) & (velocity < 0.1)
    
    return torch.where(successful_grasp, torch.tensor(bonus_value, device=env.device), torch.tensor(0.0, device=env.device))
