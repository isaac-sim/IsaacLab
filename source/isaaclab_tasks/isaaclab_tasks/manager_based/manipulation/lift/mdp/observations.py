# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
# from isaaclab.cube_prediction_mv.play import load_model
# from isaaclab.cube_prediction_mv.play import play

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    print(object_pos_b)
    return object_pos_b

# def object_position_prediction_model() -> torch.Tensor:
#     position = play(env.scene["camera_bird"], env.scene["camera_ext1"], env.scene["camera_ext2"], model)
#     return position

def log_object_position(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Log object position (used in object_is_lifted and object_goal_distance)."""
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w  # Shape: [num_envs, 3]

def log_ee_position(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """Log end-effector position (used in object_ee_distance)."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]  # Shape: [num_envs, 3]

def log_goal_position(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Log goal position (used in object_goal_distance)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    return des_pos_w  # Shape: [num_envs, 3]

def log_joint_velocities(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Log joint velocities for the specified joints (used in joint_vel_l2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]  # Shape: [num_envs, num_joints]

def log_current_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Log current action (used in action_rate_l2)."""
    return env.action_manager.action  # Shape: [num_envs, action_dim]

def log_previous_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Log previous action (used in action_rate_l2)."""
    return env.action_manager.prev_action  # Shape: [num_envs, action_dim]

def log_action_difference(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Log action difference (used in action_rate_l2)."""
    return env.action_manager.action - env.action_manager.prev_action  # Shape: [num_envs, action_dim]