# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    maximal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(
        (object.data.root_pos_w[:, 2] > minimal_height) & (object.data.root_pos_w[:, 2] < maximal_height), 1.0, 0.0
    )


def object_is_placed(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    height_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    place_command = env.command_manager.get_command("place_pose")

    object_pos = object.data.root_pos_w
    target_pos = place_command[:, :3]

    # check xy-distance to target
    xy_distance = torch.norm(object_pos[:, :2] - target_pos[:, :2], dim=1)

    # check height difference
    height_diff = torch.abs(object_pos[:, 2] - target_pos[:, 2])

    # return 1.0 if within thresholds, 0.0 otherwise
    return torch.where((xy_distance < distance_threshold) & (height_diff < height_threshold), 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    maximal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold

    if command_name == "object_pose":
        # for lifting - reward when above minimal height
        return ((object.data.root_pos_w[:, 2] > minimal_height) & (object.data.root_pos_w[:, 2] < maximal_height)) * (
            1 - torch.tanh(distance / std)
        )
    elif command_name == "place_pose":
        # for placing - reward getting close to place position
        target_height = des_pos_w[:, 2]  # Get height from command
        height_diff = torch.abs(object.data.root_pos_w[:, 2] - target_height)
        return 1 - torch.tanh((distance + height_diff) / std)
