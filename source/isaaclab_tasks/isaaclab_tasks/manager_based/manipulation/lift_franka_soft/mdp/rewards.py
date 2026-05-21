# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward and termination functions for the Franka deformable lifting environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def deformable_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Reward if the deformable COM is above a minimum height.

    Args:
        env: The environment instance.
        minimal_height: Minimum COM height [m].
        asset_cfg: The deformable object entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = wp.to_torch(asset.data.root_pos_w)[:, 2]
    return torch.where(com_z > minimal_height, 1.0, 0.0)


def deformable_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward reaching the deformable's nearest nodal point with the end-effector.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation [m].
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    nodal_pos_w = wp.to_torch(asset.data.nodal_pos_w)
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    distance = torch.linalg.norm(nodal_pos_w - ee_w.unsqueeze(1), dim=2).min(dim=1).values
    return 1.0 - torch.tanh(distance / std)


def deformable_com_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Reward tracking of the goal position by the deformable's COM (tanh kernel).

    Only credits when the COM is above ``minimal_height`` (i.e. the object is lifted).
    The command is interpreted as ``[x, y, z, qw, qx, qy, qz]`` in the robot's root frame.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    asset: DeformableObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w), des_pos_b
    )
    com_w = wp.to_torch(asset.data.root_pos_w)
    distance = torch.linalg.norm(des_pos_w - com_w, dim=1)
    return (com_w[:, 2] > minimal_height) * (1.0 - torch.tanh(distance / std))


def gripper_close_action(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """Penalty signal for commanding the gripper to close.

    The binary gripper action uses negative float actions for close commands and
    non-negative actions for open commands.

    Args:
        env: The environment instance.
        action_name: Name of the gripper action term.

    Returns:
        Tensor with shape ``(num_envs,)`` containing ``1`` when the gripper is
        commanded closed and ``0`` otherwise.
    """
    gripper_action = env.action_manager.get_term(action_name).raw_actions
    return torch.any(gripper_action < 0.0, dim=1).float()


def deformable_com_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Termination signal when the deformable's COM falls below ``minimum_height`` [m]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = wp.to_torch(asset.data.root_pos_w)[:, 2]
    return com_z < minimum_height


def deformable_outside_table_bounds(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Terminate if any deformable nodal point leaves the table footprint.

    Args:
        env: The environment instance.
        x_bounds: Allowed x-position range in the environment frame [m].
        y_bounds: Allowed y-position range in the environment frame [m].
        asset_cfg: The deformable object entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = wp.to_torch(asset.data.nodal_pos_w) - env.scene.env_origins.unsqueeze(1)
    outside_x = (nodal_pos[..., 0] < x_bounds[0]) | (nodal_pos[..., 0] > x_bounds[1])
    outside_y = (nodal_pos[..., 1] < y_bounds[0]) | (nodal_pos[..., 1] > y_bounds[1])
    return torch.any(outside_x | outside_y, dim=1)


def ee_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Termination signal when the end-effector falls below ``minimum_height`` [m].

    Height is measured in the environment frame (``z`` of the EE position with the env
    origin subtracted), so the threshold is independent of the environment's xy offset.

    Args:
        env: The environment instance.
        minimum_height: Minimum allowed EE height in the environment frame [m].
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, 2] - env.scene.env_origins[:, 2]
    return ee_z < minimum_height
