# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward and termination functions for the Franka deformable lifting environment.

Reward terms target either a volumetric/surface :class:`~isaaclab.assets.DeformableObject`
(particle data lives on ``data.nodal_pos_w`` / ``data.root_pos_w``) or a cable that is an
:class:`~isaaclab.assets.Articulation` whose per-segment positions live on ``data.body_pos_w``.
Each reward picks the right access path via :func:`_points_w` / :func:`_com_w`; downstream
math is shared.
"""

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


def _points_w(asset: DeformableObject | Articulation) -> torch.Tensor:
    """Return per-asset point cloud positions in world frame, shape ``[num_envs, K, 3]`` [m].

    For a deformable object, ``K`` is the number of FEM nodes. For a cable
    articulation, ``K`` is the number of segments.
    """
    if hasattr(asset.data, "nodal_pos_w"):
        return wp.to_torch(asset.data.nodal_pos_w)
    return asset.data.body_pos_w.torch


def _com_w(asset: DeformableObject | Articulation) -> torch.Tensor:
    """Return the asset's centre of mass in world frame, shape ``[num_envs, 3]`` [m].

    For a deformable object this is :attr:`DeformableObject.data.root_pos_w` (already
    the COM). For a cable articulation it is the mean of the per-segment positions.
    """
    if hasattr(asset.data, "nodal_pos_w"):
        return wp.to_torch(asset.data.root_pos_w)
    return asset.data.body_pos_w.torch.mean(dim=1)


def object_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward if the asset's COM is above a minimum height.

    Args:
        env: The environment instance.
        minimal_height: Minimum COM height [m].
        asset_cfg: The deformable or cable entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset = env.scene[asset_cfg.name]
    com_z = _com_w(asset)[:, 2]
    return torch.where(com_z > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward reaching the asset's nearest point with the end-effector.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation [m].
        asset_cfg: The deformable or cable entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    points_w = _points_w(asset)
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    distance = torch.linalg.norm(points_w - ee_w.unsqueeze(1), dim=2).min(dim=1).values
    return 1.0 - torch.tanh(distance / std)


def object_com_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of the goal position by the asset's COM (tanh kernel).

    Only credits when the COM is above ``minimal_height`` (i.e. the object is lifted).
    The command is interpreted as ``[x, y, z, qw, qx, qy, qz]`` in the robot's root frame.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w), des_pos_b
    )
    com_w = _com_w(asset)
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


def object_com_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Termination signal when the asset's COM falls below ``minimum_height`` [m]."""
    asset = env.scene[asset_cfg.name]
    com_z = _com_w(asset)[:, 2]
    return com_z < minimum_height


def object_outside_table_bounds(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate if any asset point leaves the table footprint.

    Args:
        env: The environment instance.
        x_bounds: Allowed x-position range in the environment frame [m].
        y_bounds: Allowed y-position range in the environment frame [m].
        asset_cfg: The deformable or cable entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    asset = env.scene[asset_cfg.name]
    points = _points_w(asset) - env.scene.env_origins.unsqueeze(1)
    outside_x = (points[..., 0] < x_bounds[0]) | (points[..., 0] > x_bounds[1])
    outside_y = (points[..., 1] < y_bounds[0]) | (points[..., 1] > y_bounds[1])
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
