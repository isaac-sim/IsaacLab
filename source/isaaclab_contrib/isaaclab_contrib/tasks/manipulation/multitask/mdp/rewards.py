# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for multi-task environments.

Group-local kernels use the ``@scatterable`` decorator (see :func:`.utils.scatterable`).
Each wrapped function computes on ``SceneEntityCfg.view_ids`` rows and returns
``(env_ids, group_local_result)``; the decorator scatters into a full-env tensor
of shape ``(num_envs,)`` (or matching trailing dims).

Terms that compose already-scattered full-env tensors (for example tanh wrappers)
remain plain functions.

All terms exposed to the reward manager return shape ``(num_envs,)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils

from .utils import ScatterResult, scatterable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


# ===========================================================
# Joint-space rewards
# ===========================================================


@scatterable
def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> ScatterResult:
    """Joint velocity L2 penalty."""
    robot = env.scene[asset_cfg.name]
    vel_data = wp.to_torch(robot.data.joint_vel)[asset_cfg.view_ids]
    if asset_cfg.joint_ids != slice(None):
        vel_data = vel_data[:, asset_cfg.joint_ids]
    return asset_cfg.env_ids, torch.sum(vel_data.square(), dim=-1)


# ===========================================================
# Position tracking rewards
# ===========================================================


@scatterable
def position_command_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "ee_pose",
) -> ScatterResult:
    """Position command tracking L2 error [m]."""
    robot = env.scene[asset_cfg.name]
    body_idx = asset_cfg.body_ids[0]
    root_pos = wp.to_torch(robot.data.root_pos_w)[asset_cfg.view_ids]
    root_quat = wp.to_torch(robot.data.root_quat_w)[asset_cfg.view_ids]
    body_pos = wp.to_torch(robot.data.body_pos_w)[asset_cfg.view_ids, body_idx]
    cmd_pos = env.command_manager.get_command(command_name)[asset_cfg.env_ids, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(root_pos, root_quat, cmd_pos)
    return asset_cfg.env_ids, torch.linalg.norm(body_pos - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    *,
    asset_cfg: SceneEntityCfg,
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Position command tracking with tanh kernel."""
    return 1.0 - torch.tanh(position_command_error(env, asset_cfg=asset_cfg, command_name=command_name) / std)


# ===========================================================
# Orientation tracking rewards
# ===========================================================


@scatterable
def orientation_command_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "ee_pose",
) -> ScatterResult:
    """Orientation command tracking error (shortest path)."""
    robot = env.scene[asset_cfg.name]
    body_idx = asset_cfg.body_ids[0]
    root_quat = wp.to_torch(robot.data.root_quat_w)[asset_cfg.view_ids]
    body_quat = wp.to_torch(robot.data.body_quat_w)[asset_cfg.view_ids, body_idx]
    cmd_quat = env.command_manager.get_command(command_name)[asset_cfg.env_ids, 3:7]
    des_quat_w = math_utils.quat_mul(root_quat, cmd_quat)
    return asset_cfg.env_ids, math_utils.quat_error_magnitude(body_quat, des_quat_w)


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    *,
    asset_cfg: SceneEntityCfg,
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Orientation command tracking with tanh kernel."""
    return 1.0 - torch.tanh(orientation_command_error(env, asset_cfg=asset_cfg, command_name=command_name) / std)


# ===========================================================
# Object manipulation rewards
# ===========================================================


@scatterable
def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    *,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Object-to-EE distance with tanh kernel."""
    obj_pos = wp.to_torch(env.scene[object_cfg.name].data.root_pos_w)[object_cfg.view_ids]
    ee_pos = wp.to_torch(env.scene[ee_frame_cfg.name].data.target_pos_w)[ee_frame_cfg.view_ids, 0, :]
    norm = torch.linalg.norm(obj_pos - ee_pos, dim=1)
    return object_cfg.env_ids, 1.0 - torch.tanh(norm / std)


@scatterable
def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.04,
    *,
    object_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Object lifted above threshold."""
    height = wp.to_torch(env.scene[object_cfg.name].data.root_pos_w)[object_cfg.view_ids, 2]
    return object_cfg.env_ids, torch.where(height > minimal_height, 1.0, 0.0)


@scatterable
def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    minimal_height: float = 0.04,
    command_name: str = "ee_pose",
    *,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Object-to-goal distance with tanh kernel."""
    robot = env.scene[robot_cfg.name]
    rigid_object = env.scene[object_cfg.name]
    root_pos = wp.to_torch(robot.data.root_pos_w)[robot_cfg.view_ids]
    root_quat = wp.to_torch(robot.data.root_quat_w)[robot_cfg.view_ids]
    obj_pos = wp.to_torch(rigid_object.data.root_pos_w)[object_cfg.view_ids]
    cmd_pos = env.command_manager.get_command(command_name)[robot_cfg.env_ids, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(root_pos, root_quat, cmd_pos)
    dist = torch.linalg.norm(des_pos_w - obj_pos, dim=1)
    return robot_cfg.env_ids, (obj_pos[:, 2] > minimal_height) * (1.0 - torch.tanh(dist / std))


# ===========================================================
# Cabinet rewards
# ===========================================================


@scatterable
def cabinet_approach_ee_handle(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    *,
    ee_frame_cfg: SceneEntityCfg,
    cabinet_frame_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Reward for reaching the cabinet handle."""
    ee_pos = wp.to_torch(env.scene[ee_frame_cfg.name].data.target_pos_w)[ee_frame_cfg.view_ids, 0, :]
    handle_pos = wp.to_torch(env.scene[cabinet_frame_cfg.name].data.target_pos_w)[cabinet_frame_cfg.view_ids, 0, :]
    distance = torch.linalg.norm(handle_pos - ee_pos, dim=-1, ord=2)
    reward = torch.pow(1.0 / (1.0 + distance**2), 2)
    reward = torch.where(distance <= threshold, 2 * reward, reward)
    return ee_frame_cfg.env_ids, reward


@scatterable
def cabinet_align_ee_handle(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    cabinet_frame_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Reward for aligning with the cabinet handle."""
    ee_quat = wp.to_torch(env.scene[ee_frame_cfg.name].data.target_quat_w)[ee_frame_cfg.view_ids, 0, :]
    handle_quat = wp.to_torch(env.scene[cabinet_frame_cfg.name].data.target_quat_w)[cabinet_frame_cfg.view_ids, 0, :]
    ee_rot = math_utils.matrix_from_quat(ee_quat)
    handle_rot = math_utils.matrix_from_quat(handle_quat)
    handle_x, handle_y = handle_rot[..., 0], handle_rot[..., 1]
    ee_x, ee_z = ee_rot[..., 0], ee_rot[..., 2]
    align_z = torch.bmm(ee_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return ee_frame_cfg.env_ids, 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


@scatterable
def cabinet_align_grasp_around_handle(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    cabinet_frame_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Bonus when fingers straddle the drawer handle."""
    handle_pos = wp.to_torch(env.scene[cabinet_frame_cfg.name].data.target_pos_w)[cabinet_frame_cfg.view_ids, 0, :]
    fingertips = wp.to_torch(env.scene[ee_frame_cfg.name].data.target_pos_w)[ee_frame_cfg.view_ids, 1:, :]
    left = fingertips[:, 0, :]
    right = fingertips[:, 1, :]
    is_grasp = (right[:, 2] < handle_pos[:, 2]) & (left[:, 2] > handle_pos[:, 2])
    return ee_frame_cfg.env_ids, is_grasp.float()


@scatterable
def cabinet_approach_gripper_handle(
    env: ManagerBasedRLEnv,
    offset: float = 0.04,
    *,
    ee_frame_cfg: SceneEntityCfg,
    cabinet_frame_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Reward for finger placement around the handle."""
    handle_pos = wp.to_torch(env.scene[cabinet_frame_cfg.name].data.target_pos_w)[cabinet_frame_cfg.view_ids, 0, :]
    fingertips = wp.to_torch(env.scene[ee_frame_cfg.name].data.target_pos_w)[ee_frame_cfg.view_ids, 1:, :]
    left = fingertips[:, 0, :]
    right = fingertips[:, 1, :]
    left_dist = torch.abs(left[:, 2] - handle_pos[:, 2])
    right_dist = torch.abs(right[:, 2] - handle_pos[:, 2])
    is_graspable = (right[:, 2] < handle_pos[:, 2]) & (left[:, 2] > handle_pos[:, 2])
    return ee_frame_cfg.env_ids, is_graspable * ((offset - left_dist) + (offset - right_dist))


@scatterable
def cabinet_grasp_handle(
    env: ManagerBasedRLEnv,
    threshold: float = 0.03,
    open_joint_pos: float = 0.04,
    *,
    asset_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    cabinet_frame_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Reward for closing fingers near the handle."""
    ee_pos = wp.to_torch(env.scene[ee_frame_cfg.name].data.target_pos_w)[ee_frame_cfg.view_ids, 0, :]
    handle_pos = wp.to_torch(env.scene[cabinet_frame_cfg.name].data.target_pos_w)[cabinet_frame_cfg.view_ids, 0, :]
    robot = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    joint_pos_g = wp.to_torch(robot.data.joint_pos)[asset_cfg.view_ids, :]
    gripper = torch.sum(open_joint_pos - joint_pos_g[:, joint_ids], dim=-1)
    distance = torch.linalg.norm(handle_pos - ee_pos, dim=-1, ord=2)
    return asset_cfg.env_ids, (distance <= threshold) * gripper


@scatterable
def cabinet_open_drawer_bonus(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    cabinet_frame_cfg: SceneEntityCfg,
    cabinet_asset_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Drawer opening bonus."""
    cabinet = env.scene[cabinet_asset_cfg.name]
    drawer_pos = wp.to_torch(cabinet.data.joint_pos)[cabinet_asset_cfg.view_ids, cabinet_asset_cfg.joint_ids].squeeze(
        -1
    )
    handle_pos = wp.to_torch(env.scene[cabinet_frame_cfg.name].data.target_pos_w)[cabinet_frame_cfg.view_ids, 0, :]
    fingertips = wp.to_torch(env.scene[ee_frame_cfg.name].data.target_pos_w)[ee_frame_cfg.view_ids, 1:, :]
    left = fingertips[:, 0, :]
    right = fingertips[:, 1, :]
    is_graspable = ((right[:, 2] < handle_pos[:, 2]) & (left[:, 2] > handle_pos[:, 2])).float()
    return cabinet_asset_cfg.env_ids, (is_graspable + 1.0) * drawer_pos


@scatterable
def cabinet_multi_stage_open_drawer(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    cabinet_frame_cfg: SceneEntityCfg,
    cabinet_asset_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Multi-stage drawer opening bonus."""
    cabinet = env.scene[cabinet_asset_cfg.name]
    drawer_pos = wp.to_torch(cabinet.data.joint_pos)[cabinet_asset_cfg.view_ids, cabinet_asset_cfg.joint_ids].squeeze(
        -1
    )
    handle_pos = wp.to_torch(env.scene[cabinet_frame_cfg.name].data.target_pos_w)[cabinet_frame_cfg.view_ids, 0, :]
    fingertips = wp.to_torch(env.scene[ee_frame_cfg.name].data.target_pos_w)[ee_frame_cfg.view_ids, 1:, :]
    left = fingertips[:, 0, :]
    right = fingertips[:, 1, :]
    is_graspable = ((right[:, 2] < handle_pos[:, 2]) & (left[:, 2] > handle_pos[:, 2])).float()
    open_easy = (drawer_pos > 0.01) * 0.5
    open_medium = (drawer_pos > 0.2) * is_graspable
    open_hard = (drawer_pos > 0.3) * is_graspable
    return cabinet_asset_cfg.env_ids, open_easy + open_medium + open_hard
