# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# TODO: Clean this up
@wp.func
def base_yaw_roll_wp_func(root_pose_w: wp.transformf) -> wp.vec2f:
    quat = wp.transform_get_rotation(root_pose_w)
    sin_roll = 2.0 * (quat[3] * quat[0] + quat[1] * quat[2])
    cos_roll = 1.0 - 2.0 * (quat[0] * quat[0] + quat[1] * quat[1])
    roll = wp.atan2(sin_roll, cos_roll)

    sin_yaw = 2.0 * (quat[3] * quat[2] + quat[0] * quat[1])
    cos_yaw = 1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2])
    yaw = wp.atan2(sin_yaw, cos_yaw)
    return wp.vec2f(yaw, roll)


def base_yaw_roll(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw and roll of the base in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, _, yaw = math_utils.euler_xyz_from_quat(wp.to_torch(asset.data.root_quat_w))
    # normalize angle to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return torch.cat((yaw.unsqueeze(-1), roll.unsqueeze(-1)), dim=-1)
    # torch_out = torch.zeros((env.num_envs, 2), device=env.device)
    # wp.map(base_yaw_roll_wp_func, asset.data.root_pose_w, out=wp.from_torch(torch_out, dtype=wp.vec2f))
    # return torch_out


@wp.func
def base_up_proj_wp_func(projected_gravity_b: wp.vec3f) -> wp.float32:
    return -projected_gravity_b[2]


def base_up_proj(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Projection of the base up vector onto the world up vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute base up vector
    base_up_vec = -wp.to_torch(asset.data.projected_gravity_b)
    return base_up_vec[:, 2].unsqueeze(-1)
    # torch_out = torch.zeros((env.num_envs), device=env.device)
    # wp.map(base_up_proj_wp_func, asset.data.projected_gravity_b, out=wp.from_torch(torch_out, dtype=wp.float32))
    # return torch_out.unsqueeze(-1)


@wp.func
def base_heading_proj_wp_func(target_pos: wp.vec3f, root_pose_w: wp.transformf) -> wp.float32:
    pos = wp.transform_get_translation(root_pose_w)
    quat = wp.transform_get_rotation(root_pose_w)
    # compute desired heading direction
    to_target_pos = target_pos - pos
    to_target_pos[2] = 0.0
    to_target_dir = wp.normalize(to_target_pos)
    # Compute base forward vector
    heading_vec = wp.quat_rotate(quat, wp.static(wp.vec3f(1, 0, 0)))
    # compute dot product between heading and target direction
    heading_proj = wp.dot(heading_vec, to_target_dir)
    return heading_proj


def base_heading_proj(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Projection of the base forward vector onto the world forward vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - wp.to_torch(asset.data.root_pos_w)[:, :3]
    to_target_pos[:, 2] = 0.0
    to_target_dir = math_utils.normalize(to_target_pos)
    # compute base forward vector
    heading_vec = math_utils.quat_apply(wp.to_torch(asset.data.root_quat_w), asset.data.FORWARD_VEC_B_TORCH)
    # compute dot product between heading and target direction
    heading_proj = torch.bmm(heading_vec.view(env.num_envs, 1, 3), to_target_dir.view(env.num_envs, 3, 1))

    return heading_proj.view(env.num_envs, 1)
    # torch_out = torch.zeros((env.num_envs), device=env.device)
    # target = wp.vec3f(target_pos[0], target_pos[1], target_pos[2])
    # wp.map(base_heading_proj_wp_func, target, asset.data.root_pose_w, out=wp.from_torch(torch_out, dtype=wp.float32))
    # return torch_out.unsqueeze(-1)


# @wp.func
# def base_angle_to_target_wp_func(target_pos: wp.vec3f, root_pose_w: wp.transformf) -> wp.float32:


def base_angle_to_target(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angle between the base forward vector and the vector to the target."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - wp.to_torch(asset.data.root_pos_w)[:, :3]
    walk_target_angle = torch.atan2(to_target_pos[:, 1], to_target_pos[:, 0])
    # compute base forward vector
    _, _, yaw = math_utils.euler_xyz_from_quat(wp.to_torch(asset.data.root_quat_w))
    # normalize angle to target to [-pi, pi]
    angle_to_target = walk_target_angle - yaw
    angle_to_target = torch.atan2(torch.sin(angle_to_target), torch.cos(angle_to_target))
    return angle_to_target.unsqueeze(-1)
