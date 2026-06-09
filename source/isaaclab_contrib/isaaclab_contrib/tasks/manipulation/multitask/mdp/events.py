# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for multi-task environments.

These handle the dual-indexing needed when assets don't span all envs:
- ``env_ids``: global env indices for output buffer / env_origins indexing
- ``view_ids``: indices into the asset's data buffer
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def reset_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    reset_joint_targets: bool = False,
    *,
    asset_cfgs: list[SceneEntityCfg],
):
    """Reset one or more assets to their default state.

    Works with both articulations and rigid objects.  For articulations,
    joint positions/velocities are also reset.

    Args:
        env: The environment instance.
        env_ids: Global env indices that are being reset this step.
        reset_joint_targets: Whether to also reset joint position/velocity targets
            (only applies to articulations).
        asset_cfgs: List of asset configurations to reset.
    """
    selector = env.scene.selector

    for asset_cfg in asset_cfgs:
        env_ids_filtered, view_ids = selector.filter_reset_ids(asset_cfg.name, env_ids)
        if env_ids_filtered.numel() == 0:
            continue

        asset = env.scene[asset_cfg.name]
        pose = wp.to_torch(asset.data.default_root_pose)[view_ids].clone()
        vel = wp.to_torch(asset.data.default_root_vel)[view_ids].clone()
        pose[:, :3] += env.scene.env_origins[env_ids_filtered]
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=view_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=view_ids)

        if hasattr(asset.data, "default_joint_pos"):
            joint_pos = wp.to_torch(asset.data.default_joint_pos)[view_ids].clone()
            joint_vel = wp.to_torch(asset.data.default_joint_vel)[view_ids].clone()
            asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=view_ids)
            asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=view_ids)
            if reset_joint_targets:
                asset.set_joint_position_target_index(target=joint_pos, env_ids=view_ids)
                asset.set_joint_velocity_target_index(target=joint_vel, env_ids=view_ids)


def reset_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float] = (1.0, 1.0),
    velocity_range: tuple[float, float] = (0.0, 0.0),
    *,
    asset_cfg: SceneEntityCfg,
):
    """Reset robot joints by scaling default positions."""
    selector = env.scene.selector

    env_ids, view_ids = selector.filter_reset_ids(asset_cfg.name, env_ids)
    if env_ids.numel() == 0:
        return

    robot = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    env_slice = view_ids[:, None] if joint_ids != slice(None) else view_ids
    joint_pos = wp.to_torch(robot.data.default_joint_pos)[env_slice, joint_ids].clone()
    joint_vel = wp.to_torch(robot.data.default_joint_vel)[env_slice, joint_ids].clone()
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)
    pos_limits = wp.to_torch(robot.data.soft_joint_pos_limits)[env_slice, joint_ids]
    joint_pos = joint_pos.clamp_(pos_limits[..., 0], pos_limits[..., 1])
    vel_limits = wp.to_torch(robot.data.soft_joint_vel_limits)[env_slice, joint_ids]
    joint_vel = joint_vel.clamp_(-vel_limits, vel_limits)
    robot.write_joint_position_to_sim_index(position=joint_pos, joint_ids=joint_ids, env_ids=view_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=joint_ids, env_ids=view_ids)


def reset_object_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    *,
    object_cfg: SceneEntityCfg,
):
    """Reset objects with random position/velocity offsets."""
    selector = env.scene.selector

    env_ids, view_ids = selector.filter_reset_ids(object_cfg.name, env_ids)
    if env_ids.numel() == 0:
        return

    rigid_object = env.scene[object_cfg.name]
    pose = wp.to_torch(rigid_object.data.default_root_pose)[view_ids].clone()
    vel = wp.to_torch(rigid_object.data.default_root_vel)[view_ids].clone()
    ranges = torch.tensor(
        [pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=rigid_object.device,
    )
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(view_ids), 6), device=rigid_object.device)
    pose[:, :3] += env.scene.env_origins[env_ids] + samples[:, :3]
    quat_delta = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    pose[:, 3:7] = math_utils.quat_mul(pose[:, 3:7], quat_delta)
    vel_ranges = torch.tensor(
        [velocity_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=rigid_object.device,
    )
    vel_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(view_ids), 6), device=rigid_object.device
    )
    vel += vel_samples
    rigid_object.write_root_pose_to_sim_index(root_pose=pose, env_ids=view_ids)
    rigid_object.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=view_ids)
