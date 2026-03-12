# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-robot event terms for heterogeneous scenes.

**Per-asset functions** (use with ``per_robot=True``):
    Accept ``asset_cfg: SceneEntityCfg`` (auto-injected by the
    manager from :class:`RobotSpec`) and group-local ``env_ids``.

**Scatter-based functions** (self-dispatching):
    Iterate :attr:`EnvLayout.robot_specs` and map env-ids internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ===========================================================
# Per-asset event functions  (use with per_robot=True)
# ===========================================================


def reset_asset_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reset_joint_targets: bool = False,
) -> None:
    """Reset a single articulation to its default root and joint state.

    ``env_ids`` are group-local (0-based) when dispatched via
    ``per_robot=True``.  The function recovers global env indices
    for :attr:`env_origins` via :meth:`EnvLayout.local_to_global`.
    """
    layout = env.scene.layout
    group_key = layout.group_for_asset(asset_cfg.name)
    global_ids = layout.local_to_global(group_key, env_ids)

    art = env.scene[asset_cfg.name]
    default_pose = wp.to_torch(art.data.default_root_pose)[env_ids].clone()
    default_vel = wp.to_torch(art.data.default_root_vel)[env_ids].clone()
    default_pose[:, :3] += env.scene.env_origins[global_ids]
    art.write_root_pose_to_sim_index(root_pose=default_pose, env_ids=env_ids)
    art.write_root_velocity_to_sim_index(root_velocity=default_vel, env_ids=env_ids)

    default_jpos = wp.to_torch(art.data.default_joint_pos)[env_ids].clone()
    default_jvel = wp.to_torch(art.data.default_joint_vel)[env_ids].clone()
    art.write_joint_position_to_sim_index(position=default_jpos, env_ids=env_ids)
    art.write_joint_velocity_to_sim_index(velocity=default_jvel, env_ids=env_ids)
    if reset_joint_targets:
        art.set_joint_position_target_index(target=default_jpos, env_ids=env_ids)
        art.set_joint_velocity_target_index(target=default_jvel, env_ids=env_ids)


# ===========================================================
# Scatter-based functions (self-dispatching)
# ===========================================================


def reset_multitask_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor, reset_joint_targets: bool = False):
    """Reset the scene to the default state specified in the scene configuration.

    Supports heterogeneous scenes where per-task assets only exist in a
    subset of environments.  Uses the centralized :class:`EnvLayout` for
    global-to-local env-id mapping rather than per-asset filter methods.

    If :attr:`reset_joint_targets` is True, the joint position and velocity
    targets of the articulations are also reset to their default values.
    """
    layout = env.scene.layout
    for name, rigid_object in env.scene.rigid_objects.items():
        key = layout.group_for_asset(name)
        local_ids, global_ids = layout.filter_and_split(key, env_ids) if key else (env_ids, env_ids)
        if local_ids.numel() == 0:
            continue
        default_pose = wp.to_torch(rigid_object.data.default_root_pose)[local_ids].clone()
        default_vel = wp.to_torch(rigid_object.data.default_root_vel)[local_ids].clone()
        default_pose[:, :3] += env.scene.env_origins[global_ids]
        rigid_object.write_root_pose_to_sim_index(root_pose=default_pose, env_ids=local_ids)
        rigid_object.write_root_velocity_to_sim_index(root_velocity=default_vel, env_ids=local_ids)
    for name, articulation_asset in env.scene.articulations.items():
        key = layout.group_for_asset(name)
        local_ids, global_ids = layout.filter_and_split(key, env_ids) if key else (env_ids, env_ids)
        if local_ids.numel() == 0:
            continue
        default_pose = wp.to_torch(articulation_asset.data.default_root_pose)[local_ids].clone()
        default_vel = wp.to_torch(articulation_asset.data.default_root_vel)[local_ids].clone()
        default_pose[:, :3] += env.scene.env_origins[global_ids]
        articulation_asset.write_root_pose_to_sim_index(root_pose=default_pose, env_ids=local_ids)
        articulation_asset.write_root_velocity_to_sim_index(root_velocity=default_vel, env_ids=local_ids)
        default_joint_pos = wp.to_torch(articulation_asset.data.default_joint_pos)[local_ids].clone()
        default_joint_vel = wp.to_torch(articulation_asset.data.default_joint_vel)[local_ids].clone()
        articulation_asset.write_joint_position_to_sim_index(position=default_joint_pos, env_ids=local_ids)
        articulation_asset.write_joint_velocity_to_sim_index(velocity=default_joint_vel, env_ids=local_ids)
        if reset_joint_targets:
            articulation_asset.set_joint_position_target_index(target=default_joint_pos, env_ids=local_ids)
            articulation_asset.set_joint_velocity_target_index(target=default_joint_vel, env_ids=local_ids)


def multi_robot_reset_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float] = (0.5, 1.5),
    velocity_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Reset joints with randomized offsets per robot group.

    Iterates :attr:`EnvLayout.robot_specs` so this function works with
    any heterogeneous scene configuration without hardcoded robot lists.
    Resulting positions and velocities are clamped to the asset's soft
    joint limits.
    """
    layout = env.scene.layout
    for spec in layout.robot_specs:
        key = layout.group_for_asset(spec.asset_name)
        if key is None:
            continue
        local, _glob = layout.filter_and_split(key, env_ids)
        if local.numel() == 0:
            continue
        art = env.scene[spec.asset_name]
        jids, _ = art.find_joints(spec.joint_patterns)

        dfl_pos = wp.to_torch(art.data.default_joint_pos)[local][:, jids].clone()
        dfl_vel = wp.to_torch(art.data.default_joint_vel)[local][:, jids].clone()

        n, nj = local.shape[0], len(jids)
        p_lo, p_hi = position_range
        v_lo, v_hi = velocity_range

        pos_scaled = dfl_pos * (torch.rand(n, nj, device=env.device) * (p_hi - p_lo) + p_lo)
        vel_scaled = dfl_vel * (torch.rand(n, nj, device=env.device) * (v_hi - v_lo) + v_lo)

        pos_limits = wp.to_torch(art.data.soft_joint_pos_limits)[local][:, jids]
        pos_scaled = pos_scaled.clamp_(pos_limits[..., 0], pos_limits[..., 1])
        vel_limits = wp.to_torch(art.data.soft_joint_vel_limits)[local][:, jids]
        vel_scaled = vel_scaled.clamp_(-vel_limits, vel_limits)

        art.write_joint_position_to_sim_index(position=pos_scaled, joint_ids=jids, env_ids=local)
        art.write_joint_velocity_to_sim_index(velocity=vel_scaled, joint_ids=jids, env_ids=local)
