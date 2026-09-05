# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selection-aware reset events for heterogeneous manipulation scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..selection_utils import SceneEntitySelectionCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def _reset_root(env: ManagerBasedEnv, asset_cfg: SceneEntitySelectionCfg, env_ids: torch.Tensor) -> None:
    """Write an asset's default root state for selected global environments."""
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    rows, selected_env_ids = asset_cfg.select(env_ids)
    if rows.numel() == 0:
        return
    root_pose = asset.data.default_root_pose.torch[rows].clone()
    root_pose[:, :3] += env.scene.env_origins[selected_env_ids]
    asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=rows)
    asset.write_root_velocity_to_sim_index(root_velocity=asset.data.default_root_vel.torch[rows], env_ids=rows)


def _reset_joints_default(env: ManagerBasedEnv, asset_cfg: SceneEntitySelectionCfg, env_ids: torch.Tensor) -> None:
    """Write default articulation joint states for selected global environments."""
    asset: Articulation = env.scene[asset_cfg.name]
    rows, _ = asset_cfg.select(env_ids)
    if rows.numel() == 0:
        return
    joint_pos = asset.data.default_joint_pos.torch[rows].clone()
    joint_vel = asset.data.default_joint_vel.torch[rows].clone()
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=rows)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=rows)
    asset.actuators.target_command.set_position_index(value=joint_pos, env_ids=rows)
    asset.actuators.target_command.set_velocity_index(value=joint_vel, env_ids=rows)


def _randomize_joint_offset(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntitySelectionCfg,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
) -> None:
    """Reset articulation joints with uniform offsets from their defaults."""
    asset: Articulation = env.scene[asset_cfg.name]
    rows, _ = asset_cfg.select(env_ids)
    if rows.numel() == 0:
        return
    joint_pos = asset.data.default_joint_pos.torch[rows].clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(*position_range)
    limits = asset.data.soft_joint_pos_limits.torch[rows]
    joint_pos.clamp_(limits[..., 0], limits[..., 1])
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=rows)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=rows)
    asset.actuators.target_command.set_position_index(value=joint_pos, env_ids=rows)


def _randomize_joint_scale(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntitySelectionCfg,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
) -> None:
    """Reset articulation joints by uniformly scaling their default positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    rows, _ = asset_cfg.select(env_ids)
    if rows.numel() == 0:
        return
    joint_pos = asset.data.default_joint_pos.torch[rows].clone()
    joint_pos *= torch.empty_like(joint_pos).uniform_(*position_range)
    limits = asset.data.soft_joint_pos_limits.torch[rows]
    joint_pos.clamp_(limits[..., 0], limits[..., 1])
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=rows)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=rows)
    asset.actuators.target_command.set_position_index(value=joint_pos, env_ids=rows)


def reset_multitask_scene(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    root_asset_cfgs: tuple[SceneEntitySelectionCfg, ...],
    lift_robot_cfg: SceneEntitySelectionCfg,
    lift_object_cfg: SceneEntitySelectionCfg,
    cabinet_robot_cfg: SceneEntitySelectionCfg,
    cabinet_cfg: SceneEntitySelectionCfg,
    reach_robot_cfg: SceneEntitySelectionCfg,
) -> None:
    """Reset active assets and apply task-specific initial-state randomization."""
    for asset_cfg in root_asset_cfgs:
        _reset_root(env, asset_cfg, env_ids)

    _reset_joints_default(env, lift_robot_cfg, env_ids)
    _reset_joints_default(env, cabinet_cfg, env_ids)
    _randomize_joint_offset(env, cabinet_robot_cfg, env_ids, (-0.1, 0.1))
    _randomize_joint_scale(env, reach_robot_cfg, env_ids, (0.75, 1.25))

    lift_object: RigidObject = env.scene[lift_object_cfg.name]
    object_rows, object_env_ids = lift_object_cfg.select(env_ids)
    if object_rows.numel() == 0:
        return
    object_pose = lift_object.data.default_root_pose.torch[object_rows].clone()
    object_pose[:, :3] += env.scene.env_origins[object_env_ids]
    object_pose[:, 0] += torch.empty(len(object_rows), device=env.device).uniform_(-0.1, 0.1)
    object_pose[:, 1] += torch.empty(len(object_rows), device=env.device).uniform_(-0.25, 0.25)
    lift_object.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=object_rows)
    lift_object.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(len(object_rows), 6, device=env.device), env_ids=object_rows
    )
