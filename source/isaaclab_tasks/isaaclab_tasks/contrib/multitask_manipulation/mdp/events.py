# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selection-aware reset events for heterogeneous manipulation scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from ..selection_utils import SceneEntitySelectionCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def _row_mask(asset_cfg: SceneEntitySelectionCfg, env_mask: torch.Tensor) -> wp.array:
    """Return the physics-view rows of an asset selected by a global environment mask."""
    return wp.from_torch(env_mask[asset_cfg.env_ids].contiguous(), dtype=wp.bool)


def _reset_root(env: ManagerBasedEnv, asset_cfg: SceneEntitySelectionCfg, env_mask: torch.Tensor) -> None:
    """Write an asset's default root state for selected global environments."""
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    rows = _row_mask(asset_cfg, env_mask)
    root_pose = asset.data.default_root_pose.torch.clone()
    root_pose[:, :3] += env.scene.env_origins[asset_cfg.env_ids]
    asset.write_root_pose_to_sim_mask(root_pose=root_pose, env_mask=rows)
    asset.write_root_velocity_to_sim_mask(root_velocity=asset.data.default_root_vel.torch.clone(), env_mask=rows)


def _write_joint_state(
    asset: Articulation, rows: wp.array, joint_pos: torch.Tensor, joint_vel: torch.Tensor, set_velocity_target: bool
) -> None:
    """Write joint states and position targets for the selected physics-view rows."""
    asset.write_joint_position_to_sim_mask(position=joint_pos, env_mask=rows)
    asset.write_joint_velocity_to_sim_mask(velocity=joint_vel, env_mask=rows)
    asset.actuators.target_command.set_position_mask(value=joint_pos, env_mask=rows)
    if set_velocity_target:
        asset.actuators.target_command.set_velocity_mask(value=joint_vel, env_mask=rows)


def _reset_joints_default(env: ManagerBasedEnv, asset_cfg: SceneEntitySelectionCfg, env_mask: torch.Tensor) -> None:
    """Write default articulation joint states for selected global environments."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos.torch.clone()
    joint_vel = asset.data.default_joint_vel.torch.clone()
    _write_joint_state(asset, _row_mask(asset_cfg, env_mask), joint_pos, joint_vel, set_velocity_target=True)


def _randomize_joint_offset(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntitySelectionCfg,
    env_mask: torch.Tensor,
    position_range: tuple[float, float],
) -> None:
    """Reset articulation joints with uniform offsets from their defaults."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos.torch.clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(*position_range)
    limits = asset.data.soft_joint_pos_limits.torch
    joint_pos.clamp_(limits[..., 0], limits[..., 1])
    joint_vel = torch.zeros_like(joint_pos)
    _write_joint_state(asset, _row_mask(asset_cfg, env_mask), joint_pos, joint_vel, set_velocity_target=False)


def _randomize_joint_scale(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntitySelectionCfg,
    env_mask: torch.Tensor,
    position_range: tuple[float, float],
) -> None:
    """Reset articulation joints by uniformly scaling their default positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos.torch.clone()
    joint_pos *= torch.empty_like(joint_pos).uniform_(*position_range)
    limits = asset.data.soft_joint_pos_limits.torch
    joint_pos.clamp_(limits[..., 0], limits[..., 1])
    joint_vel = torch.zeros_like(joint_pos)
    _write_joint_state(asset, _row_mask(asset_cfg, env_mask), joint_pos, joint_vel, set_velocity_target=False)


def reset_multitask_scene(
    env: ManagerBasedEnv,
    env_mask: torch.Tensor,
    root_asset_cfgs: tuple[SceneEntitySelectionCfg, ...],
    lift_robot_cfg: SceneEntitySelectionCfg,
    lift_object_cfg: SceneEntitySelectionCfg,
    cabinet_robot_cfg: SceneEntitySelectionCfg,
    cabinet_cfg: SceneEntitySelectionCfg,
    reach_robot_cfg: SceneEntitySelectionCfg,
) -> None:
    """Reset active assets and apply task-specific initial-state randomization.

    Args:
        env: The environment.
        env_mask: Boolean mask of the global environments to reset. Shape is (num_envs,).
        root_asset_cfgs: Assets whose default root state is restored.
        lift_robot_cfg: Lift-task robot.
        lift_object_cfg: Lift-task object.
        cabinet_robot_cfg: Cabinet-task robot.
        cabinet_cfg: Cabinet articulation.
        reach_robot_cfg: Reach-task robot.
    """
    for asset_cfg in root_asset_cfgs:
        _reset_root(env, asset_cfg, env_mask)

    _reset_joints_default(env, lift_robot_cfg, env_mask)
    _reset_joints_default(env, cabinet_cfg, env_mask)
    _randomize_joint_offset(env, cabinet_robot_cfg, env_mask, (-0.1, 0.1))
    _randomize_joint_scale(env, reach_robot_cfg, env_mask, (0.75, 1.25))

    lift_object: RigidObject = env.scene[lift_object_cfg.name]
    object_pose = lift_object.data.default_root_pose.torch.clone()
    object_pose[:, :3] += env.scene.env_origins[lift_object_cfg.env_ids]
    object_pose[:, 0] += torch.empty(object_pose.shape[0], device=env.device).uniform_(-0.1, 0.1)
    object_pose[:, 1] += torch.empty(object_pose.shape[0], device=env.device).uniform_(-0.25, 0.25)
    rows = _row_mask(lift_object_cfg, env_mask)
    lift_object.write_root_pose_to_sim_mask(root_pose=object_pose, env_mask=rows)
    lift_object.write_root_velocity_to_sim_mask(root_velocity=torch.zeros_like(object_pose[:, :6]), env_mask=rows)
