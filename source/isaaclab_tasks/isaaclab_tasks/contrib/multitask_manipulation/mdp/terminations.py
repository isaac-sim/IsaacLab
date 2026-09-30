# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selection-aware terminations for heterogeneous manipulation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..selection_utils import SceneEntitySelectionCfg
from .rewards import reach_orientation_error, reach_position_error

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def reach_success(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    command_name: str,
    position_threshold: float,
    orientation_threshold: float,
) -> torch.Tensor:
    """Terminate successful UR10 reach environments."""
    position_error = reach_position_error(env, robot_cfg, command_name)
    orientation_error = reach_orientation_error(env, robot_cfg, command_name)
    return (
        (robot_cfg.instance_ids >= 0)
        & (position_error < position_threshold)
        & (orientation_error < orientation_threshold)
    )


def lift_object_out_of_bounds(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntitySelectionCfg,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> torch.Tensor:
    """Terminate lift environments whose object leaves configured local bounds [m]."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    position = object_asset.data.root_pos_w.torch - env.scene.env_origins[object_cfg.env_ids]
    lower = position.new_tensor([axis[0] for axis in bounds])
    upper = position.new_tensor([axis[1] for axis in bounds])
    outside = torch.any((position < lower) | (position > upper), dim=-1)
    return object_cfg.scatter_to_envs(outside)


def lift_object_dropped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntitySelectionCfg,
    minimum_height: float,
) -> torch.Tensor:
    """Terminate lift environments whose object falls below a local height [m]."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    height = object_asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[object_cfg.env_ids, 2]
    return object_cfg.scatter_to_envs(height < minimum_height)


def articulation_state_invalid(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntitySelectionCfg,
    max_joint_velocity: float,
    joint_position_margin: float,
) -> torch.Tensor:
    """Terminate selected articulations with non-finite or implausible joint state.

    Args:
        env: Manager-based RL environment.
        asset_cfg: Selection-aware articulation and joints.
        max_joint_velocity: Maximum absolute joint velocity [m/s or rad/s, depending on joint type].
        joint_position_margin: Allowed distance beyond soft position limits [m or rad, depending on joint type].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids]
    invalid = ~torch.isfinite(joint_pos).all(dim=-1) | ~torch.isfinite(joint_vel).all(dim=-1)
    invalid |= torch.any(torch.abs(joint_vel) > max_joint_velocity, dim=-1)
    invalid |= torch.any(
        (joint_pos < limits[..., 0] - joint_position_margin) | (joint_pos > limits[..., 1] + joint_position_margin),
        dim=-1,
    )
    return asset_cfg.scatter_to_envs(invalid)


def task_time_out(
    env: ManagerBasedRLEnv,
    task_asset_cfgs: tuple[SceneEntitySelectionCfg, ...],
    episode_lengths_s: tuple[float, ...],
) -> torch.Tensor:
    """Time out each task at its homogeneous episode duration [s]."""
    timed_out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for asset_cfg, episode_length_s in zip(task_asset_cfgs, episode_lengths_s):
        max_steps = round(episode_length_s / env.step_dt)
        timed_out |= (asset_cfg.instance_ids >= 0) & (env.episode_length_buf >= max_steps)
    return timed_out
