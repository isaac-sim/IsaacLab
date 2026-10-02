# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat-walking events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
_ENCODER_BIAS_ATTR = "_microduck_encoder_bias"


def encoder_bias(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The per-environment, per-joint encoder bias [rad] of an articulation."""
    store: dict[str, torch.Tensor] | None = getattr(env, _ENCODER_BIAS_ATTR, None)
    if store is None:
        store = {}
        setattr(env, _ENCODER_BIAS_ATTR, store)
    bias = store.get(asset_cfg.name)
    if bias is None:
        asset: Articulation = env.scene[asset_cfg.name]
        bias = torch.zeros(env.num_envs, asset.num_joints, device=env.device)
        store[asset_cfg.name] = bias
    return bias


def randomize_encoder_bias(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    bias_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample joint-encoder calibration errors uniformly within ``bias_range`` [rad]."""
    bias = encoder_bias(env, asset_cfg)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    joint_ids = asset_cfg.joint_ids
    num_joints = bias.shape[1] if isinstance(joint_ids, slice) else len(joint_ids)
    samples = torch.empty(len(env_ids), num_joints, device=env.device).uniform_(*bias_range)
    if isinstance(joint_ids, slice):
        bias[env_ids] = samples
    else:
        bias[env_ids[:, None], torch.as_tensor(joint_ids, device=env.device)] = samples


def randomize_bam_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    scale_range: tuple[float, float] = (0.9, 1.1),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample one friction-budget multiplier per environment and broadcast it to all BAM servos."""
    from isaaclab.actuators import BamActuatorCfg
    from isaaclab.actuators.newton import read_group_parameter, write_group_parameter

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    for name, actuator_cfg in asset.cfg.actuators.items():
        if not isinstance(actuator_cfg, BamActuatorCfg):
            continue
        scales = torch.empty(len(env_ids), 1, device=env.device).uniform_(*scale_range)
        num_group_joints = read_group_parameter(asset.actuators, name, "drive", "friction_scale").shape[1]
        write_group_parameter(
            asset.actuators,
            name,
            "drive",
            "friction_scale",
            values=scales.expand(len(env_ids), num_group_joints),
            env_ids=env_ids,
        )
