# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck walking events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import BamActuatorCfg
from isaaclab.actuators.newton import read_group_parameter, write_group_parameter
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def randomize_encoder_bias(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, bias_range: tuple[float, float], action_name: str = "joint_pos"
) -> None:
    """Sample joint-encoder calibration errors [rad] into the biased joint-position action term."""
    bias = env.action_manager.get_term(action_name).encoder_bias
    rows = slice(None) if env_ids is None else env_ids
    bias[rows] = torch.empty_like(bias[rows]).uniform_(*bias_range)


def randomize_bam_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    scale_range: tuple[float, float] = (0.9, 1.1),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample one friction-budget multiplier per environment and broadcast it to all BAM servos."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    scales = torch.empty(len(env_ids), 1, device=env.device).uniform_(*scale_range)
    for name, actuator_cfg in asset.cfg.actuators.items():
        if not isinstance(actuator_cfg, BamActuatorCfg):
            continue
        num_group_joints = read_group_parameter(asset.actuators, name, "drive", "friction_scale").shape[1]
        write_group_parameter(
            asset.actuators,
            name,
            "drive",
            "friction_scale",
            values=scales.expand(len(env_ids), num_group_joints),
            env_ids=env_ids,
        )
