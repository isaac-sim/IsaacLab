# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the deformable lift tasks.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


def deformable_outside_bounds(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Terminate if any deformable nodal point leaves the allowed workspace box.

    Covers both leaving the table footprint (x, y) and being dropped off it (z).

    Args:
        env: The environment instance.
        x_bounds: Allowed x-position range in the environment frame [m].
        y_bounds: Allowed y-position range in the environment frame [m].
        z_bounds: Allowed z-position range in the environment frame [m].
        asset_cfg: The deformable object entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = asset.data.nodal_pos_w.torch - env.scene.env_origins.unsqueeze(1)
    lower = torch.tensor([x_bounds[0], y_bounds[0], z_bounds[0]], device=nodal_pos.device)
    upper = torch.tensor([x_bounds[1], y_bounds[1], z_bounds[1]], device=nodal_pos.device)
    return ((nodal_pos < lower) | (nodal_pos > upper)).flatten(1).any(dim=1)


def deformable_nodal_vel_above_maximum(
    env: ManagerBasedRLEnv,
    maximum_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Terminate when any deformable node moves faster than ``maximum_velocity`` [m/s].

    Guards against solver blow-up, where penalty contact ejects nodes at implausible speeds.

    Args:
        env: The environment instance.
        maximum_velocity: Maximum allowed nodal speed [m/s].
        asset_cfg: The deformable object entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    speed = torch.linalg.norm(asset.data.nodal_vel_w.torch, dim=-1)
    return speed.max(dim=1).values > maximum_velocity


def joint_vel_out_of_sim_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when joint velocities exceed actuator simulator limits [m/s or rad/s, depending on joint type]."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    limits = torch.full_like(asset.data.joint_vel.torch, torch.inf)
    for actuator in asset.actuators.values():
        limits[:, actuator.joint_indices] = actuator.velocity_limit_sim
    return torch.any(torch.abs(asset.data.joint_vel.torch[:, joint_ids]) > limits[:, joint_ids], dim=1)
