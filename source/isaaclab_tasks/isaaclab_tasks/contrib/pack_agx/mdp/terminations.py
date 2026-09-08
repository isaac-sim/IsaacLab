# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination conditions for packing an AGX Orin into its protective box."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Offset convention: AGX root position minus box root position, expressed in
# world coordinates. Thresholds are calibrated from a successful settled replay:
# AGX=(-0.564771, -0.018734, 0.945549),
# box=(-0.565000, -0.025000, 0.965000), so dz=-0.019451 m.
def agx_is_horizontal_from_quat(
    agx_orin_quat: torch.Tensor,
    rotation_tolerance: float = 0.2,
) -> torch.Tensor:
    """Return whether AGX roll and pitch remain within the horizontal tolerance."""
    roll, pitch, _ = euler_xyz_from_quat(agx_orin_quat)
    return (torch.abs(roll) <= rotation_tolerance) & (torch.abs(pitch) <= rotation_tolerance)


def agx_in_box_from_pose(
    agx_orin_pos: torch.Tensor,
    agx_orin_quat: torch.Tensor,
    box_pos: torch.Tensor,
    xy_tolerance: float = 0.01,
    z_tolerance: float = 0.03,
    rotation_tolerance: float = 0.2,
) -> torch.Tensor:
    """Return whether AGX poses satisfy the task's seated-success thresholds."""
    in_xy = torch.all(
        torch.abs(agx_orin_pos[:, :2] - box_pos[:, :2]) <= xy_tolerance,
        dim=-1,
    )
    in_z = torch.abs(agx_orin_pos[:, 2] - box_pos[:, 2]) <= z_tolerance

    horizontal = agx_is_horizontal_from_quat(
        agx_orin_quat,
        rotation_tolerance=rotation_tolerance,
    )
    return in_xy & in_z & horizontal


def agx_in_box(
    env: ManagerBasedRLEnv,
    agx_orin_cfg: SceneEntityCfg = SceneEntityCfg("agx_orin"),
    protective_box_cfg: SceneEntityCfg = SceneEntityCfg("protective_box"),
    xy_tolerance: float = 0.01,
    z_tolerance: float = 0.03,
    rotation_tolerance: float = 0.2,
) -> torch.Tensor:
    """Return whether the AGX Orin is seated inside the protective box."""
    agx_orin = env.scene[agx_orin_cfg.name]
    protective_box = env.scene[protective_box_cfg.name]

    agx_orin_pos = agx_orin.data.root_pos_w.torch
    box_pos = protective_box.data.root_pos_w.torch
    agx_quat = agx_orin.data.root_quat_w.torch
    return agx_in_box_from_pose(
        agx_orin_pos,
        agx_quat,
        box_pos,
        xy_tolerance=xy_tolerance,
        z_tolerance=z_tolerance,
        rotation_tolerance=rotation_tolerance,
    )


def task_success_termination(
    env: ManagerBasedRLEnv,
    success_stage: int = 3,
) -> torch.Tensor:
    """Terminate after all monotonic Pack-AGX reward stages complete."""
    from .rewards import get_task_stage

    return get_task_stage(env) >= success_stage
