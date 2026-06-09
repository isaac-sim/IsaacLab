# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for multi-task environments.

Functions are decorated with ``@scatterable`` to compute on group
rows only and scatter into a full-env buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from .utils import ScatterResult, scatterable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


@scatterable(output_dim=0, dtype=torch.bool)
def object_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    *,
    object_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Terminate when lift object falls below a minimum height."""
    rigid_object = env.scene[object_cfg.name]
    height = wp.to_torch(rigid_object.data.root_pos_w)[object_cfg.view_ids, 2]
    return object_cfg.env_ids, (height < minimum_height)


@scatterable(output_dim=0, dtype=torch.bool)
def cabinet_drawer_opened(
    env: ManagerBasedRLEnv,
    threshold: float = 0.39,
    *,
    cabinet_asset_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Terminate cabinet episodes once the drawer is sufficiently open."""
    cabinet = env.scene[cabinet_asset_cfg.name]
    drawer_pos = wp.to_torch(cabinet.data.joint_pos)[cabinet_asset_cfg.view_ids, cabinet_asset_cfg.joint_ids]
    return cabinet_asset_cfg.env_ids, (drawer_pos.squeeze(-1) > threshold)
