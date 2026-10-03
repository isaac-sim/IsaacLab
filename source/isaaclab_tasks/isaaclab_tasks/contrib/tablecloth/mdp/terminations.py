# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from ._metrics import cloth_pull_distance, tableware_assets

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tablecloth_success(
    env: ManagerBasedRLEnv,
    asset_names: tuple[str, ...],
    table_height: float,
    pull_distance: float,
    maximum_tilt: float,
    maximum_displacement: float,
    cloth_cfg: SceneEntityCfg = SceneEntityCfg("cloth"),
) -> torch.Tensor:
    """Detect a pull that leaves every object upright and near its start position.

    Args:
        env: Manager-based tablecloth environment.
        asset_names: Names of the tableware assets.
        table_height: Tabletop height [m].
        pull_distance: Required cloth travel [m].
        maximum_tilt: Allowed tableware tilt [rad].
        maximum_displacement: Allowed horizontal tableware travel [m].
        cloth_cfg: Cloth asset selector.

    Returns:
        Success flags for all environments.
    """
    success = cloth_pull_distance(env, cloth_cfg) >= pull_distance
    minimum_up_axis = math.cos(maximum_tilt)
    for asset in tableware_assets(env, asset_names):
        on_table = asset.data.root_pos_w.torch[:, 2] >= table_height - 0.05
        upright = -asset.data.projected_gravity_b.torch[:, 2] >= minimum_up_axis
        initial_position = asset.data.default_root_pose.torch[:, :2] + env.scene.env_origins[:, :2]
        stationary = torch.linalg.vector_norm(asset.data.root_pos_w.torch[:, :2] - initial_position, dim=1)
        success &= on_table & upright & (stationary <= maximum_displacement)
    return success


def tableware_fallen(
    env: ManagerBasedRLEnv,
    asset_names: tuple[str, ...],
    minimum_height: float,
) -> torch.Tensor:
    """Terminate when any tableware object falls below the support surface."""
    fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for asset in tableware_assets(env, asset_names):
        fallen |= asset.data.root_pos_w.torch[:, 2] < minimum_height
    return fallen
