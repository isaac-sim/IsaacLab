# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

"""
Terrain size limits.
"""


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when agents move too close to the edge of the terrain."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    def get_map_size(env: ManagerBasedRLEnv) -> tuple[float, float]:
        grid_width, grid_length = env.scene.terrain.cfg.terrain_generator.size
        n_cols = env.scene.terrain.cfg.terrain_generator.num_cols
        n_rows = env.scene.terrain.cfg.terrain_generator.num_rows
        border_width = env.scene.terrain.cfg.terrain_generator.border_width
        length = n_cols * grid_length + 2 * border_width
        width = n_rows * grid_width + 2 * border_width
        return (width, length)

    if env.scene.cfg.terrain.terrain_type == "plane":
        return False
    elif env.scene.cfg.terrain.terrain_type == "generator":
        map_width, map_height = get_map_size(env)
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'")
