# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first termination functions for the velocity locomotion environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel
def _terrain_out_of_bounds_kernel(
    root_pos_w: wp.array(dtype=wp.vec3f),
    half_width: float,
    half_height: float,
    distance_buffer: float,
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    px = wp.abs(root_pos_w[i][0])
    py = wp.abs(root_pos_w[i][1])
    out[i] = px > half_width - distance_buffer or py > half_height - distance_buffer


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> None:
    """Terminate when the actor moves too close to the edge of the terrain."""
    terrain_type = env.scene.cfg.terrain.terrain_type
    if terrain_type == "plane":
        out.zero_()
        return
    if terrain_type != "generator":
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")

    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    grid_width, grid_length = terrain_gen_cfg.size
    n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
    border_width = terrain_gen_cfg.border_width
    half_width = 0.5 * (n_rows * grid_width + 2 * border_width)
    half_height = 0.5 * (n_cols * grid_length + 2 * border_width)

    asset: Articulation = env.scene[asset_cfg.name]
    env._warp_launch.launch(
        _terrain_out_of_bounds_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_pos_w.warp, half_width, half_height, distance_buffer, out],
        site=(out, float(half_width), float(half_height), float(distance_buffer)),
    )
