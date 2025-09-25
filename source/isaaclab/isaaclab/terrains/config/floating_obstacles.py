# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

FLOATING_OBSTACLES_CFG = TerrainGeneratorCfg(
    size=(12.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "floating_obstacles": terrain_gen.MeshFloatingObstaclesTerrainCfg(
            max_num_obstacles=100,
            env_size=(12.0, 8.0, 6.0),
        ),
    },
)
"""floating obstacles terrain configuration."""