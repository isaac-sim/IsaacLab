# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for ray-caster benchmark workloads."""

from __future__ import annotations

import math


def rough_terrain_size(num_envs: int, env_spacing: float, ray_grid_size: float) -> float:
    """Compute a square terrain size covering every environment ray grid.

    Args:
        num_envs: Number of environments.
        env_spacing: Distance between adjacent environment origins [m].
        ray_grid_size: Width and length of each ray grid [m].

    Returns:
        Terrain width and length [m].
    """
    num_columns = math.ceil(math.sqrt(num_envs))
    num_rows = math.ceil(num_envs / num_columns)
    environment_extent = max(num_columns - 1, num_rows - 1) * env_spacing
    return max(3.0, environment_extent + ray_grid_size)
