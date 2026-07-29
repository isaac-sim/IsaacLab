# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared ray-caster workload geometry."""

import math

from isaaclab.benchmark.sensor_suites import rough_terrain_size


def test_default_rough_terrain_covers_every_environment_ray_grid() -> None:
    """The default 4096-environment grid must fit inside the generated terrain."""
    num_envs = 4096
    env_spacing = 2.0
    ray_grid_size = 1.0

    terrain_size = rough_terrain_size(num_envs, env_spacing, ray_grid_size)

    columns = math.ceil(math.sqrt(num_envs))
    rows = math.ceil(num_envs / columns)
    required_extent = max(columns - 1, rows - 1) * env_spacing + ray_grid_size
    assert terrain_size >= required_extent


def test_single_environment_rough_terrain_keeps_minimum_size() -> None:
    """Small smoke workloads should still generate a useful rough surface."""
    assert rough_terrain_size(1, 2.0, 1.0) == 3.0
