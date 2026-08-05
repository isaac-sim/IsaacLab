# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-contract tests for the Franka Pour MPM grid."""

import numpy as np
import pytest
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg

from isaaclab_tasks.contrib.franka_pour.media import build_media_object_cfg, media_particle_count
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import FrankaPourResetDatasetEnvCfg


def test_media_grid_matches_the_authored_source_cup_and_dataset_contract() -> None:
    cfg = FrankaPourResetDatasetEnvCfg()
    media = build_media_object_cfg(cfg, cfg.cup_reset_pos, (0.0, 0.0, 0.0, 1.0))
    grid = media.spawn
    assert isinstance(grid, MPMGridCfg)

    # Match the spawner's float32 geometry conversion before resolving the grid dimensions.
    lower = np.asarray(grid.lower, dtype=np.float32)
    upper = np.asarray(grid.upper, dtype=np.float32)
    resolution = np.ceil(grid.particles_per_cell * (upper - lower) / grid.voxel_size).astype(int)
    cell = (upper - lower) / resolution
    first = lower + 0.5 * cell
    last = upper - 0.5 * cell
    jitter_half_width = 0.5 * grid.jitter
    cavity_lower = np.asarray(
        (-0.5 * cfg.source_cup_inner_width, -0.5 * cfg.source_cup_inner_depth, cfg.source_cup_bottom_thickness)
    )
    cavity_upper = np.asarray(
        (
            0.5 * cfg.source_cup_inner_width,
            0.5 * cfg.source_cup_inner_depth,
            cfg.source_cup_bottom_thickness + cfg.source_cup_cavity_depth,
        )
    )

    np.testing.assert_array_equal(resolution, (7, 7, 5))
    assert media_particle_count(cfg) == int(np.prod(resolution)) == 245
    assert grid.particle_placement == "cell_center"
    assert grid.voxel_size == pytest.approx(cfg.media_particle_spacing)
    assert grid.mass is None and grid.radius is None
    assert np.all(first - jitter_half_width >= cavity_lower + cfg.mpm_collider_margin)
    assert np.all(last + jitter_half_width <= cavity_upper - cfg.mpm_collider_margin)
    assert media.init_state.pos == cfg.cup_reset_pos


def test_media_layout_is_independent_of_solver_voxel_size() -> None:
    cfg = FrankaPourResetDatasetEnvCfg()
    original = build_media_object_cfg(cfg, cfg.cup_reset_pos, (0.0, 0.0, 0.0, 1.0)).spawn
    cfg.voxel_size = 0.005
    refined = build_media_object_cfg(cfg, cfg.cup_reset_pos, (0.0, 0.0, 0.0, 1.0)).spawn

    assert refined.lower == original.lower
    assert refined.upper == original.upper
    assert refined.voxel_size == original.voxel_size
    assert media_particle_count(cfg) == 245
