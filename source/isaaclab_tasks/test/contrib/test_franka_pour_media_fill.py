# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Franka pour granular-media fill (no simulator)."""

from types import SimpleNamespace

import numpy as np

from isaaclab_tasks.contrib.franka_pour.cube_bowl_mesh import cube_bowl_inner_bounds
from isaaclab_tasks.contrib.franka_pour.cup_media import cup_cavity_lattice, particle_mass_and_radius
from isaaclab_tasks.contrib.franka_pour.media_fill import cube_fill_points, expected_fill_count
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import FrankaPourEnvCfg

LO, HI = cube_bowl_inner_bounds(0.037, 0.037, 0.045, 0.009)
CLR = max(0.003, 3 * 0.002)


def _min_neighbor_distance(pts: np.ndarray, sample: int = 200) -> float:
    """Smallest nearest-neighbour distance over a subsample (numpy-only, no scipy)."""
    idx = np.linspace(0, len(pts) - 1, min(sample, len(pts))).astype(int)
    sub = pts[idx]
    best = np.inf
    for i in range(len(sub)):
        d = np.linalg.norm(pts - sub[i], axis=1)
        d[np.argmin(d)] = np.inf  # drop the self-distance (0)
        best = min(best, float(d.min()))
    return best


def test_points_nonempty_and_inside_cavity():
    pts = cube_fill_points(LO, HI, spacing=0.003, fill_frac=1.0, jitter=0.0)
    assert pts.dtype == np.float32 and pts.shape[1] == 3 and len(pts) > 200
    assert np.all(pts[:, 0] >= LO[0] + CLR - 1e-6) and np.all(pts[:, 0] <= HI[0] - CLR + 1e-6)
    assert np.all(pts[:, 1] >= LO[1] + CLR - 1e-6) and np.all(pts[:, 1] <= HI[1] - CLR + 1e-6)
    assert np.all(pts[:, 2] >= LO[2] + CLR - 1e-6)


def test_fill_frac_limits_height():
    full = cube_fill_points(LO, HI, spacing=0.003, fill_frac=1.0, jitter=0.0)
    half = cube_fill_points(LO, HI, spacing=0.003, fill_frac=0.5, jitter=0.0)
    assert float(half[:, 2].max()) < float(full[:, 2].max())
    assert len(half) < len(full)


def test_deterministic_seed():
    a = cube_fill_points(LO, HI, spacing=0.003, seed=7)
    b = cube_fill_points(LO, HI, spacing=0.003, seed=7)
    assert np.array_equal(a, b)


def test_no_overlap_min_spacing():
    pts = cube_fill_points(LO, HI, spacing=0.003, jitter=0.0)
    assert _min_neighbor_distance(pts) > 0.5 * 0.003


def test_expected_count_matches_actual():
    n = expected_fill_count(LO, HI, spacing=0.003, fill_frac=1.0)
    pts = cube_fill_points(LO, HI, spacing=0.003, fill_frac=1.0, jitter=0.0)
    assert n == len(pts)


def test_particle_mass_and_radius_represent_one_full_lattice_cell():
    """Implicit MPM derives particle volume as 8*r^3, so r must be half the lattice spacing."""
    cfg = SimpleNamespace(
        voxel_size=0.006,
        particles_per_cell=2.0,
        media_material=SimpleNamespace(density=1500.0),
    )
    mass, radius = particle_mass_and_radius(cfg)
    spacing = cfg.voxel_size / cfg.particles_per_cell
    represented_volume = 8.0 * radius**3

    assert np.isclose(radius, 0.5 * spacing)
    assert np.isclose(represented_volume, spacing**3)
    assert np.isclose(mass / represented_volume, cfg.media_material.density)


def test_task_fill_fraction_means_represented_cavity_volume_not_inset_point_height():
    cfg = FrankaPourEnvCfg()
    # Fill-fraction fidelity needs multiple lattice layers. The task's intentionally coarse
    # rollout resolution is covered separately by the environment-config regression tests.
    cfg.voxel_size = 0.006
    points, cell = cup_cavity_lattice(cfg)
    cavity_volume = cfg.source_cup_inner_width * cfg.source_cup_inner_depth * cfg.source_cup_cavity_depth
    represented_fill = len(points) * float(np.prod(cell)) / cavity_volume
    points_per_layer = len(np.unique(np.round(points[:, :2], decimals=5), axis=0))
    one_layer_fraction = points_per_layer * float(np.prod(cell)) / cavity_volume

    # The safe wall/rim inset can leave the nearest higher layer infeasible in a shallow cup.
    assert abs(represented_fill - cfg.media_fill_frac) <= one_layer_fraction
