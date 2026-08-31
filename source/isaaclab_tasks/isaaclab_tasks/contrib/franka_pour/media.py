# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative MPM media configuration for the Franka Pour source cup."""

from __future__ import annotations

import math

import numpy as np
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

_MEDIA_JITTER_FRACTION = 0.3
"""Full Newton jitter interval relative to the particle spacing."""


def _media_grid_bounds(
    *,
    source_inner_width: float,
    source_inner_depth: float,
    source_cavity_depth: float,
    source_bottom_thickness: float,
    fill_level: float,
    fill_resolution: float,
    collider_margin: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Resolve fill-volume bounds up to a requested fraction of the cup height."""
    level = float(fill_level)
    if not math.isfinite(level) or not 0.0 < level <= 1.0:
        raise ValueError(f"Franka Pour fill_level must lie in (0, 1], got {fill_level}.")
    spacing = float(fill_resolution)
    clearance = max(spacing, float(collider_margin))

    def horizontal_axis(size: float) -> tuple[float, float]:
        usable = float(size) - 2.0 * clearance
        intervals = max(0, math.floor(usable / spacing + 1.0e-9))
        count = intervals + 1
        first = -0.5 * float(size) + clearance + 0.5 * (usable - intervals * spacing)
        lower = first - 0.5 * spacing
        # Newton computes the grid resolution from float32 bounds. Step the float32 upper
        # bound inward so an intended integral resolution cannot round just above the integer.
        upper = float(np.nextafter(np.float32(lower + count * spacing), np.float32(lower)))
        return lower, upper

    lower_x, upper_x = horizontal_axis(source_inner_width)
    lower_y, upper_y = horizontal_axis(source_inner_depth)
    max_count_z = max(1, math.floor((float(source_cavity_depth) - 2.0 * clearance) / spacing) + 1)
    # The fill level is a water-height fraction. Quantize it to the nearest safe lattice layer;
    # unlike the former volume-derived count, adding particles therefore raises the free surface.
    count_z = max(1, min(max_count_z, math.floor(level * max_count_z + 0.5)))
    first_z = float(source_bottom_thickness) + clearance
    lower_z = first_z - 0.5 * spacing
    upper_z = float(np.nextafter(np.float32(lower_z + count_z * spacing), np.float32(lower_z)))
    return (lower_x, lower_y, lower_z), (upper_x, upper_y, upper_z)


def media_particle_count(cfg: MPMObjectCfg) -> int:
    """Return the number of particles represented by a cell-centred MPM grid asset."""
    if not isinstance(cfg.spawn, MPMGridCfg):
        raise TypeError(f"Franka Pour media requires MPMGridCfg, got {type(cfg.spawn).__name__}.")
    lower = np.asarray(cfg.spawn.lower, dtype=np.float32)
    upper = np.asarray(cfg.spawn.upper, dtype=np.float32)
    extent = upper - lower
    resolution = np.maximum(
        np.ceil(float(cfg.spawn.particles_per_cell) * extent / float(cfg.spawn.voxel_size)), 1
    ).astype(np.int32)
    return int(np.prod(resolution, dtype=np.int64))


def configure_media_object_fill(
    cfg: MPMObjectCfg,
    *,
    source_inner_width: float,
    source_inner_depth: float,
    source_cavity_depth: float,
    source_bottom_thickness: float,
    fill_level: float,
    fill_resolution: float,
    collider_margin: float,
) -> None:
    """Apply a source-cup fill height to an existing grid asset configuration."""
    if not isinstance(cfg.spawn, MPMGridCfg):
        raise TypeError(f"Franka Pour media requires MPMGridCfg, got {type(cfg.spawn).__name__}.")
    lower, upper = _media_grid_bounds(
        source_inner_width=source_inner_width,
        source_inner_depth=source_inner_depth,
        source_cavity_depth=source_cavity_depth,
        source_bottom_thickness=source_bottom_thickness,
        fill_level=fill_level,
        fill_resolution=fill_resolution,
        collider_margin=collider_margin,
    )
    cfg.spawn.lower = lower
    cfg.spawn.upper = upper


def build_media_object_cfg(
    *,
    cup_pos: tuple[float, float, float],
    cup_quat_xyzw: tuple[float, float, float, float],
    source_inner_width: float,
    source_inner_depth: float,
    source_cavity_depth: float,
    source_bottom_thickness: float,
    fill_level: float,
    fill_resolution: float,
    voxel_size: float,
    particles_per_cell: float,
    collider_margin: float,
    material: MPMParticleMaterialCfg,
) -> MPMObjectCfg:
    """Build the source-cup media asset using Isaac Lab's standard Newton grid spawner."""
    lower, upper = _media_grid_bounds(
        source_inner_width=source_inner_width,
        source_inner_depth=source_inner_depth,
        source_cavity_depth=source_cavity_depth,
        source_bottom_thickness=source_bottom_thickness,
        fill_level=fill_level,
        fill_resolution=fill_resolution,
        collider_margin=collider_margin,
    )
    voxel_size = float(voxel_size)
    particles_per_cell = float(particles_per_cell)
    particle_spacing = voxel_size / particles_per_cell
    return MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Media",
        spawn=MPMGridCfg(
            lower=lower,
            upper=upper,
            voxel_size=voxel_size,
            particles_per_cell=particles_per_cell,
            particle_placement="cell_center",
            jitter=_MEDIA_JITTER_FRACTION * particle_spacing,
            material=material,
            visual_color=(0.85, 0.72, 0.45),
        ),
        init_state=MPMObjectCfg.InitialStateCfg(pos=cup_pos, rot=cup_quat_xyzw),
    )
