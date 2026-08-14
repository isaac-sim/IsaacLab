# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative MPM media configuration for the Franka Pour source cup."""

from __future__ import annotations

import math

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

_MEDIA_JITTER_FRACTION = 0.1
"""Full Newton jitter interval relative to the particle spacing."""


def _media_grid_geometry(
    *,
    source_inner_width: float,
    source_inner_depth: float,
    source_cavity_depth: float,
    source_bottom_thickness: float,
    fill_fraction: float,
    particle_spacing: float,
    collider_margin: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[int, int, int]]:
    """Resolve cell-centred grid bounds [m] and dimensions inside the authored source cup."""
    spacing = float(particle_spacing)
    clearance = max(spacing, float(collider_margin))

    def horizontal_axis(size: float) -> tuple[float, float, int]:
        usable = float(size) - 2.0 * clearance
        intervals = max(0, math.floor(usable / spacing + 1.0e-9))
        count = intervals + 1
        first = -0.5 * float(size) + clearance + 0.5 * (usable - intervals * spacing)
        lower = first - 0.5 * spacing
        return lower, lower + count * spacing, count

    lower_x, upper_x, count_x = horizontal_axis(source_inner_width)
    lower_y, upper_y, count_y = horizontal_axis(source_inner_depth)
    cavity_volume = float(source_inner_width) * float(source_inner_depth) * float(source_cavity_depth)
    target_count = float(fill_fraction) * cavity_volume / spacing**3
    count_z = max(1, round(target_count / (count_x * count_y)))
    max_count_z = max(1, math.floor((float(source_cavity_depth) - 2.0 * clearance) / spacing) + 1)
    count_z = min(count_z, max_count_z)
    first_z = float(source_bottom_thickness) + clearance
    lower_z = first_z - 0.5 * spacing
    upper_z = lower_z + count_z * spacing
    return (lower_x, lower_y, lower_z), (upper_x, upper_y, upper_z), (count_x, count_y, count_z)


def media_particle_count(cfg: MPMObjectCfg) -> int:
    """Return the number of particles represented by a cell-centred MPM grid asset."""
    if not isinstance(cfg.spawn, MPMGridCfg):
        raise TypeError(f"Franka Pour media requires MPMGridCfg, got {type(cfg.spawn).__name__}.")
    spacing = float(cfg.spawn.voxel_size)
    counts = tuple(
        round(float(cfg.spawn.particles_per_cell) * (float(upper) - float(lower)) / spacing)
        for lower, upper in zip(cfg.spawn.lower, cfg.spawn.upper, strict=True)
    )
    if any(count <= 0 for count in counts):
        raise ValueError(f"Franka Pour media grid must contain particles on every axis, got counts={counts}.")
    return math.prod(counts)


def build_media_object_cfg(
    *,
    cup_pos: tuple[float, float, float],
    cup_quat_xyzw: tuple[float, float, float, float],
    source_inner_width: float,
    source_inner_depth: float,
    source_cavity_depth: float,
    source_bottom_thickness: float,
    fill_fraction: float,
    particle_spacing: float,
    collider_margin: float,
    material: MPMParticleMaterialCfg,
) -> MPMObjectCfg:
    """Build the source-cup media asset using Isaac Lab's standard Newton grid spawner."""
    lower, upper, _ = _media_grid_geometry(
        source_inner_width=source_inner_width,
        source_inner_depth=source_inner_depth,
        source_cavity_depth=source_cavity_depth,
        source_bottom_thickness=source_bottom_thickness,
        fill_fraction=fill_fraction,
        particle_spacing=particle_spacing,
        collider_margin=collider_margin,
    )
    spacing = float(particle_spacing)
    return MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Media",
        spawn=MPMGridCfg(
            lower=lower,
            upper=upper,
            voxel_size=spacing,
            particles_per_cell=1.0,
            particle_placement="cell_center",
            jitter=_MEDIA_JITTER_FRACTION * spacing,
            material=material,
            visual_color=(0.85, 0.72, 0.45),
        ),
        init_state=MPMObjectCfg.InitialStateCfg(pos=cup_pos, rot=cup_quat_xyzw),
    )
