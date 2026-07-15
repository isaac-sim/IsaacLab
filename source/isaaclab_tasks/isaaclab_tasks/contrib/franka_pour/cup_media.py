# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config-side MPM media generation for the dynamic hollow-cube source cup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.spawners.mpm import MPMPointsCfg

from .cube_bowl_mesh import cube_bowl_inner_bounds
from .media_fill import cube_fill_points

if TYPE_CHECKING:
    from .pour_env_cfg import FrankaPourEnvCfg

MEDIA_SPAWN_SEED = 7
"""Fixed seed for media spawn sampling: every env must emit identical particles."""


def particle_spacing(cfg: FrankaPourEnvCfg) -> float:
    """Particle lattice spacing [m]: ``voxel_size / particles_per_cell``."""
    return float(cfg.voxel_size) / max(float(cfg.particles_per_cell), 1.0)


def particle_mass_and_radius(cfg: FrankaPourEnvCfg) -> tuple[float, float]:
    """Return mass [kg] and radius [m] for one MPM lattice cell.

    Newton's implicit MPM backend treats each particle as a cube with volume
    ``8 * radius**3``.  A radius of half the lattice spacing therefore makes the
    represented particle volume exactly match the cell volume used to compute
    mass, preserving the configured material density.
    """
    spacing = particle_spacing(cfg)
    volume = spacing**3
    return float(volume * cfg.media_material.density), float(0.5 * spacing)


def cup_cavity_lattice(cfg: FrankaPourEnvCfg) -> tuple[np.ndarray, np.ndarray]:
    """Jittered lattice filling the source cup cavity in its local frame.

    Args:
        cfg: The pour env config (cup geometry + MPM spacing fields).

    Returns:
        ``(points, cell)`` with cup-local points ``(N, 3)`` float32 and the lattice cell size
        ``(3,)`` float32 [m] used for per-particle mass/radius derivation.
    """
    spacing = particle_spacing(cfg)
    # Keep particle centres at least one lattice spacing / collider margin from the wall. The old
    # ``max(voxel_size, 3 * margin)`` inset removed 6 mm on every side of a 37 mm cup; consequently
    # a requested 70% fill represented only 39% of the cavity volume.
    clearance = max(spacing, float(cfg.collider_margin))
    lo, hi = cube_bowl_inner_bounds(
        float(cfg.source_cup_inner_width),
        float(cfg.source_cup_inner_depth),
        float(cfg.source_cup_cavity_depth),
        float(cfg.source_cup_bottom_thickness),
    )
    # ``cube_fill_points.fill_frac`` controls seed *height* inside an inset footprint, whereas the
    # task config describes the represented MPM *volume*. Choose the nearest whole number of z
    # layers whose cubic particle volumes match that requested cavity-volume fraction.
    spans = np.maximum((hi - lo)[:2] - 2.0 * clearance, 0.0)
    nx, ny = (int(np.floor(span / spacing)) + 1 for span in spans)
    cavity_volume = float(np.prod(hi - lo))
    target_count = float(cfg.media_fill_frac) * cavity_volume / spacing**3
    nz = max(1, int(round(target_count / max(nx * ny, 1))))
    fill_depth = min((nz - 1) * spacing + 1.0e-6 * spacing, float(hi[2] - lo[2]) - 2.0 * clearance)
    seed_height_frac = max(fill_depth, 0.0) / float(hi[2] - lo[2])
    points = cube_fill_points(
        lo,
        hi,
        spacing=spacing,
        fill_frac=seed_height_frac,
        clearance=clearance,
        jitter=0.05,
        seed=MEDIA_SPAWN_SEED,
    )
    if points.shape[0] == 0:
        raise RuntimeError("Cup media initialization produced no particles; reduce voxel size or clearance.")
    cell = np.full(3, spacing, dtype=np.float32)
    return points.astype(np.float32, copy=False), cell


def transform_points(points: np.ndarray, pos, quat_xyzw) -> np.ndarray:
    """Rotate + translate cup-local ``points`` ``(N, 3)`` by an xyzw quaternion and a translation."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1.0e-12)
    xyz = q[:3]
    v = points.astype(np.float64)
    t = 2.0 * np.cross(np.broadcast_to(xyz, v.shape), v)
    rotated = v + float(q[3]) * t + np.cross(np.broadcast_to(xyz, v.shape), t)
    return (rotated + np.asarray(pos, dtype=np.float64)).astype(np.float32)


def build_media_object_cfg(cfg: FrankaPourEnvCfg, cup_pos, cup_quat_xyzw) -> MPMObjectCfg:
    """Build the declarative cup-media :class:`MPMObjectCfg` from the env config.

    Args:
        cfg: The pour env config.
        cup_pos: World position [m] of the cup body at reset (the cup-local frame origin).
        cup_quat_xyzw: World orientation (xyzw quaternion) of the cup body at reset.

    Returns:
        An :class:`MPMObjectCfg` whose spawn points fill the cup cavity at the reset pose, with
        per-particle mass/radius derived from the lattice cell.
    """
    local_points, cell = cup_cavity_lattice(cfg)
    world_points = transform_points(local_points, cup_pos, cup_quat_xyzw)
    mass, radius = particle_mass_and_radius(cfg)
    return MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Media",
        spawn=MPMPointsCfg(
            positions=world_points.tolist(),
            mass=mass,
            radius=radius,
            material=cfg.media_material,
            visual_color=(0.85, 0.72, 0.45),
        ),
    )
