# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from isaaclab.sim.utils import clone, create_prim

if TYPE_CHECKING:
    from newton import ModelBuilder

    from pxr import Usd

from .mpm_cfg import MPMGridCfg, MPMParticleMaterialCfg, MPMParticleSpawnerCfg, MPMPointsCfg


@clone
def spawn_mpm_particles(
    prim_path: str,
    cfg: MPMParticleSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Create a lightweight placeholder prim for a Newton MPM particle object.

    MPM particles are inserted directly into the Newton model builder during
    Newton replication. The USD prim exists so Isaac Lab's normal asset spawning
    and clone-planning machinery can reason about the scene entity.

    Args:
        prim_path: Prim path or pattern at which to create the placeholder.
        cfg: MPM particle spawner configuration.
        translation: Translation relative to the parent prim [m]. If ``None``, the
            placeholder uses the origin.
        orientation: Orientation relative to the parent prim as an ``(x, y, z, w)``
            quaternion. If ``None``, the placeholder uses the identity quaternion.
        **kwargs: Additional keyword arguments consumed by the :func:`~isaaclab.sim.utils.clone`
            decorator.

    Returns:
        The created placeholder prim.
    """
    return create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)


def _material_custom_attributes(material: MPMParticleMaterialCfg) -> dict[str, float]:
    """Map material cfg values to Newton ``mpm:*`` custom attributes for ``add_particles``.

    ``density`` is intentionally absent: it is consumed by the particle
    generators to derive per-particle mass, not forwarded to the solver.
    """
    return {
        "mpm:young_modulus": float(material.young_modulus),
        "mpm:poisson_ratio": float(material.poisson_ratio),
        "mpm:viscosity": float(material.viscosity),
        "mpm:friction": float(material.friction),
        "mpm:damping": float(material.damping),
        "mpm:yield_pressure": float(material.yield_pressure),
        "mpm:tensile_yield_ratio": float(material.tensile_yield_ratio),
        "mpm:yield_stress": float(material.yield_stress),
        "mpm:hardening": float(material.hardening),
        "mpm:dilatancy": float(material.dilatancy),
    }


def emit_mpm_particles(
    builder: ModelBuilder,
    cfg: MPMParticleSpawnerCfg,
    *,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> None:
    """Emit particles from a declarative configuration into a Newton model builder.

    Args:
        builder: Newton model builder that receives the particles.
        cfg: Grid or explicit-point particle spawner configuration.
        position: World-space translation applied to local particle positions [m].
        orientation: World-space orientation applied to local particle positions and
            velocities as an ``(x, y, z, w)`` quaternion.

    Raises:
        TypeError: If :paramref:`cfg` is not a supported MPM particle spawner configuration.
        ValueError: If the configuration contains invalid particle geometry or physical values.
    """
    if isinstance(cfg, MPMGridCfg):
        _emit_grid(builder, cfg, position=position, orientation=orientation)
    elif isinstance(cfg, MPMPointsCfg):
        _emit_points(builder, cfg, position=position, orientation=orientation)
    else:
        raise TypeError(f"Unsupported MPM particle spawner config type: {type(cfg).__name__}")


def _emit_grid(
    builder: ModelBuilder,
    cfg: MPMGridCfg,
    *,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> None:
    lower = _as_finite_vector3(cfg.lower, "lower")
    upper = _as_finite_vector3(cfg.upper, "upper")
    extent = upper - lower
    if np.any(extent <= 0.0):
        raise ValueError(f"MPMGridCfg upper corner must be greater than lower corner. Got {cfg.lower=} {cfg.upper=}.")

    voxel_size = float(cfg.voxel_size)
    _validate_positive_finite(voxel_size, "voxel_size")
    particles_per_cell = float(cfg.particles_per_cell)
    _validate_positive_finite(particles_per_cell, "particles_per_cell")
    jitter = float(cfg.jitter)
    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError(f"MPMGridCfg `jitter` must be finite and non-negative. Got {cfg.jitter}.")

    resolution = np.maximum(np.ceil(particles_per_cell * extent / voxel_size), 1).astype(np.int32)
    cell_size = extent / resolution
    cell_volume = float(np.prod(cell_size))
    if cfg.mass is None:
        density = float(cfg.material.density)
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError(
                "MPMGridCfg `material.density` must be finite and positive when deriving particle mass. "
                f"Got {cfg.material.density}."
            )
        mass = cell_volume * density
    else:
        mass = float(cfg.mass)
    _validate_positive_finite(mass, "mass")

    if cfg.particle_placement == "boundary":
        first_particle = lower
        dimensions = resolution + 1
    elif cfg.particle_placement == "cell_center":
        first_particle = lower + 0.5 * cell_size
        dimensions = resolution
    else:
        raise ValueError(
            f"MPMGridCfg particle_placement must be 'boundary' or 'cell_center'. Got {cfg.particle_placement!r}."
        )

    if cfg.radius is not None:
        radius = float(cfg.radius)
    elif cfg.particle_placement == "cell_center":
        # Newton represents MPM particle volume as ``(2 * radius) ** 3``.
        radius = 0.5 * float(np.cbrt(cell_volume))
    else:
        # Preserve the historical boundary-placement behavior.
        radius = 0.5 * float(np.max(cell_size))
    _validate_positive_finite(radius, "radius")

    world_pos = _transform_point(first_particle, position, orientation)
    builder.add_particle_grid(
        pos=wp.vec3(*world_pos.tolist()),
        rot=wp.quat(*orientation),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=int(dimensions[0]),
        dim_y=int(dimensions[1]),
        dim_z=int(dimensions[2]),
        cell_x=float(cell_size[0]),
        cell_y=float(cell_size[1]),
        cell_z=float(cell_size[2]),
        mass=mass,
        jitter=jitter,
        radius_mean=radius,
        custom_attributes=_material_custom_attributes(cfg.material),
    )


def _emit_points(
    builder: ModelBuilder,
    cfg: MPMPointsCfg,
    *,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> None:
    points = np.asarray(cfg.positions, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"MPMPointsCfg positions must have shape (N, 3). Got {points.shape}.")
    if points.shape[0] == 0:
        raise ValueError("MPMPointsCfg positions must contain at least one particle.")
    if not np.all(np.isfinite(points)):
        raise ValueError("MPMPointsCfg `positions` must contain only finite values.")

    velocities = np.zeros_like(points) if cfg.velocities is None else np.asarray(cfg.velocities, dtype=np.float32)
    if velocities.shape != points.shape:
        raise ValueError(f"MPMPointsCfg velocities must match positions shape {points.shape}. Got {velocities.shape}.")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("MPMPointsCfg `velocities` must contain only finite values.")

    mass = _expand_scalar_or_sequence(cfg.mass, points.shape[0], "mass")
    radius = _expand_scalar_or_sequence(cfg.radius, points.shape[0], "radius")

    world_points = _transform_points(points, position, orientation)
    world_velocities = _rotate_vectors(velocities, orientation)

    builder.add_particles(
        pos=world_points.tolist(),
        vel=world_velocities.tolist(),
        mass=mass,
        radius=radius,
        custom_attributes=_material_custom_attributes(cfg.material),
    )


def _expand_scalar_or_sequence(value: float | Sequence[float], count: int, name: str) -> list[float]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        values = [float(value)] * count
    else:
        if len(value) != count:
            raise ValueError(
                f"MPMPointsCfg {name} must be scalar or have one value per particle. Got {len(value)} values."
            )
        values = [float(v) for v in value]
    if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0.0):
        raise ValueError(f"MPMPointsCfg `{name}` must contain only finite positive values.")
    return values


def _as_finite_vector3(value: Sequence[float], name: str) -> np.ndarray:
    """Convert a sequence to a finite three-component float32 vector."""
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (3,):
        raise ValueError(f"MPMGridCfg `{name}` must have shape (3,). Got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"MPMGridCfg `{name}` must contain only finite values.")
    return vector


def _validate_positive_finite(value: float, name: str) -> None:
    """Validate a positive finite scalar used for grid particle generation."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"MPMGridCfg `{name}` must be finite and positive. Got {value}.")


def _transform_point(
    point: np.ndarray,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> np.ndarray:
    return _transform_points(point.reshape(1, 3), position, orientation)[0]


def _transform_points(
    points: np.ndarray,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> np.ndarray:
    return _rotate_vectors(points, orientation) + np.asarray(position, dtype=np.float32)


def _rotate_vectors(vectors: np.ndarray, orientation: tuple[float, float, float, float]) -> np.ndarray:
    q = np.asarray(orientation, dtype=np.float32)
    q_vec = q[:3]
    q_w = q[3]
    uv = np.cross(q_vec, vectors)
    uuv = np.cross(q_vec, uv)
    return vectors + 2.0 * (q_w * uv + uuv)
