# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from isaaclab.sim.utils import clone, create_prim

if TYPE_CHECKING:
    from pxr import Usd

from .mpm_cfg import MPMGridCfg, MPMParticleMaterialCfg, MPMParticleSpawnerCfg, MPMPointsCfg

_SIMULATION_POINTS_SUFFIX = "/geometry/points"
_PHYSICS_MATERIAL_SUFFIX = "/geometry/physics_material"
_GRID_JITTER_SEED = 42


@clone
def spawn_mpm_particles(
    prim_path: str,
    cfg: MPMParticleSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Author a Newton MPM particle object as schema-valid USD.

    The asset root remains an ``Xform`` for normal Isaac Lab pose and cloning
    workflows. Explicit simulation points and their physics material are authored
    below it and imported by Newton during scene replication. The simulation
    points stay invisible because :class:`~isaaclab_newton.assets.MPMObject`
    maintains a separate mutable render cloud at ``<asset>/Particles``.

    Args:
        prim_path: Prim path or pattern at which to create the particle asset.
        cfg: MPM particle spawner configuration.
        translation: Translation relative to the parent prim [m]. If ``None``, the
            asset root uses the origin.
        orientation: Orientation relative to the parent prim as an ``(x, y, z, w)``
            quaternion. If ``None``, the asset root uses the identity quaternion.
        **kwargs: Additional keyword arguments consumed by the :func:`~isaaclab.sim.utils.clone`
            decorator.

    Returns:
        The created asset root prim.
    """
    if isinstance(cfg, MPMGridCfg):
        positions, velocities, masses, radii = _generate_grid_particles(cfg)
    elif isinstance(cfg, MPMPointsCfg):
        positions, velocities, masses, radii = _generate_explicit_particles(cfg)
    else:
        raise TypeError(f"Unsupported MPM particle spawner config type: {type(cfg).__name__}")
    cfg.spawn_path = prim_path

    # Register codeless schemas before consulting USD's schema registry. Keep the
    # imports local so importing config classes before SimulationApp remains safe.
    import newton_usd_schemas  # noqa: F401, PLC0415

    from pxr import Sdf, UsdGeom, UsdShade, Vt  # noqa: PLC0415

    from isaaclab.sim.utils import get_current_stage  # noqa: PLC0415

    stage = get_current_stage()
    scene_prim = _find_owning_mpm_scene(stage)

    root_prim = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    UsdGeom.Scope.Define(stage, f"{prim_path}/geometry")

    points = UsdGeom.Points.Define(stage, f"{prim_path}{_SIMULATION_POINTS_SUFFIX}")
    points_prim = points.GetPrim()
    if not points_prim.ApplyAPI("NewtonPointsDeformableSimAPI"):
        raise RuntimeError(f"Failed to apply NewtonPointsDeformableSimAPI to '{points_prim.GetPath()}'.")
    if not points_prim.AddAppliedSchema("PhysicsDeformableBodyAPI"):
        raise RuntimeError(f"Failed to apply PhysicsDeformableBodyAPI to '{points_prim.GetPath()}'.")

    points.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(positions, dtype=np.float32)))
    points.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(velocities, dtype=np.float32)))
    points.CreateWidthsAttr(Vt.FloatArray.FromNumpy(np.ascontiguousarray(2.0 * radii, dtype=np.float32)))
    points.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    points_prim.GetAttribute("physics:masses").Set(
        Vt.FloatArray.FromNumpy(np.ascontiguousarray(masses, dtype=np.float32))
    )
    points_prim.CreateRelationship("physics:simulationOwner").SetTargets([Sdf.Path(scene_prim.GetPath())])
    points.MakeInvisible()

    material = UsdShade.Material.Define(stage, f"{prim_path}{_PHYSICS_MATERIAL_SUFFIX}")
    _author_mpm_material(material.GetPrim(), cfg.material)
    UsdShade.MaterialBindingAPI.Apply(points_prim).Bind(material, materialPurpose="physics")
    return root_prim


def _find_owning_mpm_scene(stage: Usd.Stage) -> Usd.Prim:
    """Return the single MPM physics scene that owns newly authored particles."""
    from pxr import UsdPhysics  # noqa: PLC0415

    from isaaclab.sim import SimulationContext  # noqa: PLC0415

    sim = SimulationContext.instance()
    if sim is not None and sim.stage is stage:
        scene_prim = stage.GetPrimAtPath(sim.cfg.physics_prim_path)
        candidates = [scene_prim] if scene_prim and scene_prim.IsA(UsdPhysics.Scene) else []
    else:
        candidates = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]

    if len(candidates) != 1:
        raise RuntimeError(f"MPM particle spawning requires exactly one owning PhysicsScene; found {len(candidates)}.")
    scene_prim = candidates[0]
    if "NewtonMPMSceneAPI" not in scene_prim.GetAppliedSchemas() and not scene_prim.ApplyAPI("NewtonMPMSceneAPI"):
        raise RuntimeError(f"Failed to apply NewtonMPMSceneAPI to '{scene_prim.GetPath()}'.")
    return scene_prim


def _author_mpm_material(material_prim: Usd.Prim, material: MPMParticleMaterialCfg) -> None:
    """Apply ``NewtonMPMMaterialAPI`` and author every supported material value."""
    if not material_prim.ApplyAPI("NewtonMPMMaterialAPI"):
        raise RuntimeError(f"Failed to apply NewtonMPMMaterialAPI to '{material_prim.GetPath()}'.")

    attributes = {
        "physics:density": material.density,
        "newton:mpm:youngsModulus": material.young_modulus,
        "newton:mpm:poissonsRatio": material.poisson_ratio,
        # Isaac Lab stores a relaxation time; USD stores absolute viscosity.
        "newton:mpm:elasticDamping": material.damping * material.young_modulus,
        "newton:mpm:internalFriction": material.friction,
        "newton:mpm:yieldPressure": material.yield_pressure,
        "newton:mpm:tensileYieldRatio": material.tensile_yield_ratio,
        "newton:mpm:yieldStress": material.yield_stress,
        "newton:mpm:viscosity": material.viscosity,
        "newton:mpm:hardening": material.hardening,
        "newton:mpm:hardeningRate": material.hardening_rate,
        "newton:mpm:softeningRate": material.softening_rate,
        "newton:mpm:dilatancy": material.dilatancy,
    }
    for name, value in attributes.items():
        material_prim.GetAttribute(name).Set(float(value))


def _generate_grid_particles(cfg: MPMGridCfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a deterministic explicit particle grid for USD authoring."""
    first_particle, dimensions, cell_size, mass, radius, jitter = _grid_parameters(cfg)
    px = np.arange(int(dimensions[0])) * float(cell_size[0])
    py = np.arange(int(dimensions[1])) * float(cell_size[1])
    pz = np.arange(int(dimensions[2])) * float(cell_size[2])
    positions = np.stack(np.meshgrid(px, py, pz, indexing="xy")).reshape(3, -1).T
    # ``add_particle_grid`` historically received this origin through ``wp.vec3``.
    positions += np.asarray(first_particle, dtype=np.float32)
    positions += _grid_jitter_offsets(positions.shape, jitter, _GRID_JITTER_SEED)

    count = positions.shape[0]
    return (
        np.ascontiguousarray(positions, dtype=np.float32),
        np.zeros((count, 3), dtype=np.float32),
        np.full(count, mass, dtype=np.float32),
        np.full(count, radius, dtype=np.float32),
    )


def _grid_jitter_offsets(shape: tuple[int, ...], jitter: float, seed: int) -> np.ndarray:
    """Return deterministic particle jitter offsets."""
    if jitter == 0.0:
        return np.zeros(shape, dtype=np.float32)
    return (np.random.default_rng(seed).random(shape) - 0.5) * jitter


def _generate_explicit_particles(cfg: MPMPointsCfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate explicit particle values and return contiguous host arrays."""
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

    masses = _expand_scalar_or_sequence(cfg.mass, points.shape[0], "mass")
    radii = _expand_scalar_or_sequence(cfg.radius, points.shape[0], "radius")
    return (
        np.ascontiguousarray(points),
        np.ascontiguousarray(velocities),
        np.ascontiguousarray(masses, dtype=np.float64),
        np.ascontiguousarray(radii, dtype=np.float64),
    )


def _grid_parameters(
    cfg: MPMGridCfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Validate a grid configuration and derive its explicit lattice parameters."""
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

    return first_particle, dimensions, cell_size, mass, radius, jitter


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
