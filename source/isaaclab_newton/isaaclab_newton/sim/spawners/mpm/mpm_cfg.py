# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import MISSING, dataclass
from typing import Literal

from isaaclab.sim.spawners.materials.visual_materials_cfg import VisualMaterialCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import config_field


@dataclass
class MPMParticleMaterialCfg:
    """Per-particle material values consumed by Newton's implicit MPM solver.

    This lightweight value configuration does not create or bind a USD material.
    Its values are forwarded to Newton as ``mpm:*`` custom attributes when
    particles are emitted into the model builder. Density is used only when a
    particle generator derives mass.

    The defaults model a dry sand-like granular material.
    """

    density: float = config_field(1000.0)
    """Particle material density [kg/m^3] used to derive particle mass."""

    young_modulus: float = config_field(1.0e15)
    """Young's modulus [Pa]."""

    poisson_ratio: float = config_field(0.3)
    """Dimensionless Poisson's ratio for elasticity."""

    viscosity: float = config_field(0.0)
    """Plastic viscosity [Pa·s]."""

    friction: float = config_field(0.68)
    """Dimensionless particle friction coefficient."""

    damping: float = config_field(0.0)
    """Elastic damping relaxation time [s]."""

    yield_pressure: float = config_field(1.0e12)
    """Pressure at which the material yields [Pa]."""

    tensile_yield_ratio: float = config_field(0.0)
    """Dimensionless tensile-to-compressive yield ratio."""

    yield_stress: float = config_field(0.0)
    """Von Mises yield stress [Pa]."""

    hardening: float = config_field(0.0)
    """Dimensionless plastic hardening factor."""

    dilatancy: float = config_field(0.0)
    """Dimensionless granular dilatancy factor."""


@dataclass
class MPMParticleSpawnerCfg(SpawnerCfg):
    """Base configuration for declarative Newton MPM particle generation.

    Particle geometry is emitted directly into Newton during scene replication.
    The USD spawner creates only a lightweight placeholder prim used by Isaac
    Lab's scene and cloning machinery.
    """

    func: Callable | str = config_field("{DIR}.mpm:spawn_mpm_particles")

    material: MPMParticleMaterialCfg = config_field(MPMParticleMaterialCfg())
    """Physical material values applied to generated particles."""

    visual_color: Sequence[float] = config_field((0.7, 0.6, 0.4))
    """RGB display color for particle visualization."""

    visual_material: VisualMaterialCfg | None = config_field(None)
    """Optional visual-material spawner configuration bound to each particle cloud."""

    visual_update_frequency: int = config_field(1)
    """USD-stage particle visualization update frequency in render frames."""


@dataclass
class MPMGridCfg(MPMParticleSpawnerCfg):
    """Generate a regular MPM particle lattice in an axis-aligned local box."""

    lower: Sequence[float] = config_field(MISSING)
    """Lower local-space corner [m], shape ``(3,)``."""

    upper: Sequence[float] = config_field(MISSING)
    """Upper local-space corner [m], shape ``(3,)``."""

    voxel_size: float = config_field(MISSING)
    """Target MPM voxel size [m], used with :attr:`particles_per_cell` to choose lattice resolution."""

    particles_per_cell: float = config_field(1.0)
    """Particle resolution multiplier applied independently along each axis.

    For example, ``2`` doubles the lattice resolution along every axis and
    therefore creates approximately eight times as many particles in 3D.
    """

    particle_placement: Literal["boundary", "cell_center"] = config_field("boundary")
    """Particle placement convention.

    ``"boundary"`` preserves the original behavior by emitting particles on
    both box boundaries. ``"cell_center"`` emits one equal-volume particle at
    each lattice-cell center, so derived particle masses sum to the requested
    box mass.
    """

    jitter: float = config_field(0.0)
    """Width of Newton's uniform per-axis jitter interval [m].

    Newton samples each position component in ``[-jitter / 2, jitter / 2]``.
    """

    mass: float | None = config_field(None)
    """Per-particle mass [kg].

    If ``None``, mass is derived from the generated lattice-cell volume and
    :attr:`material` density.
    """

    radius: float | None = config_field(None)
    """Particle radius [m].

    If ``None``, cell-centered placement uses the equal-volume radius whose
    represented cube volume matches the lattice-cell volume. Boundary placement
    uses half the largest lattice-cell size.
    """


@dataclass
class MPMPointsCfg(MPMParticleSpawnerCfg):
    """Generate MPM particles from explicit local-space positions."""

    positions: Sequence[Sequence[float]] = config_field(MISSING)
    """Local-space particle positions [m], shape ``(num_particles, 3)``."""

    velocities: Sequence[Sequence[float]] | None = config_field(None)
    """Local-space particle velocities [m/s], shape ``(num_particles, 3)``.

    If ``None``, all initial velocities are zero.
    """

    mass: float | Sequence[float] = config_field(1.0)
    """Positive particle masses [kg], either scalar or one value per particle."""

    radius: float | Sequence[float] = config_field(0.01)
    """Positive particle radii [m], either scalar or one value per particle."""
