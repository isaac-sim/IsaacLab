# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import Literal

from isaaclab.sim.spawners.materials.visual_materials_cfg import VisualMaterialCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils.configclass import configclass


@configclass
class MPMParticleMaterialCfg:
    """Per-particle material values consumed by Newton's implicit MPM solver.

    This lightweight value configuration does not create or bind a USD material.
    Its values are forwarded to Newton as ``mpm:*`` custom attributes when
    particles are emitted into the model builder. Density is used only when a
    particle generator derives mass.

    The defaults model a dry sand-like granular material.
    """

    density: float = 1000.0
    """Particle material density [kg/m^3] used to derive particle mass."""

    young_modulus: float = 1.0e15
    """Young's modulus [Pa]."""

    poisson_ratio: float = 0.3
    """Dimensionless Poisson's ratio for elasticity."""

    viscosity: float = 0.0
    """Plastic viscosity [Pa·s]."""

    friction: float = 0.68
    """Dimensionless particle friction coefficient."""

    damping: float = 0.0
    """Elastic damping relaxation time [s]."""

    yield_pressure: float = 1.0e12
    """Pressure at which the material yields [Pa]."""

    tensile_yield_ratio: float = 0.0
    """Dimensionless tensile-to-compressive yield ratio."""

    yield_stress: float = 0.0
    """Von Mises yield stress [Pa]."""

    hardening: float = 0.0
    """Dimensionless plastic hardening factor."""

    dilatancy: float = 0.0
    """Dimensionless granular dilatancy factor."""


@configclass
class MPMParticleSpawnerCfg(SpawnerCfg):
    """Base configuration for declarative Newton MPM particle generation.

    Particle geometry is emitted directly into Newton during scene replication.
    The USD spawner creates only a lightweight placeholder prim used by Isaac
    Lab's scene and cloning machinery.
    """

    func: Callable | str = "{DIR}.mpm:spawn_mpm_particles"

    material: MPMParticleMaterialCfg = MPMParticleMaterialCfg()
    """Physical material values applied to generated particles."""

    visual_color: Sequence[float] = (0.7, 0.6, 0.4)
    """RGB display color for particle visualization."""

    visual_material: VisualMaterialCfg | None = None
    """Optional visual-material spawner configuration bound to each particle cloud."""

    visual_update_frequency: int = 1
    """USD-stage particle visualization update frequency in render frames."""


@configclass
class MPMGridCfg(MPMParticleSpawnerCfg):
    """Generate a regular MPM particle lattice in an axis-aligned local box."""

    lower: Sequence[float] = MISSING
    """Lower local-space corner [m], shape ``(3,)``."""

    upper: Sequence[float] = MISSING
    """Upper local-space corner [m], shape ``(3,)``."""

    voxel_size: float = MISSING
    """Target MPM voxel size [m], used with :attr:`particles_per_cell` to choose lattice resolution."""

    particles_per_cell: float = 1.0
    """Particle resolution multiplier applied independently along each axis.

    For example, ``2`` doubles the lattice resolution along every axis and
    therefore creates approximately eight times as many particles in 3D.
    """

    particle_placement: Literal["boundary", "cell_center"] = "boundary"
    """Particle placement convention.

    ``"boundary"`` preserves the original behavior by emitting particles on
    both box boundaries. ``"cell_center"`` emits one equal-volume particle at
    each lattice-cell center, so derived particle masses sum to the requested
    box mass.
    """

    jitter: float = 0.0
    """Width of Newton's uniform per-axis jitter interval [m].

    Newton samples each position component in ``[-jitter / 2, jitter / 2]``.
    """

    mass: float | None = None
    """Per-particle mass [kg].

    If ``None``, mass is derived from the generated lattice-cell volume and
    :attr:`material` density.
    """

    radius: float | None = None
    """Particle radius [m].

    If ``None``, cell-centered placement uses the equal-volume radius whose
    represented cube volume matches the lattice-cell volume. Boundary placement
    preserves the legacy default of half the largest lattice-cell size.
    """


@configclass
class MPMPointsCfg(MPMParticleSpawnerCfg):
    """Generate MPM particles from explicit local-space positions."""

    positions: Sequence[Sequence[float]] = MISSING
    """Local-space particle positions [m], shape ``(num_particles, 3)``."""

    velocities: Sequence[Sequence[float]] | None = None
    """Local-space particle velocities [m/s], shape ``(num_particles, 3)``.

    If ``None``, all initial velocities are zero.
    """

    mass: float | Sequence[float] = 1.0
    """Positive particle masses [kg], either scalar or one value per particle."""

    radius: float | Sequence[float] = 0.01
    """Positive particle radii [m], either scalar or one value per particle."""
