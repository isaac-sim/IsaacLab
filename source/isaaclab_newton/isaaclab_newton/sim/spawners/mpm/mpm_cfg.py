# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import MISSING

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils.configclass import configclass


@configclass
class MPMParticleMaterialCfg:
    """Per-particle material values consumed by Newton's implicit MPM solver.

    This is a lightweight value config. It does not create or bind a USD material
    prim; values are forwarded to Newton as ``mpm:*`` custom attributes when
    particles are added to the model builder.

    The defaults model a dry sand-like granular material.
    """

    density: float = 1000.0
    """Particle material density [kg/m^3], used to derive particle mass when a generator does not override it."""

    young_modulus: float = 1.0e15
    """Young's modulus [Pa]."""

    poisson_ratio: float = 0.3
    """Poisson ratio."""

    viscosity: float = 0.0
    """Viscosity coefficient."""

    friction: float = 0.68
    """Particle friction coefficient."""

    damping: float = 0.0
    """Material damping."""

    yield_pressure: float = 1.0e12
    """Pressure at which the material yields."""

    tensile_yield_ratio: float = 0.0
    """Tensile/compressive yield ratio."""

    yield_stress: float = 0.0
    """Von-Mises yield stress."""

    hardening: float = 0.0
    """Plastic hardening coefficient."""

    dilatancy: float = 0.0
    """Granular dilatancy coefficient."""


@configclass
class MPMParticleSpawnerCfg(SpawnerCfg):
    """Base configuration for declarative Newton MPM particle generation."""

    func: Callable | str = "{DIR}.mpm:spawn_mpm_particles"

    material: MPMParticleMaterialCfg = MPMParticleMaterialCfg()
    """Material values applied to generated particles."""

    visual_color: Sequence[float] = (0.7, 0.6, 0.4)
    """Display color for Kit particle visualization."""

    visual_update_frequency: int = 1
    """Kit particle visualization update frequency in render frames."""


@configclass
class MPMGridCfg(MPMParticleSpawnerCfg):
    """Generate a regular MPM particle lattice inside an axis-aligned local box."""

    lower: Sequence[float] = MISSING
    """Lower local-space corner of the particle box [m]."""

    upper: Sequence[float] = MISSING
    """Upper local-space corner of the particle box [m]."""

    voxel_size: float = MISSING
    """Target MPM voxel size [m], used with :attr:`particles_per_cell` to choose lattice resolution."""

    particles_per_cell: float = 1.0
    """Particle lattice density relative to the MPM grid resolution."""

    jitter: float = 0.0
    """Newton particle-grid jitter value [m]."""

    mass: float | None = None
    """Per-particle mass [kg]. If ``None``, mass is derived from cell volume and material density."""

    radius: float | None = None
    """Particle radius [m]. If ``None``, radius is half the largest generated cell size."""


@configclass
class MPMPointsCfg(MPMParticleSpawnerCfg):
    """Generate MPM particles from explicit local-space point positions."""

    positions: Sequence[Sequence[float]] = MISSING
    """Local-space particle positions [m]."""

    velocities: Sequence[Sequence[float]] | None = None
    """Optional local-space particle velocities [m/s]. If ``None``, velocities are zero."""

    mass: float | Sequence[float] = 1.0
    """Particle mass values [kg], either scalar or one value per particle."""

    radius: float | Sequence[float] = 0.01
    """Particle radius values [m], either scalar or one value per particle."""
