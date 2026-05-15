# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    DeformableBodyMaterialBaseCfg,
    PhysicsMaterialCfg,
    SurfaceDeformableBodyMaterialBaseCfg,
)
from isaaclab.utils.configclass import configclass


@configclass
class NewtonDeformableMaterialCfg:
    """Newton-specific material properties for a deformable body.

    These properties are set with the prefix ``newton:<property_name>``.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}

    density: float = 1.0
    """The material density [kg/m^3]. Defaults to 1.0 kg/m^3."""

    particle_radius: float = 0.008
    """Particle radius [m] used by the Newton backend."""


@configclass
class NewtonDeformableBodyMaterialCfg(DeformableBodyMaterialBaseCfg, NewtonDeformableMaterialCfg):
    """Newton-specific physics material parameters for volume deformable bodies."""

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}

    func: Callable | str = "isaaclab.sim.spawners.materials.physics_materials:spawn_deformable_body_material"

    k_mu: float = 1e5
    """First Lame material parameter [Pa]. Defaults to 1e5 Pa."""

    k_lambda: float = 1e5
    """Second Lame material parameter [Pa]. Defaults to 1e5 Pa."""

    k_damp: float = 0.0
    """Damping stiffness for tetrahedral elements [Pa*s]. Defaults to 0.0."""


@configclass
class NewtonSurfaceDeformableBodyMaterialCfg(SurfaceDeformableBodyMaterialBaseCfg, NewtonDeformableMaterialCfg):
    """Newton-specific physics material parameters for surface deformable bodies."""

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}

    func: Callable | str = "isaaclab.sim.spawners.materials.physics_materials:spawn_deformable_body_material"

    tri_ke: float = 1e4
    """Triangle area-preserving stiffness [Pa]. Used by Newton backend for cloth meshes."""

    tri_ka: float = 1e4
    """Triangle area stiffness [Pa]. Used by Newton backend for cloth meshes."""

    tri_kd: float = 1.5e-6
    """Triangle area damping [Pa*s]. Used by Newton backend for cloth meshes."""

    edge_ke: float = 5.0
    """Bending stiffness [N*m]. Used by Newton backend for cloth meshes."""

    edge_kd: float = 1e-2
    """Bending damping [N*m*s]. Used by Newton backend for cloth meshes."""


@configclass
class NewtonCableMaterialCfg(PhysicsMaterialCfg):
    """Newton-specific physics material for cable rods.

    Authored as a ``UsdShade.Material`` prim with ``newton:*`` attributes via the
    generic :func:`spawn_deformable_body_material` helper. :class:`CableObject`
    reads these fields directly from ``cfg.physics_material`` when constructing
    the registry entry.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}

    func: Callable | str = "isaaclab.sim.spawners.materials.physics_materials:spawn_deformable_body_material"

    stretch_stiffness: float = 1.0e9
    """Material-like axial stiffness EA [N]; normalized internally by segment length."""

    bend_stiffness: float = 0.0
    """Material-like bend/twist stiffness EI [N*m^2]; normalized internally by segment length."""

    stretch_damping: float = 0.0
    """Per-joint stretch damping [N*s/m]."""

    bend_damping: float = 0.0
    """Per-joint bend/twist damping [N*m*s/rad]."""

    density: float = 1500.0
    """Material density [kg/m^3]. Converted to per-segment mass via the capsule
    shape's volume (``pi * radius^2 * segment_length * density``) by the cable
    replicate hook before calling :meth:`newton.ModelBuilder.add_rod_graph`."""
