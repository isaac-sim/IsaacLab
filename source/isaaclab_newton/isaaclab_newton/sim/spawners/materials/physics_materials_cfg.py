# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    DeformableBodyMaterialBaseCfg,
    RigidBodyMaterialFragment,
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
class NewtonMaterialCfg(RigidBodyMaterialFragment):
    """``newton:*`` rigid-body material attributes read by Newton's USD material schema resolver.

    Single-namespace fragment (see
    :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment`) for the Newton-only
    friction knobs (torsional and rolling friction) and the per-material contact model (contact
    stiffness/damping, friction gain, adhesion) that replaces the deprecated per-shape
    ``ke``/``kd``/``kf``/``ka`` parameters. The ``NewtonMaterialAPI`` schema is applied (and the
    ``newton:*`` attributes authored) by the generic :func:`~isaaclab.sim.schemas.apply_namespaced`
    writer. ``None`` fields are left unchanged.

    .. note::
        The generated ``NewtonMaterialAPI`` USD schema currently only declares the two friction
        attributes; the four contact attributes are still authored as raw ``newton:*`` USD
        attributes and are read directly by Newton's schema resolver.

    Composes with other rigid-body material fragments (e.g.
    :class:`~isaaclab.sim.spawners.materials.UsdPhysicsRigidBodyMaterialCfg`) in the same fragment
    list passed to
    :func:`~isaaclab.sim.spawners.materials.spawn_rigid_body_material_from_fragments`. For the
    legacy (non-fragment) equivalent, see
    :class:`~isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg`.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = "NewtonMaterialAPI"

    torsional_friction: float | None = None
    """Torsional friction coefficient (resistance to spinning at a contact point) [dimensionless].

    Writes ``newton:torsionalFriction``. Range: [0, inf).
    """

    rolling_friction: float | None = None
    """Rolling friction coefficient (resistance to rolling motion) [dimensionless].

    Writes ``newton:rollingFriction``. Range: [0, inf).
    """

    contact_stiffness: float | None = None
    """Contact normal-force stiffness [N/m].

    Writes ``newton:contactStiffness``. Replaces the deprecated per-shape ``ke`` contact parameter;
    used by the SemiImplicit, Featherstone, MuJoCo, and VBD solvers.
    """

    contact_damping: float | None = None
    """Contact normal-force damping coefficient [N·s/m].

    Writes ``newton:contactDamping``. Replaces the deprecated per-shape ``kd`` contact parameter;
    used by the SemiImplicit, Featherstone, MuJoCo, and VBD solvers.
    """

    contact_friction_gain: float | None = None
    """Friction-force stiffness gain used by the tangential (friction) contact response [N·s/m].

    Writes ``newton:contactFrictionGain``. Replaces the deprecated per-shape ``kf`` contact
    parameter; used by the SemiImplicit and Featherstone solvers.
    """

    contact_adhesion: float | None = None
    """Contact adhesion distance: shapes closer than this threshold experience an attractive
    (adhesive) force [m].

    Writes ``newton:contactAdhesion``. Replaces the deprecated per-shape ``ka`` contact parameter;
    used by the SemiImplicit and Featherstone solvers.
    """
