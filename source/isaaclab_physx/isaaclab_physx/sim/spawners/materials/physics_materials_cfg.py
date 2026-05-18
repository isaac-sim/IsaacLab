# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import ClassVar, Literal

from isaaclab.sim.spawners.materials import PhysicsMaterialCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass


@configclass
class OmniPhysicsDeformableMaterialCfg:
    """OmniPhysics material properties for a deformable body.

    These properties are set with the prefix ``omniphysics:<property_name>``. For example, to set the density of the
    deformable body, you would set the property ``omniphysics:density``.

    See the OmniPhysics documentation for more information on the available properties.
    """

    density: float | None = None
    """The material density in [kg/m^3]. Defaults to None, in which case the simulation decides the default density."""

    static_friction: float = 0.25
    """The static friction. Defaults to 0.25."""

    dynamic_friction: float = 0.25
    """The dynamic friction. Defaults to 0.25."""

    youngs_modulus: float = 1000000.0
    """The Young's modulus, which defines the body's stiffness. Defaults to 1[MPa].

    The Young's modulus is a measure of the material's ability to deform under stress. It is measured in Pascals ([Pa]).
    """

    poissons_ratio: float = 0.45
    """The Poisson's ratio which defines the body's volume preservation. Defaults to 0.45.

    The Poisson's ratio is a measure of the material's ability to expand in the lateral direction when compressed
    in the axial direction. It is a dimensionless number between 0 and 0.5. Using a value of 0.5 will make the
    material incompressible.
    """


@configclass
class OmniPhysicsSurfaceDeformableMaterialCfg(OmniPhysicsDeformableMaterialCfg):
    """OmniPhysics material properties for a surface deformable body,
    extending on :class:`OmniPhysicsDeformableMaterialCfg` with additional parameters for surface deformable bodies.

    These properties are set with the prefix ``omniphysics:<property_name>``.
    For example, to set the surface thickness of the surface deformable body,
    you would set the property ``omniphysics:surfaceThickness``.

    See the OmniPhysics documentation for more information on the available properties.
    """

    surface_thickness: float = 0.01
    """The thickness of the deformable body's surface. Defaults to 0.01 meters ([m])."""

    surface_stretch_stiffness: float = 0.0
    """The stretch stiffness of the deformable body's surface. Defaults to 0.0."""

    surface_shear_stiffness: float = 0.0
    """The shear stiffness of the deformable body's surface. Defaults to 0.0."""

    surface_bend_stiffness: float = 0.0
    """The bend stiffness of the deformable body's surface. Defaults to 0.0."""

    bend_damping: float = 0.0
    """The bend damping for the deformable body's surface. Defaults to 0.0."""


@configclass
class PhysXDeformableMaterialCfg:
    """PhysX-specific material properties for a deformable body.

    These properties are set with the prefix ``physxDeformableBody:<property_name>``.
    For example, to set the elasticity damping of the deformable body,
    you would set the property ``physxDeformableBody:elasticityDamping``.

    See the PhysX documentation for more information on the available properties.
    """

    elasticity_damping: float = 0.005
    """The elasticity damping for the deformable material. Defaults to 0.005."""


@configclass
class DeformableBodyMaterialCfg(PhysicsMaterialCfg, OmniPhysicsDeformableMaterialCfg, PhysXDeformableMaterialCfg):
    """Physics material parameters for deformable bodies.

    See :meth:`spawn_deformable_body_material` for more information.
    """

    func: Callable | str = "{DIR}.physics_materials:spawn_deformable_body_material"

    _property_prefix: dict[str, list[str]] = {
        "omniphysics": [field.name for field in dataclasses.fields(OmniPhysicsDeformableMaterialCfg)],
        "physxDeformableBody": [field.name for field in dataclasses.fields(PhysXDeformableMaterialCfg)],
    }
    """Mapping between the property prefixes and the properties that fall under each prefix."""


@configclass
class SurfaceDeformableBodyMaterialCfg(DeformableBodyMaterialCfg, OmniPhysicsSurfaceDeformableMaterialCfg):
    """Physics material parameters for surface deformable bodies,
    extending on :class:`DeformableBodyMaterialCfg` with additional parameters for surface deformable bodies.

    See :meth:`spawn_deformable_body_material` for more information.
    """

    func: Callable | str = "{DIR}.physics_materials:spawn_deformable_body_material"

    _property_prefix: dict[str, list[str]] = {
        "omniphysics": [field.name for field in dataclasses.fields(OmniPhysicsSurfaceDeformableMaterialCfg)],
        "physxDeformableBody": [field.name for field in dataclasses.fields(PhysXDeformableMaterialCfg)],
    }
    """Extend DeformableBodyMaterialCfg properties under each prefix."""


@configclass
class PhysxRigidBodyMaterialCfg(RigidBodyMaterialBaseCfg):
    """PhysX-specific physics-material parameters for rigid bodies.

    Extends :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` with the
    `PhysxMaterialAPI`_ schema fields: compliant-contact spring (stiffness/damping) and the
    friction/restitution combine-mode tokens. None of these fields have a Newton consumer
    today; they are PhysX-engine-only knobs.

    See :meth:`~isaaclab.sim.spawners.materials.spawn_rigid_body_material` for more information.

    .. _PhysxMaterialAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_material_a_p_i.html
    """

    # -- Class metadata (not dataclass fields) --
    # USD applied schema written when at least one PhysX-namespaced field is set.
    _usd_applied_schema: ClassVar[str | None] = "PhysxMaterialAPI"
    # Prim attribute namespace for PhysX-specific fields.
    _usd_namespace: ClassVar[str | None] = "physxMaterial"

    compliant_contact_stiffness: float | None = None
    """Spring stiffness for a compliant contact model using implicit springs.

    A higher stiffness results in behavior closer to a rigid contact. The compliant contact model
    is only enabled if the stiffness is larger than 0. PhysX-only; not consumed by Newton.
    """

    compliant_contact_damping: float | None = None
    """Damping coefficient for a compliant contact model using implicit springs.

    Irrelevant if compliant contacts are disabled when :attr:`compliant_contact_stiffness` is set
    to zero and rigid contacts are active. PhysX-only; not consumed by Newton.
    """

    friction_combine_mode: Literal["average", "min", "multiply", "max"] | None = None
    """Determines the way friction will be combined during collisions.

    .. attention::

        When two physics materials with different combine modes collide, the combine mode with
        the higher priority will be used. The priority order is provided `here
        <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCombineMode.html>`__.
    """

    restitution_combine_mode: Literal["average", "min", "multiply", "max"] | None = None
    """Determines the way restitution coefficient will be combined during collisions.

    .. attention::

        When two physics materials with different combine modes collide, the combine mode with
        the higher priority will be used. The priority order is provided `here
        <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCombineMode.html>`__.
    """


@configclass
class RigidBodyMaterialCfg(PhysxRigidBodyMaterialCfg):
    """Deprecated: use :class:`PhysxRigidBodyMaterialCfg` or
    :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg`.

    .. deprecated:: 4.6.22
        ``RigidBodyMaterialCfg`` has been split into
        :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` (solver-common) and
        :class:`PhysxRigidBodyMaterialCfg` (PhysX-specific) and relocated to
        :mod:`isaaclab_physx.sim.spawners.materials`. This alias preserves backwards compatibility
        and is scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'RigidBodyMaterialCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg' for PhysX"
            " properties, or 'isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg' for"
            " solver-common properties only.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()
