# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import ClassVar, Literal

from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    DeformableBodyMaterialBaseCfg,
    RigidBodyMaterialBaseCfg,
    SurfaceDeformableBodyMaterialBaseCfg,
)
from isaaclab.utils.configclass import configclass


@configclass
class OmniPhysicsDeformableMaterialCfg:
    """OmniPhysics material properties for a deformable body.

    These properties are set with the prefix ``omniphysics:<property_name>``.
    """

    _usd_namespace: ClassVar[str | None] = "omniphysics"
    _usd_applied_schema: ClassVar[str | None] = "OmniPhysicsDeformableMaterialAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    density: float = 1000.0
    """The material density [kg/m^3]. Defaults to 1000.0 kg/m^3."""

    static_friction: float = 0.25
    """The static friction coefficient. Defaults to 0.25."""

    dynamic_friction: float = 0.25
    """The dynamic friction coefficient. Defaults to 0.25."""

    youngs_modulus: float = 1000000.0
    """The Young's modulus, which defines the body's stiffness [Pa]. Defaults to 1 MPa."""

    poissons_ratio: float = 0.45
    """The Poisson's ratio which defines the body's volume preservation."""


@configclass
class OmniPhysicsSurfaceDeformableMaterialCfg(OmniPhysicsDeformableMaterialCfg):
    """OmniPhysics material properties for a surface deformable body."""

    _usd_namespace: ClassVar[str | None] = "omniphysics"
    _usd_applied_schema: ClassVar[str | None] = "OmniPhysicsSurfaceDeformableMaterialAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    surface_thickness: float = 0.01
    """The thickness of the deformable body's surface [m]. Defaults to 0.01."""

    surface_stretch_stiffness: float = 0.0
    """The stretch stiffness of the deformable body's surface. Defaults to 0.0."""

    surface_shear_stiffness: float = 0.0
    """The shear stiffness of the deformable body's surface. Defaults to 0.0."""

    surface_bend_stiffness: float = 0.0
    """The bend stiffness of the deformable body's surface. Defaults to 0.0."""


@configclass
class PhysXDeformableMaterialCfg:
    """PhysX-specific material properties for a deformable body.

    These properties are set with the prefix ``physxDeformableMaterial:<property_name>``.
    """

    _usd_namespace: ClassVar[str | None] = "physxDeformableMaterial"
    _usd_applied_schema: ClassVar[str | None] = "PhysxDeformableMaterialAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    elasticity_damping: float = 0.005
    """The elasticity damping for the deformable material. Defaults to 0.005."""


@configclass
class PhysxDeformableBodyMaterialCfg(
    DeformableBodyMaterialBaseCfg,
    OmniPhysicsDeformableMaterialCfg,
    PhysXDeformableMaterialCfg,
):
    """PhysX-specific physics material parameters for deformable bodies."""

    func: Callable | str = "isaaclab.sim.spawners.materials.physics_materials:spawn_deformable_body_material"


@configclass
class PhysxSurfaceDeformableBodyMaterialCfg(
    SurfaceDeformableBodyMaterialBaseCfg,
    OmniPhysicsSurfaceDeformableMaterialCfg,
    PhysXDeformableMaterialCfg,
):
    """PhysX-specific physics material parameters for surface deformable bodies."""

    _usd_namespace: ClassVar[str | None] = "physxDeformableMaterial"
    _usd_applied_schema: ClassVar[str | None] = "PhysxSurfaceDeformableMaterialAPI"

    func: Callable | str = "isaaclab.sim.spawners.materials.physics_materials:spawn_deformable_body_material"

    bend_damping: float = 0.0
    """Damping acting against bend-resistance forces [1/s]. Defaults to 0.0."""


@configclass
class DeformableBodyMaterialCfg(PhysxDeformableBodyMaterialCfg):
    """Deprecated: use :class:`PhysxDeformableBodyMaterialCfg`.

    .. deprecated:: 4.6.x
        ``DeformableBodyMaterialCfg`` has moved to
        :class:`PhysxDeformableBodyMaterialCfg` for PhysX-specific deformable materials
        and is scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'DeformableBodyMaterialCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.spawners.materials.PhysxDeformableBodyMaterialCfg' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class SurfaceDeformableBodyMaterialCfg(PhysxSurfaceDeformableBodyMaterialCfg):
    """Deprecated: use :class:`PhysxSurfaceDeformableBodyMaterialCfg`.

    .. deprecated:: 4.6.x
        ``SurfaceDeformableBodyMaterialCfg`` has moved to
        :class:`PhysxSurfaceDeformableBodyMaterialCfg` for PhysX-specific surface
        deformable materials and is scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'SurfaceDeformableBodyMaterialCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.spawners.materials.PhysxSurfaceDeformableBodyMaterialCfg' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


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
