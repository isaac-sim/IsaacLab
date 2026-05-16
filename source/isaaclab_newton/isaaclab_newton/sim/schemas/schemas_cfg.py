# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import ClassVar

from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootBaseCfg,
    CollisionBaseCfg,
    JointDriveBaseCfg,
    MeshCollisionBaseCfg,
    RigidBodyBaseCfg,
)
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass


@configclass
class NewtonRigidBodyPropertiesCfg(RigidBodyBaseCfg):
    """Newton-targeted rigid body properties.

    Base class for cfgs that author rigid-body attributes consumed by any of
    Newton's solver options (MuJoCo, XPBD, Featherstone, Semi-implicit, Kamino).
    Newton has no native ``newton:*`` rigid-body attributes today, so this class
    is currently empty — solver-specific subclasses (e.g.,
    :class:`MujocoRigidBodyPropertiesCfg`) carry the actual fields.

    The ``newton:`` namespace is reserved here so future Newton-native
    rigid-body fields can be added without an API change.

    See :meth:`~isaaclab.sim.schemas.modify_rigid_body_properties` for more information.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}


@configclass
class MujocoRigidBodyPropertiesCfg(NewtonRigidBodyPropertiesCfg):
    """MuJoCo-solver-specific rigid body properties.

    Extends :class:`NewtonRigidBodyPropertiesCfg` with body-level gravity
    compensation, consumed only when running Newton's MuJoCo solver.

    See :meth:`~isaaclab.sim.schemas.modify_rigid_body_properties` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "mjc"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}

    gravcomp: float | None = None
    """Gravity compensation scale for the body [dimensionless].

    ``0.0`` = no compensation; ``1.0`` = full compensation.
    Written to ``mjc:gravcomp`` on the rigid-body prim.
    Body-level gravcomp must be set for joint-level actuatorgravcomp to have any effect.
    """


@configclass
class NewtonJointDrivePropertiesCfg(JointDriveBaseCfg):
    """Newton-targeted joint drive properties.

    Base class for cfgs that author joint-drive attributes consumed by any of
    Newton's solver options. Newton has no native ``newton:*`` joint-drive
    attributes today, so this class is currently empty — solver-specific
    subclasses (e.g., :class:`MujocoJointDrivePropertiesCfg`) carry the actual
    fields.

    The ``newton:`` namespace is reserved here so future Newton-native
    joint-drive fields can be added without an API change.

    See :meth:`~isaaclab.sim.schemas.modify_joint_drive_properties` for more information.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}


@configclass
class MujocoJointDrivePropertiesCfg(NewtonJointDrivePropertiesCfg):
    """MuJoCo-solver-specific joint drive properties.

    Extends :class:`NewtonJointDrivePropertiesCfg` with joint-level gravity
    compensation routing, consumed only when running Newton's MuJoCo solver.

    See :meth:`~isaaclab.sim.schemas.modify_joint_drive_properties` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "mjc"
    _usd_applied_schema: ClassVar[str | None] = "MjcJointAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    actuatorgravcomp: bool | None = None
    """Route gravity compensation forces through the actuator channel.

    When ``True``, compensation forces go to ``qfrc_actuator`` (subject to force limits).
    Requires body-level :attr:`MujocoRigidBodyPropertiesCfg.gravcomp`.
    Written to ``mjc:actuatorgravcomp`` via ``MjcJointAPI``.
    """


@configclass
class NewtonCollisionPropertiesCfg(CollisionBaseCfg):
    """Newton-specific collision properties.

    Extends :class:`~isaaclab.sim.schemas.CollisionBaseCfg` with Newton-native
    contact geometry attributes.

    See :meth:`~isaaclab.sim.schemas.modify_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = "NewtonCollisionAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    contact_margin: float | None = None
    """Outward inflation of the collision surface [m].

    Extends the effective collision surface outward. Sum of both bodies' margins is
    used for collision detection. Essential for thin shells and cloth.
    Written to ``newton:contactMargin`` via ``NewtonCollisionAPI``.
    Range: [0, inf).
    """

    contact_gap: float | None = None
    """Additional contact detection gap [m].

    AABBs are expanded by this value; contacts detected earlier to avoid tunneling.
    Written to ``newton:contactGap`` via ``NewtonCollisionAPI``.
    Set to ``-inf`` to use Newton's builder default. Range: [0, inf).
    """


@configclass
class NewtonMeshCollisionPropertiesCfg(NewtonCollisionPropertiesCfg, MeshCollisionBaseCfg):
    """Newton-specific mesh collision properties.

    Extends :class:`NewtonCollisionPropertiesCfg` with convex-hull vertex limit.

    See :meth:`~isaaclab.sim.schemas.modify_mesh_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = "NewtonMeshCollisionAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    max_hull_vertices: int | None = None
    """Maximum vertices in the convex hull approximation [dimensionless].

    Only relevant when ``physics:approximation = "convexHull"``.
    Written to ``newton:maxHullVertices`` via ``NewtonMeshCollisionAPI``.
    Set to ``-1`` to use as many vertices as needed for a perfect hull.
    """


@configclass
class NewtonMaterialPropertiesCfg(RigidBodyMaterialBaseCfg):
    """Newton-specific rigid body material properties.

    Extends :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg`
    with Newton-native friction attributes.

    See :meth:`~isaaclab.sim.spawners.materials.spawn_rigid_body_material` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = "NewtonMaterialAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    torsional_friction: float | None = None
    """Torsional friction coefficient (resistance to spinning at a contact point) [dimensionless].

    Written to ``newton:torsionalFriction`` via ``NewtonMaterialAPI``.
    Range: [0, inf).
    """

    rolling_friction: float | None = None
    """Rolling friction coefficient (resistance to rolling motion) [dimensionless].

    Written to ``newton:rollingFriction`` via ``NewtonMaterialAPI``.
    Range: [0, inf).
    """


@configclass
class NewtonArticulationRootPropertiesCfg(ArticulationRootBaseCfg):
    """Newton-specific articulation root properties.

    Extends :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg` with
    Newton-native self-collision control.

    See :meth:`~isaaclab.sim.schemas.modify_articulation_root_properties` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = "NewtonArticulationRootAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    self_collision_enabled: bool | None = None
    """Whether self-collisions between bodies in this articulation are enabled.

    Written to ``newton:selfCollisionEnabled`` via ``NewtonArticulationRootAPI``.
    Newton's resolver checks this native attribute first before falling back to
    ``physxArticulation:enabledSelfCollisions``.
    """
