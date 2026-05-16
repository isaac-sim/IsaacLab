# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import ClassVar, Literal

from isaaclab.utils.configclass import configclass

# Names that moved out of this submodule into ``isaaclab_physx.sim.schemas.schemas_cfg``.
# Resolved lazily so callers using ``from isaaclab.sim.schemas.schemas_cfg import
# RigidBodyPropertiesCfg`` continue to work without importing ``isaaclab_physx`` at module
# load time.
_PHYSX_FORWARDS = frozenset(
    {
        "RigidBodyPropertiesCfg",
        "JointDrivePropertiesCfg",
        "PhysxRigidBodyPropertiesCfg",
        "PhysxJointDrivePropertiesCfg",
        "CollisionPropertiesCfg",
        "PhysxCollisionPropertiesCfg",
        "PhysXCollisionPropertiesCfg",
        "PhysxDeformableCollisionPropertiesCfg",
        "ArticulationRootPropertiesCfg",
        "PhysxArticulationRootPropertiesCfg",
        "MeshCollisionPropertiesCfg",
        "ConvexHullPropertiesCfg",
        "ConvexDecompositionPropertiesCfg",
        "TriangleMeshPropertiesCfg",
        "TriangleMeshSimplificationPropertiesCfg",
        "SDFMeshPropertiesCfg",
        "PhysxConvexHullPropertiesCfg",
        "PhysxConvexDecompositionPropertiesCfg",
        "PhysxTriangleMeshPropertiesCfg",
        "PhysxTriangleMeshSimplificationPropertiesCfg",
        "PhysxSDFMeshPropertiesCfg",
        "FixedTendonPropertiesCfg",
        "SpatialTendonPropertiesCfg",
        "PhysxFixedTendonPropertiesCfg",
        "PhysxSpatialTendonPropertiesCfg",
    }
)

_NEWTON_FORWARDS = frozenset(
    {
        "MujocoRigidBodyPropertiesCfg",
        "MujocoJointDrivePropertiesCfg",
        "NewtonRigidBodyPropertiesCfg",
        "NewtonJointDrivePropertiesCfg",
        "NewtonCollisionPropertiesCfg",
        "NewtonMeshCollisionPropertiesCfg",
        "NewtonMaterialPropertiesCfg",
        "NewtonArticulationRootPropertiesCfg",
    }
)


def __getattr__(name):
    if name in _PHYSX_FORWARDS:
        try:
            from isaaclab_physx.sim.schemas import schemas_cfg as _physx_cfg
        except ImportError as e:
            raise ImportError(
                f"'isaaclab.sim.schemas.schemas_cfg.{name}' has moved to"
                " 'isaaclab_physx.sim.schemas.schemas_cfg'. Install the isaaclab_physx"
                " extension or update your import. This forwarding shim is scheduled for"
                " removal in 4.0."
            ) from e
        return getattr(_physx_cfg, name)
    if name in _NEWTON_FORWARDS:
        try:
            from isaaclab_newton.sim.schemas import schemas_cfg as _newton_cfg
        except ImportError as e:
            raise ImportError(
                f"'isaaclab.sim.schemas.schemas_cfg.{name}' has moved to"
                " 'isaaclab_newton.sim.schemas.schemas_cfg'. Install the isaaclab_newton"
                " extension or update your import. This forwarding shim is scheduled for"
                " removal in 4.0."
            ) from e
        return getattr(_newton_cfg, name)
    raise AttributeError(f"module 'isaaclab.sim.schemas.schemas_cfg' has no attribute {name!r}")


def _deprecate_field_alias(cfg, alias: str, canonical: str) -> None:
    """Forward a deprecated cfg field to its canonical replacement.

    If ``alias`` is set on the cfg instance, emit a ``DeprecationWarning`` and copy the
    value to ``canonical`` (when ``canonical`` is unset). The alias is then nulled so
    downstream metadata-driven writers see only the canonical name.
    """
    value = getattr(cfg, alias, None)
    if value is None:
        return
    warnings.warn(
        f"'{alias}' is deprecated; use '{canonical}' instead. The alias is scheduled for removal in 4.0.",
        DeprecationWarning,
        stacklevel=3,
    )
    if getattr(cfg, canonical, None) is None:
        setattr(cfg, canonical, value)
    setattr(cfg, alias, None)


@configclass
class ArticulationRootBaseCfg:
    """Solver-common properties to apply to the root of an articulation.

    Carries :attr:`fix_root_link` (writer-side; materializes a
    :class:`UsdPhysics.FixedJoint` between the world frame and the root link) and
    :attr:`articulation_enabled` whose only USD path today is the PhysX-namespaced
    ``physxArticulation:articulationEnabled`` attribute. The base class itself
    declares no USD namespace; the writer consults :attr:`_usd_field_exceptions`
    to route ``articulation_enabled`` to its non-base namespace and apply
    ``PhysxArticulationAPI`` only when the user authored that one field.
    For PhysX-only articulation-root properties (self-collisions, TGS solver
    iterations, sleep / stabilization thresholds), use
    :class:`~isaaclab_physx.sim.schemas.PhysxArticulationRootPropertiesCfg`.

    See :meth:`modify_articulation_root_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    # -- Class metadata (not dataclass fields) --
    # No base-native namespace today: every field is either solver-common (typed
    # UsdPhysics API) or routed through ``_usd_field_exceptions``.
    _usd_namespace: ClassVar[str | None] = None
    _usd_applied_schema: ClassVar[str | None] = None
    # Per-field exceptions: applied_schema -> (namespace, [cfg_field, ...]). The USD
    # attribute name is the auto snake -> camelCase of the cfg field name (project
    # convention). When any listed field is non-None at write time, the writer applies
    # the schema and writes the attribute under the exception namespace.
    _usd_field_exceptions: ClassVar[dict] = {
        "PhysxArticulationAPI": ("physxArticulation", ["articulation_enabled"]),
    }

    articulation_enabled: bool | None = None
    """Whether to enable or disable the articulation.

    PhysX honors this per-articulation at sim time via
    ``physxArticulation:articulationEnabled``: setting False makes PhysX skip
    the articulation in its solver passes.

    On Newton, the field is read by the IsaacLab Newton wrapper at spawn time
    (``isaaclab_newton/assets/rigid_object/rigid_object.py:1035``) as a guard
    against accidentally spawning a ``RigidObject`` over a prim that still has
    ``ArticulationRootAPI`` applied; setting False suppresses the guard error.
    The Newton solver itself does not consult the flag at sim time.

    Placed on the solver-common class because the user-facing intent is
    universal and both PhysX (sim-time) and the IL Newton wrapper (spawn-time)
    honor it.
    """

    fix_root_link: bool | None = None
    """Whether to fix the root link of the articulation.

    * If set to None, the root link is not modified.
    * If the articulation already has a fixed root link, this flag will enable or disable the fixed joint.
    * If the articulation does not have a fixed root link, this flag will create a fixed joint between the world
      frame and the root link. The joint is created with the name "FixedJoint" under the articulation prim.

    .. note::
        This is a non-USD schema property. It is handled by the :meth:`modify_articulation_root_properties` function.

    """


@configclass
class RigidBodyBaseCfg:
    """Solver-common properties to apply to a rigid body.

    Contains properties from the `UsdPhysics.RigidBodyAPI`_ that are common across all
    simulation backends, plus :attr:`disable_gravity` whose USD attribute today is
    PhysX-namespaced but whose semantics (per-body gravity exclusion) are universal:
    PhysX honors it per-body; Newton's importer consumes it at the scene level
    (partial honor, documented on the field). For PhysX-only rigid-body properties,
    use :class:`PhysxRigidBodyPropertiesCfg`.

    See :meth:`modify_rigid_body_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.

    .. _UsdPhysics.RigidBodyAPI: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    # -- Class metadata (not dataclass fields) --
    # ``rigid_body_enabled`` and ``kinematic_enabled`` write to ``physics:*`` (UsdPhysics
    # standard attributes). The helper's per-declaring-class routing keeps these under
    # the base namespace even when the cfg is a PhysX subclass instance. The
    # ``UsdPhysics.RigidBodyAPI`` schema is applied upstream by ``define_rigid_body_properties``
    # so ``_usd_applied_schema`` here stays None. ``disable_gravity`` is routed via
    # ``_usd_field_exceptions`` to ``physxRigidBody:disableGravity``.
    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {
        "PhysxRigidBodyAPI": ("physxRigidBody", ["disable_gravity"]),
    }

    rigid_body_enabled: bool | None = None
    """Whether to enable or disable the rigid body."""

    kinematic_enabled: bool | None = None
    """Determines whether the body is kinematic or not.

    A kinematic body is a body that is moved through animated poses or through user defined poses. The simulation
    still derives velocities for the kinematic body based on the external motion.

    For more information on kinematic bodies, please refer to the `documentation <https://openusd.org/release/wp_rigid_body_physics.html#kinematic-bodies>`_.
    """

    disable_gravity: bool | None = None
    """Disable gravity for the body.

    PhysX honors this per-body via ``physxRigidBody:disableGravity``: setting True
    excludes the body from world gravity integration.

    Newton currently consumes the same USD attribute at the **scene level** --
    Newton's importer reads ``physxRigidBody:disableGravity`` on the scene prim
    and uses it to drive the scene-wide ``builder.gravity`` flag (``import_usd.py:1212``).
    Per-body intent is therefore partially honored on Newton: whichever rigid body
    has the attribute authored ends up controlling scene-wide gravity, and other
    bodies cannot be selectively excluded.

    The field is placed on the base because the user-facing intent (per-body
    gravity exclusion for markers, sensors, kinematic targets) is universal physics
    and PhysX honors it fully. Closing the Newton gap is a kernel-level fix
    (introduce ``Model.body_disable_gravity`` boolean array consumed by the
    integrator) that does not require a cfg-API change.
    """


@configclass
class CollisionBaseCfg:
    """Solver-common properties to apply to colliders.

    Contains :attr:`collision_enabled` from the `UsdPhysics.CollisionAPI`_ and the
    :attr:`contact_offset` / :attr:`rest_offset` knobs whose USD attributes today are
    PhysX-namespaced (``physxCollision:contactOffset``, ``physxCollision:restOffset``)
    but whose semantics (collision-pair generation distance, rest separation gap) are
    universal physics: PhysX consumes them natively, Newton's importer consumes them
    via the PhysX bridge resolver and populates ``Model.shape_collision_radius`` /
    ``Model.shape_collision_thickness`` from the ``gap`` and ``margin`` keys (see
    ``import_usd.py:2104, 2111``). For PhysX-only collision properties (e.g. torsional
    patch friction), use :class:`~isaaclab_physx.sim.schemas.PhysxCollisionPropertiesCfg`.

    See :meth:`modify_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.

    .. _UsdPhysics.CollisionAPI: https://openusd.org/dev/api/class_usd_physics_collision_a_p_i.html
    """

    # -- Class metadata (not dataclass fields) --
    # ``collision_enabled`` writes to ``physics:collisionEnabled`` (UsdPhysics standard).
    # The helper's per-declaring-class routing keeps it under ``physics:*`` even when
    # the cfg is a PhysX subclass instance. ``contact_offset`` / ``rest_offset`` are
    # routed via ``_usd_field_exceptions`` to ``physxCollision:*``.
    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {
        "PhysxCollisionAPI": ("physxCollision", ["contact_offset", "rest_offset"]),
    }

    collision_enabled: bool | None = None
    """Whether to enable or disable collisions.

    Writes ``physics:collisionEnabled`` via :class:`UsdPhysics.CollisionAPI`.
    """

    contact_offset: float | None = None
    """Contact offset for the collision shape [m].

    The collision detector generates contact points as soon as two shapes get closer than the sum of their
    contact offsets. This quantity should be non-negative which means that contact generation can potentially start
    before the shapes actually penetrate.

    Writes ``physxCollision:contactOffset``. Newton's USD importer consumes the same
    attribute via its PhysX-bridge resolver.
    """

    rest_offset: float | None = None
    """Rest offset for the collision shape [m].

    The rest offset quantifies how close a shape gets to others at rest, At rest, the distance between two
    vertically stacked objects is the sum of their rest offsets. If a pair of shapes have a positive rest
    offset, the shapes will be separated at rest by an air gap.

    Writes ``physxCollision:restOffset``. Newton's USD importer consumes the same
    attribute via its PhysX-bridge resolver.
    """


@configclass
class MassPropertiesCfg:
    """Properties to define explicit mass properties of a rigid body.

    See :meth:`modify_mass_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    # -- Class metadata (not dataclass fields) --
    # ``mass`` / ``density`` write to ``physics:*`` (UsdPhysics standard attributes).
    # The ``UsdPhysics.MassAPI`` schema is applied upstream by ``define_mass_properties``.
    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}

    mass: float | None = None
    """The mass of the rigid body (in kg).

    Note:
        If non-zero, the mass is ignored and the density is used to compute the mass.
    """

    density: float | None = None
    """The density of the rigid body (in kg/m^3).

    The density indirectly defines the mass of the rigid body. It is generally computed using the collision
    approximation of the body.
    """


@configclass
class JointDriveBaseCfg:
    """Solver-common properties to define the drive mechanism of a joint.

    Contains properties from the `UsdPhysics.DriveAPI`_ that are common across all
    simulation backends, plus :attr:`max_joint_velocity` whose USD attribute today is
    PhysX-namespaced but whose semantics (per-DOF velocity limit) are universal:
    Newton's importer consumes ``physxJoint:maxJointVelocity`` and populates
    ``Model.joint_velocity_limit``; PhysX consumes it natively. For PhysX-only
    drive properties, use :class:`PhysxJointDrivePropertiesCfg`.

    See :meth:`modify_joint_drive_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.

    .. _UsdPhysics.DriveAPI: https://openusd.org/dev/api/class_usd_physics_drive_a_p_i.html
    """

    # -- Class metadata (not dataclass fields) --
    # No base-native namespace today: drive-type / max-effort / stiffness / damping are
    # written via the typed ``UsdPhysics.DriveAPI``; ``max_joint_velocity`` is routed
    # through ``_usd_field_exceptions`` to ``physxJoint:maxJointVelocity`` (the only
    # USD path to ``Model.joint_velocity_limit`` today).
    _usd_namespace: ClassVar[str | None] = None
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {
        "PhysxJointAPI": ("physxJoint", ["max_joint_velocity"]),
    }

    def __post_init__(self):
        # Deprecation aliases: project convention is that python ``snake_case`` cfg field
        # names map identity-style to USD ``camelCase`` attrs. Legacy short names that
        # diverged are forwarded here.
        _deprecate_field_alias(self, "max_velocity", "max_joint_velocity")
        _deprecate_field_alias(self, "max_effort", "max_force")

    drive_type: Literal["force", "acceleration"] | None = None
    """Joint drive type to apply.

    If the drive type is "force", then the joint is driven by a force. If the drive type is "acceleration",
    then the joint is driven by an acceleration (usually used for kinematic joints).
    """

    max_force: float | None = None
    """Maximum force/torque that can be applied to the joint [N for linear joints, N-m for angular joints].

    Writes ``drive:<linear|angular>:physics:maxForce`` via :class:`UsdPhysics.DriveAPI`.
    """

    max_effort: float | None = None
    """Deprecated alias for :attr:`max_force`.

    .. deprecated:: 4.6.25
        Use :attr:`max_force` instead. The cfg field is renamed so its
        snake_case name maps identity-style to the USD camelCase attribute
        (``maxForce`` on ``UsdPhysics.DriveAPI``). The alias is forwarded to
        :attr:`max_force` in :meth:`__post_init__` and will be removed in 4.0.
    """

    stiffness: float | None = None
    """Stiffness of the joint drive.

    The unit depends on the joint model:

    * For linear joints, the unit is kg-m/s^2 (N/m).
    * For angular joints, the unit is kg-m^2/s^2/rad (N-m/rad).
    """

    damping: float | None = None
    """Damping of the joint drive.

    The unit depends on the joint model:

    * For linear joints, the unit is kg-m/s (N-s/m).
    * For angular joints, the unit is kg-m^2/s/rad (N-m-s/rad).
    """

    ensure_drives_exist: bool = False
    """If True, ensure every joint has a non-zero drive so that physics backends
    (e.g. Newton) create proper actuators for it.

    When a USD asset defines ``PhysicsDriveAPI`` with ``stiffness=0`` and
    ``damping=0``, some backends treat the joint as passive (no PD control).
    Enabling this flag writes a minimal stiffness (``1e-3``) to any drive whose
    stiffness *and* damping are both zero, guaranteeing that the backend
    recognises the drive as active.  The actual gains are expected to be
    overridden later by the actuator model.
    """

    max_joint_velocity: float | None = None
    """Maximum velocity of the joint [m/s for linear joints, rad/s for angular joints].

    Notes:
        Today this writes ``physxJoint:maxJointVelocity`` (a PhysX add-on schema attribute).
        Newton's USD importer consumes the same attribute via its PhysX-bridge resolver and
        populates ``Model.joint_velocity_limit``; the PhysX engine consumes it natively. The
        Kamino solver honors the limit at the simulation step. The XPBD, Featherstone, and
        Semi-implicit Newton solvers import the value but do not consume it in their kernels;
        the MuJoCo (MJC) solver explicitly drops it. When Newton ships ``newton:maxJointVelocity``
        as a registered applied API, the writer namespace will switch transparently and this
        docstring caveat will be removed.
    """

    max_velocity: float | None = None
    """Deprecated alias for :attr:`max_joint_velocity`.

    .. deprecated:: 4.6.25
        Use :attr:`max_joint_velocity` instead. The cfg field is renamed so its
        snake_case name maps identity-style to the USD camelCase attribute
        (``physxJoint:maxJointVelocity``). The alias is forwarded to
        :attr:`max_joint_velocity` in :meth:`__post_init__` and will be removed in 4.0.
    """


@configclass
class MeshCollisionBaseCfg:
    """Solver-common properties to apply to a mesh in regards to collision.

    Carries only the standard ``UsdPhysics:MeshCollisionAPI`` token
    (:attr:`mesh_approximation_name` -> ``physics:approximation``). For PhysX-cooking
    tunables (convex hull / decomposition / triangle mesh / SDF), use the
    ``Physx*PropertiesCfg`` subclasses in :mod:`isaaclab_physx.sim.schemas`.

    See :meth:`modify_mesh_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to
        set only a subset of the properties and leave the rest as-is.
    """

    # -- Class metadata (not dataclass fields) --
    # The standard ``UsdPhysics.MeshCollisionAPI`` is always applied by the writer when a
    # mesh-collision cfg is supplied; ``_usd_applied_schema`` here records the standard
    # API name so subclasses that author no PhysX namespace can rely on the writer's
    # standard-vs-PhysX gating logic. PhysX-cooking subclasses override this.
    _usd_applied_schema: ClassVar[str | None] = "MeshCollisionAPI"
    # Base class authors no PhysX-namespaced fields, so no namespace is defined.
    _usd_namespace: ClassVar[str | None] = None
    _usd_attr_name_map: ClassVar[dict] = {}
    _usd_field_exceptions: ClassVar[dict] = {}

    mesh_approximation_name: str = "none"
    """Name of mesh collision approximation method. Default: "none".

    Writes ``physics:approximation`` via :class:`UsdPhysics.MeshCollisionAPI`.
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """

    def __getattr__(self, name: str):
        """Deprecated read-only access to the legacy ``usd_api`` / ``physx_api`` instance attrs.

        Falls back here only when the attribute is not found on the dataclass instance.
        Returns the legacy-mapped string value derived from the class-level
        ``_usd_applied_schema`` metadata and emits a ``DeprecationWarning``.
        """
        if name == "usd_api":
            warnings.warn(
                "'usd_api' attribute is deprecated and will be removed in 4.0. Use class-level"
                " metadata via getattr(cfg, '_usd_applied_schema').",
                DeprecationWarning,
                stacklevel=2,
            )
            schema = self.__dict__.get("_usd_applied_schema", None)
            # Every PhysX cooking subclass legacy-mapped to ``"MeshCollisionAPI"``; the base
            # class also wrote that token. Return ``None`` only when no schema is declared.
            return "MeshCollisionAPI" if schema is not None else None
        if name == "physx_api":
            warnings.warn(
                "'physx_api' attribute is deprecated and will be removed in 4.0. Use class-level"
                " metadata via getattr(cfg, '_usd_applied_schema').",
                DeprecationWarning,
                stacklevel=2,
            )
            schema = self.__dict__.get("_usd_applied_schema", None)
            if schema and schema.startswith("Physx"):
                return schema
            return None
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


@configclass
class BoundingCubePropertiesCfg(MeshCollisionBaseCfg):
    """Bounding-cube mesh collision approximation. USD-only; authors no PhysX schema.

    Writes the ``boundingCube`` token to ``physics:approximation`` via
    :class:`UsdPhysics.MeshCollisionAPI`.

    Original USD Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html
    """

    mesh_approximation_name: str = "boundingCube"
    """Name of mesh collision approximation method. Default: "boundingCube"."""


@configclass
class BoundingSpherePropertiesCfg(MeshCollisionBaseCfg):
    """Bounding-sphere mesh collision approximation. USD-only; authors no PhysX schema.

    Writes the ``boundingSphere`` token to ``physics:approximation`` via
    :class:`UsdPhysics.MeshCollisionAPI`.

    Original USD Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html
    """

    mesh_approximation_name: str = "boundingSphere"
    """Name of mesh collision approximation method. Default: "boundingSphere"."""
