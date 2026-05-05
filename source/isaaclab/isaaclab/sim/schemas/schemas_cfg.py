# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass

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
                " removal in 5.0."
            ) from e
        return getattr(_physx_cfg, name)
    raise AttributeError(f"module 'isaaclab.sim.schemas.schemas_cfg' has no attribute {name!r}")


@configclass
class ArticulationRootBaseCfg:
    """Solver-common properties to apply to the root of an articulation.

    Carries :attr:`fix_root_link` (writer-side; materializes a
    :class:`UsdPhysics.FixedJoint` between the world frame and the root link) and
    :attr:`articulation_enabled` whose USD attribute today is PhysX-namespaced
    (``physxArticulation:articulationEnabled``) but is consumed by the IsaacLab
    Newton wrapper as a spawn-time guard. For PhysX-only articulation-root
    properties (self-collisions, TGS solver iterations, sleep / stabilization
    thresholds), use
    :class:`~isaaclab_physx.sim.schemas.PhysxArticulationRootPropertiesCfg`.

    See :meth:`modify_articulation_root_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    # Documented exception: this base class carries PhysX namespace metadata because
    # ``articulation_enabled`` writes to ``physxArticulation:articulationEnabled``.
    # The IL Newton wrapper consumes the same PhysX attribute as a spawn-time guard;
    # PhysX honors it at sim time. See docs/superpowers/schema-cfg-placement-path2.md.
    # -- Class metadata (not dataclass fields) --
    # USD applied schema written when at least one solver-specific field is set.
    _usd_applied_schema = "PhysxArticulationAPI"
    # Prim attribute namespace for solver-specific fields.
    _usd_namespace = "physxArticulation"
    # Mapping from cfg field names to USD attribute names (already in camelCase).
    # ``articulation_enabled`` -> ``articulationEnabled`` is the auto-conversion result;
    # listed here explicitly for clarity.
    _usd_attr_name_map = {"articulation_enabled": "articulationEnabled"}

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

    # Documented exception: this base class carries PhysX namespace metadata because
    # ``disable_gravity`` writes to ``physxRigidBody:disableGravity`` -- Newton consumes
    # the same PhysX attribute via its bridge resolver. See
    # ``docs/superpowers/schema-cfg-placement-path2.md`` for the placement rule.
    # -- Class metadata (not dataclass fields) --
    # USD applied schema written when at least one solver-specific field is set.
    _usd_applied_schema = "PhysxRigidBodyAPI"
    # Prim attribute namespace for solver-specific fields.
    _usd_namespace = "physxRigidBody"

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

    # Documented exception: this base class carries PhysX namespace metadata because
    # ``contact_offset`` and ``rest_offset`` write to ``physxCollision:*`` -- Newton
    # consumes them via the bridge resolver (gap = contactOffset - restOffset,
    # margin = restOffset). See docs/superpowers/schema-cfg-placement-path2.md.
    # -- Class metadata (not dataclass fields) --
    # USD applied schema written when at least one solver-specific field is set.
    _usd_applied_schema = "PhysxCollisionAPI"
    # Prim attribute namespace for solver-specific fields.
    _usd_namespace = "physxCollision"

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
    simulation backends, plus :attr:`max_velocity` whose USD attribute today is
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

    # Documented exception: this base class carries PhysX namespace metadata because
    # ``max_velocity`` writes to ``physxJoint:maxJointVelocity`` -- the only USD path to
    # ``Model.joint_velocity_limit`` (no ``newton:*`` equivalent today). Newton consumes
    # the PhysX attribute via its bridge resolver. See
    # ``docs/superpowers/schema-cfg-placement-path2.md`` for the placement rule.
    # -- Class metadata (not dataclass fields) --
    # USD applied schema written when at least one solver-specific field is set.
    _usd_applied_schema = "PhysxJointAPI"
    # Prim attribute namespace for solver-specific fields.
    _usd_namespace = "physxJoint"
    # Mapping from cfg field names to USD attribute names (already in camelCase).
    _usd_attr_name_map = {"max_velocity": "maxJointVelocity"}

    drive_type: Literal["force", "acceleration"] | None = None
    """Joint drive type to apply.

    If the drive type is "force", then the joint is driven by a force. If the drive type is "acceleration",
    then the joint is driven by an acceleration (usually used for kinematic joints).
    """

    max_effort: float | None = None
    """Maximum effort that can be applied to the joint (in kg-m^2/s^2)."""

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

    max_velocity: float | None = None
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


@configclass
class FixedTendonPropertiesCfg:
    """Properties to define fixed tendons of an articulation.

    See :meth:`modify_fixed_tendon_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    tendon_enabled: bool | None = None
    """Whether to enable or disable the tendon."""

    stiffness: float | None = None
    """Spring stiffness term acting on the tendon's length."""

    damping: float | None = None
    """The damping term acting on both the tendon length and the tendon-length limits."""

    limit_stiffness: float | None = None
    """Limit stiffness term acting on the tendon's length limits."""

    offset: float | None = None
    """Length offset term for the tendon.

    It defines an amount to be added to the accumulated length computed for the tendon. This allows the application
    to actuate the tendon by shortening or lengthening it.
    """

    rest_length: float | None = None
    """Spring rest length of the tendon."""


@configclass
class SpatialTendonPropertiesCfg:
    """Properties to define spatial tendons of an articulation.

    See :meth:`modify_spatial_tendon_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    tendon_enabled: bool | None = None
    """Whether to enable or disable the tendon."""

    stiffness: float | None = None
    """Spring stiffness term acting on the tendon's length."""

    damping: float | None = None
    """The damping term acting on both the tendon length and the tendon-length limits."""

    limit_stiffness: float | None = None
    """Limit stiffness term acting on the tendon's length limits."""

    offset: float | None = None
    """Length offset term for the tendon.

    It defines an amount to be added to the accumulated length computed for the tendon. This allows the application
    to actuate the tendon by shortening or lengthening it.
    """


@configclass
class MeshCollisionPropertiesCfg:
    """Properties to apply to a mesh in regards to collision.
    See :meth:`set_mesh_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    usd_api: str | None = None
    """USD API name for mesh collision (e.g. 'MeshCollisionAPI')."""

    physx_api: str | None = None
    """PhysX schema name for mesh collision (e.g. 'PhysxConvexDecompositionCollisionAPI')."""

    mesh_approximation_name: str = "none"
    """Name of mesh collision approximation method. Default: "none".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """


@configclass
class BoundingCubePropertiesCfg(MeshCollisionPropertiesCfg):
    usd_api: str = "MeshCollisionAPI"
    """Original USD Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html
    """

    mesh_approximation_name: str = "boundingCube"
    """Name of mesh collision approximation method. Default: "boundingCube".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """


@configclass
class BoundingSpherePropertiesCfg(MeshCollisionPropertiesCfg):
    usd_api: str = "MeshCollisionAPI"
    """Original USD Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html
    """

    mesh_approximation_name: str = "boundingSphere"
    """Name of mesh collision approximation method. Default: "boundingSphere".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """


@configclass
class ConvexDecompositionPropertiesCfg(MeshCollisionPropertiesCfg):
    usd_api: str = "MeshCollisionAPI"
    """Original USD Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html
    """

    physx_api: str = "PhysxConvexDecompositionCollisionAPI"
    """Original PhysX Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_convex_decomposition_collision_a_p_i.html
    """

    mesh_approximation_name: str = "convexDecomposition"
    """Name of mesh collision approximation method. Default: "convexDecomposition".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """

    hull_vertex_limit: int | None = None
    """Convex hull vertex limit used for convex hull cooking.

    Defaults to 64.
    """
    max_convex_hulls: int | None = None
    """Maximum of convex hulls created during convex decomposition.
    Default value is 32.
    """
    min_thickness: float | None = None
    """Convex hull min thickness.

    Range: [0, inf). Units are distance. Default value is 0.001.
    """
    voxel_resolution: int | None = None
    """Voxel resolution used for convex decomposition.

    Defaults to 500,000 voxels.
    """
    error_percentage: float | None = None
    """Convex decomposition error percentage parameter.

    Defaults to 10 percent. Units are percent.
    """
    shrink_wrap: bool | None = None
    """Attempts to adjust the convex hull points so that they are projected onto the surface of the original graphics
    mesh.

    Defaults to False.
    """


@configclass
class ConvexHullPropertiesCfg(MeshCollisionPropertiesCfg):
    usd_api: str = "MeshCollisionAPI"
    """Original USD Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html
    """

    physx_api: str = "PhysxConvexHullCollisionAPI"
    """Original PhysX Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_convex_hull_collision_a_p_i.html
    """

    mesh_approximation_name: str = "convexHull"
    """Name of mesh collision approximation method. Default: "convexHull".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """

    hull_vertex_limit: int | None = None
    """Convex hull vertex limit used for convex hull cooking.

    Defaults to 64.
    """
    min_thickness: float | None = None
    """Convex hull min thickness.

    Range: [0, inf). Units are distance. Default value is 0.001.
    """


@configclass
class TriangleMeshPropertiesCfg(MeshCollisionPropertiesCfg):
    physx_api: str = "PhysxTriangleMeshCollisionAPI"
    """Triangle mesh is only supported by PhysX API.

    Original PhysX Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_triangle_mesh_collision_a_p_i.html
    """

    mesh_approximation_name: str = "none"
    """Name of mesh collision approximation method. Default: "none" (uses triangle mesh).
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """

    weld_tolerance: float | None = None
    """Mesh weld tolerance, controls the distance at which vertices are welded.

    Default -inf will autocompute the welding tolerance based on the mesh size. Zero value will disable welding.
    Range: [0, inf) Units: distance
    """


@configclass
class TriangleMeshSimplificationPropertiesCfg(MeshCollisionPropertiesCfg):
    usd_api: str = "MeshCollisionAPI"
    """Original USD Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html
    """

    physx_api: str = "PhysxTriangleMeshSimplificationCollisionAPI"
    """Original PhysX Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_triangle_mesh_simplification_collision_a_p_i.html
    """

    mesh_approximation_name: str = "meshSimplification"
    """Name of mesh collision approximation method. Default: "meshSimplification".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """

    simplification_metric: float | None = None
    """Mesh simplification accuracy.

    Defaults to 0.55.
    """
    weld_tolerance: float | None = None
    """Mesh weld tolerance, controls the distance at which vertices are welded.

    Default -inf will autocompute the welding tolerance based on the mesh size. Zero value will disable welding.
    Range: [0, inf) Units: distance
    """


@configclass
class SDFMeshPropertiesCfg(MeshCollisionPropertiesCfg):
    physx_api: str = "PhysxSDFMeshCollisionAPI"
    """SDF mesh is only supported by PhysX API.

    Original PhysX documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_s_d_f_mesh_collision_a_p_i.html

    More details and steps for optimizing SDF results can be found here:
    https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/docs/RigidBodyCollision.html#dynamic-triangle-meshes-with-sdfs
    """

    mesh_approximation_name: str = "sdf"
    """Name of mesh collision approximation method. Default: "sdf".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """

    sdf_margin: float | None = None
    """Margin to increase the size of the SDF relative to the bounding box diagonal length of the mesh.


    A sdf margin value of 0.01 means the sdf boundary will be enlarged in any direction by 1% of the mesh's bounding
    box diagonal length. Representing the margin relative to the bounding box diagonal length ensures that it is scale
    independent. Margins allow for precise distance queries in a region slightly outside of the mesh's bounding box.

    Default value is 0.01.
    Range: [0, inf) Units: dimensionless
    """
    sdf_narrow_band_thickness: float | None = None
    """Size of the narrow band around the mesh surface where high resolution SDF samples are available.

    Outside of the narrow band, only low resolution samples are stored. Representing the narrow band thickness as a
    fraction of the mesh's bounding box diagonal length ensures that it is scale independent. A value of 0.01 is
    usually large enough. The smaller the narrow band thickness, the smaller the memory consumption of the sparse SDF.

    Default value is 0.01.
    Range: [0, 1] Units: dimensionless
    """
    sdf_resolution: int | None = None
    """The spacing of the uniformly sampled SDF is equal to the largest AABB extent of the mesh,
    divided by the resolution.

    Choose the lowest possible resolution that provides acceptable performance; very high resolution results in large
    memory consumption, and slower cooking and simulation performance.

    Default value is 256.
    Range: (1, inf)
    """
    sdf_subgrid_resolution: int | None = None
    """A positive subgrid resolution enables sparsity on signed-distance-fields (SDF) while a value of 0 leads to the
    usage of a dense SDF.

    A value in the range of 4 to 8 is a reasonable compromise between block size and the overhead introduced by block
    addressing. The smaller a block, the more memory is spent on the address table. The bigger a block, the less
    precisely the sparse SDF can adapt to the mesh's surface. In most cases sparsity reduces the memory consumption of
    a SDF significantly.

    Default value is 6.
    Range: [0, inf)
    """
