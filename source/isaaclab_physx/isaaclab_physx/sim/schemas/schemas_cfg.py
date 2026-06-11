# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import ClassVar

from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootBaseCfg,
    CollisionBaseCfg,
    DeformableBodyPropertiesBaseCfg,
    JointDriveBaseCfg,
    MeshCollisionBaseCfg,
    RigidBodyBaseCfg,
)
from isaaclab.utils.configclass import configclass


@configclass
class OmniPhysicsDeformableBodyPropertiesCfg(DeformableBodyPropertiesBaseCfg):
    """OmniPhysics properties for a deformable body.

    These properties are set with the prefix ``omniphysics:<property_name>``.
    """

    _usd_namespace: ClassVar[str | None] = "omniphysics"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}

    deformable_body_enabled: bool | None = None
    """Enables deformable body."""

    kinematic_enabled: bool = False
    """Enables kinematic body. Defaults to False, which means that the body is not kinematic."""

    mass: float | None = None
    """The material mass [kg]. Defaults to None, in which case the material density is used to compute the mass."""


@configclass
class PhysXDeformableBodyPropertiesCfg:
    """PhysX-specific properties for a deformable body.

    These properties are set with the prefix ``physxDeformableBody:<property_name>``

    For more information on the available properties, please refer to the `documentation <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/physx_deformable_schema.html#physxbasedeformablebodyapi>`_.
    """

    _usd_namespace: ClassVar[str | None] = "physxDeformableBody"
    _usd_applied_schema: ClassVar[str | None] = "PhysxBaseDeformableBodyAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    solver_position_iteration_count: int = 16
    """Number of the solver positional iterations per step. Range is [1,255], default to 16."""

    linear_damping: float | None = None
    """Linear damping coefficient, in units of [1/s] and constrained to the range [0, inf)."""

    max_linear_velocity: float | None = None
    """Maximum allowable linear velocity for the deformable body, in units of distance/second and constrained to the
    range [0, inf). A negative value allows the simulation to choose suitable a per vertex value dynamically,
    currently only supported for surface deformables. This can help prevent surface-surface intersections."""

    settling_damping: float | None = None
    """Additional damping applied when a vertex's velocity falls below :attr:`settling_threshold`.
    Specified in units of [1/s] and constrained to the range [0, inf)."""

    settling_threshold: float | None = None
    """Velocity threshold below which :attr:`settling_damping` is applied in addition to standard damping.
    Specified in units of distance/second and constrained to the range [0, inf)."""

    sleep_threshold: float | None = None
    """Velocity threshold below which a vertex becomes a candidate for sleeping.
    Specified in units of distance/seconds and constrained to the range [0, inf)."""

    max_depenetration_velocity: float | None = None
    """Maximum velocity that the solver may apply to resolve intersections.
    Specified in units of distance/seconds and constrained to the range [0, inf)."""

    self_collision: bool | None = None
    """Enables self-collisions for the deformable body, preventing self-intersections."""

    self_collision_filter_distance: float | None = None
    r"""Distance below which self-collision is disabled [m].

    The default value of -inf indicates that the simulation selects a suitable value.
    Constrained to range [:attr:`rest_offset` \* 2, inf].
    """

    enable_speculative_c_c_d: bool | None = None
    """Enables dynamic adjustment of contact offset based on velocity (speculative continuous collision detection)."""

    disable_gravity: bool | None = None
    """Disables gravity for the deformable body."""

    # specific to surface deformables
    collision_pair_update_frequency: int | None = None
    """Determines how often surface-to-surface collision pairs are updated during each time step.
    Increasing this value results in more frequent updates to the contact pairs, which provides better contact points.

    For example, a value of 2 means collision pairs are updated twice per time step:
    once at the beginning and once in the middle of the time step (i.e., during the middle solver iteration).
    If set to 0, the solver adaptively determines when to update the surface-to-surface contact pairs,
    instead of using a fixed frequency.

    Valid range: [1, :attr:`solver_position_iteration_count`].
    """

    collision_iteration_multiplier: float | None = None
    """Determines how many collision subiterations are used in each solver iteration.
    By default, collision constraints are applied once per solver iteration.
    Increasing this value applies collision constraints more frequently within each solver iteration.

    For example, a value of 2 means collision constraints are applied twice per solver iteration
    (i.e., collision constraints are applied 2 x :attr:`solver_position_iteration_count` times per time step).
    Increasing this value does not update collision pairs more frequently;
    refer to :attr:`collision_pair_update_frequency` for that.

    Valid range: [1, :attr:`solver_position_iteration_count` / 2].
    """


@configclass
class PhysxDeformableCollisionPropertiesCfg:
    """PhysX-specific collision properties for a deformable body.

    These properties are set with the prefix ``physxCollision:<property_name>``.

    See the PhysX documentation for more information on the available properties.

    .. note::
        This class is distinct from
        :class:`~isaaclab_physx.sim.schemas.PhysxCollisionPropertiesCfg` (lowercase x),
        which is the rigid-body collision cfg layered on
        :class:`~isaaclab.sim.schemas.CollisionBaseCfg`. This class is used internally
        as a base of :class:`DeformableBodyPropertiesCfg`.
    """

    _usd_namespace: ClassVar[str | None] = "physxCollision"
    _usd_applied_schema: ClassVar[str | None] = "PhysxCollisionAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    contact_offset: float | None = None
    """Contact offset for the collision shape [m].

    The collision detector generates contact points as soon as two shapes get closer than the sum of their
    contact offsets. This quantity should be non-negative which means that contact generation can potentially start
    before the shapes actually penetrate.
    """

    rest_offset: float | None = None
    """Rest offset for the collision shape [m].

    The rest offset quantifies how close a shape gets to others at rest, At rest, the distance between two
    vertically stacked objects is the sum of their rest offsets. If a pair of shapes have a positive rest
    offset, the shapes will be separated at rest by an air gap.
    """


@configclass
class PhysxDeformableBodyPropertiesCfg(
    OmniPhysicsDeformableBodyPropertiesCfg,
    PhysXDeformableBodyPropertiesCfg,
    PhysxDeformableCollisionPropertiesCfg,
):
    """PhysX-specific properties to apply to a deformable body.

    A deformable body is a body that can deform under forces, both surface and volume deformables.
    The configuration allows users to specify the properties of the deformable body,
    such as the solver iteration counts, damping, and self-collision.

    An FEM-based deformable body is created by providing a collision mesh and simulation mesh. The collision mesh
    is used for collision detection and the simulation mesh is used for simulation.

    See :meth:`modify_deformable_body_properties` for more information.

    .. note::
        If the values are :obj:`None`, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """


@configclass
class DeformableBodyPropertiesCfg(PhysxDeformableBodyPropertiesCfg):
    """Deprecated: use :class:`PhysxDeformableBodyPropertiesCfg`.

    .. deprecated:: 4.6.x
        ``DeformableBodyPropertiesCfg`` has moved to
        :class:`PhysxDeformableBodyPropertiesCfg` for PhysX-specific deformable properties
        and is scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'DeformableBodyPropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.schemas.PhysxDeformableBodyPropertiesCfg' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class PhysxRigidBodyPropertiesCfg(RigidBodyBaseCfg):
    """PhysX-specific rigid body properties.

    Extends :class:`~isaaclab.sim.schemas.RigidBodyBaseCfg` with properties from the `PhysxRigidBodyAPI`_ schema.

    See :meth:`~isaaclab.sim.schemas.modify_rigid_body_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.

    .. _PhysxRigidBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_rigid_body_a_p_i.html
    """

    # PhysX-specific fields below all live under the ``PhysxRigidBodyAPI`` schema's
    # ``physxRigidBody:*`` namespace. The ``disable_gravity`` field on the base remains
    # routed via ``_usd_field_exceptions`` (inherited).
    _usd_applied_schema: ClassVar[str | None] = "PhysxRigidBodyAPI"
    _usd_namespace: ClassVar[str | None] = "physxRigidBody"

    linear_damping: float | None = None
    """Linear damping for the body."""

    angular_damping: float | None = None
    """Angular damping for the body."""

    max_linear_velocity: float | None = None
    """Maximum linear velocity for rigid bodies (in m/s)."""

    max_angular_velocity: float | None = None
    """Maximum angular velocity for rigid bodies (in deg/s)."""

    max_depenetration_velocity: float | None = None
    """Maximum depenetration velocity permitted to be introduced by the solver (in m/s)."""

    max_contact_impulse: float | None = None
    """The limit on the impulse that may be applied at a contact."""

    enable_gyroscopic_forces: bool | None = None
    """Enables computation of gyroscopic forces on the rigid body."""

    retain_accelerations: bool | None = None
    """Carries over forces/accelerations over sub-steps."""

    solver_position_iteration_count: int | None = None
    """Solver position iteration counts for the body."""

    solver_velocity_iteration_count: int | None = None
    """Solver velocity iteration counts for the body."""

    sleep_threshold: float | None = None
    """Mass-normalized kinetic energy threshold below which an actor may go to sleep."""

    stabilization_threshold: float | None = None
    """The mass-normalized kinetic energy threshold below which an actor may participate in stabilization."""


@configclass
class RigidBodyPropertiesCfg(PhysxRigidBodyPropertiesCfg):
    """Deprecated: use :class:`PhysxRigidBodyPropertiesCfg` or :class:`~isaaclab.sim.schemas.RigidBodyBaseCfg`.

    .. deprecated:: 4.6.22
        ``RigidBodyPropertiesCfg`` has been split into
        :class:`~isaaclab.sim.schemas.RigidBodyBaseCfg` (solver-common) and
        :class:`PhysxRigidBodyPropertiesCfg` (PhysX-specific) and relocated to
        :mod:`isaaclab_physx.sim.schemas`. This alias preserves backwards compatibility and is
        scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'RigidBodyPropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.schemas.PhysxRigidBodyPropertiesCfg' for PhysX properties, or"
            " 'isaaclab.sim.schemas.RigidBodyBaseCfg' for solver-common properties only.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class PhysxJointDrivePropertiesCfg(JointDriveBaseCfg):
    """PhysX-specific joint drive properties.

    Currently empty after the consumption-gated split moved :attr:`max_joint_velocity`
    to :class:`~isaaclab.sim.schemas.JointDriveBaseCfg`. This class is retained
    as the deprecation-alias target for the legacy :class:`JointDrivePropertiesCfg`
    name and as the home for any future PhysX-only joint-drive fields (e.g.
    PhysX-specific drive force-limit modes).

    Inherits all fields and USD metadata from
    :class:`~isaaclab.sim.schemas.JointDriveBaseCfg`.

    See :meth:`~isaaclab.sim.schemas.modify_joint_drive_properties` for more information.

    .. _PhysxJointAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_joint_a_p_i.html
    """

    # ``max_joint_velocity`` on the base remains routed via ``_usd_field_exceptions``
    # (inherited). Future PhysX-only joint-drive fields would be written under this
    # namespace.
    _usd_applied_schema: ClassVar[str | None] = "PhysxJointAPI"
    _usd_namespace: ClassVar[str | None] = "physxJoint"


@configclass
class JointDrivePropertiesCfg(PhysxJointDrivePropertiesCfg):
    """Deprecated: use :class:`PhysxJointDrivePropertiesCfg` or :class:`~isaaclab.sim.schemas.JointDriveBaseCfg`.

    .. deprecated:: 4.6.22
        ``JointDrivePropertiesCfg`` has been split into
        :class:`~isaaclab.sim.schemas.JointDriveBaseCfg` (solver-common) and
        :class:`PhysxJointDrivePropertiesCfg` (PhysX-specific) and relocated to
        :mod:`isaaclab_physx.sim.schemas`. This alias preserves backwards compatibility and is
        scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'JointDrivePropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.schemas.PhysxJointDrivePropertiesCfg' for PhysX properties, or"
            " 'isaaclab.sim.schemas.JointDriveBaseCfg' for solver-common properties only.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class PhysxCollisionPropertiesCfg(CollisionBaseCfg):
    """PhysX-specific rigid-body collision properties.

    Extends :class:`~isaaclab.sim.schemas.CollisionBaseCfg` with the PhysX-only torsional
    patch friction approximations (:attr:`torsional_patch_radius`,
    :attr:`min_torsional_patch_radius`). These fields have no Newton equivalent and are
    consumed only by the PhysX solver.

    See :meth:`~isaaclab.sim.schemas.modify_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.

    .. _PhysxCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_collision_a_p_i.html
    """

    # PhysX torsional-friction fields below live under the ``PhysxCollisionAPI`` schema's
    # ``physxCollision:*`` namespace. Base ``contact_offset`` / ``rest_offset`` remain
    # routed via ``_usd_field_exceptions`` (inherited).
    _usd_applied_schema: ClassVar[str | None] = "PhysxCollisionAPI"
    _usd_namespace: ClassVar[str | None] = "physxCollision"

    torsional_patch_radius: float | None = None
    """Radius of the contact patch for applying torsional friction [m].

    It is used to approximate rotational friction introduced by the compression of contacting surfaces.
    If the radius is zero, no torsional friction is applied.
    """

    min_torsional_patch_radius: float | None = None
    """Minimum radius of the contact patch for applying torsional friction [m]."""


@configclass
class PhysxArticulationRootPropertiesCfg(ArticulationRootBaseCfg):
    """PhysX-specific articulation-root properties.

    Extends :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg` with the
    `PhysxArticulationAPI`_ schema fields that are PhysX-only or dual-namespace
    (Rule 2 — the conceptual quantity also has a ``newton:*`` attribute, and a
    future ``NewtonArticulationRootPropertiesCfg`` would carry it on the Newton
    side). Use this class when authoring PhysX-specific articulation knobs;
    use :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg` when only the
    solver-common ``fix_root_link`` / ``articulation_enabled`` fields are needed.

    See :meth:`~isaaclab.sim.schemas.modify_articulation_root_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.

    .. _PhysxArticulationAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_articulation_a_p_i.html
    """

    # PhysX articulation-root fields below live under the ``PhysxArticulationAPI`` schema's
    # ``physxArticulation:*`` namespace. Base ``articulation_enabled`` remains routed via
    # ``_usd_field_exceptions`` (inherited).
    _usd_applied_schema: ClassVar[str | None] = "PhysxArticulationAPI"
    _usd_namespace: ClassVar[str | None] = "physxArticulation"

    enabled_self_collisions: bool | None = None
    """Whether self-collisions between bodies in the same articulation are enabled.

    The conceptual quantity exists in two USD namespaces simultaneously:

    * ``physxArticulation:enabledSelfCollisions`` (PhysX, ``PhysxArticulationAPI``)
    * ``newton:selfCollisionEnabled`` (Newton-native, on a future ``NewtonArticulationRootAPI``)

    Newton's resolver checks the native ``newton:*`` attribute first and falls back
    to the PhysX namespace. Both backends honor the field end-to-end.

    Because the conceptual quantity has a dedicated USD attribute in each backend's
    namespace, this field is placed on the **PhysX subclass** (one cfg per namespace).
    A future ``NewtonArticulationRootPropertiesCfg`` will carry the same field over the
    ``newton:*`` namespace.
    """

    solver_position_iteration_count: int | None = None
    """Solver position iteration counts for the body."""

    solver_velocity_iteration_count: int | None = None
    """Solver velocity iteration counts for the body."""

    sleep_threshold: float | None = None
    """Mass-normalized kinetic energy threshold below which an actor may go to sleep."""

    stabilization_threshold: float | None = None
    """The mass-normalized kinetic energy threshold below which an articulation may participate in stabilization."""


@configclass
class ArticulationRootPropertiesCfg(PhysxArticulationRootPropertiesCfg):
    """Deprecated: use :class:`PhysxArticulationRootPropertiesCfg` or the solver-common base class.

    Use :class:`PhysxArticulationRootPropertiesCfg` for PhysX-specific properties or
    :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg` for solver-common properties only.

    .. deprecated:: 4.6.24
        ``ArticulationRootPropertiesCfg`` has been split into
        :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg` (solver-common
        ``fix_root_link`` and the PhysX-namespaced but IL-Newton-consumed
        ``articulation_enabled``) and
        :class:`PhysxArticulationRootPropertiesCfg` (PhysX-specific
        self-collisions, TGS solver iter / sleep / stabilization thresholds)
        and relocated to :mod:`isaaclab_physx.sim.schemas`. This alias preserves
        backwards compatibility and is scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'ArticulationRootPropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.schemas.PhysxArticulationRootPropertiesCfg' for PhysX properties, or"
            " 'isaaclab.sim.schemas.ArticulationRootBaseCfg' for solver-common properties only.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class CollisionPropertiesCfg(PhysxCollisionPropertiesCfg):
    """Deprecated: use :class:`PhysxCollisionPropertiesCfg` or :class:`~isaaclab.sim.schemas.CollisionBaseCfg`.

    .. deprecated:: 4.6.23
        ``CollisionPropertiesCfg`` has been split into
        :class:`~isaaclab.sim.schemas.CollisionBaseCfg` (solver-common) and
        :class:`PhysxCollisionPropertiesCfg` (PhysX-specific) and relocated to
        :mod:`isaaclab_physx.sim.schemas`. This alias preserves backwards compatibility and is
        scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'CollisionPropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.schemas.PhysxCollisionPropertiesCfg' for PhysX properties, or"
            " 'isaaclab.sim.schemas.CollisionBaseCfg' for solver-common properties only.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class PhysxConvexHullPropertiesCfg(MeshCollisionBaseCfg):
    """PhysX convex-hull cooking properties for a mesh collider.

    Extends :class:`~isaaclab.sim.schemas.MeshCollisionBaseCfg` with the
    ``PhysxConvexHullCollisionAPI`` schema's tuning fields. The ``convexHull`` token is
    written to ``physics:approximation``; the cooking schema is applied only when at
    least one tuning field is set (consistent with the other consumption-gated writers).

    See :meth:`~isaaclab.sim.schemas.modify_mesh_collision_properties` for more information.

    Original PhysX Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_convex_hull_collision_a_p_i.html
    """

    _usd_applied_schema: ClassVar[str | None] = "PhysxConvexHullCollisionAPI"
    _usd_namespace: ClassVar[str | None] = "physxConvexHullCollision"

    mesh_approximation_name: str = "convexHull"
    """Name of mesh collision approximation method. Default: "convexHull"."""

    hull_vertex_limit: int | None = None
    """Convex hull vertex limit used for convex hull cooking.

    Defaults to 64.
    """
    min_thickness: float | None = None
    """Convex hull min thickness.

    Range: [0, inf). Units are distance. Default value is 0.001.
    """


@configclass
class PhysxConvexDecompositionPropertiesCfg(MeshCollisionBaseCfg):
    """PhysX convex-decomposition cooking properties for a mesh collider.

    See :meth:`~isaaclab.sim.schemas.modify_mesh_collision_properties` for more information.

    Original PhysX Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_convex_decomposition_collision_a_p_i.html
    """

    _usd_applied_schema: ClassVar[str | None] = "PhysxConvexDecompositionCollisionAPI"
    _usd_namespace: ClassVar[str | None] = "physxConvexDecompositionCollision"

    mesh_approximation_name: str = "convexDecomposition"
    """Name of mesh collision approximation method. Default: "convexDecomposition"."""

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
class PhysxTriangleMeshPropertiesCfg(MeshCollisionBaseCfg):
    """PhysX triangle-mesh cooking properties for a mesh collider.

    Triangle-mesh colliders are PhysX-only.

    See :meth:`~isaaclab.sim.schemas.modify_mesh_collision_properties` for more information.

    Original PhysX Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_triangle_mesh_collision_a_p_i.html
    """

    _usd_applied_schema: ClassVar[str | None] = "PhysxTriangleMeshCollisionAPI"
    _usd_namespace: ClassVar[str | None] = "physxTriangleMeshCollision"

    mesh_approximation_name: str = "none"
    """Name of mesh collision approximation method. Default: "none" (uses triangle mesh)."""

    weld_tolerance: float | None = None
    """Mesh weld tolerance, controls the distance at which vertices are welded.

    Default -inf will autocompute the welding tolerance based on the mesh size. Zero value will disable welding.
    Range: [0, inf) Units: distance
    """


@configclass
class PhysxTriangleMeshSimplificationPropertiesCfg(MeshCollisionBaseCfg):
    """PhysX triangle-mesh-simplification cooking properties for a mesh collider.

    See :meth:`~isaaclab.sim.schemas.modify_mesh_collision_properties` for more information.

    Original PhysX Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_triangle_mesh_simplification_collision_a_p_i.html
    """

    _usd_applied_schema: ClassVar[str | None] = "PhysxTriangleMeshSimplificationCollisionAPI"
    _usd_namespace: ClassVar[str | None] = "physxTriangleMeshSimplificationCollision"

    mesh_approximation_name: str = "meshSimplification"
    """Name of mesh collision approximation method. Default: "meshSimplification"."""

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
class PhysxSDFMeshPropertiesCfg(MeshCollisionBaseCfg):
    """PhysX SDF-mesh cooking properties for a mesh collider.

    SDF-mesh colliders are PhysX-only.

    See :meth:`~isaaclab.sim.schemas.modify_mesh_collision_properties` for more information.

    Original PhysX documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_s_d_f_mesh_collision_a_p_i.html

    More details and steps for optimizing SDF results can be found here:
    https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/docs/RigidBodyCollision.html#dynamic-triangle-meshes-with-sdfs
    """

    _usd_applied_schema: ClassVar[str | None] = "PhysxSDFMeshCollisionAPI"
    _usd_namespace: ClassVar[str | None] = "physxSDFMeshCollision"

    mesh_approximation_name: str = "sdf"
    """Name of mesh collision approximation method. Default: "sdf"."""

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


@configclass
class MeshCollisionPropertiesCfg(MeshCollisionBaseCfg):
    """Deprecated: use :class:`~isaaclab.sim.schemas.MeshCollisionBaseCfg`.

    .. deprecated:: 4.6.25
        ``MeshCollisionPropertiesCfg`` was the flat (non-leaf) base of the legacy
        mesh-collision cfg family. It has been renamed to
        :class:`~isaaclab.sim.schemas.MeshCollisionBaseCfg` to match the rest of the
        consumption-gated split. This alias preserves backwards compatibility and is
        scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'MeshCollisionPropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab.sim.schemas.MeshCollisionBaseCfg' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class ConvexHullPropertiesCfg(PhysxConvexHullPropertiesCfg):
    """Deprecated: use :class:`PhysxConvexHullPropertiesCfg`.

    .. deprecated:: 4.6.25
        Renamed and relocated. This alias preserves backwards compatibility and is
        scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'ConvexHullPropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.schemas.PhysxConvexHullPropertiesCfg' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class ConvexDecompositionPropertiesCfg(PhysxConvexDecompositionPropertiesCfg):
    """Deprecated: use :class:`PhysxConvexDecompositionPropertiesCfg`.

    .. deprecated:: 4.6.25
        Renamed and relocated. This alias preserves backwards compatibility and is
        scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'ConvexDecompositionPropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.schemas.PhysxConvexDecompositionPropertiesCfg' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class TriangleMeshPropertiesCfg(PhysxTriangleMeshPropertiesCfg):
    """Deprecated: use :class:`PhysxTriangleMeshPropertiesCfg`.

    .. deprecated:: 4.6.25
        Renamed and relocated. This alias preserves backwards compatibility and is
        scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'TriangleMeshPropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.schemas.PhysxTriangleMeshPropertiesCfg' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class TriangleMeshSimplificationPropertiesCfg(PhysxTriangleMeshSimplificationPropertiesCfg):
    """Deprecated: use :class:`PhysxTriangleMeshSimplificationPropertiesCfg`.

    .. deprecated:: 4.6.25
        Renamed and relocated. This alias preserves backwards compatibility and is
        scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'TriangleMeshSimplificationPropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.schemas.PhysxTriangleMeshSimplificationPropertiesCfg' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class SDFMeshPropertiesCfg(PhysxSDFMeshPropertiesCfg):
    """Deprecated: use :class:`PhysxSDFMeshPropertiesCfg`.

    .. deprecated:: 4.6.25
        Renamed and relocated. This alias preserves backwards compatibility and is
        scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'SDFMeshPropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.schemas.PhysxSDFMeshPropertiesCfg' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class PhysxFixedTendonPropertiesCfg:
    """PhysX fixed-tendon properties for an articulation.

    Tendons are a PhysX-only feature -- Newton has no tendon system -- so this class
    is a pure data carrier that is consumed by the PhysX-specific writer
    :func:`~isaaclab.sim.schemas.modify_fixed_tendon_properties`. The writer authors
    the multi-instance ``PhysxTendonAxisRootAPI`` schema; this cfg class declares no
    metadata-driven writer plumbing of its own.

    See :func:`~isaaclab.sim.schemas.modify_fixed_tendon_properties` for more information.

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
class FixedTendonPropertiesCfg(PhysxFixedTendonPropertiesCfg):
    """Deprecated: use :class:`PhysxFixedTendonPropertiesCfg`.

    .. deprecated:: 4.6.x
        ``FixedTendonPropertiesCfg`` was relocated to
        :mod:`isaaclab_physx.sim.schemas` and renamed to
        :class:`PhysxFixedTendonPropertiesCfg`. The legacy name remains as a
        deprecation alias and is scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'FixedTendonPropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.schemas.PhysxFixedTendonPropertiesCfg' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()


@configclass
class PhysxSpatialTendonPropertiesCfg:
    """PhysX spatial-tendon properties for an articulation.

    Tendons are a PhysX-only feature -- Newton has no tendon system -- so this class
    is a pure data carrier that is consumed by the PhysX-specific writer
    :func:`~isaaclab.sim.schemas.modify_spatial_tendon_properties`. The writer authors
    the multi-instance ``PhysxTendonAttachmentRootAPI`` / ``PhysxTendonAttachmentLeafAPI``
    schemas; this cfg class declares no metadata-driven writer plumbing of its own.

    See :func:`~isaaclab.sim.schemas.modify_spatial_tendon_properties` for more information.

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
class SpatialTendonPropertiesCfg(PhysxSpatialTendonPropertiesCfg):
    """Deprecated: use :class:`PhysxSpatialTendonPropertiesCfg`.

    .. deprecated:: 4.6.x
        ``SpatialTendonPropertiesCfg`` was relocated to
        :mod:`isaaclab_physx.sim.schemas` and renamed to
        :class:`PhysxSpatialTendonPropertiesCfg`. The legacy name remains as a
        deprecation alias and is scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'SpatialTendonPropertiesCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.schemas.PhysxSpatialTendonPropertiesCfg' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()
