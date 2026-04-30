# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import warnings

from isaaclab.sim.schemas.schemas_cfg import JointDriveBaseCfg, RigidBodyBaseCfg
from isaaclab.utils import configclass


@configclass
class OmniPhysicsPropertiesCfg:
    """OmniPhysics properties for a deformable body.

    These properties are set with the prefix ``omniphysics:<property_name>``. For example, to set the mass of the
    deformable body, you would set the property ``omniphysics:mass``.

    See the OmniPhysics documentation for more information on the available properties.
    """

    deformable_body_enabled: bool | None = None
    """Enables deformable body."""

    kinematic_enabled: bool = False
    """Enables kinematic body. Defaults to False, which means that the body is not kinematic."""

    mass: float | None = None
    """The material mass in [kg]. Defaults to None, in which case the material density is used to compute the mass."""


@configclass
class PhysXDeformableBodyPropertiesCfg:
    """PhysX-specific properties for a deformable body.

    These properties are set with the prefix ``physxDeformableBody:<property_name>``

    For more information on the available properties, please refer to the `documentation <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/physx_deformable_schema.html#physxbasedeformablebodyapi>`_.
    """

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
class PhysXCollisionPropertiesCfg:
    """PhysX-specific collision properties for a deformable body.

    These properties are set with the prefix ``physxCollision:<property_name>``.

    See the PhysX documentation for more information on the available properties.
    """

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
class DeformableBodyPropertiesCfg(
    OmniPhysicsPropertiesCfg, PhysXDeformableBodyPropertiesCfg, PhysXCollisionPropertiesCfg
):
    """Properties to apply to a deformable body.

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

    _property_prefix: dict[str, list[str]] = {
        "omniphysics": [field.name for field in dataclasses.fields(OmniPhysicsPropertiesCfg)],
        "physxDeformableBody": [field.name for field in dataclasses.fields(PhysXDeformableBodyPropertiesCfg)],
        "physxCollision": [field.name for field in dataclasses.fields(PhysXCollisionPropertiesCfg)],
    }
    """Mapping between the property prefixes and the properties that fall under each prefix."""


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

    # -- Class metadata (not dataclass fields) --
    # USD schema to apply on the prim before writing solver-specific attributes.
    _usd_applied_schema = "PhysxRigidBodyAPI"
    # Prim attribute namespace for solver-specific fields.
    _usd_namespace = "physxRigidBody"

    disable_gravity: bool | None = None
    """Disable gravity for the actor."""

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

    Extends :class:`~isaaclab.sim.schemas.JointDriveBaseCfg` with properties from the `PhysxJointAPI`_ schema.

    See :meth:`~isaaclab.sim.schemas.modify_joint_drive_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.

    .. _PhysxJointAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_joint_a_p_i.html
    """

    # -- Class metadata (not dataclass fields) --
    # USD schema to apply on the prim before writing solver-specific attributes.
    _usd_applied_schema = "PhysxJointAPI"
    # Prim attribute namespace for solver-specific fields.
    _usd_namespace = "physxJoint"
    # Mapping from cfg field names to USD attribute names (already in camelCase).
    _usd_attr_name_map = {"max_velocity": "maxJointVelocity"}

    max_velocity: float | None = None
    """Maximum velocity of the joint.

    The unit depends on the joint model:

    * For linear joints, the unit is m/s.
    * For angular joints, the unit is rad/s.
    """


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
