# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

from isaaclab.utils import configclass


@configclass
class ArticulationRootPropertiesCfg:
    """Properties to apply to the root of an articulation.

    See :meth:`modify_articulation_root_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    articulation_enabled: bool | None = None
    """Whether to enable or disable articulation."""

    enabled_self_collisions: bool | None = None
    """Whether to enable or disable self-collisions."""

    solver_position_iteration_count: int | None = None
    """Solver position iteration counts for the body."""

    solver_velocity_iteration_count: int | None = None
    """Solver position iteration counts for the body."""

    sleep_threshold: float | None = None
    """Mass-normalized kinetic energy threshold below which an actor may go to sleep."""

    stabilization_threshold: float | None = None
    """The mass-normalized kinetic energy threshold below which an articulation may participate in stabilization."""

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
class RigidBodyPropertiesCfg:
    """Properties to apply to a rigid body.

    See :meth:`modify_rigid_body_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    rigid_body_enabled: bool | None = None
    """Whether to enable or disable the rigid body."""

    kinematic_enabled: bool | None = None
    """Determines whether the body is kinematic or not.

    A kinematic body is a body that is moved through animated poses or through user defined poses. The simulation
    still derives velocities for the kinematic body based on the external motion.

    For more information on kinematic bodies, please refer to the `documentation <https://openusd.org/release/wp_rigid_body_physics.html#kinematic-bodies>`_.
    """

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
    """Solver position iteration counts for the body."""

    sleep_threshold: float | None = None
    """Mass-normalized kinetic energy threshold below which an actor may go to sleep."""

    stabilization_threshold: float | None = None
    """The mass-normalized kinetic energy threshold below which an actor may participate in stabilization."""


@configclass
class CollisionPropertiesCfg:
    """Properties to apply to colliders in a rigid body.

    See :meth:`modify_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    collision_enabled: bool | None = None
    """Whether to enable or disable collisions."""

    contact_offset: float | None = None
    """Contact offset for the collision shape (in m).

    The collision detector generates contact points as soon as two shapes get closer than the sum of their
    contact offsets. This quantity should be non-negative which means that contact generation can potentially start
    before the shapes actually penetrate.
    """

    rest_offset: float | None = None
    """Rest offset for the collision shape (in m).

    The rest offset quantifies how close a shape gets to others at rest, At rest, the distance between two
    vertically stacked objects is the sum of their rest offsets. If a pair of shapes have a positive rest
    offset, the shapes will be separated at rest by an air gap.
    """

    torsional_patch_radius: float | None = None
    """Radius of the contact patch for applying torsional friction (in m).

    It is used to approximate rotational friction introduced by the compression of contacting surfaces.
    If the radius is zero, no torsional friction is applied.
    """

    min_torsional_patch_radius: float | None = None
    """Minimum radius of the contact patch for applying torsional friction (in m)."""


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
class JointDrivePropertiesCfg:
    """Properties to define the drive mechanism of a joint.

    See :meth:`modify_joint_drive_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    drive_type: Literal["force", "acceleration"] | None = None
    """Joint drive type to apply.

    If the drive type is "force", then the joint is driven by a force. If the drive type is "acceleration",
    then the joint is driven by an acceleration (usually used for kinematic joints).
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
class DeformableBodyPropertiesCfg:
    """Properties to apply to a deformable body.

    A deformable body is a body that can deform under forces. The configuration allows users to specify
    the properties of the deformable body, such as the solver iteration counts, damping, and self-collision.

    An FEM-based deformable body is created by providing a collision mesh and simulation mesh. The collision mesh
    is used for collision detection and the simulation mesh is used for simulation. The collision mesh is usually
    a simplified version of the simulation mesh.

    Based on the above, the PhysX team provides APIs to either set the simulation and collision mesh directly
    (by specifying the points) or to simplify the collision mesh based on the simulation mesh. The simplification
    process involves remeshing the collision mesh and simplifying it based on the target triangle count.

    Since specifying the collision mesh points directly is not a common use case, we only expose the parameters
    to simplify the collision mesh based on the simulation mesh. If you want to provide the collision mesh points,
    please open an issue on the repository and we can add support for it.

    See :meth:`modify_deformable_body_properties` for more information.

    .. note::
        If the values are :obj:`None`, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    deformable_enabled: bool | None = None
    """Enables deformable body."""

    kinematic_enabled: bool = False
    """Enables kinematic body. Defaults to False, which means that the body is not kinematic.

    Similar to rigid bodies, this allows setting user-driven motion for the deformable body. For more information,
    please refer to the `documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html#kinematic-soft-bodies>`__.
    """

    self_collision: bool | None = None
    """Whether to enable or disable self-collisions for the deformable body based on the rest position distances."""

    self_collision_filter_distance: float | None = None
    """Penetration value that needs to get exceeded before contacts for self collision are generated.

    This parameter must be greater than of equal to twice the :attr:`rest_offset` value.

    This value has an effect only if :attr:`self_collision` is enabled.
    """

    settling_threshold: float | None = None
    """Threshold vertex velocity (in m/s) under which sleep damping is applied in addition to velocity damping."""

    sleep_damping: float | None = None
    """Coefficient for the additional damping term if fertex velocity drops below setting threshold."""

    sleep_threshold: float | None = None
    """The velocity threshold (in m/s) under which the vertex becomes a candidate for sleeping in the next step."""

    solver_position_iteration_count: int | None = None
    """Number of the solver positional iterations per step. Range is [1,255]"""

    vertex_velocity_damping: float | None = None
    """Coefficient for artificial damping on the vertex velocity.

    This parameter can be used to approximate the effect of air drag on the deformable body.
    """

    simulation_hexahedral_resolution: int = 10
    """The target resolution for the hexahedral mesh used for simulation. Defaults to 10.

    Note:
        This value is ignored if the user provides the simulation mesh points directly. However, we assume that
        most users will not provide the simulation mesh points directly. If you want to provide the simulation mesh
        directly, please set this value to :obj:`None`.
    """

    collision_simplification: bool = True
    """Whether or not to simplify the collision mesh before creating a soft body out of it. Defaults to True.

    Note:
        This flag is ignored if the user provides the simulation mesh points directly. However, we assume that
        most users will not provide the simulation mesh points directly. Hence, this flag is enabled by default.

        If you want to provide the simulation mesh points directly, please set this flag to False.
    """

    collision_simplification_remeshing: bool = True
    """Whether or not the collision mesh should be remeshed before simplification. Defaults to True.

    This parameter is ignored if :attr:`collision_simplification` is False.
    """

    collision_simplification_remeshing_resolution: int = 0
    """The resolution used for remeshing. Defaults to 0, which means that a heuristic is used to determine the
    resolution.

    This parameter is ignored if :attr:`collision_simplification_remeshing` is False.
    """

    collision_simplification_target_triangle_count: int = 0
    """The target triangle count used for the simplification. Defaults to 0, which means that a heuristic based on
    the :attr:`simulation_hexahedral_resolution` is used to determine the target count.

    This parameter is ignored if :attr:`collision_simplification` is False.
    """

    collision_simplification_force_conforming: bool = True
    """Whether or not the simplification should force the output mesh to conform to the input mesh. Defaults to True.

    The flag indicates that the tretrahedralizer used to generate the collision mesh should produce tetrahedra
    that conform to the triangle mesh. If False, the simplifier uses the output from the tretrahedralizer used.

    This parameter is ignored if :attr:`collision_simplification` is False.
    """

    contact_offset: float | None = None
    """Contact offset for the collision shape (in m).

    The collision detector generates contact points as soon as two shapes get closer than the sum of their
    contact offsets. This quantity should be non-negative which means that contact generation can potentially start
    before the shapes actually penetrate.
    """

    rest_offset: float | None = None
    """Rest offset for the collision shape (in m).

    The rest offset quantifies how close a shape gets to others at rest, At rest, the distance between two
    vertically stacked objects is the sum of their rest offsets. If a pair of shapes have a positive rest
    offset, the shapes will be separated at rest by an air gap.
    """

    max_depenetration_velocity: float | None = None
    """Maximum depenetration velocity permitted to be introduced by the solver (in m/s)."""
