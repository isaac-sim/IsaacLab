# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABC, abstractmethod

import warp as wp

from isaaclab.utils.leapp import (
    POSE6_ELEMENT_NAMES,
    POSE7_ELEMENT_NAMES,
    QUAT_XYZW_ELEMENT_NAMES,
    XYZ_ELEMENT_NAMES,
    InputKindEnum,
    body_pose6_resolver,
    body_pose_resolver,
    body_quat_resolver,
    body_xyz_resolver,
    joint_names_resolver,
    leapp_tensor_semantics,
)
from isaaclab.utils.warp import ProxyArray


class BaseArticulationData(ABC):
    """Data container for an articulation.

    This class contains the data for an articulation in the simulation. The data includes the state of
    the root rigid body, the state of all the bodies in the articulation, and the joint state. The data is
    stored in the simulation world frame unless otherwise specified.

    An articulation is comprised of multiple rigid bodies or links. For a rigid body, there are two frames
    of reference that are used:

    - Actor frame: The frame of reference of the rigid body prim. This typically corresponds to the Xform prim
      with the rigid body schema.
    - Center of mass frame: The frame of reference of the center of mass of the rigid body.

    Depending on the settings, the two frames may not coincide with each other. In the robotics sense, the actor frame
    can be interpreted as the link frame.
    """

    def __init__(self, root_view, device: str):
        """Initializes the articulation data.

        Args:
            root_view: The root articulation view.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device

    @abstractmethod
    def update(self, dt: float) -> None:
        raise NotImplementedError

    ##
    # Names.
    ##

    body_names: list[str] | None = None
    """Body names in the order parsed by the simulation view."""

    joint_names: list[str] | None = None
    """Joint names in the order parsed by the simulation view."""

    fixed_tendon_names: list[str] | None = None
    """Fixed tendon names in the order parsed by the simulation view."""

    spatial_tendon_names: list[str] | None = None
    """Spatial tendon names in the order parsed by the simulation view."""

    ##
    # Defaults - Initial state.
    ##

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def default_root_pose(self) -> ProxyArray:
        """Default root pose ``[pos, quat]`` in the local environment frame.

        The position and quaternion are of the articulation root's actor frame. Shape is (num_instances),
        dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def default_root_vel(self) -> ProxyArray:
        """Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame.

        The linear and angular velocities are of the articulation root's center of mass frame.
        Shape is (num_instances), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def default_root_state(self) -> ProxyArray:
        """Deprecated, same as :attr:`default_root_pose` and :attr:`default_root_vel`."""
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def default_joint_pos(self) -> ProxyArray:
        """Default joint positions of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def default_joint_vel(self) -> ProxyArray:
        """Default joint velocities of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        raise NotImplementedError

    ##
    # Joint commands -- Set into simulation.
    ##

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.COMMAND_JOINT_POSITION)
    def joint_pos_target(self) -> ProxyArray:
        """Joint position targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.COMMAND_JOINT_VELOCITY)
    def joint_vel_target(self) -> ProxyArray:
        """Joint velocity targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.COMMAND_JOINT_TORQUES)
    def joint_effort_target(self) -> ProxyArray:
        """Joint effort targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        raise NotImplementedError

    ##
    # Joint commands -- Explicit actuators.
    ##

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind="state/joint/computed_torque")
    def computed_torque(self) -> ProxyArray:
        """Joint torques computed from the actuator model (before clipping).

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        This quantity is the raw torque output from the actuator mode, before any clipping is applied.
        It is exposed for users who want to inspect the computations inside the actuator model.
        For instance, to penalize the learning agent for a difference between the computed and applied torques.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind="state/joint/applied_torque")
    def applied_torque(self) -> ProxyArray:
        """Joint torques applied from the actuator model (after clipping).

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
        actuator model.
        """
        raise NotImplementedError

    ##
    # Joint properties.
    ##

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def joint_stiffness(self) -> ProxyArray:
        """Joint stiffness provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def joint_damping(self) -> ProxyArray:
        """Joint damping provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def joint_armature(self) -> ProxyArray:
        """Joint armature provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def joint_friction_coeff(self) -> ProxyArray:
        """Joint static friction coefficient provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def joint_pos_limits(self) -> ProxyArray:
        """Joint position limits provided to the simulation.

        Shape is (num_instances, num_joints, 2), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def joint_vel_limits(self) -> ProxyArray:
        """Joint maximum velocity provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def joint_effort_limits(self) -> ProxyArray:
        """Joint maximum effort provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        raise NotImplementedError

    ##
    # Joint properties - Custom.
    ##

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def soft_joint_pos_limits(self) -> ProxyArray:
        r"""Soft joint positions limits for all joints.

        Shape is (num_instances, num_joints), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.The soft joint position limits are computed as
        a sub-region of the :attr:`joint_pos_limits` based on the
        :attr:`~isaaclab.assets.ArticulationCfg.soft_joint_pos_limit_factor` parameter.

        Consider the joint position limits :math:`[lower, upper]` and the soft joint position limits
        :math:`[soft_lower, soft_upper]`. The soft joint position limits are computed as:

        .. math::

            soft\_lower = (lower + upper) / 2 - factor * (upper - lower) / 2
            soft\_upper = (lower + upper) / 2 + factor * (upper - lower) / 2

        The soft joint position limits help specify a safety region around the joint limits. It isn't used by the
        simulation, but is useful for learning agents to prevent the joint positions from violating the limits.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def soft_joint_vel_limits(self) -> ProxyArray:
        """Soft joint velocity limits for all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
        has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def gear_ratio(self) -> ProxyArray:
        """Gear ratio for relating motor torques to applied Joint torques.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        raise NotImplementedError

    ##
    # Fixed tendon properties.
    ##

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def fixed_tendon_stiffness(self) -> ProxyArray:
        """Fixed tendon stiffness provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def fixed_tendon_damping(self) -> ProxyArray:
        """Fixed tendon damping provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def fixed_tendon_limit_stiffness(self) -> ProxyArray:
        """Fixed tendon limit stiffness provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def fixed_tendon_rest_length(self) -> ProxyArray:
        """Fixed tendon rest length provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def fixed_tendon_offset(self) -> ProxyArray:
        """Fixed tendon offset provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def fixed_tendon_pos_limits(self) -> ProxyArray:
        """Fixed tendon position limits provided to the simulation.

        Shape is (num_instances, num_fixed_tendons, 2), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_fixed_tendons, 2).
        """
        raise NotImplementedError

    ##
    # Spatial tendon properties.
    ##

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def spatial_tendon_stiffness(self) -> ProxyArray:
        """Spatial tendon stiffness provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def spatial_tendon_damping(self) -> ProxyArray:
        """Spatial tendon damping provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def spatial_tendon_limit_stiffness(self) -> ProxyArray:
        """Spatial tendon limit stiffness provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def spatial_tendon_offset(self) -> ProxyArray:
        """Spatial tendon offset provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    ##
    # Root state properties.
    ##

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names=POSE7_ELEMENT_NAMES)
    def root_link_pose_w(self) -> ProxyArray:
        """Root link pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).

        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_VEL, element_names=POSE6_ELEMENT_NAMES)
    def root_link_vel_w(self) -> ProxyArray:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names=POSE7_ELEMENT_NAMES)
    def root_com_pose_w(self) -> ProxyArray:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).

        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_VEL, element_names=POSE6_ELEMENT_NAMES)
    def root_com_vel_w(self) -> ProxyArray:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind="state/body/state")
    def root_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`root_link_pose_w` and :attr:`root_com_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind="state/body/link_state")
    def root_link_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`root_link_pose_w` and :attr:`root_link_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind="state/body/com_state")
    def root_com_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`root_com_pose_w` and :attr:`root_com_vel_w`."""
        raise NotImplementedError

    ##
    # Body state properties.
    ##

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def body_mass(self) -> ProxyArray:
        """Body mass ``wp.float32`` in the world frame.

        Shape is (num_instances, num_bodies), dtype = wp.float32. In torch this resolves to (num_instances, num_bodies).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(const=True)
    def body_inertia(self) -> ProxyArray:
        """Flattened body inertia in the world frame.

        Shape is (num_instances, num_bodies, 9), dtype = wp.float32. In torch this resolves to
        (num_instances, num_bodies, 9).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names_resolver=body_pose_resolver)
    def body_link_pose_w(self) -> ProxyArray:
        """Body link pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_VEL, element_names_resolver=body_pose6_resolver)
    def body_link_vel_w(self) -> ProxyArray:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names_resolver=body_pose_resolver)
    def body_com_pose_w(self) -> ProxyArray:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_VEL, element_names_resolver=body_pose6_resolver)
    def body_com_vel_w(self) -> ProxyArray:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind="state/body/state")
    def body_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`body_link_pose_w` and :attr:`body_com_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind="state/body/link_state")
    def body_link_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`body_link_pose_w` and :attr:`body_link_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind="state/body/com_state")
    def body_com_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`body_com_pose_w` and :attr:`body_com_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ACC, element_names_resolver=body_pose6_resolver)
    def body_com_acc_w(self) -> ProxyArray:
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]``.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names_resolver=body_pose_resolver)
    def body_com_pose_b(self) -> ProxyArray:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    ##
    # Joint state properties.
    ##

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.JOINT_POSITION, element_names_resolver=joint_names_resolver)
    def joint_pos(self) -> ProxyArray:
        """Joint positions of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.JOINT_VELOCITY, element_names_resolver=joint_names_resolver)
    def joint_vel(self) -> ProxyArray:
        """Joint velocities of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind="state/joint/acceleration", element_names_resolver=joint_names_resolver)
    def joint_acc(self) -> ProxyArray:
        """Joint acceleration of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        raise NotImplementedError

    ##
    # Derived Properties.
    ##

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def projected_gravity_b(self) -> ProxyArray:
        """Projection of the gravity direction on base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind="state/body/heading")
    def heading_w(self) -> ProxyArray:
        """Yaw heading of the base frame (in radians).

        Shape is (num_instances), dtype = wp.float32. In torch this resolves to (num_instances,).

        .. note::
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def root_link_lin_vel_b(self) -> ProxyArray:
        """Root link linear velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to
        its actor frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def root_link_ang_vel_b(self) -> ProxyArray:
        """Root link angular velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to
        its actor frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def root_com_lin_vel_b(self) -> ProxyArray:
        """Root center of mass linear velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to
        its actor frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def root_com_ang_vel_b(self) -> ProxyArray:
        """Root center of mass angular velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to
        its actor frame.
        """
        raise NotImplementedError

    ##
    # Sliced properties.
    ##

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names=XYZ_ELEMENT_NAMES)
    def root_link_pos_w(self) -> ProxyArray:
        """Root link position in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names=QUAT_XYZW_ELEMENT_NAMES)
    def root_link_quat_w(self) -> ProxyArray:
        """Root link orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances), dtype = wp.quatf. In torch this resolves to (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def root_link_lin_vel_w(self) -> ProxyArray:
        """Root linear velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def root_link_ang_vel_w(self) -> ProxyArray:
        """Root link angular velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names=XYZ_ELEMENT_NAMES)
    def root_com_pos_w(self) -> ProxyArray:
        """Root center of mass position in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the position of the center of mass frame of the root rigid body relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names=QUAT_XYZW_ELEMENT_NAMES)
    def root_com_quat_w(self) -> ProxyArray:
        """Root center of mass orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances), dtype = wp.quatf. In torch this resolves to (num_instances, 4).

        This quantity is the orientation of the principal axes of inertia of the root rigid body relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def root_com_lin_vel_w(self) -> ProxyArray:
        """Root center of mass linear velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def root_com_ang_vel_w(self) -> ProxyArray:
        """Root center of mass angular velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names_resolver=body_xyz_resolver)
    def body_link_pos_w(self) -> ProxyArray:
        """Positions of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names_resolver=body_quat_resolver)
    def body_link_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_VELOCITY, element_names_resolver=body_xyz_resolver)
    def body_link_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_VELOCITY, element_names_resolver=body_xyz_resolver)
    def body_link_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names_resolver=body_xyz_resolver)
    def body_com_pos_w(self) -> ProxyArray:
        """Positions of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names_resolver=body_quat_resolver)
    def body_com_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the principal axes of inertia of the articulation bodies.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_VELOCITY, element_names_resolver=body_xyz_resolver)
    def body_com_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_VELOCITY, element_names_resolver=body_xyz_resolver)
    def body_com_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_ACCELERATION, element_names_resolver=body_xyz_resolver)
    def body_com_lin_acc_w(self) -> ProxyArray:
        """Linear acceleration of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_ACCELERATION, element_names_resolver=body_xyz_resolver)
    def body_com_ang_acc_w(self) -> ProxyArray:
        """Angular acceleration of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names_resolver=body_xyz_resolver)
    def body_com_pos_b(self) -> ProxyArray:
        """Center of mass position of all of the bodies in their respective link frames.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body's link frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names_resolver=body_quat_resolver)
    def body_com_quat_b(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all of the bodies in their respective link
        frames.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the principal axes of inertia relative to its body's link frame.
        """
        raise NotImplementedError

    def _create_buffers(self) -> None:
        # -- Defaults (Lazy allocation of default values)
        self._default_mass = None
        self._default_inertia = None
        self._default_joint_stiffness = None
        self._default_joint_damping = None
        self._default_joint_armature = None
        self._default_joint_friction_coeff = None
        self._default_joint_viscous_friction_coeff = None
        self._default_joint_pos_limits = None
        self._default_fixed_tendon_stiffness = None
        self._default_fixed_tendon_damping = None
        self._default_fixed_tendon_limit_stiffness = None
        self._default_fixed_tendon_rest_length = None
        self._default_fixed_tendon_offset = None
        self._default_fixed_tendon_pos_limits = None
        self._default_spatial_tendon_stiffness = None
        self._default_spatial_tendon_damping = None
        self._default_spatial_tendon_limit_stiffness = None
        self._default_spatial_tendon_offset = None

    """
    Shorthands for commonly used properties.
    """

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names=POSE7_ELEMENT_NAMES)
    def root_pose_w(self) -> ProxyArray:
        """Shorthand for :attr:`root_link_pose_w`."""
        return self.root_link_pose_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names=XYZ_ELEMENT_NAMES)
    def root_pos_w(self) -> ProxyArray:
        """Shorthand for :attr:`root_link_pos_w`."""
        return self.root_link_pos_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names=QUAT_XYZW_ELEMENT_NAMES)
    def root_quat_w(self) -> ProxyArray:
        """Shorthand for :attr:`root_link_quat_w`."""
        return self.root_link_quat_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_VEL, element_names=POSE6_ELEMENT_NAMES)
    def root_vel_w(self) -> ProxyArray:
        """Shorthand for :attr:`root_com_vel_w`."""
        return self.root_com_vel_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def root_lin_vel_w(self) -> ProxyArray:
        """Shorthand for :attr:`root_com_lin_vel_w`."""
        return self.root_com_lin_vel_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def root_ang_vel_w(self) -> ProxyArray:
        """Shorthand for :attr:`root_com_ang_vel_w`."""
        return self.root_com_ang_vel_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def root_lin_vel_b(self) -> ProxyArray:
        """Shorthand for :attr:`root_com_lin_vel_b`."""
        return self.root_com_lin_vel_b

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def root_ang_vel_b(self) -> ProxyArray:
        """Shorthand for :attr:`root_com_ang_vel_b`."""
        return self.root_com_ang_vel_b

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names_resolver=body_pose_resolver)
    def body_pose_w(self) -> ProxyArray:
        """Shorthand for :attr:`body_link_pose_w`."""
        return self.body_link_pose_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names_resolver=body_xyz_resolver)
    def body_pos_w(self) -> ProxyArray:
        """Shorthand for :attr:`body_link_pos_w`."""
        return self.body_link_pos_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names_resolver=body_quat_resolver)
    def body_quat_w(self) -> ProxyArray:
        """Shorthand for :attr:`body_link_quat_w`."""
        return self.body_link_quat_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_VEL, element_names_resolver=body_pose6_resolver)
    def body_vel_w(self) -> ProxyArray:
        """Shorthand for :attr:`body_com_vel_w`."""
        return self.body_com_vel_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_VELOCITY, element_names_resolver=body_xyz_resolver)
    def body_lin_vel_w(self) -> ProxyArray:
        """Shorthand for :attr:`body_com_lin_vel_w`."""
        return self.body_com_lin_vel_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_VELOCITY, element_names_resolver=body_xyz_resolver)
    def body_ang_vel_w(self) -> ProxyArray:
        """Shorthand for :attr:`body_com_ang_vel_w`."""
        return self.body_com_ang_vel_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ACC, element_names_resolver=body_pose6_resolver)
    def body_acc_w(self) -> ProxyArray:
        """Shorthand for :attr:`body_com_acc_w`."""
        return self.body_com_acc_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_ACCELERATION, element_names_resolver=body_xyz_resolver)
    def body_lin_acc_w(self) -> ProxyArray:
        """Shorthand for :attr:`body_com_lin_acc_w`."""
        return self.body_com_lin_acc_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_ACCELERATION, element_names_resolver=body_xyz_resolver)
    def body_ang_acc_w(self) -> ProxyArray:
        """Shorthand for :attr:`body_com_ang_acc_w`."""
        return self.body_com_ang_acc_w

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names_resolver=body_xyz_resolver)
    def com_pos_b(self) -> ProxyArray:
        """Shorthand for :attr:`body_com_pos_b`."""
        return self.body_com_pos_b

    @property
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names_resolver=body_quat_resolver)
    def com_quat_b(self) -> ProxyArray:
        """Shorthand for :attr:`body_com_quat_b`."""
        return self.body_com_quat_b

    @property
    @leapp_tensor_semantics(const=True)
    def joint_limits(self) -> ProxyArray:
        """Shorthand for :attr:`joint_pos_limits`."""
        return self.joint_pos_limits

    @property
    @leapp_tensor_semantics(const=True)
    def default_joint_limits(self) -> ProxyArray:
        """Shorthand for :attr:`default_joint_pos_limits`."""
        return self.default_joint_pos_limits

    @property
    @leapp_tensor_semantics(const=True)
    def joint_velocity_limits(self) -> ProxyArray:
        """Shorthand for :attr:`joint_vel_limits`."""
        return self.joint_vel_limits

    @property
    @leapp_tensor_semantics(const=True)
    def joint_friction(self) -> ProxyArray:
        """Shorthand for :attr:`joint_friction_coeff`."""
        return self.joint_friction_coeff

    @property
    @leapp_tensor_semantics(const=True)
    def fixed_tendon_limit(self) -> ProxyArray:
        """Shorthand for :attr:`fixed_tendon_pos_limits`."""
        return self.fixed_tendon_pos_limits

    """
    Defaults - Default values will no longer be stored.
    """

    @property
    @leapp_tensor_semantics(const=True)
    def default_mass(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`body_mass` instead and manage the default mass manually."""
        warnings.warn(
            "The `default_mass` property will be deprecated in a IsaacLab 4.0. Please use `body_mass` instead. "
            "The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_mass is None:
            self._default_mass = wp.clone(self.body_mass.warp, self.device)
        return ProxyArray(self._default_mass)

    @property
    @leapp_tensor_semantics(const=True)
    def default_inertia(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`body_inertia` instead and manage the default inertia manually."""
        warnings.warn(
            "The `default_inertia` property will be deprecated in a IsaacLab 4.0. Please use `body_inertia` instead. "
            "The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_inertia is None:
            self._default_inertia = wp.clone(self.body_inertia.warp, self.device)
        return ProxyArray(self._default_inertia)

    @property
    @leapp_tensor_semantics(const=True)
    def default_joint_stiffness(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`joint_stiffness` instead and manage the default joint stiffness
        manually."""
        warnings.warn(
            "The `default_joint_stiffness` property will be deprecated in a IsaacLab 4.0. Please use `joint_stiffness` "
            "instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_stiffness is None:
            self._default_joint_stiffness = wp.clone(self.joint_stiffness.warp, self.device)
        return ProxyArray(self._default_joint_stiffness)

    @property
    @leapp_tensor_semantics(const=True)
    def default_joint_damping(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`joint_damping` instead and manage the default joint damping
        manually."""
        warnings.warn(
            "The `default_joint_damping` property will be deprecated in a IsaacLab 4.0. Please use `joint_damping` "
            "instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_damping is None:
            self._default_joint_damping = wp.clone(self.joint_damping.warp, self.device)
        return ProxyArray(self._default_joint_damping)

    @property
    @leapp_tensor_semantics(const=True)
    def default_joint_armature(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`joint_armature` instead and manage the default joint armature
        manually."""
        warnings.warn(
            "The `default_joint_armature` property will be deprecated in a IsaacLab 4.0. Please use `joint_armature` "
            "instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_armature is None:
            self._default_joint_armature = wp.clone(self.joint_armature.warp, self.device)
        return ProxyArray(self._default_joint_armature)

    @property
    @leapp_tensor_semantics(const=True)
    def default_joint_friction_coeff(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead and manage the default joint friction
        coefficient manually."""
        warnings.warn(
            "The `default_joint_friction_coeff` property will be deprecated in a IsaacLab 4.0. Please use "
            "`joint_friction_coeff` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_friction_coeff is None:
            self._default_joint_friction_coeff = wp.clone(self.joint_friction_coeff.warp, self.device)
        return ProxyArray(self._default_joint_friction_coeff)

    @property
    @leapp_tensor_semantics(const=True)
    def default_joint_viscous_friction_coeff(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`joint_viscous_friction_coeff` instead and manage the default joint
        viscous friction coefficient manually."""
        warnings.warn(
            "The `default_joint_viscous_friction_coeff` property will be deprecated in a IsaacLab 4.0. Please use "
            "`joint_viscous_friction_coeff` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_viscous_friction_coeff is None:
            self._default_joint_viscous_friction_coeff = wp.clone(
                getattr(self, "joint_viscous_friction_coeff").warp, self.device
            )
        return ProxyArray(self._default_joint_viscous_friction_coeff)

    @property
    @leapp_tensor_semantics(const=True)
    def default_joint_pos_limits(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead and manage the default joint position
        limits manually."""
        warnings.warn(
            "The `default_joint_pos_limits` property will be deprecated in a IsaacLab 4.0. Please use "
            "`joint_pos_limits` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_pos_limits is None:
            self._default_joint_pos_limits = wp.clone(self.joint_pos_limits.warp, self.device)
        return ProxyArray(self._default_joint_pos_limits)

    @property
    @leapp_tensor_semantics(const=True)
    def default_fixed_tendon_stiffness(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`fixed_tendon_stiffness` instead and manage the default fixed tendon
        stiffness manually."""
        warnings.warn(
            "The `default_fixed_tendon_stiffness` property will be deprecated in a IsaacLab 4.0. Please use "
            "`fixed_tendon_stiffness` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_stiffness is None:
            self._default_fixed_tendon_stiffness = wp.clone(self.fixed_tendon_stiffness.warp, self.device)
        return ProxyArray(self._default_fixed_tendon_stiffness)

    @property
    @leapp_tensor_semantics(const=True)
    def default_fixed_tendon_damping(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`fixed_tendon_damping` instead and manage the default fixed tendon
        damping manually."""
        warnings.warn(
            "The `default_fixed_tendon_damping` property will be deprecated in a IsaacLab 4.0. Please use "
            "`fixed_tendon_damping` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_damping is None:
            self._default_fixed_tendon_damping = wp.clone(self.fixed_tendon_damping.warp, self.device)
        return ProxyArray(self._default_fixed_tendon_damping)

    @property
    @leapp_tensor_semantics(const=True)
    def default_fixed_tendon_limit_stiffness(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`fixed_tendon_limit_stiffness` instead and manage the default fixed
        tendon limit stiffness manually."""
        warnings.warn(
            "The `default_fixed_tendon_limit_stiffness` property will be deprecated in a IsaacLab 4.0. Please use "
            "`fixed_tendon_limit_stiffness` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_limit_stiffness is None:
            self._default_fixed_tendon_limit_stiffness = wp.clone(self.fixed_tendon_limit_stiffness.warp, self.device)
        return ProxyArray(self._default_fixed_tendon_limit_stiffness)

    @property
    @leapp_tensor_semantics(const=True)
    def default_fixed_tendon_rest_length(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`fixed_tendon_rest_length` instead and manage the default fixed tendon
        rest length manually."""
        warnings.warn(
            "The `default_fixed_tendon_rest_length` property will be deprecated in a IsaacLab 4.0. Please use "
            "`fixed_tendon_rest_length` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_rest_length is None:
            self._default_fixed_tendon_rest_length = wp.clone(self.fixed_tendon_rest_length.warp, self.device)
        return ProxyArray(self._default_fixed_tendon_rest_length)

    @property
    @leapp_tensor_semantics(const=True)
    def default_fixed_tendon_offset(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`fixed_tendon_offset` instead and manage the default fixed tendon
        offset manually."""
        warnings.warn(
            "The `default_fixed_tendon_offset` property will be deprecated in a IsaacLab 4.0. Please use "
            "`fixed_tendon_offset` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_offset is None:
            self._default_fixed_tendon_offset = wp.clone(self.fixed_tendon_offset.warp, self.device)
        return ProxyArray(self._default_fixed_tendon_offset)

    @property
    @leapp_tensor_semantics(const=True)
    def default_fixed_tendon_pos_limits(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`fixed_tendon_pos_limits` instead and manage the default fixed tendon
        position limits manually."""
        warnings.warn(
            "The `default_fixed_tendon_pos_limits` property will be deprecated in a IsaacLab 4.0. Please use "
            "`fixed_tendon_pos_limits` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_pos_limits is None:
            self._default_fixed_tendon_pos_limits = wp.clone(self.fixed_tendon_pos_limits.warp, self.device)
        return ProxyArray(self._default_fixed_tendon_pos_limits)

    @property
    @leapp_tensor_semantics(const=True)
    def default_spatial_tendon_stiffness(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`spatial_tendon_stiffness` instead and manage the default spatial
        tendon stiffness manually."""
        warnings.warn(
            "The `default_spatial_tendon_stiffness` property will be deprecated in a IsaacLab 4.0. Please use "
            "`spatial_tendon_stiffness` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_spatial_tendon_stiffness is None:
            self._default_spatial_tendon_stiffness = wp.clone(self.spatial_tendon_stiffness.warp, self.device)
        return ProxyArray(self._default_spatial_tendon_stiffness)

    @property
    @leapp_tensor_semantics(const=True)
    def default_spatial_tendon_damping(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`spatial_tendon_damping` instead and manage the default spatial tendon
        damping manually."""
        warnings.warn(
            "The `default_spatial_tendon_damping` property will be deprecated in a IsaacLab 4.0. Please use "
            "`spatial_tendon_damping` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_spatial_tendon_damping is None:
            self._default_spatial_tendon_damping = wp.clone(self.spatial_tendon_damping.warp, self.device)
        return ProxyArray(self._default_spatial_tendon_damping)

    @property
    @leapp_tensor_semantics(const=True)
    def default_spatial_tendon_limit_stiffness(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`spatial_tendon_limit_stiffness` instead and manage the default
        spatial tendon limit stiffness manually."""
        warnings.warn(
            "The `default_spatial_tendon_limit_stiffness` property will be deprecated in a IsaacLab 4.0. Please use "
            "`spatial_tendon_limit_stiffness` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_spatial_tendon_limit_stiffness is None:
            self._default_spatial_tendon_limit_stiffness = wp.clone(
                self.spatial_tendon_limit_stiffness.warp, self.device
            )
        return ProxyArray(self._default_spatial_tendon_limit_stiffness)

    @property
    @leapp_tensor_semantics(const=True)
    def default_spatial_tendon_offset(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`spatial_tendon_offset` instead and manage the default spatial tendon
        offset manually."""
        warnings.warn(
            "The `default_spatial_tendon_offset` property will be deprecated in a IsaacLab 4.0. Please use "
            "`spatial_tendon_offset` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_spatial_tendon_offset is None:
            self._default_spatial_tendon_offset = wp.clone(self.spatial_tendon_offset.warp, self.device)
        return ProxyArray(self._default_spatial_tendon_offset)

    @property
    @leapp_tensor_semantics(const=True)
    def default_fixed_tendon_limit(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`default_fixed_tendon_pos_limits` instead."""
        warnings.warn(
            "The `default_fixed_tendon_limit` property will be deprecated in a IsaacLab 4.0. Please use"
            " `default_fixed_tendon_pos_limits` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_fixed_tendon_pos_limits

    @property
    @leapp_tensor_semantics(const=True)
    def default_joint_friction(self) -> ProxyArray:
        """Deprecated property. Please use :attr:`default_joint_friction_coeff` instead."""
        warnings.warn(
            "The `default_joint_friction` property will be deprecated in a IsaacLab 4.0. Please use"
            " `default_joint_friction_coeff` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_joint_friction_coeff
