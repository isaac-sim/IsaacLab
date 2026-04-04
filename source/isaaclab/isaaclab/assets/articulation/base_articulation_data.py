# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABC, abstractmethod

import warp as wp


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

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    joint_names: list[str] = None
    """Joint names in the order parsed by the simulation view."""

    fixed_tendon_names: list[str] = None
    """Fixed tendon names in the order parsed by the simulation view."""

    spatial_tendon_names: list[str] = None
    """Spatial tendon names in the order parsed by the simulation view."""

    ##
    # Defaults - Initial state.
    ##

    @property
    @abstractmethod
    def default_root_pose(self) -> wp.array:
        """Default root pose ``[pos, quat]`` in the local environment frame.

        The position and quaternion are of the articulation root's actor frame. Shape is (num_instances),
        dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def default_root_vel(self) -> wp.array:
        """Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame.

        The linear and angular velocities are of the articulation root's center of mass frame.
        Shape is (num_instances), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def default_root_state(self) -> wp.array:
        """Deprecated, same as :attr:`default_root_pose` and :attr:`default_root_vel`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_joint_pos(self) -> wp.array:
        """Default joint positions of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def default_joint_vel(self) -> wp.array:
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
    def joint_pos_target(self) -> wp.array:
        """Joint position targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_vel_target(self) -> wp.array:
        """Joint velocity targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_effort_target(self) -> wp.array:
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
    def computed_torque(self) -> wp.array:
        """Joint torques computed from the actuator model (before clipping).

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        This quantity is the raw torque output from the actuator mode, before any clipping is applied.
        It is exposed for users who want to inspect the computations inside the actuator model.
        For instance, to penalize the learning agent for a difference between the computed and applied torques.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def applied_torque(self) -> wp.array:
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
    def joint_stiffness(self) -> wp.array:
        """Joint stiffness provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_damping(self) -> wp.array:
        """Joint damping provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_armature(self) -> wp.array:
        """Joint armature provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_friction_coeff(self) -> wp.array:
        """Joint static friction coefficient provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_pos_limits(self) -> wp.array:
        """Joint position limits provided to the simulation.

        Shape is (num_instances, num_joints, 2), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_vel_limits(self) -> wp.array:
        """Joint maximum velocity provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_effort_limits(self) -> wp.array:
        """Joint maximum effort provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        raise NotImplementedError

    ##
    # Joint properties - Custom.
    ##

    @property
    @abstractmethod
    def soft_joint_pos_limits(self) -> wp.array:
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
    def soft_joint_vel_limits(self) -> wp.array:
        """Soft joint velocity limits for all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
        has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def gear_ratio(self) -> wp.array:
        """Gear ratio for relating motor torques to applied Joint torques.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        raise NotImplementedError

    ##
    # Fixed tendon properties.
    ##

    @property
    @abstractmethod
    def fixed_tendon_stiffness(self) -> wp.array:
        """Fixed tendon stiffness provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def fixed_tendon_damping(self) -> wp.array:
        """Fixed tendon damping provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def fixed_tendon_limit_stiffness(self) -> wp.array:
        """Fixed tendon limit stiffness provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def fixed_tendon_rest_length(self) -> wp.array:
        """Fixed tendon rest length provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def fixed_tendon_offset(self) -> wp.array:
        """Fixed tendon offset provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def fixed_tendon_pos_limits(self) -> wp.array:
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
    def spatial_tendon_stiffness(self) -> wp.array:
        """Spatial tendon stiffness provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def spatial_tendon_damping(self) -> wp.array:
        """Spatial tendon damping provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def spatial_tendon_limit_stiffness(self) -> wp.array:
        """Spatial tendon limit stiffness provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def spatial_tendon_offset(self) -> wp.array:
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
    def root_link_pose_w(self) -> wp.array:
        """Root link pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).

        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_vel_w(self) -> wp.array:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_pose_w(self) -> wp.array:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).

        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_vel_w(self) -> wp.array:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`root_link_pose_w` and :attr:`root_com_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`root_link_pose_w` and :attr:`root_link_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`root_com_pose_w` and :attr:`root_com_vel_w`."""
        raise NotImplementedError

    ##
    # Body state properties.
    ##

    @property
    @abstractmethod
    def body_mass(self) -> wp.array:
        """Body mass ``wp.float32`` in the world frame.

        Shape is (num_instances, num_bodies), dtype = wp.float32. In torch this resolves to (num_instances, num_bodies).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_inertia(self) -> wp.array:
        """Flattened body inertia in the world frame.

        Shape is (num_instances, num_bodies, 9), dtype = wp.float32. In torch this resolves to
        (num_instances, num_bodies, 9).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_pose_w(self) -> wp.array:
        """Body link pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_vel_w(self) -> wp.array:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_pose_w(self) -> wp.array:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_vel_w(self) -> wp.array:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`body_link_pose_w` and :attr:`body_com_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`body_link_pose_w` and :attr:`body_link_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`body_com_pose_w` and :attr:`body_com_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_acc_w(self) -> wp.array:
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]``.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_pose_b(self) -> wp.array:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_incoming_joint_wrench_b(self) -> wp.array:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.

        For more information on joint wrenches, please check the `PhysX documentation`_ and the
        underlying `PhysX Tensor API`_.

        .. _PhysX documentation: https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force
        .. _PhysX Tensor API: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force
        """
        raise NotImplementedError

    ##
    # Joint state properties.
    ##

    @property
    @abstractmethod
    def joint_pos(self) -> wp.array:
        """Joint positions of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_vel(self) -> wp.array:
        """Joint velocities of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_acc(self) -> wp.array:
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
    def projected_gravity_b(self) -> wp.array:
        """Projection of the gravity direction on base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def heading_w(self) -> wp.array:
        """Yaw heading of the base frame (in radians).

        Shape is (num_instances), dtype = wp.float32. In torch this resolves to (num_instances,).

        .. note::
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_lin_vel_b(self) -> wp.array:
        """Root link linear velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to
        its actor frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_ang_vel_b(self) -> wp.array:
        """Root link angular velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to
        its actor frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_lin_vel_b(self) -> wp.array:
        """Root center of mass linear velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to
        its actor frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_ang_vel_b(self) -> wp.array:
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
    def root_link_pos_w(self) -> wp.array:
        """Root link position in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_quat_w(self) -> wp.array:
        """Root link orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances), dtype = wp.quatf. In torch this resolves to (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_lin_vel_w(self) -> wp.array:
        """Root linear velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_ang_vel_w(self) -> wp.array:
        """Root link angular velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_pos_w(self) -> wp.array:
        """Root center of mass position in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the position of the center of mass frame of the root rigid body relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_quat_w(self) -> wp.array:
        """Root center of mass orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances), dtype = wp.quatf. In torch this resolves to (num_instances, 4).

        This quantity is the orientation of the principal axes of inertia of the root rigid body relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_lin_vel_w(self) -> wp.array:
        """Root center of mass linear velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_ang_vel_w(self) -> wp.array:
        """Root center of mass angular velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the principal axes of inertia of the articulation bodies.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_lin_acc_w(self) -> wp.array:
        """Linear acceleration of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_ang_acc_w(self) -> wp.array:
        """Angular acceleration of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_pos_b(self) -> wp.array:
        """Center of mass position of all of the bodies in their respective link frames.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body's link frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_quat_b(self) -> wp.array:
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
    def root_pose_w(self) -> wp.array:
        """Shorthand for :attr:`root_link_pose_w`."""
        return self.root_link_pose_w

    @property
    def root_pos_w(self) -> wp.array:
        """Shorthand for :attr:`root_link_pos_w`."""
        return self.root_link_pos_w

    @property
    def root_quat_w(self) -> wp.array:
        """Shorthand for :attr:`root_link_quat_w`."""
        return self.root_link_quat_w

    @property
    def root_vel_w(self) -> wp.array:
        """Shorthand for :attr:`root_com_vel_w`."""
        return self.root_com_vel_w

    @property
    def root_lin_vel_w(self) -> wp.array:
        """Shorthand for :attr:`root_com_lin_vel_w`."""
        return self.root_com_lin_vel_w

    @property
    def root_ang_vel_w(self) -> wp.array:
        """Shorthand for :attr:`root_com_ang_vel_w`."""
        return self.root_com_ang_vel_w

    @property
    def root_lin_vel_b(self) -> wp.array:
        """Shorthand for :attr:`root_com_lin_vel_b`."""
        return self.root_com_lin_vel_b

    @property
    def root_ang_vel_b(self) -> wp.array:
        """Shorthand for :attr:`root_com_ang_vel_b`."""
        return self.root_com_ang_vel_b

    @property
    def body_pose_w(self) -> wp.array:
        """Shorthand for :attr:`body_link_pose_w`."""
        return self.body_link_pose_w

    @property
    def body_pos_w(self) -> wp.array:
        """Shorthand for :attr:`body_link_pos_w`."""
        return self.body_link_pos_w

    @property
    def body_quat_w(self) -> wp.array:
        """Shorthand for :attr:`body_link_quat_w`."""
        return self.body_link_quat_w

    @property
    def body_vel_w(self) -> wp.array:
        """Shorthand for :attr:`body_com_vel_w`."""
        return self.body_com_vel_w

    @property
    def body_lin_vel_w(self) -> wp.array:
        """Shorthand for :attr:`body_com_lin_vel_w`."""
        return self.body_com_lin_vel_w

    @property
    def body_ang_vel_w(self) -> wp.array:
        """Shorthand for :attr:`body_com_ang_vel_w`."""
        return self.body_com_ang_vel_w

    @property
    def body_acc_w(self) -> wp.array:
        """Shorthand for :attr:`body_com_acc_w`."""
        return self.body_com_acc_w

    @property
    def body_lin_acc_w(self) -> wp.array:
        """Shorthand for :attr:`body_com_lin_acc_w`."""
        return self.body_com_lin_acc_w

    @property
    def body_ang_acc_w(self) -> wp.array:
        """Shorthand for :attr:`body_com_ang_acc_w`."""
        return self.body_com_ang_acc_w

    @property
    def com_pos_b(self) -> wp.array:
        """Shorthand for :attr:`body_com_pos_b`."""
        return self.body_com_pos_b

    @property
    def com_quat_b(self) -> wp.array:
        """Shorthand for :attr:`body_com_quat_b`."""
        return self.body_com_quat_b

    @property
    def joint_limits(self) -> wp.array:
        """Shorthand for :attr:`joint_pos_limits`."""
        return self.joint_pos_limits

    @property
    def default_joint_limits(self) -> wp.array:
        """Shorthand for :attr:`default_joint_pos_limits`."""
        return self.default_joint_pos_limits

    @property
    def joint_velocity_limits(self) -> wp.array:
        """Shorthand for :attr:`joint_vel_limits`."""
        return self.joint_vel_limits

    @property
    def joint_friction(self) -> wp.array:
        """Shorthand for :attr:`joint_friction_coeff`."""
        return self.joint_friction_coeff

    @property
    def fixed_tendon_limit(self) -> wp.array:
        """Shorthand for :attr:`fixed_tendon_pos_limits`."""
        return self.fixed_tendon_pos_limits

    """
    Defaults - Default values will no longer be stored.
    """

    @property
    def default_mass(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_mass` instead and manage the default mass manually."""
        warnings.warn(
            "The `default_mass` property will be deprecated in a IsaacLab 4.0. Please use `body_mass` instead. "
            "The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_mass is None:
            self._default_mass = wp.clone(self.body_mass, self.device)
        return self._default_mass

    @property
    def default_inertia(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_inertia` instead and manage the default inertia manually."""
        warnings.warn(
            "The `default_inertia` property will be deprecated in a IsaacLab 4.0. Please use `body_inertia` instead. "
            "The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_inertia is None:
            self._default_inertia = wp.clone(self.body_inertia, self.device)
        return self._default_inertia

    @property
    def default_joint_stiffness(self) -> wp.array:
        """Deprecated property. Please use :attr:`joint_stiffness` instead and manage the default joint stiffness
        manually."""
        warnings.warn(
            "The `default_joint_stiffness` property will be deprecated in a IsaacLab 4.0. Please use `joint_stiffness` "
            "instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_stiffness is None:
            self._default_joint_stiffness = wp.clone(self.joint_stiffness, self.device)
        return self._default_joint_stiffness

    @property
    def default_joint_damping(self) -> wp.array:
        """Deprecated property. Please use :attr:`joint_damping` instead and manage the default joint damping
        manually."""
        warnings.warn(
            "The `default_joint_damping` property will be deprecated in a IsaacLab 4.0. Please use `joint_damping` "
            "instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_damping is None:
            self._default_joint_damping = wp.clone(self.joint_damping, self.device)
        return self._default_joint_damping

    @property
    def default_joint_armature(self) -> wp.array:
        """Deprecated property. Please use :attr:`joint_armature` instead and manage the default joint armature
        manually."""
        warnings.warn(
            "The `default_joint_armature` property will be deprecated in a IsaacLab 4.0. Please use `joint_armature` "
            "instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_armature is None:
            self._default_joint_armature = wp.clone(self.joint_armature, self.device)
        return self._default_joint_armature

    @property
    def default_joint_friction_coeff(self) -> wp.array:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead and manage the default joint friction
        coefficient manually."""
        warnings.warn(
            "The `default_joint_friction_coeff` property will be deprecated in a IsaacLab 4.0. Please use "
            "`joint_friction_coeff` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_friction_coeff is None:
            self._default_joint_friction_coeff = wp.clone(self.joint_friction_coeff, self.device)
        return self._default_joint_friction_coeff

    @property
    def default_joint_viscous_friction_coeff(self) -> wp.array:
        """Deprecated property. Please use :attr:`joint_viscous_friction_coeff` instead and manage the default joint
        viscous friction coefficient manually."""
        warnings.warn(
            "The `default_joint_viscous_friction_coeff` property will be deprecated in a IsaacLab 4.0. Please use "
            "`joint_viscous_friction_coeff` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_viscous_friction_coeff is None:
            self._default_joint_viscous_friction_coeff = wp.clone(self.joint_viscous_friction_coeff, self.device)
        return self._default_joint_viscous_friction_coeff

    @property
    def default_joint_pos_limits(self) -> wp.array:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead and manage the default joint position
        limits manually."""
        warnings.warn(
            "The `default_joint_pos_limits` property will be deprecated in a IsaacLab 4.0. Please use "
            "`joint_pos_limits` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_pos_limits is None:
            self._default_joint_pos_limits = wp.clone(self.joint_pos_limits, self.device)
        return self._default_joint_pos_limits

    @property
    def default_fixed_tendon_stiffness(self) -> wp.array:
        """Deprecated property. Please use :attr:`fixed_tendon_stiffness` instead and manage the default fixed tendon
        stiffness manually."""
        warnings.warn(
            "The `default_fixed_tendon_stiffness` property will be deprecated in a IsaacLab 4.0. Please use "
            "`fixed_tendon_stiffness` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_stiffness is None:
            self._default_fixed_tendon_stiffness = wp.clone(self.fixed_tendon_stiffness, self.device)
        return self._default_fixed_tendon_stiffness

    @property
    def default_fixed_tendon_damping(self) -> wp.array:
        """Deprecated property. Please use :attr:`fixed_tendon_damping` instead and manage the default fixed tendon
        damping manually."""
        warnings.warn(
            "The `default_fixed_tendon_damping` property will be deprecated in a IsaacLab 4.0. Please use "
            "`fixed_tendon_damping` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_damping is None:
            self._default_fixed_tendon_damping = wp.clone(self.fixed_tendon_damping, self.device)
        return self._default_fixed_tendon_damping

    @property
    def default_fixed_tendon_limit_stiffness(self) -> wp.array:
        """Deprecated property. Please use :attr:`fixed_tendon_limit_stiffness` instead and manage the default fixed
        tendon limit stiffness manually."""
        warnings.warn(
            "The `default_fixed_tendon_limit_stiffness` property will be deprecated in a IsaacLab 4.0. Please use "
            "`fixed_tendon_limit_stiffness` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_limit_stiffness is None:
            self._default_fixed_tendon_limit_stiffness = wp.clone(self.fixed_tendon_limit_stiffness, self.device)
        return self._default_fixed_tendon_limit_stiffness

    @property
    def default_fixed_tendon_rest_length(self) -> wp.array:
        """Deprecated property. Please use :attr:`fixed_tendon_rest_length` instead and manage the default fixed tendon
        rest length manually."""
        warnings.warn(
            "The `default_fixed_tendon_rest_length` property will be deprecated in a IsaacLab 4.0. Please use "
            "`fixed_tendon_rest_length` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_rest_length is None:
            self._default_fixed_tendon_rest_length = wp.clone(self.fixed_tendon_rest_length, self.device)
        return self._default_fixed_tendon_rest_length

    @property
    def default_fixed_tendon_offset(self) -> wp.array:
        """Deprecated property. Please use :attr:`fixed_tendon_offset` instead and manage the default fixed tendon
        offset manually."""
        warnings.warn(
            "The `default_fixed_tendon_offset` property will be deprecated in a IsaacLab 4.0. Please use "
            "`fixed_tendon_offset` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_offset is None:
            self._default_fixed_tendon_offset = wp.clone(self.fixed_tendon_offset, self.device)
        return self._default_fixed_tendon_offset

    @property
    def default_fixed_tendon_pos_limits(self) -> wp.array:
        """Deprecated property. Please use :attr:`fixed_tendon_pos_limits` instead and manage the default fixed tendon
        position limits manually."""
        warnings.warn(
            "The `default_fixed_tendon_pos_limits` property will be deprecated in a IsaacLab 4.0. Please use "
            "`fixed_tendon_pos_limits` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_pos_limits is None:
            self._default_fixed_tendon_pos_limits = wp.clone(self.fixed_tendon_pos_limits, self.device)
        return self._default_fixed_tendon_pos_limits

    @property
    def default_spatial_tendon_stiffness(self) -> wp.array:
        """Deprecated property. Please use :attr:`spatial_tendon_stiffness` instead and manage the default spatial
        tendon stiffness manually."""
        warnings.warn(
            "The `default_spatial_tendon_stiffness` property will be deprecated in a IsaacLab 4.0. Please use "
            "`spatial_tendon_stiffness` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_spatial_tendon_stiffness is None:
            self._default_spatial_tendon_stiffness = wp.clone(self.spatial_tendon_stiffness, self.device)
        return self._default_spatial_tendon_stiffness

    @property
    def default_spatial_tendon_damping(self) -> wp.array:
        """Deprecated property. Please use :attr:`spatial_tendon_damping` instead and manage the default spatial tendon
        damping manually."""
        warnings.warn(
            "The `default_spatial_tendon_damping` property will be deprecated in a IsaacLab 4.0. Please use "
            "`spatial_tendon_damping` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_spatial_tendon_damping is None:
            self._default_spatial_tendon_damping = wp.clone(self.spatial_tendon_damping, self.device)
        return self._default_spatial_tendon_damping

    @property
    def default_spatial_tendon_limit_stiffness(self) -> wp.array:
        """Deprecated property. Please use :attr:`spatial_tendon_limit_stiffness` instead and manage the default
        spatial tendon limit stiffness manually."""
        warnings.warn(
            "The `default_spatial_tendon_limit_stiffness` property will be deprecated in a IsaacLab 4.0. Please use "
            "`spatial_tendon_limit_stiffness` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_spatial_tendon_limit_stiffness is None:
            self._default_spatial_tendon_limit_stiffness = wp.clone(self.spatial_tendon_limit_stiffness, self.device)
        return self._default_spatial_tendon_limit_stiffness

    @property
    def default_spatial_tendon_offset(self) -> wp.array:
        """Deprecated property. Please use :attr:`spatial_tendon_offset` instead and manage the default spatial tendon
        offset manually."""
        warnings.warn(
            "The `default_spatial_tendon_offset` property will be deprecated in a IsaacLab 4.0. Please use "
            "`spatial_tendon_offset` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_spatial_tendon_offset is None:
            self._default_spatial_tendon_offset = wp.clone(self.spatial_tendon_offset, self.device)
        return self._default_spatial_tendon_offset

    @property
    def default_fixed_tendon_limit(self) -> wp.array:
        """Deprecated property. Please use :attr:`default_fixed_tendon_pos_limits` instead."""
        warnings.warn(
            "The `default_fixed_tendon_limit` property will be deprecated in a IsaacLab 4.0. Please use"
            " `default_fixed_tendon_pos_limits` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_fixed_tendon_pos_limits

    @property
    def default_joint_friction(self) -> wp.array:
        """Deprecated property. Please use :attr:`default_joint_friction_coeff` instead."""
        warnings.warn(
            "The `default_joint_friction` property will be deprecated in a IsaacLab 4.0. Please use"
            " `default_joint_friction_coeff` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_joint_friction_coeff
