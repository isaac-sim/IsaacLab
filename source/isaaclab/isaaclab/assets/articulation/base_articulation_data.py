# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import warp as wp
from abc import ABC, abstractmethod


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

    default_root_state: torch.Tensor | wp.array = None
    """Default root state ``[pos, quat, lin_vel, ang_vel]`` in the local environment frame. Shape is (num_instances, 13).

    The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
    velocities are of its center of mass frame.

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """

    default_joint_pos: torch.Tensor | wp.array = None
    """Default joint positions of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """

    default_joint_vel: torch.Tensor | wp.array = None
    """Default joint velocities of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """

    ##
    # Defaults - Physical properties.
    ##

    default_mass: torch.Tensor | wp.array = None
    """Default mass for all the bodies in the articulation. Shape is (num_instances, num_bodies).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_inertia: torch.Tensor | wp.array = None
    """Default inertia for all the bodies in the articulation. Shape is (num_instances, num_bodies, 9).

    The inertia tensor should be given with respect to the center of mass, expressed in the articulation links' actor frame.
    The values are stored in the order :math:`[I_{xx}, I_{yx}, I_{zx}, I_{xy}, I_{yy}, I_{zy}, I_{xz}, I_{yz}, I_{zz}]`.
    However, due to the symmetry of inertia tensors, row- and column-major orders are equivalent.

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_joint_stiffness: torch.Tensor | wp.array = None
    """Default joint stiffness of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.stiffness`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    .. attention::
        The default stiffness is the value configured by the user or the value parsed from the USD schema.
        It should not be confused with :attr:`joint_stiffness`, which is the value set into the simulation.
    """

    default_joint_damping: torch.Tensor | wp.array = None
    """Default joint damping of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.damping`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    .. attention::
        The default stiffness is the value configured by the user or the value parsed from the USD schema.
        It should not be confused with :attr:`joint_damping`, which is the value set into the simulation.
    """

    default_joint_armature: torch.Tensor | wp.array = None
    """Default joint armature of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.armature`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.
    """

    default_joint_friction_coeff: torch.Tensor | wp.array = None
    """Default joint static friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.friction`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.
    """

    default_joint_dynamic_friction_coeff: torch.Tensor | wp.array = None
    """Default joint dynamic friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.dynamic_friction`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.
    """

    default_joint_viscous_friction_coeff: torch.Tensor | wp.array = None
    """Default joint viscous friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.viscous_friction`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.
    """

    default_joint_pos_limits: torch.Tensor | wp.array = None
    """Default joint position limits of all joints. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`. They are parsed from the USD schema at the time of initialization.
    """
    default_fixed_tendon_stiffness: torch.Tensor | wp.array = None
    """Default tendon stiffness of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_damping: torch.Tensor | wp.array = None
    """Default tendon damping of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_limit_stiffness: torch.Tensor | wp.array = None
    """Default tendon limit stiffness of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_rest_length: torch.Tensor | wp.array = None
    """Default tendon rest length of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_offset: torch.Tensor | wp.array = None
    """Default tendon offset of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_pos_limits: torch.Tensor | wp.array = None
    """Default tendon position limits of all fixed tendons. Shape is (num_instances, num_fixed_tendons, 2).

    The position limits are in the order :math:`[lower, upper]`. They are parsed from the USD schema at the time of
    initialization.
    """

    default_spatial_tendon_stiffness: torch.Tensor | wp.array = None
    """Default tendon stiffness of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_spatial_tendon_damping: torch.Tensor | wp.array = None
    """Default tendon damping of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_spatial_tendon_limit_stiffness: torch.Tensor | wp.array = None
    """Default tendon limit stiffness of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_spatial_tendon_offset: torch.Tensor | wp.array = None
    """Default tendon offset of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    ##
    # Joint commands -- Set into simulation.
    ##

    joint_pos_target: torch.Tensor | wp.array = None
    """Joint position targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    joint_vel_target: torch.Tensor | wp.array = None
    """Joint velocity targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    joint_effort_target: torch.Tensor | wp.array = None
    """Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    ##
    # Joint commands -- Explicit actuators.
    ##

    computed_torque: torch.Tensor | wp.array = None
    """Joint torques computed from the actuator model (before clipping). Shape is (num_instances, num_joints).

    This quantity is the raw torque output from the actuator mode, before any clipping is applied.
    It is exposed for users who want to inspect the computations inside the actuator model.
    For instance, to penalize the learning agent for a difference between the computed and applied torques.
    """

    applied_torque: torch.Tensor | wp.array = None
    """Joint torques applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

    These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
    actuator model.
    """

    ##
    # Joint properties.
    ##

    joint_stiffness: torch.Tensor | wp.array = None
    """Joint stiffness provided to the simulation. Shape is (num_instances, num_joints).

    In the case of explicit actuators, the value for the corresponding joints is zero.
    """

    joint_damping: torch.Tensor | wp.array = None
    """Joint damping provided to the simulation. Shape is (num_instances, num_joints)

    In the case of explicit actuators, the value for the corresponding joints is zero.
    """

    joint_armature: torch.Tensor | wp.array = None
    """Joint armature provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_friction_coeff: torch.Tensor | wp.array = None
    """Joint static friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_dynamic_friction_coeff: torch.Tensor | wp.array = None
    """Joint dynamic friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_viscous_friction_coeff: torch.Tensor | wp.array = None
    """Joint viscous friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_pos_limits: torch.Tensor | wp.array = None
    """Joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`.
    """

    joint_vel_limits: torch.Tensor | wp.array = None
    """Joint maximum velocity provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_effort_limits: torch.Tensor | wp.array = None
    """Joint maximum effort provided to the simulation. Shape is (num_instances, num_joints)."""

    ##
    # Joint properties - Custom.
    ##

    soft_joint_pos_limits: torch.Tensor | wp.array = None
    r"""Soft joint positions limits for all joints. Shape is (num_instances, num_joints, 2).

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

    soft_joint_vel_limits: torch.Tensor | wp.array = None
    """Soft joint velocity limits for all joints. Shape is (num_instances, num_joints).

    These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
    has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
    """

    gear_ratio: torch.Tensor | wp.array = None
    """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""

    ##
    # Fixed tendon properties.
    ##

    fixed_tendon_stiffness: torch.Tensor | wp.array = None
    """Fixed tendon stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_damping: torch.Tensor | wp.array = None
    """Fixed tendon damping provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_limit_stiffness: torch.Tensor | wp.array = None
    """Fixed tendon limit stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_rest_length: torch.Tensor | wp.array = None
    """Fixed tendon rest length provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_offset: torch.Tensor | wp.array = None
    """Fixed tendon offset provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_pos_limits: torch.Tensor | wp.array = None
    """Fixed tendon position limits provided to the simulation. Shape is (num_instances, num_fixed_tendons, 2)."""

    ##
    # Spatial tendon properties.
    ##

    spatial_tendon_stiffness: torch.Tensor | wp.array = None
    """Spatial tendon stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    spatial_tendon_damping: torch.Tensor | wp.array = None
    """Spatial tendon damping provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    spatial_tendon_limit_stiffness: torch.Tensor | wp.array = None
    """Spatial tendon limit stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    spatial_tendon_offset: torch.Tensor | wp.array = None
    """Spatial tendon offset provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    ##
    # Root state properties.
    ##

    @property
    @abstractmethod
    def root_link_pose_w(self) -> torch.Tensor | wp.array:
        """Root link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_vel_w(self) -> torch.Tensor | wp.array:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_pose_w(self) -> torch.Tensor | wp.array:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_vel_w(self) -> torch.Tensor | wp.array:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_state_w(self) -> torch.Tensor | wp.array:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame relative to the world. Meanwhile,
        the linear and angular velocities are of the articulation root's center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_state_w(self) -> torch.Tensor | wp.array:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root's actor frame relative to the
        world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_state_w(self) -> torch.Tensor | wp.array:
        """Root center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root link's center of mass frame
        relative to the world. Center of mass frame is assumed to be the same orientation as the link rather than the
        orientation of the principle inertia.
        """
        raise NotImplementedError

    ##
    # Body state properties.
    ##

    @property
    @abstractmethod
    def body_link_pose_w(self) -> torch.Tensor | wp.array:
        """Body link pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_vel_w(self) -> torch.Tensor | wp.array:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_pose_w(self) -> torch.Tensor | wp.array:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_vel_w(self) -> torch.Tensor | wp.array:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_state_w(self) -> torch.Tensor | wp.array:
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position and quaternion are of all the articulation links' actor frame. Meanwhile, the linear and angular
        velocities are of the articulation links's center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_state_w(self) -> torch.Tensor | wp.array:
        """State of all bodies' link frame`[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_state_w(self) -> torch.Tensor | wp.array:
        """State of all bodies center of mass `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_acc_w(self) -> torch.Tensor | wp.array:
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]``.
        Shape is (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_pose_b(self) -> torch.Tensor | wp.array:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.
        Shape is (num_instances, 1, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (w, x, y, z) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_incoming_joint_wrench_b(self) -> torch.Tensor | wp.array:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.

        For more information on joint wrenches, please check the`PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force>`__
        and the underlying `PhysX Tensor API <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force>`__ .
        """
        raise NotImplementedError

    ##
    # Joint state properties.
    ##

    @property
    @abstractmethod
    def joint_pos(self) -> torch.Tensor | wp.array:
        """Joint positions of all joints. Shape is (num_instances, num_joints)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_vel(self) -> torch.Tensor | wp.array:
        """Joint velocities of all joints. Shape is (num_instances, num_joints)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_acc(self) -> torch.Tensor | wp.array:
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        raise NotImplementedError

    ##
    # Derived Properties.
    ##

    @property
    @abstractmethod
    def projected_gravity_b(self) -> torch.Tensor | wp.array:
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def heading_w(self) -> torch.Tensor | wp.array:
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_lin_vel_b(self) -> torch.Tensor | wp.array:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_ang_vel_b(self) -> torch.Tensor | wp.array:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_lin_vel_b(self) -> torch.Tensor | wp.array:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_ang_vel_b(self) -> torch.Tensor | wp.array:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        raise NotImplementedError

    ##
    # Sliced properties.
    ##

    @property
    @abstractmethod
    def root_link_pos_w(self) -> torch.Tensor | wp.array:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_quat_w(self) -> torch.Tensor | wp.array:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_lin_vel_w(self) -> torch.Tensor | wp.array:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_link_ang_vel_w(self) -> torch.Tensor | wp.array:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_pos_w(self) -> torch.Tensor | wp.array:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_quat_w(self) -> torch.Tensor | wp.array:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_lin_vel_w(self) -> torch.Tensor | wp.array:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def root_com_ang_vel_w(self) -> torch.Tensor | wp.array:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_pos_w(self) -> torch.Tensor | wp.array:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_quat_w(self) -> torch.Tensor | wp.array:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_lin_vel_w(self) -> torch.Tensor | wp.array:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_link_ang_vel_w(self) -> torch.Tensor | wp.array:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame relative to the world.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_pos_w(self) -> torch.Tensor | wp.array:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_quat_w(self) -> torch.Tensor | wp.array:
        """Orientation (w, x, y, z) of the principle axis of inertia of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_lin_vel_w(self) -> torch.Tensor | wp.array:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_ang_vel_w(self) -> torch.Tensor | wp.array:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_lin_acc_w(self) -> torch.Tensor | wp.array:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_ang_acc_w(self) -> torch.Tensor | wp.array:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_pos_b(self) -> torch.Tensor | wp.array:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body'slink frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_com_quat_b(self) -> torch.Tensor | wp.array:
        """Orientation (w, x, y, z) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        raise NotImplementedError

    ##
    # Backward compatibility.
    ##

    @property
    @abstractmethod
    def root_pose_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`root_link_pose_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def root_pos_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`root_link_pos_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def root_quat_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`root_link_quat_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def root_vel_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`root_com_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def root_lin_vel_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`root_com_lin_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def root_ang_vel_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`root_com_ang_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def root_lin_vel_b(self) -> torch.Tensor | wp.array:
        """Same as :attr:`root_com_lin_vel_b`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def root_ang_vel_b(self) -> torch.Tensor | wp.array:
        """Same as :attr:`root_com_ang_vel_b`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_pose_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`body_link_pose_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_pos_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`body_link_pos_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_quat_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`body_link_quat_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_vel_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`body_com_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_lin_vel_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`body_com_lin_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_ang_vel_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`body_com_ang_vel_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_acc_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`body_com_acc_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_lin_acc_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`body_com_lin_acc_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_ang_acc_w(self) -> torch.Tensor | wp.array:
        """Same as :attr:`body_com_ang_acc_w`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def com_pos_b(self) -> torch.Tensor | wp.array:
        """Same as :attr:`body_com_pos_b`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def com_quat_b(self) -> torch.Tensor | wp.array:
        """Same as :attr:`body_com_quat_b`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_limits(self) -> torch.Tensor | wp.array:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_joint_limits(self) -> torch.Tensor | wp.array:
        """Deprecated property. Please use :attr:`default_joint_pos_limits` instead."""
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_velocity_limits(self) -> torch.Tensor | wp.array:
        """Deprecated property. Please use :attr:`joint_vel_limits` instead."""
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_friction(self) -> torch.Tensor | wp.array:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_joint_friction(self) -> torch.Tensor | wp.array:
        """Deprecated property. Please use :attr:`default_joint_friction_coeff` instead."""
        raise NotImplementedError

    @property
    @abstractmethod
    def fixed_tendon_limit(self) -> torch.Tensor | wp.array:
        """Deprecated property. Please use :attr:`fixed_tendon_pos_limits` instead."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_fixed_tendon_limit(self) -> torch.Tensor | wp.array:
        """Deprecated property. Please use :attr:`default_fixed_tendon_pos_limits` instead."""
        raise NotImplementedError
