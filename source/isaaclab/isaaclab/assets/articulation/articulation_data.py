# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import weakref

import omni.log
import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer


class ArticulationData:
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

    def __init__(self, root_physx_view: physx.ArticulationView, device: str):
        """Initializes the articulation data.

        Args:
            root_physx_view: The root articulation view.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_physx_view: physx.ArticulationView = weakref.proxy(root_physx_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        gravity = self._physics_sim_view.get_gravity()
        # Convert to direction vector
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)

        # Initialize constants
        self.GRAVITY_VEC_W = gravity_dir.repeat(self._root_physx_view.count, 1)
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_physx_view.count, 1)

        # Initialize history for finite differencing
        self._previous_joint_vel = self._root_physx_view.get_dof_velocities().clone()

        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._root_link_pose_w = TimestampedBuffer()
        self._root_link_vel_w = TimestampedBuffer()
        self._body_link_pose_w = TimestampedBuffer()
        self._body_link_vel_w = TimestampedBuffer()
        # -- com frame w.r.t. link frame
        self._body_com_pose_b = TimestampedBuffer()
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedBuffer()
        self._root_com_vel_w = TimestampedBuffer()
        self._body_com_pose_w = TimestampedBuffer()
        self._body_com_vel_w = TimestampedBuffer()
        self._body_com_acc_w = TimestampedBuffer()
        # -- combined state (these are cached as they concatenate)
        self._root_state_w = TimestampedBuffer()
        self._root_link_state_w = TimestampedBuffer()
        self._root_com_state_w = TimestampedBuffer()
        self._body_state_w = TimestampedBuffer()
        self._body_link_state_w = TimestampedBuffer()
        self._body_com_state_w = TimestampedBuffer()
        # -- joint state
        self._joint_pos = TimestampedBuffer()
        self._joint_vel = TimestampedBuffer()
        self._joint_acc = TimestampedBuffer()
        self._body_incoming_joint_wrench_b = TimestampedBuffer()

    def update(self, dt: float):
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the joint acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.joint_acc

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

    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat, lin_vel, ang_vel]`` in the local environment frame. Shape is (num_instances, 13).

    The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
    velocities are of its center of mass frame.

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """

    default_joint_pos: torch.Tensor = None
    """Default joint positions of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """

    default_joint_vel: torch.Tensor = None
    """Default joint velocities of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """

    ##
    # Defaults - Physical properties.
    ##

    default_mass: torch.Tensor = None
    """Default mass for all the bodies in the articulation. Shape is (num_instances, num_bodies).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_inertia: torch.Tensor = None
    """Default inertia for all the bodies in the articulation. Shape is (num_instances, num_bodies, 9).

    The inertia is the inertia tensor relative to the center of mass frame. The values are stored in
    the order :math:`[I_{xx}, I_{xy}, I_{xz}, I_{yx}, I_{yy}, I_{yz}, I_{zx}, I_{zy}, I_{zz}]`.

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_joint_stiffness: torch.Tensor = None
    """Default joint stiffness of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.stiffness`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    .. attention::
        The default stiffness is the value configured by the user or the value parsed from the USD schema.
        It should not be confused with :attr:`joint_stiffness`, which is the value set into the simulation.
    """

    default_joint_damping: torch.Tensor = None
    """Default joint damping of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.damping`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    .. attention::
        The default stiffness is the value configured by the user or the value parsed from the USD schema.
        It should not be confused with :attr:`joint_damping`, which is the value set into the simulation.
    """

    default_joint_armature: torch.Tensor = None
    """Default joint armature of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.armature`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.
    """

    default_joint_friction_coeff: torch.Tensor = None
    """Default joint static friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.friction`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.
    """

    default_joint_dynamic_friction_coeff: torch.Tensor = None
    """Default joint dynamic friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.dynamic_friction`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.
    """

    default_joint_viscous_friction_coeff: torch.Tensor = None
    """Default joint viscous friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.viscous_friction`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.
    """

    default_joint_pos_limits: torch.Tensor = None
    """Default joint position limits of all joints. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`. They are parsed from the USD schema at the time of initialization.
    """
    default_fixed_tendon_stiffness: torch.Tensor = None
    """Default tendon stiffness of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_damping: torch.Tensor = None
    """Default tendon damping of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_limit_stiffness: torch.Tensor = None
    """Default tendon limit stiffness of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_rest_length: torch.Tensor = None
    """Default tendon rest length of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_offset: torch.Tensor = None
    """Default tendon offset of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_fixed_tendon_pos_limits: torch.Tensor = None
    """Default tendon position limits of all fixed tendons. Shape is (num_instances, num_fixed_tendons, 2).

    The position limits are in the order :math:`[lower, upper]`. They are parsed from the USD schema at the time of
    initialization.
    """

    default_spatial_tendon_stiffness: torch.Tensor = None
    """Default tendon stiffness of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_spatial_tendon_damping: torch.Tensor = None
    """Default tendon damping of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_spatial_tendon_limit_stiffness: torch.Tensor = None
    """Default tendon limit stiffness of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    default_spatial_tendon_offset: torch.Tensor = None
    """Default tendon offset of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """

    ##
    # Joint commands -- Set into simulation.
    ##

    joint_pos_target: torch.Tensor = None
    """Joint position targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    joint_vel_target: torch.Tensor = None
    """Joint velocity targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    joint_effort_target: torch.Tensor = None
    """Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    ##
    # Joint commands -- Explicit actuators.
    ##

    computed_torque: torch.Tensor = None
    """Joint torques computed from the actuator model (before clipping). Shape is (num_instances, num_joints).

    This quantity is the raw torque output from the actuator mode, before any clipping is applied.
    It is exposed for users who want to inspect the computations inside the actuator model.
    For instance, to penalize the learning agent for a difference between the computed and applied torques.
    """

    applied_torque: torch.Tensor = None
    """Joint torques applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

    These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
    actuator model.
    """

    ##
    # Joint properties.
    ##

    joint_stiffness: torch.Tensor = None
    """Joint stiffness provided to the simulation. Shape is (num_instances, num_joints).

    In the case of explicit actuators, the value for the corresponding joints is zero.
    """

    joint_damping: torch.Tensor = None
    """Joint damping provided to the simulation. Shape is (num_instances, num_joints)

    In the case of explicit actuators, the value for the corresponding joints is zero.
    """

    joint_armature: torch.Tensor = None
    """Joint armature provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_friction_coeff: torch.Tensor = None
    """Joint static friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_dynamic_friction_coeff: torch.Tensor = None
    """Joint dynamic friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_viscous_friction_coeff: torch.Tensor = None
    """Joint viscous friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_pos_limits: torch.Tensor = None
    """Joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`.
    """

    joint_vel_limits: torch.Tensor = None
    """Joint maximum velocity provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_effort_limits: torch.Tensor = None
    """Joint maximum effort provided to the simulation. Shape is (num_instances, num_joints)."""

    ##
    # Joint properties - Custom.
    ##

    soft_joint_pos_limits: torch.Tensor = None
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

    soft_joint_vel_limits: torch.Tensor = None
    """Soft joint velocity limits for all joints. Shape is (num_instances, num_joints).

    These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
    has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
    """

    gear_ratio: torch.Tensor = None
    """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""

    ##
    # Fixed tendon properties.
    ##

    fixed_tendon_stiffness: torch.Tensor = None
    """Fixed tendon stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_damping: torch.Tensor = None
    """Fixed tendon damping provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_limit_stiffness: torch.Tensor = None
    """Fixed tendon limit stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_rest_length: torch.Tensor = None
    """Fixed tendon rest length provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_offset: torch.Tensor = None
    """Fixed tendon offset provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_pos_limits: torch.Tensor = None
    """Fixed tendon position limits provided to the simulation. Shape is (num_instances, num_fixed_tendons, 2)."""

    ##
    # Spatial tendon properties.
    ##

    spatial_tendon_stiffness: torch.Tensor = None
    """Spatial tendon stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    spatial_tendon_damping: torch.Tensor = None
    """Spatial tendon damping provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    spatial_tendon_limit_stiffness: torch.Tensor = None
    """Spatial tendon limit stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    spatial_tendon_offset: torch.Tensor = None
    """Spatial tendon offset provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    ##
    # Root state properties.
    ##

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        """Root link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._root_link_pose_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_physx_view.get_root_transforms().clone()
            pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._root_link_pose_w.data = pose
            self._root_link_pose_w.timestamp = self._sim_timestamp

        return self._root_link_pose_w.data

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            # read the CoM velocity
            vel = self.root_com_vel_w.clone()
            # adjust linear velocity to link from center of mass
            vel[:, :3] += torch.linalg.cross(
                vel[:, 3:], math_utils.quat_apply(self.root_link_quat_w, -self.body_com_pos_b[:, 0]), dim=-1
            )
            # set the buffer data and timestamp
            self._root_link_vel_w.data = vel
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w.data

    @property
    def root_com_pose_w(self) -> torch.Tensor:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            pos, quat = math_utils.combine_frame_transforms(
                self.root_link_pos_w, self.root_link_quat_w, self.body_com_pos_b[:, 0], self.body_com_quat_b[:, 0]
            )
            # set the buffer data and timestamp
            self._root_com_pose_w.data = torch.cat((pos, quat), dim=-1)
            self._root_com_pose_w.timestamp = self._sim_timestamp

        return self._root_com_pose_w.data

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        if self._root_com_vel_w.timestamp < self._sim_timestamp:
            self._root_com_vel_w.data = self._root_physx_view.get_root_velocities()
            self._root_com_vel_w.timestamp = self._sim_timestamp

        return self._root_com_vel_w.data

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame relative to the world. Meanwhile,
        the linear and angular velocities are of the articulation root's center of mass frame.
        """
        if self._root_state_w.timestamp < self._sim_timestamp:
            self._root_state_w.data = torch.cat((self.root_link_pose_w, self.root_com_vel_w), dim=-1)
            self._root_state_w.timestamp = self._sim_timestamp

        return self._root_state_w.data

    @property
    def root_link_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root's actor frame relative to the
        world.
        """
        if self._root_link_state_w.timestamp < self._sim_timestamp:
            self._root_link_state_w.data = torch.cat((self.root_link_pose_w, self.root_link_vel_w), dim=-1)
            self._root_link_state_w.timestamp = self._sim_timestamp

        return self._root_link_state_w.data

    @property
    def root_com_state_w(self):
        """Root center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root link's center of mass frame
        relative to the world. Center of mass frame is assumed to be the same orientation as the link rather than the
        orientation of the principle inertia.
        """
        if self._root_com_state_w.timestamp < self._sim_timestamp:
            self._root_com_state_w.data = torch.cat((self.root_com_pose_w, self.root_com_vel_w), dim=-1)
            self._root_com_state_w.timestamp = self._sim_timestamp

        return self._root_com_state_w.data

    ##
    # Body state properties.
    ##

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        """Body link pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._body_link_pose_w.timestamp < self._sim_timestamp:
            # perform forward kinematics (shouldn't cause overhead if it happened already)
            self._physics_sim_view.update_articulations_kinematic()
            # read data from simulation
            poses = self._root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._body_link_pose_w.data = poses
            self._body_link_pose_w.timestamp = self._sim_timestamp

        return self._body_link_pose_w.data

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        if self._body_link_vel_w.timestamp < self._sim_timestamp:
            # read data from simulation
            velocities = self.body_com_vel_w.clone()
            # adjust linear velocity to link from center of mass
            velocities[..., :3] += torch.linalg.cross(
                velocities[..., 3:], math_utils.quat_apply(self.body_link_quat_w, -self.body_com_pos_b), dim=-1
            )
            # set the buffer data and timestamp
            self._body_link_vel_w.data = velocities
            self._body_link_vel_w.timestamp = self._sim_timestamp

        return self._body_link_vel_w.data

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            pos, quat = math_utils.combine_frame_transforms(
                self.body_link_pos_w, self.body_link_quat_w, self.body_com_pos_b, self.body_com_quat_b
            )
            # set the buffer data and timestamp
            self._body_com_pose_w.data = torch.cat((pos, quat), dim=-1)
            self._body_com_pose_w.timestamp = self._sim_timestamp

        return self._body_com_pose_w.data

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        if self._body_com_vel_w.timestamp < self._sim_timestamp:
            self._body_com_vel_w.data = self._root_physx_view.get_link_velocities()
            self._body_com_vel_w.timestamp = self._sim_timestamp

        return self._body_com_vel_w.data

    @property
    def body_state_w(self):
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position and quaternion are of all the articulation links' actor frame. Meanwhile, the linear and angular
        velocities are of the articulation links's center of mass frame.
        """
        if self._body_state_w.timestamp < self._sim_timestamp:
            self._body_state_w.data = torch.cat((self.body_link_pose_w, self.body_com_vel_w), dim=-1)
            self._body_state_w.timestamp = self._sim_timestamp

        return self._body_state_w.data

    @property
    def body_link_state_w(self):
        """State of all bodies' link frame`[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """
        if self._body_link_state_w.timestamp < self._sim_timestamp:
            self._body_link_state_w.data = torch.cat((self.body_link_pose_w, self.body_link_vel_w), dim=-1)
            self._body_link_state_w.timestamp = self._sim_timestamp

        return self._body_link_state_w.data

    @property
    def body_com_state_w(self):
        """State of all bodies center of mass `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.
        """
        if self._body_com_state_w.timestamp < self._sim_timestamp:
            self._body_com_state_w.data = torch.cat((self.body_com_pose_w, self.body_com_vel_w), dim=-1)
            self._body_com_state_w.timestamp = self._sim_timestamp

        return self._body_com_state_w.data

    @property
    def body_com_acc_w(self):
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]``.
        Shape is (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._body_com_acc_w.data = self._root_physx_view.get_link_accelerations()
            self._body_com_acc_w.timestamp = self._sim_timestamp

        return self._body_com_acc_w.data

    @property
    def body_com_pose_b(self) -> torch.Tensor:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.
        Shape is (num_instances, 1, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_physx_view.get_coms().to(self.device)
            pose[..., 3:7] = math_utils.convert_quat(pose[..., 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._body_com_pose_b.data = pose
            self._body_com_pose_b.timestamp = self._sim_timestamp

        return self._body_com_pose_b.data

    @property
    def body_incoming_joint_wrench_b(self) -> torch.Tensor:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.

        For more information on joint wrenches, please check the`PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force>`__
        and the underlying `PhysX Tensor API <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force>`__ .
        """

        if self._body_incoming_joint_wrench_b.timestamp < self._sim_timestamp:
            self._body_incoming_joint_wrench_b.data = self._root_physx_view.get_link_incoming_joint_force()
            self._body_incoming_joint_wrench_b.time_stamp = self._sim_timestamp
        return self._body_incoming_joint_wrench_b.data

    ##
    # Joint state properties.
    ##

    @property
    def joint_pos(self):
        """Joint positions of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_pos.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_pos.data = self._root_physx_view.get_dof_positions()
            self._joint_pos.timestamp = self._sim_timestamp
        return self._joint_pos.data

    @property
    def joint_vel(self):
        """Joint velocities of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_vel.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_vel.data = self._root_physx_view.get_dof_velocities()
            self._joint_vel.timestamp = self._sim_timestamp
        return self._joint_vel.data

    @property
    def joint_acc(self):
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_acc.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            time_elapsed = self._sim_timestamp - self._joint_acc.timestamp
            self._joint_acc.data = (self.joint_vel - self._previous_joint_vel) / time_elapsed
            self._joint_acc.timestamp = self._sim_timestamp
            # update the previous joint velocity
            self._previous_joint_vel[:] = self.joint_vel
        return self._joint_acc.data

    ##
    # Derived Properties.
    ##

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        forward_w = math_utils.quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_ang_vel_w)

    ##
    # Sliced properties.
    ##

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self.root_link_pose_w[:, 3:7]

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self.root_link_vel_w[:, :3]

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self.root_link_vel_w[:, 3:6]

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self.root_com_pose_w[:, :3]

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        return self.root_com_pose_w[:, 3:7]

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self.root_com_vel_w[:, :3]

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self.root_com_vel_w[:, 3:6]

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        return self.body_link_pose_w[..., :3]

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        return self.body_link_pose_w[..., 3:7]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return self.body_link_vel_w[..., :3]

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return self.body_link_vel_w[..., 3:6]

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame.
        """
        return self.body_com_pose_w[..., :3]

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame.
        """
        return self.body_com_pose_w[..., 3:7]

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        return self.body_com_vel_w[..., :3]

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        return self.body_com_vel_w[..., 3:6]

    @property
    def body_com_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        return self.body_com_acc_w[..., :3]

    @property
    def body_com_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        return self.body_com_acc_w[..., 3:6]

    @property
    def body_com_pos_b(self) -> torch.Tensor:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body'slink frame.
        """
        return self.body_com_pose_b[..., :3]

    @property
    def body_com_quat_b(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        return self.body_com_pose_b[..., 3:7]

    ##
    # Backward compatibility.
    ##

    @property
    def root_pose_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_pose_w`."""
        return self.root_link_pose_w

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_pos_w`."""
        return self.root_link_pos_w

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_quat_w`."""
        return self.root_link_quat_w

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_vel_w`."""
        return self.root_com_vel_w

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_lin_vel_w`."""
        return self.root_com_lin_vel_w

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_ang_vel_w`."""
        return self.root_com_ang_vel_w

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Same as :attr:`root_com_lin_vel_b`."""
        return self.root_com_lin_vel_b

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Same as :attr:`root_com_ang_vel_b`."""
        return self.root_com_ang_vel_b

    @property
    def body_pose_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_pose_w`."""
        return self.body_link_pose_w

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_pos_w`."""
        return self.body_link_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_quat_w`."""
        return self.body_link_quat_w

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_vel_w`."""
        return self.body_com_vel_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_lin_vel_w`."""
        return self.body_com_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_ang_vel_w`."""
        return self.body_com_ang_vel_w

    @property
    def body_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_acc_w`."""
        return self.body_com_acc_w

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_lin_acc_w`."""
        return self.body_com_lin_acc_w

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_ang_acc_w`."""
        return self.body_com_ang_acc_w

    @property
    def com_pos_b(self) -> torch.Tensor:
        """Same as :attr:`body_com_pos_b`."""
        return self.body_com_pos_b

    @property
    def com_quat_b(self) -> torch.Tensor:
        """Same as :attr:`body_com_quat_b`."""
        return self.body_com_quat_b

    @property
    def joint_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead."""
        omni.log.warn(
            "The `joint_limits` property will be deprecated in a future release. Please use `joint_pos_limits` instead."
        )
        return self.joint_pos_limits

    @property
    def default_joint_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_joint_pos_limits` instead."""
        omni.log.warn(
            "The `default_joint_limits` property will be deprecated in a future release. Please use"
            " `default_joint_pos_limits` instead."
        )
        return self.default_joint_pos_limits

    @property
    def joint_velocity_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_vel_limits` instead."""
        omni.log.warn(
            "The `joint_velocity_limits` property will be deprecated in a future release. Please use"
            " `joint_vel_limits` instead."
        )
        return self.joint_vel_limits

    @property
    def joint_friction(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead."""
        omni.log.warn(
            "The `joint_friction` property will be deprecated in a future release. Please use"
            " `joint_friction_coeff` instead."
        )
        return self.joint_friction_coeff

    @property
    def default_joint_friction(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_joint_friction_coeff` instead."""
        omni.log.warn(
            "The `default_joint_friction` property will be deprecated in a future release. Please use"
            " `default_joint_friction_coeff` instead."
        )
        return self.default_joint_friction_coeff

    @property
    def fixed_tendon_limit(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`fixed_tendon_pos_limits` instead."""
        omni.log.warn(
            "The `fixed_tendon_limit` property will be deprecated in a future release. Please use"
            " `fixed_tendon_pos_limits` instead."
        )
        return self.fixed_tendon_pos_limits

    @property
    def default_fixed_tendon_limit(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_fixed_tendon_pos_limits` instead."""
        omni.log.warn(
            "The `default_fixed_tendon_limit` property will be deprecated in a future release. Please use"
            " `default_fixed_tendon_pos_limits` instead."
        )
        return self.default_fixed_tendon_pos_limits
