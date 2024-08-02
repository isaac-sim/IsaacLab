# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import weakref

import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.utils.buffers import TimestampedBuffer


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

        # Obtain global physics sim view
        physics_sim_view = physx.create_simulation_view("torch")
        physics_sim_view.set_subspace_roots("/")
        gravity = physics_sim_view.get_gravity()
        # Convert to direction vector
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)

        # Initialize constants
        self.GRAVITY_VEC_W = gravity_dir.repeat(self._root_physx_view.count, 1)
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_physx_view.count, 1)

        # Initialize history for finite differencing
        self._previous_joint_vel = self._root_physx_view.get_dof_velocities().clone()

        # Initialize the lazy buffers.
        self._root_state_w = TimestampedBuffer()
        self._body_state_w = TimestampedBuffer()
        self._body_acc_w = TimestampedBuffer()
        self._joint_pos = TimestampedBuffer()
        self._joint_acc = TimestampedBuffer()
        self._joint_vel = TimestampedBuffer()

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

    ##
    # Defaults.
    ##

    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame. Shape is (num_instances, 13).

    The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
    velocities are of its center of mass frame.
    """

    default_mass: torch.Tensor = None
    """Default mass read from the simulation. Shape is (num_instances, num_bodies)."""

    default_joint_pos: torch.Tensor = None
    """Default joint positions of all joints. Shape is (num_instances, num_joints)."""

    default_joint_vel: torch.Tensor = None
    """Default joint velocities of all joints. Shape is (num_instances, num_joints)."""

    default_joint_stiffness: torch.Tensor = None
    """Default joint stiffness of all joints. Shape is (num_instances, num_joints)."""

    default_joint_damping: torch.Tensor = None
    """Default joint damping of all joints. Shape is (num_instances, num_joints)."""

    default_joint_armature: torch.Tensor = None
    """Default joint armature of all joints. Shape is (num_instances, num_joints)."""

    default_joint_friction: torch.Tensor = None
    """Default joint friction of all joints. Shape is (num_instances, num_joints)."""

    default_joint_limits: torch.Tensor = None
    """Default joint limits of all joints. Shape is (num_instances, num_joints, 2)."""

    default_fixed_tendon_stiffness: torch.Tensor = None
    """Default tendon stiffness of all tendons. Shape is (num_instances, num_fixed_tendons)."""

    default_fixed_tendon_damping: torch.Tensor = None
    """Default tendon damping of all tendons. Shape is (num_instances, num_fixed_tendons)."""

    default_fixed_tendon_limit_stiffness: torch.Tensor = None
    """Default tendon limit stiffness of all tendons. Shape is (num_instances, num_fixed_tendons)."""

    default_fixed_tendon_rest_length: torch.Tensor = None
    """Default tendon rest length of all tendons. Shape is (num_instances, num_fixed_tendons)."""

    default_fixed_tendon_offset: torch.Tensor = None
    """Default tendon offset of all tendons. Shape is (num_instances, num_fixed_tendons)."""

    default_fixed_tendon_limit: torch.Tensor = None
    """Default tendon limits of all tendons. Shape is (num_instances, num_fixed_tendons, 2)."""

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

    Note: The torques are zero for implicit actuator models.
    """

    applied_torque: torch.Tensor = None
    """Joint torques applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

    These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
    actuator model.

    Note: The torques are zero for implicit actuator models.
    """

    ##
    # Joint properties.
    ##

    joint_stiffness: torch.Tensor = None
    """Joint stiffness provided to simulation. Shape is (num_instances, num_joints)."""

    joint_damping: torch.Tensor = None
    """Joint damping provided to simulation. Shape is (num_instances, num_joints)."""

    joint_armature: torch.Tensor = None
    """Joint armature provided to simulation. Shape is (num_instances, num_joints)."""

    joint_friction: torch.Tensor = None
    """Joint friction provided to simulation. Shape is (num_instances, num_joints)."""

    joint_limits: torch.Tensor = None
    """Joint limits provided to simulation. Shape is (num_instances, num_joints, 2)."""

    ##
    # Fixed tendon properties.
    ##

    fixed_tendon_stiffness: torch.Tensor = None
    """Fixed tendon stiffness provided to simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_damping: torch.Tensor = None
    """Fixed tendon damping provided to simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_limit_stiffness: torch.Tensor = None
    """Fixed tendon limit stiffness provided to simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_rest_length: torch.Tensor = None
    """Fixed tendon rest length provided to simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_offset: torch.Tensor = None
    """Fixed tendon offset provided to simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_limit: torch.Tensor = None
    """Fixed tendon limits provided to simulation. Shape is (num_instances, num_fixed_tendons, 2)."""

    ##
    # Other Data.
    ##

    soft_joint_pos_limits: torch.Tensor = None
    """Joint positions limits for all joints. Shape is (num_instances, num_joints, 2)."""

    soft_joint_vel_limits: torch.Tensor = None
    """Joint velocity limits for all joints. Shape is (num_instances, num_joints)."""

    gear_ratio: torch.Tensor = None
    """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""

    ##
    # Properties.
    ##

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
        velocities are of the articulation root's center of mass frame.
        """
        if self._root_state_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_physx_view.get_root_transforms().clone()
            pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            velocity = self._root_physx_view.get_root_velocities()
            # set the buffer data and timestamp
            self._root_state_w.data = torch.cat((pose, velocity), dim=-1)
            self._root_state_w.timestamp = self._sim_timestamp
        return self._root_state_w.data

    @property
    def body_state_w(self):
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position and quaternion are of all the articulation links's actor frame. Meanwhile, the linear and angular
        velocities are of the articulation links's center of mass frame.
        """
        if self._body_state_w.timestamp < self._sim_timestamp:
            # read data from simulation
            poses = self._root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            velocities = self._root_physx_view.get_link_velocities()
            # set the buffer data and timestamp
            self._body_state_w.data = torch.cat((poses, velocities), dim=-1)
            self._body_state_w.timestamp = self._sim_timestamp
        return self._body_state_w.data

    @property
    def body_acc_w(self):
        """Acceleration of all bodies. Shape is (num_instances, num_bodies, 6).

        This quantity is the acceleration of the articulation links' center of mass frame.
        """
        if self._body_acc_w.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._body_acc_w.data = self._root_physx_view.get_link_accelerations()
            self._body_acc_w.timestamp = self._sim_timestamp
        return self._body_acc_w.data

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        forward_w = math_utils.quat_apply(self.root_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

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
    # Derived properties.
    ##

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the articulation root.
        """
        return self.root_state_w[:, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the articulation root.
        """
        return self.root_state_w[:, 3:7]

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of
        mass frame.
        """
        return self.root_state_w[:, 7:13]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame.
        """
        return self.root_state_w[:, 7:10]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame.
        """
        return self.root_state_w[:, 10:13]

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with
        respect to the articulation root's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.root_lin_vel_w)

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        articulation root's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.root_ang_vel_w)

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the rigid bodies' actor frame.
        """
        return self.body_state_w[..., :3]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the rigid bodies' actor frame.
        """
        return self.body_state_w[..., 3:7]

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the rigid bodies' center of mass frame.
        """
        return self.body_state_w[..., 7:13]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        return self.body_state_w[..., 7:10]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        return self.body_state_w[..., 10:13]

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        return self.body_acc_w[..., 0:3]

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        return self.body_acc_w[..., 3:6]
