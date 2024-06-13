# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.utils.buffers import TimestampedBuffer

from ..rigid_object import RigidObjectData


class ArticulationData(RigidObjectData):
    """Data container for an articulation."""

    def __init__(self, root_physx_view: physx.ArticulationView, device):
        super().__init__(root_physx_view, device)
        self._root_physx_view: physx.ArticulationView = root_physx_view

        self._previous_joint_vel = self._root_physx_view.get_dof_velocities().clone()

        # Initialize the lazy buffers.
        self._body_state_w: TimestampedBuffer = TimestampedBuffer()
        self._joint_pos: TimestampedBuffer = TimestampedBuffer()
        self._joint_acc: TimestampedBuffer = TimestampedBuffer()
        self._joint_vel: TimestampedBuffer = TimestampedBuffer()

    def update(self, dt: float):
        self._time_stamp += dt
        # Trigger an update of the joint acceleration buffer at a higher frequency since we do finite differencing.
        self.joint_acc

    joint_names: list[str] = None
    """Joint names in the order parsed by the simulation view."""

    ##
    # Defaults.
    ##

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
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13)."""
        if self._root_state_w.update_timestamp < self._time_stamp:
            pose = self._root_physx_view.get_root_transforms().clone()
            pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            velocity = self._root_physx_view.get_root_velocities()
            self._root_state_w.data = torch.cat((pose, velocity), dim=-1)
            self._root_state_w.update_timestamp = self._time_stamp
        return self._root_state_w.data

    @property
    def body_state_w(self):
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13)."""
        if self._body_state_w.update_timestamp < self._time_stamp:
            poses = self._root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            velocities = self._root_physx_view.get_link_velocities()
            self._body_state_w.data = torch.cat((poses, velocities), dim=-1)
            self._body_state_w.update_timestamp = self._time_stamp
        return self._body_state_w.data

    @property
    def body_acc_w(self):
        """Acceleration of all bodies. Shape is (num_instances, num_bodies, 6)."""
        if self._body_acc_w.update_timestamp < self._time_stamp:
            self._body_acc_w.data = self._root_physx_view.get_link_accelerations()
            self._body_acc_w.update_timestamp = self._time_stamp
        return self._body_acc_w.data

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3)."""
        return self.body_acc_w[..., 0:3]

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3)."""
        return self.body_acc_w[..., 3:6]

    @property
    def joint_pos(self):
        """Joint positions of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_pos.update_timestamp < self._time_stamp:
            self._joint_pos.data = self._root_physx_view.get_dof_positions()
            self._joint_pos.update_timestamp = self._time_stamp
        return self._joint_pos.data

    @property
    def joint_vel(self):
        """Joint velocities of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_vel.update_timestamp < self._time_stamp:
            self._joint_vel.data = self._root_physx_view.get_dof_velocities()
            self._joint_vel.update_timestamp = self._time_stamp
        return self._joint_vel.data

    @property
    def joint_acc(self):
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_acc.update_timestamp < self._time_stamp:
            self._joint_acc.data = (self.joint_vel - self._previous_joint_vel) / (
                self._time_stamp - self._joint_acc.update_timestamp
            )
            self._previous_joint_vel[:] = self.joint_vel
            self._joint_acc.update_timestamp = self._time_stamp
        return self._joint_acc.data
