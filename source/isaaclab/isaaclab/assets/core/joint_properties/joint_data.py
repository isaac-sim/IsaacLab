# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import weakref

import warp as wp

from newton.selection import ArticulationView as NewtonArticulationView
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.buffers import TimestampedWarpBuffer
from isaaclab.utils.helpers import deprecated, warn_overhead_cost

from isaaclab.assets.core.kernels import (
    derive_joint_acceleration_from_velocity,
    make_joint_pos_limits_from_lower_and_upper_limits,
)


class JointData:
    def __init__(self, root_newton_view: NewtonArticulationView, device: str) -> None:
        """Initializes the container for joint data.

        Args:
            root_newton_view: The root articulation view.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_newton_view: NewtonArticulationView = weakref.proxy(root_newton_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0

        self._create_simulation_bindings()
        self._create_buffers()


    def update(self, dt: float):
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the joint acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.joint_acc

    def _create_simulation_bindings(self):
        """Create simulation bindings for the joint data.

        Direct simulation bindings are pointers to the simulation data, their data is not copied, and should
        only be updated using warp kernels. Any modifications made to the bindings will be reflected in the simulation.
        Hence we encourage users to carefully think about the data they modify and in which order it should be updated.

        .. caution:: This is possible if and only if the properties that we access are strided from newton and not
        indexed. Newton willing this is the case all the time, but we should pay attention to this if things look off.
        """

        # -- joint properties
        self._sim_bind_joint_pos_limits_lower = self._root_newton_view.get_attribute(
            "joint_limit_lower", NewtonManager.get_model()
        )
        self._sim_bind_joint_pos_limits_upper = self._root_newton_view.get_attribute(
            "joint_limit_upper", NewtonManager.get_model()
        )
        self._sim_bind_joint_stiffness_sim = self._root_newton_view.get_attribute(
            "joint_target_ke", NewtonManager.get_model()
        )
        self._sim_bind_joint_damping_sim = self._root_newton_view.get_attribute(
            "joint_target_kd", NewtonManager.get_model()
        )
        self._sim_bind_joint_armature = self._root_newton_view.get_attribute(
            "joint_armature", NewtonManager.get_model()
        )
        self._sim_bind_joint_friction_coeff = self._root_newton_view.get_attribute(
            "joint_friction", NewtonManager.get_model()
        )
        self._sim_bind_joint_vel_limits_sim = self._root_newton_view.get_attribute(
            "joint_velocity_limit", NewtonManager.get_model()
        )
        self._sim_bind_joint_effort_limits_sim = self._root_newton_view.get_attribute(
            "joint_effort_limit", NewtonManager.get_model()
        )
        self._sim_bind_joint_control_mode_sim = self._root_newton_view.get_attribute(
            "joint_dof_mode", NewtonManager.get_model()
        )
        # -- joint states
        self._sim_bind_joint_pos = self._root_newton_view.get_dof_positions(NewtonManager.get_state_0())
        self._sim_bind_joint_vel = self._root_newton_view.get_dof_velocities(NewtonManager.get_state_0())
        # -- joint commands (sent to the simulation)
        self._sim_bind_joint_effort = self._root_newton_view.get_attribute("joint_f", NewtonManager.get_control())
        self._sim_bind_joint_target = self._root_newton_view.get_attribute(
            "joint_target", NewtonManager.get_control()
        )

    def _create_buffers(self):
        """Create buffers for the joint data."""
        # -- default joint positions and velocities
        self._default_joint_pos = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.float32, device=self.device
        )
        self._default_joint_vel = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.float32, device=self.device
        )
        # -- joint commands (sent to the actuator from the user)
        self._joint_target = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.float32, device=self.device
        )
        self._joint_effort_target = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.float32, device=self.device
        )
        # -- computed joint efforts from the actuator models
        self._computed_effort = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.float32, device=self.device
        )
        self._applied_effort = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.float32, device=self.device
        )
        # -- joint properties for the actuator models
        self._joint_stiffness = wp.clone(self._sim_bind_joint_stiffness_sim)
        self._joint_damping = wp.clone(self._sim_bind_joint_damping_sim)
        self._joint_control_mode = wp.clone(self._sim_bind_joint_control_mode_sim)
        # -- other data that are filled based on explicit actuator models
        self._joint_dynamic_friction = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.float32, device=self.device
        )
        self._joint_viscous_friction = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.float32, device=self.device
        )
        self._soft_joint_vel_limits = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.float32, device=self.device
        )
        self._gear_ratio = wp.ones(
            (self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.float32, device=self.device
        )
        # -- update the soft joint position limits
        self._soft_joint_pos_limits = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.vec2f, device=self.device
        )

        # Initialize history for finite differencing
        self._previous_joint_vel = wp.clone(self._root_newton_view.get_dof_velocities(NewtonManager.get_state_0()))

        # Initialize the lazy buffers.
        # -- joint state
        self._joint_acc = TimestampedWarpBuffer(
            shape=(self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.float32
        )
        # self._body_incoming_joint_wrench_b = TimestampedWarpBuffer()

    ##
    # Direct simulation bindings accessors.
    ##

    @property
    def joint_control_mode_sim(self) -> wp.array:
        return self._sim_bind_joint_control_mode_sim

    @property
    def joint_stiffness_sim(self) -> wp.array:
        """Joint stiffness as defined in the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_stiffness_sim

    @property
    def joint_damping_sim(self) -> wp.array:
        """Joint damping as defined in the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_damping_sim

    @property
    def joint_armature(self) -> wp.array:
        """Joint armature as defined in the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_armature

    @property
    def joint_friction_coeff(self) -> wp.array:
        """Joint friction coefficient as defined in the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_friction_coeff

    @property
    def joint_pos_limits_lower(self) -> wp.array:
        """Joint position limits lower as defined in the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_pos_limits_lower

    @property
    def joint_pos_limits_upper(self) -> wp.array:
        """Joint position limits upper as defined in the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_pos_limits_upper

    @property
    def joint_vel_limits(self) -> wp.array:
        """Joint velocity limits as defined in the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_vel_limits_sim

    @property
    def joint_effort_limits(self) -> wp.array:
        """Joint effort limits as defined in the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_effort_limits_sim

    @property
    def joint_pos(self) -> wp.array:
        """Joint posiitons. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_pos

    @property
    def joint_vel(self) -> wp.array:
        """Joint velocities. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_vel

    @property
    def joint_target_sim(self) -> wp.array:
        """Joint targets in the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_target

    @property
    def joint_effort_sim(self) -> wp.array:
        """Joint effort in the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_effort

    ###
    # Buffers accessors.
    ##

    @property
    def joint_target(self) -> wp.array:
        """Joint targets commanded by the user. Shape is (num_instances, num_joints)."""
        return self._joint_target

    @property
    def joint_effort_target(self) -> wp.array:
        """Joint effort targets commanded by the user. Shape is (num_instances, num_joints)."""
        return self._joint_effort_target

    @property
    def computed_effort(self) -> wp.array:
        """Joint efforts computed from the actuator model (before clipping). Shape is (num_instances, num_joints)."""
        return self._computed_effort

    @property
    def applied_effort(self) -> wp.array:
        """Joint efforts applied from the actuator model (after clipping). Shape is (num_instances, num_joints)."""
        return self._applied_effort

    @property
    def joint_stiffness(self) -> wp.array:
        """Joint stiffness as defined in the actuator model. Shape is (num_instances, num_joints)."""
        return self._joint_stiffness

    @property
    def joint_damping(self) -> wp.array:
        """Joint damping as defined in the actuator model. Shape is (num_instances, num_joints)."""
        return self._joint_damping

    @property
    def joint_control_mode(self) -> wp.array:
        """Joint control mode as defined in the actuator model. Shape is (num_instances, num_joints)."""
        return self._joint_control_mode

    @property
    def joint_dynamic_friction(self) -> wp.array:
        """Joint dynamic friction as defined in the actuator model. Shape is (num_instances, num_joints)."""
        return self._joint_dynamic_friction

    @property
    def joint_viscous_friction(self) -> wp.array:
        """Joint viscous friction as defined in the actuator model. Shape is (num_instances, num_joints)."""
        return self._joint_viscous_friction

    @property
    def soft_joint_vel_limits(self) -> wp.array:
        """Soft joint velocity limits for all joints. Shape is (num_instances, num_joints).

        These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
        has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
        """
        return self._soft_joint_vel_limits

    @property
    def gear_ratio(self) -> wp.array:
        """Gear ratio as defined in the actuator model. Shape is (num_instances, num_joints)."""
        return self._gear_ratio

    @property
    def soft_joint_pos_limits(self) -> wp.array:
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
        return self._soft_joint_pos_limits

    ##
    # Default accessors.
    ##
    
    @property
    def default_joint_pos(self) -> wp.array:
        """Default joint positions of all joints. Shape is (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_joint_pos
    
    @property
    def default_joint_vel(self) -> wp.array:
        """Default joint velocities of all joints. Shape is (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_joint_vel

    @default_joint_pos.setter
    def default_joint_pos(self, value: wp.array) -> None:
        self._default_joint_pos = value
    
    @default_joint_vel.setter
    def default_joint_vel(self, value: wp.array) -> None:
        self._default_joint_vel = value

    ##
    # Joint state properties.
    ##

    @property
    def joint_acc(self) -> wp.array:
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_acc.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            wp.launch(
                derive_joint_acceleration_from_velocity,
                dim=(self._root_newton_view.count, self._root_newton_view.joint_dof_count),
                inputs=[
                    self._sim_bind_joint_vel,
                    self._previous_joint_vel,
                    NewtonManager.get_dt(),
                    self._joint_acc.data,
                ],
            )
            self._joint_acc.timestamp = self._sim_timestamp
        return self._joint_acc.data

    @property
    def body_incoming_joint_wrench_b(self) -> wp.array:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.

        For more information on joint wrenches, please check the`PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force>`__
        and the underlying `PhysX Tensor API <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force>`__ .
        """
        raise NotImplementedError("Body incoming joint wrench in body frame is not implemented for Newton.")
        if self._body_incoming_joint_wrench_b.timestamp < self._sim_timestamp:
            self._body_incoming_joint_wrench_b.data = self._root_physx_view.get_link_incoming_joint_force()
            self._body_incoming_joint_wrench_b.time_stamp = self._sim_timestamp
        return self._body_incoming_joint_wrench_b.data

    ##
    # Sliced properties.
    ##

    @property
    @warn_overhead_cost(
        "joint_pos_limits_lower or joint_pos_limits_upper",
        "This function combines both the lower and upper limits into a single array, use it only if necessary.",
    )
    def joint_pos_limits(self) -> wp.array:
        """Joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.joint_dof_count), dtype=wp.vec2f, device=self.device)
        wp.launch(
            make_joint_pos_limits_from_lower_and_upper_limits,
            dim=(self._root_newton_view.count, self._root_newton_view.joint_dof_count),
            inputs=[
                self._sim_bind_joint_pos_limits_lower,
                self._sim_bind_joint_pos_limits_upper,
                out,
            ],
        )
        return out

    ##
    # Backward compatibility. Need to nuke these properties in a future release.
    ##

    @property
    @deprecated("joint_pos_limits")
    def joint_limits(self) -> wp.array:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead."""
        return self.joint_pos_limits

    @property
    @deprecated("joint_friction_coeff")
    def joint_friction(self) -> wp.array:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead."""
        return self._sim_bind_joint_friction_coeff
