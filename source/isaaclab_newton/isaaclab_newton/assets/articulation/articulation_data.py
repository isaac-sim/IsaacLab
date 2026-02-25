# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import logging
import torch
import weakref

import warp as wp
from isaaclab_newton.kernels import (
    combine_frame_transforms_partial_batch,
    combine_frame_transforms_partial_root,
    combine_pose_and_velocity_to_state,
    combine_pose_and_velocity_to_state_batched,
    compute_heading,
    derive_body_acceleration_from_velocity_batched,
    derive_joint_acceleration_from_velocity,
    generate_pose_from_position_with_unit_quaternion_batched,
    make_joint_pos_limits_from_lower_and_upper_limits,
    project_com_velocity_to_link_frame_batch,
    project_com_velocity_to_link_frame_root,
    project_vec_from_pose_single,
    project_velocities_to_frame,
    split_spatial_vectory_array_to_angular_velocity_array,
    split_spatial_vectory_array_to_linear_velocity_array,
    split_spatial_vectory_batched_array_to_angular_velocity_batched_array,
    split_spatial_vectory_batched_array_to_linear_velocity_batched_array,
    split_transform_array_to_position_array,
    split_transform_array_to_quaternion_array,
    split_transform_batched_array_to_position_batched_array,
    split_transform_batched_array_to_quaternion_batched_array,
    vec13f,
)
from isaaclab_newton.physics import NewtonManager
from newton.selection import ArticulationView as NewtonArticulationView

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.utils.buffers import TimestampedWarpBuffer
from isaaclab.utils.helpers import deprecated, warn_overhead_cost

# import logger
logger = logging.getLogger(__name__)


class ArticulationData(BaseArticulationData):
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

    ..note:: This class is implemented so that all the properties can be accessed as either a Torch tensor or a Warp
    array. However, all the operations are performed on Warp arrays. To enable this, there is a set of internal
    only helper functions that perform all the warp operations. Internal classes depending on this class should not use
    properties directly! This is because they can either be Torch tensors or Warp arrays, and all the internal
    operations should be performed on the Warp arrays. Hence, internal operations should instead use the "private"
    helper functions / attributes.
    """

    def __init__(self, root_view, device: str):
        """Initializes the articulation data.

        Args:
            root_view: The root articulation view.
            device: The device used for processing.
            frontend: The frontend to use for the data.
        """
        # Set the parameters
        self.device = device
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_view: NewtonArticulationView = weakref.proxy(root_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0
        self._is_primed = False
        # obtain global simulation view
        gravity = wp.to_torch(NewtonManager.get_model().gravity)[0]
        gravity_dir = math_utils.normalize(gravity.unsqueeze(0)).squeeze(0)
        # Initialize constants
        self.GRAVITY_VEC_W = wp.vec3f(gravity_dir[0], gravity_dir[1], gravity_dir[2])
        self.GRAVITY_VEC_W_TORCH = torch.tensor([gravity_dir[0], gravity_dir[1], gravity_dir[2]], device=device).repeat(
            self._root_view.count, 1
        )
        self.FORWARD_VEC_B = wp.vec3f((1.0, 0.0, 0.0))
        self.FORWARD_VEC_B_TORCH = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(self._root_view.count, 1)
        # Create the simulation bindings and buffers
        self._create_simulation_bindings()
        self._create_buffers()

    @property
    def is_primed(self) -> bool:
        """Whether the articulation data is fully instantiated and ready to use."""
        return self._is_primed

    @is_primed.setter
    def is_primed(self, value: bool):
        """Set whether the articulation data is fully instantiated and ready to use.

        ..note:: Once this quantity is set to True, it cannot be changed.

        Args:
            value: Whether the articulation data is fully instantiated and ready to use.

        Raises:
            RuntimeError: If the articulation data is already fully instantiated and ready to use.
        """
        if self._is_primed:
            raise RuntimeError("Cannot set is_primed after instantiation.")
        self._is_primed = value

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
    # Defaults.
    ##

    @property
    def default_root_pose(self) -> wp.array(dtype=wp.transformf):
        """Default root pose ``[pos, quat]`` in the local environment frame. Shape is (num_instances, 7).

        The position and quaternion are of the articulation root's actor frame.

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_root_pose

    @default_root_pose.setter
    def default_root_pose(self, value: wp.array(dtype=wp.transformf)):
        """Set default root pose ``[pos, quat]`` in the local environment frame.

        ..note:: Once this quantity is set to True, it cannot be changed.

        Args:
            value: Default root pose ``[pos, quat]`` in the local environment frame.

        Raises:
            RuntimeError: If the articulation data is already fully instantiated and ready to use.
        """
        if self._is_primed:
            raise RuntimeError("Cannot set default root pose after instantiation.")
        self._default_root_pose = value

    @property
    def default_root_vel(self) -> wp.array(dtype=wp.spatial_vectorf):
        """Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame. Shape is (num_instances, 6).

        The linear and angular velocities are of the articulation root's center of mass frame.

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_root_vel

    @default_root_vel.setter
    def default_root_vel(self, value: wp.array(dtype=wp.spatial_vectorf)):
        """Set default root velocity ``[lin_vel, ang_vel]`` in the local environment frame.

        ..note:: Once this quantity is set to True, it cannot be changed.

        Args:
            value: Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame.

        Raises:
            RuntimeError: If the articulation data is already fully instantiated and ready to use.
        """
        if self._is_primed:
            raise RuntimeError("Cannot set default root velocity after instantiation.")
        self._default_root_vel = value

    @property
    def default_joint_pos(self) -> wp.array(dtype=wp.float32):
        """Default joint positions of all joints. Shape is (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_joint_pos

    @default_joint_pos.setter
    def default_joint_pos(self, value: wp.array(dtype=wp.float32)):
        """Set default joint positions of all joints.

        ..note:: Once this quantity is set to True, it cannot be changed.

        Args:
            value: Default joint positions of all joints.

        Raises:
            RuntimeError: If the articulation data is already fully instantiated and ready to use.
        """
        if self._is_primed:
            raise RuntimeError("Cannot set default joint positions after instantiation.")
        self._default_joint_pos = value

    @property
    def default_joint_vel(self) -> wp.array(dtype=wp.float32):
        """Default joint velocities of all joints. Shape is (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_joint_vel

    @default_joint_vel.setter
    def default_joint_vel(self, value: wp.array(dtype=wp.float32)):
        """Set default joint velocities of all joints.

        ..note:: Once this quantity is set to True, it cannot be changed.

        Args:
            value: Default joint velocities of all joints.

        Raises:
            RuntimeError: If the articulation data is already fully instantiated and ready to use.
        """
        if self._is_primed:
            raise RuntimeError("Cannot set default joint velocities after instantiation.")
        self._default_joint_vel = value

    ##
    # Joint commands. -- Set into the simulation
    ##

    @property
    def joint_pos_target(self) -> wp.array(dtype=wp.float32):
        return self._sim_bind_joint_position_target

    @property
    def joint_vel_target(self) -> wp.array(dtype=wp.float32):
        return self._sim_bind_joint_velocity_target

    @property
    def joint_effort(self) -> wp.array(dtype=wp.float32):
        """Joint effort. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_effort

    ##
    # Joint commands -- Explicit actuators.
    ##

    @property
    def computed_effort(self) -> wp.array(dtype=wp.float32):
        """Joint efforts computed from the actuator model (before clipping). Shape is (num_instances, num_joints).

        This quantity is the raw effort output from the actuator mode, before any clipping is applied.
        It is exposed for users who want to inspect the computations inside the actuator model.
        For instance, to penalize the learning agent for a difference between the computed and applied torques.
        """
        return self._computed_effort

    @property
    def applied_effort(self) -> wp.array(dtype=wp.float32):
        """Joint efforts applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

        These efforts are set into the simulation, after clipping the :attr:`computed_effort` based on the
        actuator model.
        """
        return self._applied_effort

    @property
    def actuator_stiffness(self) -> wp.array(dtype=wp.float32):
        """Actuator stiffness. Shape is (num_instances, num_joints)."""
        return self._actuator_stiffness

    @property
    def actuator_damping(self) -> wp.array(dtype=wp.float32):
        """Actuator damping. Shape is (num_instances, num_joints)."""
        return self._actuator_damping

    @property
    def actuator_position_target(self) -> wp.array(dtype=wp.float32):
        """Actuator position targets commanded by the user. Shape is (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint efforts (see :attr:`applied_torque`),
        which are then set into the simulation.

        Note: This is the value requested by the user. This is not the value binded to the simulation.
        """
        return self._actuator_position_target

    @property
    def actuator_velocity_target(self) -> wp.array(dtype=wp.float32):
        """Actuator velocity targets commanded by the user. Shape is (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint efforts (see :attr:`applied_torque`),
        which are then set into the simulation.

        Note: This is the value requested by the user. This is not the value binded to the simulation.
        """
        return self._actuator_velocity_target

    @property
    def actuator_effort_target(self) -> wp.array(dtype=wp.float32):
        """Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint efforts (see :attr:`applied_torque`),
        which are then set into the simulation.

        Note: This is the value requested by the user. This is not the value binded to the simulation.
        """
        return self._actuator_effort_target

    ##
    # Joint properties. -- Set into the simulation
    ##

    @property
    def joint_stiffness(self) -> wp.array(dtype=wp.float32):
        """Joint stiffness provided to the simulation. Shape is (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._sim_bind_joint_stiffness_sim

    @property
    def joint_damping(self) -> wp.array(dtype=wp.float32):
        """Joint damping provided to the simulation. Shape is (num_instances, num_joints)

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._sim_bind_joint_damping_sim

    @property
    def joint_armature(self) -> wp.array(dtype=wp.float32):
        """Joint armature provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_armature

    @property
    def joint_friction_coeff(self) -> wp.array(dtype=wp.float32):
        """Joint friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_friction_coeff

    @property
    def joint_pos_limits_lower(self) -> wp.array(dtype=wp.float32):
        """Joint position limits lower provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_pos_limits_lower

    @property
    def joint_pos_limits_upper(self) -> wp.array(dtype=wp.float32):
        """Joint position limits upper provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_pos_limits_upper

    @property
    @warn_overhead_cost(
        "joint_pos_limits",
        "Launches a kernel to compute the joint position limits from the lower and upper limits. Consider using the"
        " joint_pos_limits_lower and joint_pos_limits_upper properties instead.",
    )
    def joint_pos_limits(self) -> wp.array(dtype=wp.vec2f):
        """Joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.

        .. caution:: This property is computed on-the-fly, and while it returns a pointer, writing to that pointer
        will not affect change the joint position limits. To change the joint position limits, use the
        :attr:`joint_pos_limits_lower` and :attr:`joint_pos_limits_upper` properties.
        """
        if self._joint_pos_limits is None:
            self._joint_pos_limits = wp.zeros(
                (self._root_view.count, self._root_view.joint_dof_count), dtype=wp.vec2f, device=self.device
            )

        if self._root_view.joint_dof_count > 0:
            wp.launch(
                make_joint_pos_limits_from_lower_and_upper_limits,
                dim=(self._root_view.count, self._root_view.joint_dof_count),
                inputs=[
                    self._sim_bind_joint_pos_limits_lower,
                    self._sim_bind_joint_pos_limits_upper,
                    self._joint_pos_limits,
                ],
                device=self.device,
            )
        return self._joint_pos_limits

    @property
    def joint_vel_limits(self) -> wp.array(dtype=wp.float32):
        """Joint maximum velocity provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_vel_limits_sim

    @property
    def joint_effort_limits(self) -> wp.array(dtype=wp.float32):
        """Joint maximum effort provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_effort_limits_sim

    ##
    # Joint properties - Custom.
    ##

    @property
    def joint_dynamic_friction_coeff(self) -> wp.array(dtype=wp.float32):
        """Joint dynamic friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_dynamic_friction

    @property
    def joint_viscous_friction_coeff(self) -> wp.array(dtype=wp.float32):
        """Joint viscous friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_viscous_friction

    @property
    def soft_joint_pos_limits(self) -> wp.array(dtype=wp.vec2f):
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

    @property
    def soft_joint_vel_limits(self) -> wp.array(dtype=wp.float32):
        """Soft joint velocity limits for all joints. Shape is (num_instances, num_joints).

        These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
        has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
        """
        return self._soft_joint_vel_limits

    @property
    def gear_ratio(self) -> wp.array(dtype=wp.float32):  # TODO: Mayank got some comments
        """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""
        return self._gear_ratio

    ##
    # Fixed tendon properties.
    ##

    @property
    def fixed_tendon_stiffness(self) -> wp.array(dtype=wp.float32):
        """Fixed tendon stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        raise NotImplementedError("Fixed tendon stiffness is not supported in Newton.")

    @property
    def fixed_tendon_damping(self) -> wp.array(dtype=wp.float32):
        """Fixed tendon damping provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        raise NotImplementedError("Fixed tendon damping is not supported in Newton.")

    @property
    def fixed_tendon_limit_stiffness(self) -> wp.array(dtype=wp.float32):
        """Fixed tendon limit stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        raise NotImplementedError("Fixed tendon limit stiffness is not supported in Newton.")

    @property
    def fixed_tendon_rest_length(self) -> wp.array(dtype=wp.float32):
        """Fixed tendon rest length provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        raise NotImplementedError("Fixed tendon rest length is not supported in Newton.")

    @property
    def fixed_tendon_offset(self) -> wp.array(dtype=wp.float32):
        """Fixed tendon offset provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        raise NotImplementedError("Fixed tendon offset is not supported in Newton.")

    @property
    def fixed_tendon_pos_limits(self) -> wp.array(dtype=wp.float32):
        """Fixed tendon position limits provided to the simulation. Shape is (num_instances, num_fixed_tendons, 2)."""
        raise NotImplementedError("Fixed tendon position limits is not supported in Newton.")

    ##
    # Spatial tendon properties.
    ##

    @property
    def spatial_tendon_stiffness(self) -> wp.array(dtype=wp.float32):
        """Spatial tendon stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        raise NotImplementedError("Spatial tendon stiffness is not supported in Newton.")

    @property
    def spatial_tendon_damping(self) -> wp.array(dtype=wp.float32):
        """Spatial tendon damping provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        raise NotImplementedError("Spatial tendon damping is not supported in Newton.")

    @property
    def spatial_tendon_limit_stiffness(self) -> wp.array(dtype=wp.float32):
        """Spatial tendon limit stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        raise NotImplementedError("Spatial tendon limit stiffness is not supported in Newton.")

    @property
    def spatial_tendon_offset(self) -> wp.array(dtype=wp.float32):
        """Spatial tendon offset provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        raise NotImplementedError("Spatial tendon offset is not supported in Newton.")

    ##
    # Root state properties.
    ##

    @property
    def root_link_pose_w(self) -> wp.array(dtype=wp.transformf):
        """Root link pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return self._sim_bind_root_link_pose_w

    @property
    def root_link_vel_w(self) -> wp.array(dtype=wp.spatial_vectorf):
        """Root link velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances,). Velocities are in the form of [vx, vy, vz, wx, wy, wz].
        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                project_com_velocity_to_link_frame_root,
                dim=(self._root_view.count),
                device=self.device,
                inputs=[
                    self._sim_bind_root_com_vel_w,
                    self._sim_bind_root_link_pose_w,
                    self._sim_bind_body_com_pos_b,
                    self._root_link_vel_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w.data

    @property
    def root_com_pose_w(self) -> wp.array(dtype=wp.transformf):
        """Root center of mass pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            wp.launch(
                combine_frame_transforms_partial_root,
                dim=(self._root_view.count),
                device=self.device,
                inputs=[
                    self._sim_bind_root_link_pose_w,
                    self._sim_bind_body_com_pos_b,
                    self._root_com_pose_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._root_com_pose_w.timestamp = self._sim_timestamp

        return self._root_com_pose_w.data

    @property
    def root_com_vel_w(self) -> wp.array(dtype=wp.spatial_vectorf):
        """Root center of mass velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances,). The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        return self._sim_bind_root_com_vel_w

    @property
    @warn_overhead_cost(
        "root_link_pose_w or root_com_vel_w",
        "Launches a kernel to merge a pose and a velocity into a state. Consider using the pose and velocity arrays"
        " directly instead.",
    )
    @deprecated("root_link_pose_w or root_com_vel_w", since="3.0.0", remove_in="4.0.0")
    def root_state_w(self) -> wp.array(dtype=vec13f):
        """Root state ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances,), (num_instances,). The pose is in the form of [pos, quat].
        The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        The pose is of the articulation root's actor frame relative to the world.
        The velocity is of the articulation root's center of mass frame.
        """
        if self._root_state_w is None:
            self._root_state_w = wp.zeros((self._root_view.count), dtype=vec13f, device=self.device)
        wp.launch(
            combine_pose_and_velocity_to_state,
            dim=(self._root_view.count,),
            device=self.device,
            inputs=[
                self._sim_bind_root_link_pose_w,
                self._sim_bind_root_com_vel_w,
                self._root_state_w,
            ],
        )
        return self._root_state_w

    @property
    @warn_overhead_cost(
        "root_link_pose_w or root_link_vel_w",
        "Launches a kernel to merge a pose and a velocity into a state. Consider using the pose and velocity arrays"
        " directly instead.",
    )
    @deprecated("root_link_pose_w or root_link_vel_w", since="3.0.0", remove_in="4.0.0")
    def root_link_state_w(self) -> wp.array(dtype=vec13f):
        """Root link state ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances,), (num_instances,). The pose is in the form of [pos, quat].
        The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        The pose is of the articulation root's actor frame relative to the world.
        The velocity is of the articulation root's actor frame.
        """
        if self._root_link_state_w is None:
            self._root_link_state_w = wp.zeros((self._root_view.count), dtype=vec13f, device=self.device)

        wp.launch(
            combine_pose_and_velocity_to_state,
            dim=(self._root_view.count,),
            device=self.device,
            inputs=[
                self._sim_bind_root_link_pose_w,
                self.root_link_vel_w,
                self._root_link_state_w,
            ],
        )
        return self._root_link_state_w

    @property
    @warn_overhead_cost(
        "root_com_pose_w or root_com_vel_w",
        "Launches a kernel to merge a pose and a velocity into a state. Consider using the pose and velocity arrays"
        " directly instead.",
    )
    @deprecated("root_com_pose_w or root_com_vel_w", since="3.0.0", remove_in="4.0.0")
    def root_com_state_w(self) -> wp.array(dtype=vec13f):
        """Root center of mass state ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances,), (num_instances,). The pose is in the form of [pos, quat].
        The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        The pose is of the articulation root's center of mass frame relative to the world.
        The velocity is of the articulation root's center of mass frame.
        """
        if self._root_com_state_w is None:
            self._root_com_state_w = wp.zeros((self._root_view.count), dtype=vec13f, device=self.device)

        wp.launch(
            combine_pose_and_velocity_to_state,
            dim=(self._root_view.count,),
            device=self.device,
            inputs=[
                self.root_com_pose_w,
                self._sim_bind_root_com_vel_w,
                self._root_com_state_w,
            ],
        )
        return self._root_com_state_w

    ##
    # Body state properties.
    ##

    @property
    def body_mass(self) -> wp.array(dtype=wp.float32):
        """Body mass ``wp.float32`` in the world frame. Shape is (num_instances, num_bodies)."""
        return self._sim_bind_body_mass

    @property
    def body_inertia(self) -> wp.array(dtype=wp.mat33f):
        """Body inertia ``wp.mat33`` in the world frame. Shape is (num_instances, num_bodies, 3, 3)."""
        return self._sim_bind_body_inertia

    @property
    def body_link_pose_w(self) -> wp.array(dtype=wp.transformf):
        """Body link pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return self._sim_bind_body_link_pose_w

    @property
    def body_link_vel_w(self) -> wp.array(dtype=wp.spatial_vectorf):
        """Body link velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). Velocities are in the form of [vx, vy, vz, wx, wy, wz].
        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        if self._body_link_vel_w.timestamp < self._sim_timestamp:
            # Project the velocity from the center of mass frame to the link frame
            wp.launch(
                project_com_velocity_to_link_frame_batch,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    self._sim_bind_body_com_vel_w,
                    self._sim_bind_body_link_pose_w,
                    self._sim_bind_body_com_pos_b,
                    self._body_link_vel_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._body_link_vel_w.timestamp = self._sim_timestamp
        return self._body_link_vel_w.data

    @property
    def body_com_pose_w(self) -> wp.array(dtype=wp.transformf):
        """Body center of mass pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            # Apply local transform to center of mass frame
            wp.launch(
                combine_frame_transforms_partial_batch,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    self._sim_bind_body_link_pose_w,
                    self._sim_bind_body_com_pos_b,
                    self._body_com_pose_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._body_com_pose_w.timestamp = self._sim_timestamp
        return self._body_com_pose_w.data

    @property
    def body_com_vel_w(self) -> wp.array(dtype=wp.spatial_vectorf):
        """Body center of mass velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). Velocities are in the form of [vx, vy, vz, wx, wy, wz].
        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        return self._sim_bind_body_com_vel_w

    @property
    @warn_overhead_cost(
        "body_link_pose_w or body_com_vel_w",
        "Launches a kernel to merge a pose and a velocity into a state. Consider using the pose and velocity arrays"
        " directly instead.",
    )
    @deprecated("body_link_pose_w or body_com_vel_w", since="3.0.0", remove_in="4.0.0")
    def body_state_w(self) -> wp.array(dtype=vec13f):
        """State of all bodies ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        The pose is of the articulation links' actor frame relative to the world.
        The velocity is of the articulation links' center of mass frame.
        """
        state = wp.zeros((self._root_view.count, self._root_view.link_count), dtype=vec13f, device=self.device)
        wp.launch(
            combine_pose_and_velocity_to_state_batched,
            dim=(self._root_view.count, self._root_view.link_count),
            device=self.device,
            inputs=[
                self._sim_bind_body_link_pose_w,
                self._sim_bind_body_com_vel_w,
                state,
            ],
        )
        return state

    @property
    @warn_overhead_cost(
        "body_link_pose_w or body_link_vel_w",
        "Launches a kernel to merge a pose and a velocity into a state. Consider using the pose and velocity arrays"
        " directly instead.",
    )
    @deprecated("body_link_pose_w or body_link_vel_w", since="3.0.0", remove_in="4.0.0")
    def body_link_state_w(self) -> wp.array(dtype=vec13f):
        """State of all bodies' link frame ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """
        state = wp.zeros((self._root_view.count, self._root_view.link_count), dtype=vec13f, device=self.device)
        wp.launch(
            combine_pose_and_velocity_to_state_batched,
            dim=(self._root_view.count, self._root_view.link_count),
            device=self.device,
            inputs=[
                self._sim_bind_body_link_pose_w,
                self.body_link_vel_w,
                state,
            ],
        )
        return state

    @property
    @warn_overhead_cost(
        "body_com_pose_w or body_com_vel_w",
        "Launches a kernel to merge a pose and a velocity into a state. Consider using the pose and velocity arrays"
        " directly instead.",
    )
    @deprecated("body_com_pose_w or body_com_vel_w", since="3.0.0", remove_in="4.0.0")
    def body_com_state_w(self) -> wp.array(dtype=vec13f):
        """State of all bodies center of mass ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        The velocity is in the form of [vx, vy, vz, wx, wy, wz].

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.
        """

        state = wp.zeros((self._root_view.count, self._root_view.link_count), dtype=vec13f, device=self.device)
        wp.launch(
            combine_pose_and_velocity_to_state_batched,
            dim=(self._root_view.count, self._root_view.link_count),
            device=self.device,
            inputs=[
                self.body_com_pose_w,
                self._sim_bind_body_com_vel_w,
                state,
            ],
        )
        return state

    @property
    def body_com_acc_w(self) -> wp.array(dtype=wp.spatial_vectorf):
        """Acceleration of all bodies center of mass ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). The acceleration is in the form of [vx, vy, vz, wx, wy, wz].
        All values are relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            wp.launch(
                derive_body_acceleration_from_velocity_batched,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    self._sim_bind_body_com_vel_w,
                    self._previous_body_com_vel,
                    NewtonManager.get_dt(),
                    self._body_com_acc_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._body_com_acc_w.timestamp = self._sim_timestamp
            # update the previous velocity
            self._previous_body_com_vel.assign(self._sim_bind_body_com_vel_w)
        return self._body_com_acc_w.data

    @property
    def body_com_pose_b(self) -> wp.array(dtype=wp.transformf):
        """Center of mass pose ``wp.transformf`` of all bodies in their respective body's link frames.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_b is None:
            self._body_com_pose_b = wp.zeros(
                (self._root_view.count, self._root_view.link_count), dtype=wp.transformf, device=self.device
            )

        wp.launch(
            generate_pose_from_position_with_unit_quaternion_batched,
            dim=(self._root_view.count, self._root_view.link_count),
            device=self.device,
            inputs=[
                self._sim_bind_body_com_pos_b,
                self._body_com_pose_b,
            ],
        )
        return self._body_com_pose_b

    # TODO: Make sure this is implemented when the feature is available in Newton.
    # TODO: Waiting on https://github.com/newton-physics/newton/pull/1161 ETA: early JAN 2026.
    @property
    def body_incoming_joint_wrench_b(self) -> wp.array(dtype=wp.spatial_vectorf):
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.
        """
        raise NotImplementedError("Body incoming joint wrench in body frame is not implemented for Newton.")

    ##
    # Joint state properties.
    ##

    @property
    def joint_pos(self) -> wp.array(dtype=wp.float32):
        """Joint positions. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_pos

    @property
    def joint_vel(self) -> wp.array(dtype=wp.float32):
        """Joint velocities. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_vel

    @property
    def joint_acc(self) -> wp.array(dtype=wp.float32):
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        if self._root_view.joint_dof_count == 0:
            return self._joint_acc.data
        if self._joint_acc.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            wp.launch(
                derive_joint_acceleration_from_velocity,
                dim=(self._root_view.count, self._root_view.joint_dof_count),
                device=self.device,
                inputs=[
                    self._sim_bind_joint_vel,
                    self._previous_joint_vel,
                    NewtonManager.get_dt(),
                    self._joint_acc.data,
                ],
            )
            self._joint_acc.timestamp = self._sim_timestamp
            # update the previous joint velocity
            self._previous_joint_vel.assign(self._sim_bind_joint_vel)
        return self._joint_acc.data

    ##
    # Derived Properties.
    ##

    @property
    def projected_gravity_b(self) -> wp.array(dtype=wp.vec3f):
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            wp.launch(
                project_vec_from_pose_single,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    self.GRAVITY_VEC_W,
                    self._sim_bind_root_link_pose_w,
                    self._projected_gravity_b.data,
                ],
            )
            # set the buffer data and timestamp
            self._projected_gravity_b.timestamp = self._sim_timestamp
        return self._projected_gravity_b.data

    @property
    def heading_w(self) -> wp.array(dtype=wp.float32):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        if self._heading_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_heading,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    self.FORWARD_VEC_B,
                    self._sim_bind_root_link_pose_w,
                    self._heading_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._heading_w.timestamp = self._sim_timestamp
        return self._heading_w.data

    @property
    def root_link_vel_b(self) -> wp.array(dtype=wp.spatial_vectorf):
        """Root link velocity ``wp.spatial_vectorf`` in base frame. Shape is (num_instances).

        Velocity is provided in the form of [vx, vy, vz, wx, wy, wz].
        """
        if self._root_link_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                project_velocities_to_frame,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    self.root_link_vel_w,
                    self._sim_bind_root_link_pose_w,
                    self._root_link_vel_b.data,
                ],
            )
            # set the buffer data and timestamp
            self._root_link_vel_b.timestamp = self._sim_timestamp
        return self._root_link_vel_b.data

    @property
    def root_com_vel_b(self) -> wp.array(dtype=wp.spatial_vectorf):
        """Root center of mass velocity ``wp.spatial_vectorf`` in base frame. Shape is (num_instances).

        Velocity is provided in the form of [vx, vy, vz, wx, wy, wz].
        """
        if self._root_com_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                project_velocities_to_frame,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    self._sim_bind_root_com_vel_w,
                    self._sim_bind_root_link_pose_w,
                    self._root_com_vel_b.data,
                ],
            )
            # set the buffer data and timestamp
            self._root_com_vel_b.timestamp = self._sim_timestamp
        return self._root_com_vel_b.data

    @property
    @warn_overhead_cost(
        "root_link_vel_b",
        "Launches a kernel to split the spatial velocity array to a linear velocity array. Consider using the spatial"
        " velocity array directly instead.",
    )
    def root_link_lin_vel_b(self) -> wp.array(dtype=wp.vec3f):
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.root_link_vel_b

        # Initialize the buffer if it is not already initialized
        if self._root_link_lin_vel_b is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._root_link_lin_vel_b = wp.array(
                    ptr=data.ptr,
                    dtype=wp.vec3f,
                    shape=data.shape,
                    strides=data.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._root_link_lin_vel_b = wp.zeros((self._root_view.count,), dtype=wp.vec3f, device=self.device)

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_spatial_vectory_array_to_linear_velocity_array,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    data,
                    self._root_link_lin_vel_b,
                ],
            )
        return self._root_link_lin_vel_b

    @property
    @warn_overhead_cost(
        "root_link_vel_b",
        "Launches a kernel to split the spatial velocity array to an angular velocity array. Consider using the spatial"
        " velocity array directly instead.",
    )
    def root_link_ang_vel_b(self) -> wp.array(dtype=wp.vec3f):
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.root_link_vel_b

        # Initialize the buffer if it is not already initialized
        if self._root_link_ang_vel_b is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._root_link_ang_vel_b = wp.array(
                    ptr=data.ptr + 3 * 4,
                    dtype=wp.vec3f,
                    shape=data.shape,
                    strides=data.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._root_link_ang_vel_b = wp.zeros((self._root_view.count,), dtype=wp.vec3f, device=self.device)

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_spatial_vectory_array_to_angular_velocity_array,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    data,
                    self._root_link_ang_vel_b,
                ],
            )
        return self._root_link_ang_vel_b

    @property
    @warn_overhead_cost(
        "root_com_vel_w",
        "Launches a kernel to split the spatial velocity array to a linear velocity array. Consider using the spatial"
        " velocity array directly instead.",
    )
    def root_com_lin_vel_b(self) -> wp.array(dtype=wp.vec3f):
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.root_com_vel_b

        # Initialize the buffer if it is not already initialized
        if self._root_com_lin_vel_b is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._root_com_lin_vel_b = wp.array(
                    ptr=data.ptr,
                    dtype=wp.vec3f,
                    shape=data.shape,
                    strides=data.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._root_com_lin_vel_b = wp.zeros((self._root_view.count,), dtype=wp.vec3f, device=self.device)

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_spatial_vectory_array_to_linear_velocity_array,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    data,
                    self._root_com_lin_vel_b,
                ],
            )
        return self._root_com_lin_vel_b

    @property
    @warn_overhead_cost(
        "root_com_vel_w",
        "Launches a kernel to split the spatial velocity array to an angular velocity array. Consider using the spatial"
        " velocity array directly instead.",
    )
    def root_com_ang_vel_b(self) -> wp.array(dtype=wp.vec3f):
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.root_com_vel_b

        # Initialize the buffer if it is not already initialized
        if self._root_com_ang_vel_b is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._root_com_ang_vel_b = wp.array(
                    ptr=data.ptr + 3 * 4,
                    dtype=wp.vec3f,
                    shape=data.shape,
                    strides=data.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._root_com_ang_vel_b = wp.zeros((self._root_view.count,), dtype=wp.vec3f, device=self.device)

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_spatial_vectory_array_to_angular_velocity_array,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    data,
                    self._root_com_ang_vel_b,
                ],
            )
        return self._root_com_ang_vel_b

    ##
    # Sliced properties.
    ##

    @property
    @warn_overhead_cost(
        "root_link_pose_w",
        "Launches a kernel to split the transform array to a position array. Consider using the transform array"
        " directly instead.",
    )
    def root_link_pos_w(self) -> wp.array(dtype=wp.vec3f):
        """Root link position ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        # Not a lazy buffer, so we do not need to call it to make sure it is up to date
        # Initialize the buffer if it is not already initialized
        if self._root_link_pos_w is None:
            if self._sim_bind_root_link_pose_w.is_contiguous:
                # Create a memory view of the data
                self._root_link_pos_w = wp.array(
                    ptr=self._sim_bind_root_link_pose_w.ptr,
                    dtype=wp.vec3f,
                    shape=self._sim_bind_root_link_pose_w.shape,
                    strides=self._sim_bind_root_link_pose_w.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._root_link_pos_w = wp.zeros((self._root_view.count,), dtype=wp.vec3f, device=self.device)

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not self._sim_bind_root_link_pose_w.is_contiguous:
            wp.launch(
                split_transform_array_to_position_array,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    self._sim_bind_root_link_pose_w,
                    self._root_link_pos_w,
                ],
            )
        return self._root_link_pos_w

    @property
    @warn_overhead_cost(
        "root_link_pose_w",
        "Launches a kernel to split the transform array to a quaternion array. Consider using the transform array"
        " directly instead.",
    )
    def root_link_quat_w(self) -> wp.array(dtype=wp.quatf):
        """Root link orientation ``wp.quatf`` in simulation world frame. Shape is (num_instances,).

        Format is ``(x, y, z, w)``.
        This quantity is the orientation of the actor frame of the root rigid body.
        """
        # Not a lazy buffer, so we do not need to call it to make sure it is up to date
        # Initialize the buffer if it is not already initialized
        if self._root_link_quat_w is None:
            if self._sim_bind_root_link_pose_w.is_contiguous:
                # Create a memory view of the data
                self._root_link_quat_w = wp.array(
                    ptr=self._sim_bind_root_link_pose_w.ptr + 3 * 4,
                    dtype=wp.quatf,
                    shape=self._sim_bind_root_link_pose_w.shape,
                    strides=self._sim_bind_root_link_pose_w.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._root_link_quat_w = wp.zeros((self._root_view.count,), dtype=wp.quatf, device=self.device)

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not self._sim_bind_root_link_pose_w.is_contiguous:
            wp.launch(
                split_transform_array_to_quaternion_array,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    self._sim_bind_root_link_pose_w,
                    self._root_link_quat_w,
                ],
            )
        return self._root_link_quat_w

    @property
    @warn_overhead_cost(
        "root_link_vel_w",
        "Launches a kernel to split the spatial velocity array to a linear velocity array. Consider using the spatial"
        " velocity array directly instead.",
    )
    def root_link_lin_vel_w(self) -> wp.array(dtype=wp.vec3f):
        """Root linear velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.root_link_vel_w

        # Initialize the buffer if it is not already initialized
        if self._root_link_lin_vel_w is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._root_link_lin_vel_w = wp.array(
                    ptr=data.ptr,
                    dtype=wp.vec3f,
                    shape=data.shape,
                    strides=data.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._root_link_lin_vel_w = wp.zeros((self._root_view.count,), dtype=wp.vec3f, device=self.device)

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_spatial_vectory_array_to_linear_velocity_array,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    data,
                    self._root_link_lin_vel_w,
                ],
            )
        return self._root_link_lin_vel_w

    @property
    @warn_overhead_cost(
        "root_link_vel_w",
        "Launches a kernel to split the spatial velocity array to an angular velocity array. Consider using the spatial"
        " velocity array directly instead.",
    )
    def root_link_ang_vel_w(self) -> wp.array(dtype=wp.vec3f):
        """Root link angular velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.root_link_vel_w

        # Initialize the buffer if it is not already initialized
        if self._root_link_ang_vel_w is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._root_link_ang_vel_w = wp.array(
                    ptr=data.ptr + 3 * 4,
                    dtype=wp.vec3f,
                    shape=data.shape,
                    strides=data.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._root_link_ang_vel_w = wp.zeros((self._root_view.count), dtype=wp.vec3f, device=self.device)

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_spatial_vectory_array_to_angular_velocity_array,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    data,
                    self._root_link_ang_vel_w,
                ],
            )
        return self._root_link_ang_vel_w

    @property
    @warn_overhead_cost(
        "root_com_pose_w",
        "Launches a kernel to split the transform array to a position array. Consider using the transform array"
        " directly instead.",
    )
    def root_com_pos_w(self) -> wp.array(dtype=wp.vec3f):
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.root_com_pose_w

        # Initialize the buffer if it is not already initialized
        if self._root_com_pos_w is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._root_com_pos_w = wp.array(
                    ptr=data.ptr, dtype=wp.vec3f, shape=data.shape, strides=data.strides, device=self.device
                )
            else:
                # Create a new buffer
                self._root_com_pos_w = wp.zeros((self._root_view.count,), dtype=wp.vec3f, device=self.device)

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_transform_array_to_position_array,
                dim=self._root_view.count,
                inputs=[
                    data,
                    self._root_com_pos_w,
                ],
            )
        return self._root_com_pos_w

    @property
    @warn_overhead_cost(
        "root_com_pose_w",
        "Launches a kernel to split the transform array to a quaternion array. Consider using the transform array"
        " directly instead.",
    )
    def root_com_quat_w(self) -> wp.array(dtype=wp.quatf):
        """Root center of mass orientation ``wp.quatf`` in simulation world frame. Shape is (num_instances,).

        Format is ``(x, y, z, w)``.
        This quantity is the orientation of the root rigid body's center of mass frame.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.root_com_pose_w

        # Initialize the buffer if it is not already initialized
        if self._root_com_quat_w is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._root_com_quat_w = wp.array(
                    ptr=data.ptr + 3 * 4, dtype=wp.quatf, shape=data.shape, strides=data.strides, device=self.device
                )
            else:
                # Create a new buffer
                self._root_com_quat_w = wp.zeros((self._root_view.count,), dtype=wp.quatf, device=self.device)

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_transform_array_to_quaternion_array,
                dim=self._root_view.count,
                inputs=[
                    data,
                    self._root_com_quat_w,
                ],
            )
        return self._root_com_quat_w

    @property
    @warn_overhead_cost(
        "root_com_vel_w",
        "Launches a kernel to split the spatial velocity array to a linear velocity array. Consider using the spatial"
        " velocity array directly instead.",
    )
    def root_com_lin_vel_w(self) -> wp.array(dtype=wp.vec3f):
        """Root center of mass linear velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances,).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        # Not a lazy buffer, so we do not need to call it to make sure it is up to date
        # Initialize the buffer if it is not already initialized
        if self._root_com_lin_vel_w is None:
            if self._sim_bind_root_com_vel_w.is_contiguous:
                # Create a memory view of the data
                self._root_com_lin_vel_w = wp.array(
                    ptr=self._sim_bind_root_com_vel_w.ptr,
                    dtype=wp.vec3f,
                    shape=self._sim_bind_root_com_vel_w.shape,
                    strides=self._sim_bind_root_com_vel_w.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._root_com_lin_vel_w = wp.zeros((self._root_view.count,), dtype=wp.vec3f, device=self.device)

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not self._sim_bind_root_com_vel_w.is_contiguous:
            wp.launch(
                split_spatial_vectory_array_to_linear_velocity_array,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    self._sim_bind_root_com_vel_w,
                    self._root_com_lin_vel_w,
                ],
            )
        return self._root_com_lin_vel_w

    @property
    @warn_overhead_cost(
        "root_com_vel_w",
        "Launches a kernel to split the spatial velocity array to an angular velocity array. Consider using the spatial"
        " velocity array directly instead.",
    )
    def root_com_ang_vel_w(self) -> wp.array(dtype=wp.vec3f):
        """Root center of mass angular velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        # Not a lazy buffer, so we do not need to call it to make sure it is up to date
        # Initialize the buffer if it is not already initialized
        if self._root_com_ang_vel_w is None:
            if self.root_com_vel_w.is_contiguous:
                # Create a memory view of the data
                self._root_com_ang_vel_w = wp.array(
                    ptr=self._sim_bind_root_com_vel_w.ptr + 3 * 4,
                    dtype=wp.vec3f,
                    shape=self._sim_bind_root_com_vel_w.shape,
                    strides=self._sim_bind_root_com_vel_w.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._root_com_ang_vel_w = wp.zeros((self._root_view.count,), dtype=wp.vec3f, device=self.device)

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not self._sim_bind_root_com_vel_w.is_contiguous:
            wp.launch(
                split_spatial_vectory_array_to_angular_velocity_array,
                dim=self._root_view.count,
                device=self.device,
                inputs=[
                    self._sim_bind_root_com_vel_w,
                    self._root_com_ang_vel_w,
                ],
            )
        return self._root_com_ang_vel_w

    @property
    @warn_overhead_cost(
        "body_link_pose_w",
        "Launches a kernel to split the transform array to a position array. In a graph-based pipeline, consider using"
        " the transform array directly instead.",
    )
    def body_link_pos_w(self) -> wp.array(dtype=wp.vec3f):
        """Positions of all bodies in simulation world frame ``wp.vec3f``. Shape is (num_instances, num_bodies).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        # Not a lazy buffer, so we do not need to call it to make sure it is up to date
        # Initialize the buffer if it is not already initialized
        if self._body_link_pos_w is None:
            if self._sim_bind_body_link_pose_w.is_contiguous:
                # Create a memory view of the data
                self._body_link_pos_w = wp.array(
                    ptr=self._sim_bind_body_link_pose_w.ptr,
                    dtype=wp.vec3f,
                    shape=self._sim_bind_body_link_pose_w.shape,
                    strides=self._sim_bind_body_link_pose_w.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._body_link_pos_w = wp.zeros(
                    (self._root_view.count, self._root_view.link_count), dtype=wp.vec3f, device=self.device
                )

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not self._sim_bind_body_link_pose_w.is_contiguous:
            wp.launch(
                split_transform_batched_array_to_position_batched_array,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    self._sim_bind_body_link_pose_w,
                    self._body_link_pos_w,
                ],
            )
        return self._body_link_pos_w

    @property
    @warn_overhead_cost(
        "body_link_pose_w",
        "Launches a kernel to split the transform array to a quaternion array. In a graph-based pipeline, consider"
        " using the transform array directly instead.",
    )
    def body_link_quat_w(self) -> wp.array(dtype=wp.quatf):
        """Orientation ``wp.quatf`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        Format is ``(x, y, z, w)``.
        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        # Not a lazy buffer, so we do not need to call it to make sure it is up to date
        # Initialize the buffer if it is not already initialized
        if self._body_link_quat_w is None:
            if self._sim_bind_body_link_pose_w.is_contiguous:
                # Create a memory view of the data
                self._body_link_quat_w = wp.array(
                    ptr=self._sim_bind_body_link_pose_w.ptr + 3 * 4,
                    dtype=wp.quatf,
                    shape=self._sim_bind_body_link_pose_w.shape,
                    strides=self._sim_bind_body_link_pose_w.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._body_link_quat_w = wp.zeros(
                    (self._root_view.count, self._root_view.link_count), dtype=wp.quatf, device=self.device
                )

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not self._sim_bind_body_link_pose_w.is_contiguous:
            wp.launch(
                split_transform_batched_array_to_quaternion_batched_array,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    self._sim_bind_body_link_pose_w,
                    self._body_link_quat_w,
                ],
            )
        return self._body_link_quat_w

    @property
    @warn_overhead_cost(
        "body_link_vel_w",
        "Launches a kernel to split the velocity array to a linear velocity array. In a graph-based pipeline, consider"
        " using the velocity array directly instead.",
    )
    def body_link_lin_vel_w(self) -> wp.array(dtype=wp.vec3f):
        """Linear velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear velocity of the articulation bodies' center of mass frame relative to the world.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.body_link_vel_w

        # Initialize the buffer if it is not already initialized
        if self._body_link_lin_vel_w is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._body_link_lin_vel_w = wp.array(
                    ptr=data.ptr, dtype=wp.vec3f, shape=data.shape, strides=data.strides, device=self.device
                )
            else:
                # Create a new buffer
                self._body_link_lin_vel_w = wp.zeros(
                    (self._root_view.count, self._root_view.link_count), dtype=wp.vec3f, device=self.device
                )

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_spatial_vectory_batched_array_to_linear_velocity_batched_array,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    data,
                    self._body_link_lin_vel_w,
                ],
            )
        return self._body_link_lin_vel_w

    @property
    @warn_overhead_cost(
        "body_link_vel_w",
        "Launches a kernel to split the velocity array to an angular velocity array. In a graph-based pipeline,"
        " consider using the velocity array directly instead.",
    )
    def body_link_ang_vel_w(self) -> wp.array(dtype=wp.vec3f):
        """Angular velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.body_link_vel_w

        # Initialize the buffer if it is not already initialized
        if self._body_link_ang_vel_w is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._body_link_ang_vel_w = wp.array(
                    ptr=data.ptr + 3 * 4, dtype=wp.vec3f, shape=data.shape, strides=data.strides, device=self.device
                )
            else:
                # Create a new buffer
                self._body_link_ang_vel_w = wp.zeros(
                    (self._root_view.count, self._root_view.link_count), dtype=wp.vec3f, device=self.device
                )

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_spatial_vectory_batched_array_to_angular_velocity_batched_array,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    data,
                    self._body_link_ang_vel_w,
                ],
            )
        return self._body_link_ang_vel_w

    @property
    @warn_overhead_cost(
        "body_com_pose_w",
        "Launches a kernel to split the transform array to a position array. In a graph-based pipeline, consider using"
        " the transform array directly instead.",
    )
    def body_com_pos_w(self) -> wp.array(dtype=wp.vec3f):
        """Positions of all bodies in simulation world frame ``wp.vec3f``. Shape is (num_instances, num_bodies).

        This quantity is the position of the articulation bodies' actor frame.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.body_com_pose_w

        # Initialize the buffer if it is not already initialized
        if self._body_com_pos_w is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._body_com_pos_w = wp.array(
                    ptr=data.ptr, dtype=wp.vec3f, shape=data.shape, strides=data.strides, device=self.device
                )
            else:
                # Create a new buffer
                self._body_com_pos_w = wp.zeros(
                    (self._root_view.count, self._root_view.link_count), dtype=wp.vec3f, device=self.device
                )

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_transform_batched_array_to_position_batched_array,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    data,
                    self._body_com_pos_w,
                ],
            )
        return self._body_com_pos_w

    @property
    @warn_overhead_cost(
        "body_com_pose_w",
        "Launches a kernel to split the transform array to a quaternion array. In a graph-based pipeline, consider"
        " using the transform array directly instead.",
    )
    def body_com_quat_w(self) -> wp.array(dtype=wp.quatf):
        """Orientation ``wp.quatf`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        Format is ``(x, y, z, w)``.
        This quantity is the orientation of the articulation bodies' actor frame.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.body_com_pose_w

        # Initialize the buffer if it is not already initialized
        if self._body_com_quat_w is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._body_com_quat_w = wp.array(
                    ptr=data.ptr + 3 * 4, dtype=wp.quatf, shape=data.shape, strides=data.strides, device=self.device
                )
            else:
                # Create a new buffer
                self._body_com_quat_w = wp.zeros(
                    (self._root_view.count, self._root_view.link_count), dtype=wp.quatf, device=self.device
                )

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_transform_batched_array_to_quaternion_batched_array,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    data,
                    self._body_com_quat_w,
                ],
            )
        return self._body_com_quat_w

    @property
    @warn_overhead_cost(
        "body_com_vel_w",
        "Launches a kernel to split the velocity array to a linear velocity array. In a graph-based pipeline, consider"
        " using the velocity array directly instead.",
    )
    def body_com_lin_vel_w(self) -> wp.array(dtype=wp.vec3f):
        """Linear velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        # Not a lazy buffer, so we do not need to call it to make sure it is up to date
        # Initialize the buffer if it is not already initialized
        if self._body_com_lin_vel_w is None:
            if self._sim_bind_body_com_vel_w.is_contiguous:
                # Create a memory view of the data
                self._body_com_lin_vel_w = wp.array(
                    ptr=self._sim_bind_body_com_vel_w.ptr,
                    dtype=wp.vec3f,
                    shape=self._sim_bind_body_com_vel_w.shape,
                    strides=self._sim_bind_body_com_vel_w.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._body_com_lin_vel_w = wp.zeros(
                    (self._root_view.count, self._root_view.link_count), dtype=wp.vec3f, device=self.device
                )

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not self._sim_bind_body_com_vel_w.is_contiguous:
            wp.launch(
                split_spatial_vectory_batched_array_to_linear_velocity_batched_array,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    self._sim_bind_body_com_vel_w,
                    self._body_com_lin_vel_w,
                ],
            )
        return self._body_com_lin_vel_w

    @property
    @warn_overhead_cost(
        "body_com_vel_w",
        "Launches a kernel to split the velocity array to an angular velocity array. In a graph-based pipeline,"
        " consider using the velocity array directly instead.",
    )
    def body_com_ang_vel_w(self) -> wp.array(dtype=wp.vec3f):
        """Angular velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        # Not a lazy buffer, so we do not need to call it to make sure it is up to date
        # Initialize the buffer if it is not already initialized
        if self._body_com_ang_vel_w is None:
            if self._sim_bind_body_com_vel_w.is_contiguous:
                # Create a memory view of the data
                self._body_com_ang_vel_w = wp.array(
                    ptr=self._sim_bind_body_com_vel_w.ptr + 3 * 4,
                    dtype=wp.vec3f,
                    shape=self._sim_bind_body_com_vel_w.shape,
                    strides=self._sim_bind_body_com_vel_w.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._body_com_ang_vel_w = wp.zeros(
                    (self._root_view.count, self._root_view.link_count), dtype=wp.vec3f, device=self.device
                )

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not self._sim_bind_body_com_vel_w.is_contiguous:
            wp.launch(
                split_spatial_vectory_batched_array_to_angular_velocity_batched_array,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    self._sim_bind_body_com_vel_w,
                    self._body_com_ang_vel_w,
                ],
            )
        return self._body_com_ang_vel_w

    @property
    @warn_overhead_cost(
        "body_com_acc_w",
        "Launches a kernel to split the velocity array to a linear velocity array. In a graph-based pipeline, consider"
        " using the velocity array directly instead.",
    )
    def body_com_lin_acc_w(self) -> wp.array(dtype=wp.vec3f):
        """Linear acceleration ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.body_com_acc_w

        # Initialize the buffer if it is not already initialized
        if self._body_com_lin_acc_w is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._body_com_lin_acc_w = wp.array(
                    ptr=data.ptr, dtype=wp.vec3f, shape=data.shape, strides=data.strides, device=self.device
                )
            else:
                # Create a new buffer
                self._body_com_lin_acc_w = wp.zeros(
                    (self._root_view.count, self._root_view.link_count), dtype=wp.vec3f, device=self.device
                )

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_spatial_vectory_batched_array_to_linear_velocity_batched_array,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    data,
                    self._body_com_lin_acc_w,
                ],
            )
        return self._body_com_lin_acc_w

    @property
    @warn_overhead_cost(
        "body_com_acc_w",
        "Launches a kernel to split the velocity array to an angular velocity array. In a graph-based pipeline,"
        " consider using the velocity array directly instead.",
    )
    def body_com_ang_acc_w(self) -> wp.array(dtype=wp.vec3f):
        """Angular acceleration ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        # Call the lazy buffer to make sure it is up to date
        data = self.body_com_acc_w

        # Initialize the buffer if it is not already initialized
        if self._body_com_ang_acc_w is None:
            if data.is_contiguous:
                # Create a memory view of the data
                self._body_com_ang_acc_w = wp.array(
                    ptr=data.ptr + 3 * 4,
                    dtype=wp.vec3f,
                    shape=data.shape,
                    strides=data.strides,
                    device=self.device,
                )
            else:
                # Create a new buffer
                self._body_com_ang_acc_w = wp.zeros(
                    (self._root_view.count, self._root_view.link_count), dtype=wp.vec3f, device=self.device
                )

        # If the data is not contiguous, we need to launch a kernel to update the buffer
        if not data.is_contiguous:
            wp.launch(
                split_spatial_vectory_batched_array_to_angular_velocity_batched_array,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    data,
                    self._body_com_ang_acc_w,
                ],
            )
        return self._body_com_ang_acc_w

    @property
    def body_com_pos_b(self) -> wp.array(dtype=wp.vec3f):
        """Center of mass position ``wp.vec3f`` of all bodies in their respective body's link frames.

        Shapes are (num_instances, num_bodies,). The position is in the form of [x, y, z].
        This quantity is the position of the center of mass frame of the rigid body relative to the body's link frame.
        """
        return self._sim_bind_body_com_pos_b

    @property
    @warn_overhead_cost(
        "unit_quaternion",
        "Launches a kernel to split the pose array to a quaternion array. Consider using the pose array directly"
        " instead.",
    )
    def body_com_quat_b(self) -> wp.array(dtype=wp.quatf):
        """Orientation (x, y, z, w) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame. In Newton
        this quantity is always a unit quaternion.
        """
        if self._body_com_quat_b is None:
            self._body_com_quat_b = wp.zeros(
                (self._root_view.count, self._root_view.link_count), dtype=wp.quatf, device=self.device
            )
            self._body_com_quat_b.fill_(wp.quat_identity(wp.float32))
        return self._body_com_quat_b

    ##
    # Backward compatibility. -- Deprecated properties.
    ##

    @property
    @deprecated("default_root_pose", "default_root_vel", since="3.0.0", remove_in="4.0.0")
    def default_root_state(self) -> wp.array(dtype=vec13f):
        """Same as :attr:`default_root_pose`."""
        state = wp.zeros((self._root_view.count), dtype=vec13f, device=self.device)
        wp.launch(
            combine_pose_and_velocity_to_state,
            dim=(self._root_view.count),
            device=self.device,
            inputs=[
                self._default_root_pose,
                self._default_root_vel,
                state,
            ],
        )
        return state

    @property
    @deprecated("root_link_pose_w", since="3.0.0", remove_in="4.0.0")
    def root_pose_w(self) -> wp.array(dtype=wp.transformf):
        """Same as :attr:`root_link_pose_w`."""
        return self.root_link_pose_w

    @property
    @deprecated("root_link_pos_w", since="3.0.0", remove_in="4.0.0")
    def root_pos_w(self) -> wp.array(dtype=wp.vec3f):
        """Same as :attr:`root_link_pos_w`."""
        return self.root_link_pos_w

    @property
    @deprecated("root_link_quat_w", since="3.0.0", remove_in="4.0.0")
    def root_quat_w(self) -> wp.array(dtype=wp.quatf):
        """Same as :attr:`root_link_quat_w`."""
        return self.root_link_quat_w

    @property
    @deprecated("root_com_vel_w", since="3.0.0", remove_in="4.0.0")
    def root_vel_w(self) -> wp.array(dtype=wp.spatial_vectorf):
        """Same as :attr:`root_com_vel_w`."""
        return self.root_com_vel_w

    @property
    @deprecated("root_com_lin_vel_w", since="3.0.0", remove_in="4.0.0")
    def root_lin_vel_w(self) -> wp.array(dtype=wp.vec3f):
        """Same as :attr:`root_com_lin_vel_w`."""
        return self.root_com_lin_vel_w

    @property
    @deprecated("root_com_ang_vel_w", since="3.0.0", remove_in="4.0.0")
    def root_ang_vel_w(self) -> wp.array(dtype=wp.vec3f):
        """Same as :attr:`root_com_ang_vel_w`."""
        return self.root_com_ang_vel_w

    @property
    @deprecated("root_com_lin_vel_b", since="3.0.0", remove_in="4.0.0")
    def root_lin_vel_b(self) -> wp.array(dtype=wp.vec3f):
        """Same as :attr:`root_com_lin_vel_b`."""
        return self.root_com_lin_vel_b

    @property
    @deprecated("root_com_ang_vel_b", since="3.0.0", remove_in="4.0.0")
    def root_ang_vel_b(self) -> wp.array(dtype=wp.vec3f):
        """Same as :attr:`root_com_ang_vel_b`."""
        return self.root_com_ang_vel_b

    @property
    @deprecated("body_link_pose_w", since="3.0.0", remove_in="4.0.0")
    def body_pose_w(self) -> wp.array(dtype=wp.transformf):
        """Same as :attr:`body_link_pose_w`."""
        return self.body_link_pose_w

    @property
    @deprecated("body_link_pos_w", since="3.0.0", remove_in="4.0.0")
    def body_pos_w(self) -> wp.array(dtype=wp.vec3f):
        """Same as :attr:`body_link_pos_w`."""
        return self.body_link_pos_w

    @property
    @deprecated("body_link_quat_w", since="3.0.0", remove_in="4.0.0")
    def body_quat_w(self) -> wp.array(dtype=wp.quatf):
        """Same as :attr:`body_link_quat_w`."""
        return self.body_link_quat_w

    @property
    @deprecated("body_com_vel_w", since="3.0.0", remove_in="4.0.0")
    def body_vel_w(self) -> wp.array(dtype=wp.spatial_vectorf):
        """Same as :attr:`body_com_vel_w`."""
        return self.body_com_vel_w

    @property
    @deprecated("body_com_lin_vel_w", since="3.0.0", remove_in="4.0.0")
    def body_lin_vel_w(self) -> wp.array(dtype=wp.vec3f):
        """Same as :attr:`body_com_lin_vel_w`."""
        return self.body_com_lin_vel_w

    @property
    @deprecated("body_com_ang_vel_w", since="3.0.0", remove_in="4.0.0")
    def body_ang_vel_w(self) -> wp.array(dtype=wp.vec3f):
        """Same as :attr:`body_com_ang_vel_w`."""
        return self.body_com_ang_vel_w

    @property
    @deprecated("body_com_acc_w", since="3.0.0", remove_in="4.0.0")
    def body_acc_w(self) -> wp.array(dtype=wp.spatial_vectorf):
        """Same as :attr:`body_com_acc_w`."""
        return self.body_com_acc_w

    @property
    @deprecated("body_com_lin_acc_w", since="3.0.0", remove_in="4.0.0")
    def body_lin_acc_w(self) -> wp.array(dtype=wp.vec3f):
        """Same as :attr:`body_com_lin_acc_w`."""
        return self.body_com_lin_acc_w

    @property
    @deprecated("body_com_ang_acc_w", since="3.0.0", remove_in="4.0.0")
    def body_ang_acc_w(self) -> wp.array(dtype=wp.vec3f):
        """Same as :attr:`body_com_ang_acc_w`."""
        return self.body_com_ang_acc_w

    @property
    @deprecated("body_com_pos_b", since="3.0.0", remove_in="4.0.0")
    def com_pos_b(self) -> wp.array(dtype=wp.vec3f):
        """Same as :attr:`body_com_pos_b`."""
        return self.body_com_pos_b

    @property
    @deprecated("body_com_quat_b", since="3.0.0", remove_in="4.0.0")
    def com_quat_b(self) -> wp.array(dtype=wp.quatf):
        """Same as :attr:`body_com_quat_b`."""
        return self.body_com_quat_b

    @property
    @deprecated("joint_pos_limits", since="3.0.0", remove_in="4.0.0")
    def joint_limits(self) -> wp.array(dtype=wp.float32):
        """Deprecated property. Please use :attr:`joint_pos_limits` instead."""
        return self.joint_pos_limits

    @property
    @deprecated("joint_friction_coeff", since="3.0.0", remove_in="4.0.0")
    def joint_friction(self) -> wp.array(dtype=wp.float32):
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead."""
        return self.joint_friction_coeff

    @property
    @deprecated("fixed_tendon_pos_limits", since="3.0.0", remove_in="4.0.0")
    def fixed_tendon_limit(self) -> wp.array(dtype=wp.float32):
        """Deprecated property. Please use :attr:`fixed_tendon_pos_limits` instead."""
        return self.fixed_tendon_pos_limits

    @property
    @deprecated("applied_effort", since="3.0.0", remove_in="4.0.0")
    def applied_torque(self) -> wp.array(dtype=wp.float32):
        """Deprecated property. Please use :attr:`applied_effort` instead."""
        return self.applied_effort

    @property
    @deprecated("computed_effort", since="3.0.0", remove_in="4.0.0")
    def computed_torque(self) -> wp.array(dtype=wp.float32):
        """Deprecated property. Please use :attr:`computed_effort` instead."""
        return self.computed_effort

    @property
    @deprecated("joint_dynamic_friction_coeff", since="3.0.0", remove_in="4.0.0")
    def joint_dynamic_friction(self) -> wp.array(dtype=wp.float32):
        """Deprecated property. Please use :attr:`joint_dynamic_friction_coeff` instead."""
        return self.joint_dynamic_friction_coeff

    @property
    @deprecated("actuator_effort_target", since="3.0.0", remove_in="4.0.0")
    def joint_effort_target(self) -> wp.array(dtype=wp.float32):
        """Deprecated property. Please use :attr:`actuator_effort_target` instead."""
        return self.actuator_effort_target

    @property
    @deprecated("joint_viscous_friction_coeff", since="3.0.0", remove_in="4.0.0")
    def joint_viscous_friction(self) -> wp.array(dtype=wp.float32):
        """Deprecated property. Please use :attr:`joint_viscous_friction_coeff` instead."""
        return self.joint_viscous_friction_coeff

    @property
    @deprecated("joint_vel_limits", since="3.0.0", remove_in="4.0.0")
    def joint_velocity_limits(self) -> wp.array(dtype=wp.float32):
        """Deprecated property. Please use :attr:`joint_vel_limits` instead."""
        return self.joint_vel_limits

    ###
    # Helper functions.
    ###

    def _create_simulation_bindings(self) -> None:
        """Create simulation bindings for the root data.

        Direct simulation bindings are pointers to the simulation data, their data is not copied, and should
        only be updated using warp kernels. Any modifications made to the bindings will be reflected in the simulation.
        Hence we encourage users to carefully think about the data they modify and in which order it should be updated.

        .. caution:: This is possible if and only if the properties that we access are strided from newton and not
        indexed. Newton willing this is the case all the time, but we should pay attention to this if things look off.
        """
        # Short-hand for the number of instances, number of links, and number of joints.
        n_view = self._root_view.count
        n_dof = self._root_view.joint_dof_count

        # -- root properties
        if self._root_view.is_fixed_base:
            self._sim_bind_root_link_pose_w = self._root_view.get_root_transforms(NewtonManager.get_state_0())[:, 0, 0]
        else:
            self._sim_bind_root_link_pose_w = self._root_view.get_root_transforms(NewtonManager.get_state_0())[:, 0]
        self._sim_bind_root_com_vel_w = self._root_view.get_root_velocities(NewtonManager.get_state_0())
        if self._sim_bind_root_com_vel_w is not None:
            if self._root_view.is_fixed_base:
                self._sim_bind_root_com_vel_w = self._sim_bind_root_com_vel_w[:, 0, 0]
            else:
                self._sim_bind_root_com_vel_w = self._sim_bind_root_com_vel_w[:, 0]
        # -- body properties
        self._sim_bind_body_com_pos_b = self._root_view.get_attribute("body_com", NewtonManager.get_model())[:, 0]
        self._sim_bind_body_link_pose_w = self._root_view.get_link_transforms(NewtonManager.get_state_0())[:, 0]
        self._sim_bind_body_com_vel_w = self._root_view.get_link_velocities(NewtonManager.get_state_0())
        if self._sim_bind_body_com_vel_w is not None:
            self._sim_bind_body_com_vel_w = self._sim_bind_body_com_vel_w[:, 0]
        self._sim_bind_body_mass = self._root_view.get_attribute("body_mass", NewtonManager.get_model())[:, 0]
        self._sim_bind_body_inertia = self._root_view.get_attribute("body_inertia", NewtonManager.get_model())[:, 0]
        self._sim_bind_body_external_wrench = self._root_view.get_attribute("body_f", NewtonManager.get_state_0())[:, 0]
        # -- joint properties
        if n_dof > 0:
            self._sim_bind_joint_pos_limits_lower = self._root_view.get_attribute(
                "joint_limit_lower", NewtonManager.get_model()
            )[:, 0]
            self._sim_bind_joint_pos_limits_upper = self._root_view.get_attribute(
                "joint_limit_upper", NewtonManager.get_model()
            )[:, 0]
            self._sim_bind_joint_stiffness_sim = self._root_view.get_attribute(
                "joint_target_ke", NewtonManager.get_model()
            )[:, 0]
            self._sim_bind_joint_damping_sim = self._root_view.get_attribute(
                "joint_target_kd", NewtonManager.get_model()
            )[:, 0]
            self._sim_bind_joint_armature = self._root_view.get_attribute("joint_armature", NewtonManager.get_model())[
                :, 0
            ]
            self._sim_bind_joint_friction_coeff = self._root_view.get_attribute(
                "joint_friction", NewtonManager.get_model()
            )[:, 0]
            self._sim_bind_joint_vel_limits_sim = self._root_view.get_attribute(
                "joint_velocity_limit", NewtonManager.get_model()
            )[:, 0]
            self._sim_bind_joint_effort_limits_sim = self._root_view.get_attribute(
                "joint_effort_limit", NewtonManager.get_model()
            )[:, 0]
            # -- joint states
            print("joint pos shape:", self._root_view.get_dof_positions(NewtonManager.get_state_0()).shape)
            print("joint vel shape:", self._root_view.get_dof_velocities(NewtonManager.get_state_0()).shape)
            self._sim_bind_joint_pos = self._root_view.get_dof_positions(NewtonManager.get_state_0())[:, 0]
            self._sim_bind_joint_vel = self._root_view.get_dof_velocities(NewtonManager.get_state_0())[:, 0]
            print("joint pos shape:", self._sim_bind_joint_pos.shape)
            print("joint vel shape:", self._sim_bind_joint_vel.shape)
            # -- joint commands (sent to the simulation)
            self._sim_bind_joint_effort = self._root_view.get_attribute("joint_f", NewtonManager.get_control())[:, 0]
            self._sim_bind_joint_position_target = self._root_view.get_attribute(
                "joint_target_pos", NewtonManager.get_control()
            )[:, 0]
            self._sim_bind_joint_velocity_target = self._root_view.get_attribute(
                "joint_target_vel", NewtonManager.get_control()
            )[:, 0]
        else:
            # No joints (e.g., free-floating rigid body) - set bindings to empty arrays
            self._sim_bind_joint_pos_limits_lower = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_pos_limits_upper = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_stiffness_sim = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_damping_sim = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_armature = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_friction_coeff = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_vel_limits_sim = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_effort_limits_sim = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_pos = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_vel = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_effort = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_position_target = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_velocity_target = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)

    def _create_buffers(self) -> None:
        """Create buffers for the root data."""

        # Short-hand for the number of instances, number of links, and number of joints.
        n_view = self._root_view.count
        n_link = self._root_view.link_count
        n_dof = self._root_view.joint_dof_count

        # MASKS (used as default all-True masks and temp buffers for partial indexing)
        self.ALL_ENV_MASK = wp.ones((n_view,), dtype=wp.bool, device=self.device)
        self.ALL_BODY_MASK = wp.ones((n_link,), dtype=wp.bool, device=self.device)
        self.ALL_JOINT_MASK = wp.ones((n_dof,), dtype=wp.bool, device=self.device)
        self.ENV_MASK = wp.zeros((n_view,), dtype=wp.bool, device=self.device)
        self.BODY_MASK = wp.zeros((n_link,), dtype=wp.bool, device=self.device)
        self.JOINT_MASK = wp.zeros((n_dof,), dtype=wp.bool, device=self.device)

        # Initialize history for finite differencing. If the articulation is fixed, the root com velocity is not
        # available, so we use zeros.
        if self._root_view.get_root_velocities(NewtonManager.get_state_0()) is not None:
            if self._root_view.is_fixed_base:
                self._previous_root_com_vel = wp.clone(
                    self._root_view.get_root_velocities(NewtonManager.get_state_0())
                )[:, 0, 0]
            else:
                self._previous_root_com_vel = wp.clone(
                    self._root_view.get_root_velocities(NewtonManager.get_state_0())
                )[:, 0]
        else:
            logger.warning("Failed to get root com velocity. If the articulation is fixed, this is expected.")
            self._previous_root_com_vel = wp.zeros((n_view, n_link), dtype=wp.spatial_vectorf, device=self.device)
            logger.warning("Setting root com velocity to zeros.")
            self._sim_bind_root_com_vel_w = wp.zeros((n_view), dtype=wp.spatial_vectorf, device=self.device)
            self._sim_bind_body_com_vel_w = wp.zeros((n_view, n_link), dtype=wp.spatial_vectorf, device=self.device)
        # -- default root pose and velocity
        self._default_root_pose = wp.zeros((n_view,), dtype=wp.transformf, device=self.device)
        self._default_root_vel = wp.zeros((n_view,), dtype=wp.spatial_vectorf, device=self.device)
        # -- default joint positions and velocities
        self._default_joint_pos = wp.zeros((n_view, n_dof), dtype=wp.float32, device=self.device)
        self._default_joint_vel = wp.zeros((n_view, n_dof), dtype=wp.float32, device=self.device)
        # -- joint commands (sent to the actuator from the user)
        self._actuator_position_target = wp.zeros((n_view, n_dof), dtype=wp.float32, device=self.device)
        self._actuator_velocity_target = wp.zeros((n_view, n_dof), dtype=wp.float32, device=self.device)
        self._actuator_effort_target = wp.zeros((n_view, n_dof), dtype=wp.float32, device=self.device)
        # -- computed joint efforts from the actuator models
        self._computed_effort = wp.zeros((n_view, n_dof), dtype=wp.float32, device=self.device)
        self._applied_effort = wp.zeros((n_view, n_dof), dtype=wp.float32, device=self.device)
        # -- joint properties for the actuator models
        if n_dof > 0:
            self._actuator_stiffness = wp.clone(self._sim_bind_joint_stiffness_sim)
            self._actuator_damping = wp.clone(self._sim_bind_joint_damping_sim)
        else:
            self._actuator_stiffness = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
            self._actuator_damping = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
        # -- other data that are filled based on explicit actuator models
        self._joint_dynamic_friction = wp.zeros((n_view, n_dof), dtype=wp.float32, device=self.device)
        self._joint_viscous_friction = wp.zeros((n_view, n_dof), dtype=wp.float32, device=self.device)
        self._soft_joint_vel_limits = wp.zeros((n_view, n_dof), dtype=wp.float32, device=self.device)
        self._gear_ratio = wp.ones((n_view, n_dof), dtype=wp.float32, device=self.device)
        # -- update the soft joint position limits
        self._soft_joint_pos_limits = wp.zeros((n_view, n_dof), dtype=wp.vec2f, device=self.device)

        # Initialize history for finite differencing
        if n_dof > 0:
            self._previous_joint_vel = wp.clone(self._root_view.get_dof_velocities(NewtonManager.get_state_0()))[:, 0]
        else:
            self._previous_joint_vel = wp.zeros((n_view, 0), dtype=wp.float32, device=self.device)
        self._previous_body_com_vel = wp.clone(self._sim_bind_body_com_vel_w)

        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._root_link_vel_w = TimestampedWarpBuffer(shape=(n_view,), dtype=wp.spatial_vectorf, device=self.device)
        self._root_link_vel_b = TimestampedWarpBuffer(shape=(n_view,), dtype=wp.spatial_vectorf, device=self.device)
        self._projected_gravity_b = TimestampedWarpBuffer(shape=(n_view,), dtype=wp.vec3f, device=self.device)
        self._heading_w = TimestampedWarpBuffer(shape=(n_view,), dtype=wp.float32, device=self.device)
        self._body_link_vel_w = TimestampedWarpBuffer(
            shape=(n_view, n_link), dtype=wp.spatial_vectorf, device=self.device
        )
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedWarpBuffer(shape=(n_view,), dtype=wp.transformf, device=self.device)
        self._root_com_vel_b = TimestampedWarpBuffer(shape=(n_view,), dtype=wp.spatial_vectorf, device=self.device)
        self._root_com_acc_w = TimestampedWarpBuffer(shape=(n_view,), dtype=wp.spatial_vectorf, device=self.device)
        self._body_com_pose_w = TimestampedWarpBuffer(shape=(n_view, n_link), dtype=wp.transformf, device=self.device)
        self._body_com_acc_w = TimestampedWarpBuffer(
            shape=(n_view, n_link), dtype=wp.spatial_vectorf, device=self.device
        )
        # -- joint state
        self._joint_acc = TimestampedWarpBuffer(shape=(n_view, n_dof), dtype=wp.float32, device=self.device)
        # self._body_incoming_joint_wrench_b = TimestampedWarpBuffer(shape=(n_view, n_dof), dtype=wp.spatial_vectorf, device=self.device)
        # Empty memory pre-allocations
        self._joint_pos_limits = None
        self._root_state_w = None
        self._root_link_state_w = None
        self._root_com_state_w = None
        self._body_com_quat_b = None
        self._root_link_lin_vel_b = None
        self._root_link_ang_vel_b = None
        self._root_com_lin_vel_b = None
        self._root_com_ang_vel_b = None
        self._root_link_pos_w = None
        self._root_link_quat_w = None
        self._root_link_lin_vel_w = None
        self._root_link_ang_vel_w = None
        self._root_com_pos_w = None
        self._root_com_quat_w = None
        self._root_com_lin_vel_w = None
        self._root_com_ang_vel_w = None
        self._body_link_pos_w = None
        self._body_link_quat_w = None
        self._body_link_lin_vel_w = None
        self._body_link_ang_vel_w = None
        self._body_com_pos_w = None
        self._body_com_quat_w = None
        self._body_com_lin_vel_w = None
        self._body_com_ang_vel_w = None
        self._body_com_lin_acc_w = None
        self._body_com_ang_acc_w = None
        self._body_com_pose_b = None

    def update(self, dt: float):
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the joint and body acceleration buffers at a higher frequency since we do finite
        # differencing.
        self.joint_acc
        self.body_com_acc_w
