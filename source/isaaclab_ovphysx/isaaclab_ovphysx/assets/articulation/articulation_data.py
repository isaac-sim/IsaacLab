# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation data backed by ovphysx TensorBindingsAPI."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

import warp as wp

from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData

from isaaclab_ovphysx import tensor_types as TT


class TimestampedBuffer:
    """A warp array that tracks when it was last refreshed from the simulation."""

    __slots__ = ("data", "timestamp")

    def __init__(self, shape, device: str, dtype):
        self.data = wp.zeros(shape, dtype=dtype, device=device)
        self.timestamp: float = -1.0


class ArticulationData(BaseArticulationData):
    """Data container for an articulation backed by ovphysx tensor bindings.

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

    Uses ovphysx :class:`TensorBinding` objects to lazily read simulation state into warp
    arrays.  Writes happen via the :class:`Articulation` class.
    """

    __backend_name__: str = "ovphysx"
    """The name of the backend for the articulation data."""

    def __init__(self, bindings: dict[int, Any], device: str, binding_getter=None):
        """Initialize the articulation data.

        Args:
            bindings: Mapping from ovphysx tensor type constant to a
                live TensorBinding for this articulation.
            device: The compute device (``"cpu"`` or ``"cuda:N"``).
            binding_getter: Optional callable(tensor_type) -> TensorBinding
                that lazily creates bindings on first access.  When provided,
                ``_get_binding()`` delegates to this instead of only checking
                the static ``bindings`` dict.
        """
        super().__init__(root_view=None, device=device)
        self._bindings = bindings
        self._binding_getter = binding_getter
        self._sim_timestamp: float = 0.0

        # Metadata from an arbitrary articulation binding.
        sample = next(iter(bindings.values()))
        self._num_instances = sample.count
        self._num_joints = sample.dof_count
        self._num_bodies = sample.body_count
        self._is_fixed_base = sample.is_fixed_base

        self.body_names = list(sample.body_names)
        self.joint_names = list(sample.dof_names)
        self.fixed_tendon_names: list[str] = []
        self.spatial_tendon_names: list[str] = []

        self._num_fixed_tendons = 0
        self._num_spatial_tendons = 0

    def update(self, dt: float) -> None:
        """Update the data for the articulation.

        Args:
            dt: The time step for the update [s]. This must be a positive value.
        """
        self._sim_timestamp += dt

        # Finite-difference joint acceleration from velocity.
        if dt > 0.0 and self._previous_joint_vel is not None:
            cur_vel = self.joint_vel
            wp.launch(
                _fd_joint_acc,
                dim=(self._num_instances, self._num_joints),
                inputs=[cur_vel, self._previous_joint_vel, 1.0 / dt],
                outputs=[self._joint_acc.data],
                device=self.device,
            )
            self._joint_acc.timestamp = self._sim_timestamp
            wp.copy(self._previous_joint_vel, cur_vel)

    """
    Names.
    """

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    joint_names: list[str] = None
    """Joint names in the order parsed by the simulation view."""

    fixed_tendon_names: list[str] = None
    """Fixed tendon names in the order parsed by the simulation view."""

    spatial_tendon_names: list[str] = None
    """Spatial tendon names in the order parsed by the simulation view."""

    """
    Defaults - Initial state.
    """

    @property
    def default_root_pose(self) -> wp.array:
        """Default root pose ``[pos, quat]`` in the local environment frame.

        The position and quaternion are of the articulation root's actor frame.
        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        """
        return self._default_root_pose

    @property
    def default_root_vel(self) -> wp.array:
        """Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame.

        The linear and angular velocities are of the articulation root's center of mass frame.
        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).
        """
        return self._default_root_vel

    @property
    def default_joint_pos(self) -> wp.array:
        """Default joint positions of all joints [m or rad, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_joint_pos

    @property
    def default_joint_vel(self) -> wp.array:
        """Default joint velocities of all joints [m/s or rad/s, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_joint_vel

    """
    Joint commands -- Set into simulation.
    """

    @property
    def joint_pos_target(self) -> wp.array:
        """Joint position targets commanded by the user [m or rad, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques
        (see :attr:`applied_torque`), which are then set into the simulation.
        """
        return self._joint_pos_target

    @property
    def joint_vel_target(self) -> wp.array:
        """Joint velocity targets commanded by the user [m/s or rad/s, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques
        (see :attr:`applied_torque`), which are then set into the simulation.
        """
        return self._joint_vel_target

    @property
    def joint_effort_target(self) -> wp.array:
        """Joint effort targets commanded by the user [N or N*m, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques
        (see :attr:`applied_torque`), which are then set into the simulation.
        """
        return self._joint_effort_target

    """
    Joint commands -- Explicit actuators.
    """

    @property
    def computed_torque(self) -> wp.array:
        """Joint torques computed from the actuator model (before clipping) [N*m].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).

        This quantity is the raw torque output from the actuator model, before any clipping is applied.
        It is exposed for users who want to inspect the computations inside the actuator model.
        For instance, to penalize the learning agent for a difference between the computed and applied torques.
        """
        return self._computed_torque

    @property
    def applied_torque(self) -> wp.array:
        """Joint torques applied from the actuator model (after clipping) [N*m].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).

        These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
        actuator model.
        """
        return self._applied_torque

    """
    Joint properties
    """

    @property
    def joint_stiffness(self) -> wp.array:
        """Joint stiffness provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._joint_stiffness

    @property
    def joint_damping(self) -> wp.array:
        """Joint damping provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._joint_damping

    @property
    def joint_armature(self) -> wp.array:
        """Joint armature provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        return self._joint_armature

    @property
    def joint_friction_coeff(self) -> wp.array:
        """Joint static friction coefficient provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        return self._joint_friction_coeff

    @property
    def joint_pos_limits(self) -> wp.array:
        """Joint position limits provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        return self._joint_pos_limits

    @property
    def joint_vel_limits(self) -> wp.array:
        """Joint maximum velocity provided to the simulation [m/s or rad/s, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        return self._joint_vel_limits

    @property
    def joint_effort_limits(self) -> wp.array:
        """Joint maximum effort provided to the simulation [N or N*m, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        return self._joint_effort_limits

    """
    Joint properties - Custom.
    """

    @property
    def soft_joint_pos_limits(self) -> wp.array:
        r"""Soft joint position limits for all joints.

        Shape is (num_instances, num_joints), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`. The soft joint position limits are computed as
        a sub-region of the :attr:`joint_pos_limits` based on the
        :attr:`~isaaclab.assets.ArticulationCfg.soft_joint_pos_limit_factor` parameter.

        Consider the joint position limits :math:`[lower, upper]` and the soft joint position limits
        :math:`[soft\_lower, soft\_upper]`. The soft joint position limits are computed as:

        .. math::

            soft\_lower = (lower + upper) / 2 - factor * (upper - lower) / 2
            soft\_upper = (lower + upper) / 2 + factor * (upper - lower) / 2

        The soft joint position limits help specify a safety region around the joint limits. It isn't used by the
        simulation, but is useful for learning agents to prevent the joint positions from violating the limits.
        """
        return self._soft_joint_pos_limits

    @property
    def soft_joint_vel_limits(self) -> wp.array:
        """Soft joint velocity limits for all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).

        These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
        has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
        """
        return self._soft_joint_vel_limits

    @property
    def gear_ratio(self) -> wp.array:
        """Gear ratio for relating motor torques to applied joint torques.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        return self._gear_ratio

    """
    Fixed tendon properties.
    """

    @property
    def fixed_tendon_stiffness(self) -> wp.array:
        """Fixed tendon stiffness provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        return self._fixed_tendon_stiffness

    @property
    def fixed_tendon_damping(self) -> wp.array:
        """Fixed tendon damping provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        return self._fixed_tendon_damping

    @property
    def fixed_tendon_limit_stiffness(self) -> wp.array:
        """Fixed tendon limit stiffness provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        return self._fixed_tendon_limit_stiffness

    @property
    def fixed_tendon_rest_length(self) -> wp.array:
        """Fixed tendon rest length provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        return self._fixed_tendon_rest_length

    @property
    def fixed_tendon_offset(self) -> wp.array:
        """Fixed tendon offset provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        return self._fixed_tendon_offset

    @property
    def fixed_tendon_pos_limits(self) -> wp.array:
        """Fixed tendon position limits provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_fixed_tendons, 2).
        """
        return self._fixed_tendon_pos_limits

    """
    Spatial tendon properties.
    """

    @property
    def spatial_tendon_stiffness(self) -> wp.array:
        """Spatial tendon stiffness provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        return self._spatial_tendon_stiffness

    @property
    def spatial_tendon_damping(self) -> wp.array:
        """Spatial tendon damping provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        return self._spatial_tendon_damping

    @property
    def spatial_tendon_limit_stiffness(self) -> wp.array:
        """Spatial tendon limit stiffness provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        return self._spatial_tendon_limit_stiffness

    @property
    def spatial_tendon_offset(self) -> wp.array:
        """Spatial tendon offset provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        return self._spatial_tendon_offset

    """
    Root state properties.
    """

    @property
    def root_link_pose_w(self) -> wp.array:
        """Root link pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).

        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        self._read_transform_binding(TT.ROOT_POSE, self._root_link_pose_w)
        return self._root_link_pose_w.data

    @property
    def root_link_vel_w(self) -> wp.array:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        # ovphysx ROOT_VELOCITY is COM velocity; link velocity comes from the first
        # element of the per-link velocity tensor.
        self._read_spatial_vector_binding(
            TT.LINK_VELOCITY, self._body_link_vel_w
        )
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                _copy_first_body,
                dim=self._num_instances,
                inputs=[self._body_link_vel_w.data],
                outputs=[self._root_link_vel_w.data],
                device=self.device,
            )
            self._root_link_vel_w.timestamp = self._sim_timestamp
        return self._root_link_vel_w.data

    @property
    def root_com_pose_w(self) -> wp.array:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).

        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            wp.launch(
                _compose_root_com_pose,
                dim=self._num_instances,
                inputs=[self.root_link_pose_w, self.body_com_pose_b],
                outputs=[self._root_com_pose_w.data],
                device=self.device,
            )
            self._root_com_pose_w.timestamp = self._sim_timestamp
        return self._root_com_pose_w.data

    @property
    def root_com_vel_w(self) -> wp.array:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        self._read_spatial_vector_binding(
            TT.ROOT_VELOCITY, self._root_com_vel_w
        )
        return self._root_com_vel_w.data

    """
    Body state properties.
    """

    @property
    def body_mass(self) -> wp.array:
        """Body mass in the world frame [kg].

        Shape is (num_instances, num_bodies), dtype = wp.float32. In torch this resolves to
        (num_instances, num_bodies).
        """
        return self._body_mass

    @property
    def body_inertia(self) -> wp.array:
        """Flattened body inertia in the world frame [kg*m^2].

        Shape is (num_instances, num_bodies, 9), dtype = wp.float32. In torch this resolves to
        (num_instances, num_bodies, 9).

        Stored as a flattened 3x3 inertia matrix per body.
        """
        return self._body_inertia

    @property
    def body_link_pose_w(self) -> wp.array:
        """Body link pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        self._read_transform_binding(TT.LINK_POSE, self._body_link_pose_w)
        return self._body_link_pose_w.data

    @property
    def body_link_vel_w(self) -> wp.array:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        self._read_spatial_vector_binding(
            TT.LINK_VELOCITY, self._body_link_vel_w
        )
        return self._body_link_vel_w.data

    @property
    def body_com_pose_w(self) -> wp.array:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            wp.launch(
                _compose_body_com_poses,
                dim=(self._num_instances, self._num_bodies),
                inputs=[self.body_link_pose_w, self.body_com_pose_b],
                outputs=[self._body_com_pose_w.data],
                device=self.device,
            )
            self._body_com_pose_w.timestamp = self._sim_timestamp
        return self._body_com_pose_w.data

    @property
    def body_com_vel_w(self) -> wp.array:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.

        .. note::
            This is currently approximated using the link velocity. A proper COM velocity derivation
            accounting for the COM offset is not yet implemented.
        """
        return self.body_link_vel_w

    @property
    def body_com_acc_w(self) -> wp.array:
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]`` [m/s^2, rad/s^2].
        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        self._read_spatial_vector_binding(
            TT.LINK_ACCELERATION, self._body_com_acc_w
        )
        return self._body_com_acc_w.data

    @property
    def body_com_pose_b(self) -> wp.array:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.
        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        self._read_transform_binding(
            TT.BODY_COM_POSE, self._body_com_pose_b
        )
        return self._body_com_pose_b.data

    @property
    def body_incoming_joint_wrench_b(self) -> wp.array:
        """Incoming joint wrenches on each body in the body frame [N, N*m].

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        All body reaction wrenches are provided including the root body to the world of an articulation.
        """
        self._read_spatial_vector_binding(
            TT.LINK_INCOMING_JOINT_FORCE,
            self._body_incoming_joint_wrench_buf,
        )
        return self._body_incoming_joint_wrench_buf.data

    """
    Joint state properties.
    """

    @property
    def joint_pos(self) -> wp.array:
        """Joint positions of all joints [m or rad, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        self._read_binding_into_buf(TT.DOF_POSITION, self._joint_pos_buf)
        return self._joint_pos_buf.data

    @property
    def joint_vel(self) -> wp.array:
        """Joint velocities of all joints [m/s or rad/s, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        self._read_binding_into_buf(TT.DOF_VELOCITY, self._joint_vel_buf)
        return self._joint_vel_buf.data

    @property
    def joint_acc(self) -> wp.array:
        """Joint acceleration of all joints [m/s^2 or rad/s^2, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).

        .. note::
            This quantity is computed via finite differencing of joint velocities.
        """
        return self._joint_acc.data

    """
    Derived Properties.
    """

    @property
    def projected_gravity_b(self) -> wp.array:
        """Projection of the gravity direction on base frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            wp.launch(
                _projected_gravity,
                dim=self._num_instances,
                inputs=[self.root_link_pose_w],
                outputs=[self._projected_gravity_b.data],
                device=self.device,
            )
            self._projected_gravity_b.timestamp = self._sim_timestamp
        return self._projected_gravity_b.data

    @property
    def heading_w(self) -> wp.array:
        """Yaw heading of the base frame (in radians).
        Shape is (num_instances,), dtype = wp.float32.

        .. note::
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        if self._heading_w.timestamp < self._sim_timestamp:
            wp.launch(
                _compute_heading,
                dim=self._num_instances,
                inputs=[self.root_link_pose_w],
                outputs=[self._heading_w.data],
                device=self.device,
            )
            self._heading_w.timestamp = self._sim_timestamp
        return self._heading_w.data

    @property
    def root_link_lin_vel_b(self) -> wp.array:
        """Root link linear velocity in base frame [m/s].
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to its actor frame.
        """
        if self._root_link_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                _world_vel_to_body_lin,
                dim=self._num_instances,
                inputs=[self.root_link_pose_w, self.root_link_vel_w],
                outputs=[self._root_link_lin_vel_b.data],
                device=self.device,
            )
            self._root_link_lin_vel_b.timestamp = self._sim_timestamp
        return self._root_link_lin_vel_b.data

    @property
    def root_link_ang_vel_b(self) -> wp.array:
        """Root link angular velocity in base frame [rad/s].
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to its actor frame.
        """
        if self._root_link_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                _world_vel_to_body_ang,
                dim=self._num_instances,
                inputs=[self.root_link_pose_w, self.root_link_vel_w],
                outputs=[self._root_link_ang_vel_b.data],
                device=self.device,
            )
            self._root_link_ang_vel_b.timestamp = self._sim_timestamp
        return self._root_link_ang_vel_b.data

    @property
    def root_com_lin_vel_b(self) -> wp.array:
        """Root center of mass linear velocity in base frame [m/s].
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame
        with respect to its actor frame.
        """
        if self._root_com_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                _world_vel_to_body_lin,
                dim=self._num_instances,
                inputs=[self.root_link_pose_w, self.root_com_vel_w],
                outputs=[self._root_com_lin_vel_b.data],
                device=self.device,
            )
            self._root_com_lin_vel_b.timestamp = self._sim_timestamp
        return self._root_com_lin_vel_b.data

    @property
    def root_com_ang_vel_b(self) -> wp.array:
        """Root center of mass angular velocity in base frame [rad/s].
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame
        with respect to its actor frame.
        """
        if self._root_com_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                _world_vel_to_body_ang,
                dim=self._num_instances,
                inputs=[self.root_link_pose_w, self.root_com_vel_w],
                outputs=[self._root_com_ang_vel_b.data],
                device=self.device,
            )
            self._root_com_ang_vel_b.timestamp = self._sim_timestamp
        return self._root_com_ang_vel_b.data

    """
    Sliced properties.
    """

    @property
    def root_link_pos_w(self) -> wp.array:
        """Root link position in simulation world frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._get_pos_from_transform(self.root_link_pose_w)

    @property
    def root_link_quat_w(self) -> wp.array:
        """Root link orientation (x, y, z, w) in simulation world frame.
        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self._get_quat_from_transform(self.root_link_pose_w)

    @property
    def root_link_lin_vel_w(self) -> wp.array:
        """Root linear velocity in simulation world frame [m/s].
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._get_lin_vel_from_spatial_vector(self.root_link_vel_w)

    @property
    def root_link_ang_vel_w(self) -> wp.array:
        """Root link angular velocity in simulation world frame [rad/s].
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self._get_ang_vel_from_spatial_vector(self.root_link_vel_w)

    @property
    def root_com_pos_w(self) -> wp.array:
        """Root center of mass position in simulation world frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the position of the center of mass frame of the root rigid body relative to the world.
        """
        return self._get_pos_from_transform(self.root_com_pose_w)

    @property
    def root_com_quat_w(self) -> wp.array:
        """Root center of mass orientation (x, y, z, w) in simulation world frame.
        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).

        This quantity is the orientation of the principal axes of inertia of the root rigid body relative to the world.
        """
        return self._get_quat_from_transform(self.root_com_pose_w)

    @property
    def root_com_lin_vel_w(self) -> wp.array:
        """Root center of mass linear velocity in simulation world frame [m/s].
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._get_lin_vel_from_spatial_vector(self.root_com_vel_w)

    @property
    def root_com_ang_vel_w(self) -> wp.array:
        """Root center of mass angular velocity in simulation world frame [rad/s].
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._get_ang_vel_from_spatial_vector(self.root_com_vel_w)

    @property
    def body_link_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        return self._get_pos_from_transform(self.body_link_pose_w)

    @property
    def body_link_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        return self._get_quat_from_transform(self.body_link_pose_w)

    @property
    def body_link_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame [m/s].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' actor frame relative to the world.
        """
        return self._get_lin_vel_from_spatial_vector(self.body_link_vel_w)

    @property
    def body_link_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame [rad/s].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' actor frame relative to the world.
        """
        return self._get_ang_vel_from_spatial_vector(self.body_link_vel_w)

    @property
    def body_com_pos_w(self) -> wp.array:
        """Positions of all bodies' center of mass in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' center of mass frame.
        """
        return self._get_pos_from_transform(self.body_com_pose_w)

    @property
    def body_com_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' principal axes of inertia.
        """
        return self._get_quat_from_transform(self.body_com_pose_w)

    @property
    def body_com_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame [m/s].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        return self._get_lin_vel_from_spatial_vector(self.body_com_vel_w)

    @property
    def body_com_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame [rad/s].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        return self._get_ang_vel_from_spatial_vector(self.body_com_vel_w)

    @property
    def body_com_lin_acc_w(self) -> wp.array:
        """Linear acceleration of all bodies in simulation world frame [m/s^2].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        return self._get_lin_vel_from_spatial_vector(self.body_com_acc_w)

    @property
    def body_com_ang_acc_w(self) -> wp.array:
        """Angular acceleration of all bodies in simulation world frame [rad/s^2].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        return self._get_ang_vel_from_spatial_vector(self.body_com_acc_w)

    @property
    def body_com_pos_b(self) -> wp.array:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body's link frame.
        """
        return self._get_pos_from_transform(self.body_com_pose_b)

    @property
    def body_com_quat_b(self) -> wp.array:
        """Orientation (x, y, z, w) of the principal axes of inertia of all of the bodies in their
        respective link frames.
        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the principal axes of inertia relative to its body's link frame.
        """
        return self._get_quat_from_transform(self.body_com_pose_b)

    """
    Deprecated in base class (required by ABC for backward compatibility).
    """

    @property
    def default_root_state(self) -> wp.array:
        """Deprecated. Use :attr:`default_root_pose` and :attr:`default_root_vel` instead."""
        warnings.warn(
            "default_root_state is deprecated. Use default_root_pose and default_root_vel.",
            DeprecationWarning, stacklevel=2,
        )
        if self._root_state_w_buf is None:
            self._root_state_w_buf = wp.zeros(self._num_instances, dtype=wp.types.vector(13, wp.float32), device=self.device)
        return self._root_state_w_buf

    @property
    def root_state_w(self) -> wp.array:
        """Deprecated. Use :attr:`root_link_pose_w` and :attr:`root_com_vel_w` instead."""
        warnings.warn(
            "root_state_w is deprecated. Use root_link_pose_w and root_com_vel_w.",
            DeprecationWarning, stacklevel=2,
        )
        return self.root_link_pose_w

    @property
    def root_link_state_w(self) -> wp.array:
        """Deprecated. Use :attr:`root_link_pose_w` and :attr:`root_link_vel_w` instead."""
        warnings.warn(
            "root_link_state_w is deprecated. Use root_link_pose_w and root_link_vel_w.",
            DeprecationWarning, stacklevel=2,
        )
        return self.root_link_pose_w

    @property
    def root_com_state_w(self) -> wp.array:
        """Deprecated. Use :attr:`root_com_pose_w` and :attr:`root_com_vel_w` instead."""
        warnings.warn(
            "root_com_state_w is deprecated. Use root_com_pose_w and root_com_vel_w.",
            DeprecationWarning, stacklevel=2,
        )
        return self.root_com_pose_w

    @property
    def body_state_w(self) -> wp.array:
        """Deprecated. Use :attr:`body_link_pose_w` and :attr:`body_com_vel_w` instead."""
        warnings.warn(
            "body_state_w is deprecated. Use body_link_pose_w and body_com_vel_w.",
            DeprecationWarning, stacklevel=2,
        )
        return self.body_link_pose_w

    @property
    def body_link_state_w(self) -> wp.array:
        """Deprecated. Use :attr:`body_link_pose_w` and :attr:`body_link_vel_w` instead."""
        warnings.warn(
            "body_link_state_w is deprecated. Use body_link_pose_w and body_link_vel_w.",
            DeprecationWarning, stacklevel=2,
        )
        return self.body_link_pose_w

    @property
    def body_com_state_w(self) -> wp.array:
        """Deprecated. Use :attr:`body_com_pose_w` and :attr:`body_com_vel_w` instead."""
        warnings.warn(
            "body_com_state_w is deprecated. Use body_com_pose_w and body_com_vel_w.",
            DeprecationWarning, stacklevel=2,
        )
        return self.body_com_pose_w

    # ------------------------------------------------------------------
    # Buffer creation (called once after initialization)
    # ------------------------------------------------------------------

    def _create_buffers(self) -> None:  # noqa: C901
        super()._create_buffers()
        # Scratch buffers for _read_binding_into_* methods, allocated lazily
        # on first use and reused every subsequent step to avoid per-step
        # allocation overhead on the hot RL path.
        self._read_scratch: dict = {}

        N = self._num_instances
        D = self._num_joints
        L = self._num_bodies
        dev = self.device

        # -- Root state buffers
        self._root_link_pose_w = TimestampedBuffer(N, dev, wp.transformf)
        self._root_link_vel_w = TimestampedBuffer(N, dev, wp.spatial_vectorf)
        self._root_com_pose_w = TimestampedBuffer(N, dev, wp.transformf)
        self._root_com_vel_w = TimestampedBuffer(N, dev, wp.spatial_vectorf)

        # -- Body state buffers
        self._body_link_pose_w = TimestampedBuffer((N, L), dev, wp.transformf)
        self._body_link_vel_w = TimestampedBuffer((N, L), dev, wp.spatial_vectorf)
        self._body_com_pose_b = TimestampedBuffer((N, L), dev, wp.transformf)
        self._body_com_pose_w = TimestampedBuffer((N, L), dev, wp.transformf)
        self._body_com_vel_w = TimestampedBuffer((N, L), dev, wp.spatial_vectorf)
        self._body_com_acc_w = TimestampedBuffer((N, L), dev, wp.spatial_vectorf)
        self._body_incoming_joint_wrench_buf = TimestampedBuffer((N, L), dev, wp.spatial_vectorf)

        # -- Joint state buffers
        self._joint_pos_buf = TimestampedBuffer((N, D), dev, wp.float32)
        self._joint_vel_buf = TimestampedBuffer((N, D), dev, wp.float32)
        self._joint_acc = TimestampedBuffer((N, D), dev, wp.float32)
        self._previous_joint_vel = wp.zeros((N, D), dtype=wp.float32, device=dev)

        # -- Joint properties
        self._joint_stiffness = wp.zeros((N, D), dtype=wp.float32, device=dev)
        self._joint_damping = wp.zeros((N, D), dtype=wp.float32, device=dev)
        self._joint_armature = wp.zeros((N, D), dtype=wp.float32, device=dev)
        self._joint_friction_coeff = wp.zeros((N, D), dtype=wp.float32, device=dev)
        self._joint_pos_limits = wp.zeros((N, D), dtype=wp.vec2f, device=dev)
        self._joint_vel_limits = wp.zeros((N, D), dtype=wp.float32, device=dev)
        self._joint_effort_limits = wp.zeros((N, D), dtype=wp.float32, device=dev)

        # -- Body properties
        self._body_mass = wp.zeros((N, L), dtype=wp.float32, device=dev)
        self._body_inertia = wp.zeros((N, L, 9), dtype=wp.float32, device=dev)

        # -- Soft limits / custom properties
        self._soft_joint_pos_limits = wp.zeros((N, D), dtype=wp.vec2f, device=dev)
        self._soft_joint_vel_limits = wp.zeros((N, D), dtype=wp.float32, device=dev)
        self._gear_ratio = wp.ones((N, D), dtype=wp.float32, device=dev)

        # -- Command buffers
        self._joint_pos_target = wp.zeros((N, D), dtype=wp.float32, device=dev)
        self._joint_vel_target = wp.zeros((N, D), dtype=wp.float32, device=dev)
        self._joint_effort_target = wp.zeros((N, D), dtype=wp.float32, device=dev)
        self._computed_torque = wp.zeros((N, D), dtype=wp.float32, device=dev)
        self._applied_torque = wp.zeros((N, D), dtype=wp.float32, device=dev)

        # -- Default state
        self._default_root_pose = wp.zeros(N, dtype=wp.transformf, device=dev)
        self._default_root_vel = wp.zeros(N, dtype=wp.spatial_vectorf, device=dev)
        self._default_joint_pos = wp.zeros((N, D), dtype=wp.float32, device=dev)
        self._default_joint_vel = wp.zeros((N, D), dtype=wp.float32, device=dev)

        # -- Derived property buffers
        self._projected_gravity_b = TimestampedBuffer(N, dev, wp.vec3f)
        self._heading_w = TimestampedBuffer(N, dev, wp.float32)
        self._root_link_lin_vel_b = TimestampedBuffer(N, dev, wp.vec3f)
        self._root_link_ang_vel_b = TimestampedBuffer(N, dev, wp.vec3f)
        self._root_com_lin_vel_b = TimestampedBuffer(N, dev, wp.vec3f)
        self._root_com_ang_vel_b = TimestampedBuffer(N, dev, wp.vec3f)

        # -- Deprecated combined state buffers
        self._root_state_w_buf = None
        self._root_link_state_w_buf = None
        self._root_com_state_w_buf = None
        self._body_state_w_buf = None
        self._body_link_state_w_buf = None
        self._body_com_state_w_buf = None

        # -- Tendon property buffers
        T_fix = getattr(self, "_num_fixed_tendons", 0)
        T_spa = getattr(self, "_num_spatial_tendons", 0)
        if T_fix > 0:
            self._fixed_tendon_stiffness = wp.zeros((N, T_fix), dtype=wp.float32, device=dev)
            self._fixed_tendon_damping = wp.zeros((N, T_fix), dtype=wp.float32, device=dev)
            self._fixed_tendon_limit_stiffness = wp.zeros((N, T_fix), dtype=wp.float32, device=dev)
            self._fixed_tendon_rest_length = wp.zeros((N, T_fix), dtype=wp.float32, device=dev)
            self._fixed_tendon_offset = wp.zeros((N, T_fix), dtype=wp.float32, device=dev)
            self._fixed_tendon_pos_limits = wp.zeros((N, T_fix), dtype=wp.vec2f, device=dev)
        else:
            self._fixed_tendon_stiffness = None
            self._fixed_tendon_damping = None
            self._fixed_tendon_limit_stiffness = None
            self._fixed_tendon_rest_length = None
            self._fixed_tendon_offset = None
            self._fixed_tendon_pos_limits = None
        if T_spa > 0:
            self._spatial_tendon_stiffness = wp.zeros((N, T_spa), dtype=wp.float32, device=dev)
            self._spatial_tendon_damping = wp.zeros((N, T_spa), dtype=wp.float32, device=dev)
            self._spatial_tendon_limit_stiffness = wp.zeros((N, T_spa), dtype=wp.float32, device=dev)
            self._spatial_tendon_offset = wp.zeros((N, T_spa), dtype=wp.float32, device=dev)
        else:
            self._spatial_tendon_stiffness = None
            self._spatial_tendon_damping = None
            self._spatial_tendon_limit_stiffness = None
            self._spatial_tendon_offset = None

        # Read initial joint properties from bindings
        self._read_initial_properties()

    def _read_initial_properties(self) -> None:
        """Read static/initial joint and body properties from ovphysx bindings.

        These are one-time reads at init.  Property tensors (stiffness,
        damping, limits, mass, etc.) are CPU-resident in PhysX even in GPU
        mode, so we read them via CPU numpy buffers and then copy to the
        simulation device.
        """
        # Property reads always use CPU numpy (property tensors are host-side).
        def _read_cpu(tensor_type):
            binding = self._get_binding(tensor_type)
            if binding is None:
                return None
            np_buf = np.zeros(binding.shape, dtype=np.float32)
            binding.read(np_buf)
            return np_buf

        for tt, dst in [
            (TT.DOF_STIFFNESS, self._joint_stiffness),
            (TT.DOF_DAMPING, self._joint_damping),
            (TT.DOF_ARMATURE, self._joint_armature),
            (TT.DOF_MAX_VELOCITY, self._joint_vel_limits),
            (TT.DOF_MAX_FORCE, self._joint_effort_limits),
            (TT.BODY_MASS, self._body_mass),
        ]:
            np_buf = _read_cpu(tt)
            if np_buf is not None:
                wp.copy(dst, wp.from_numpy(np_buf, dtype=wp.float32, device=self.device))

        # Joint position limits: [N, D, 2] -> (N, D) wp.vec2f
        np_lim = _read_cpu(TT.DOF_LIMIT)
        if np_lim is not None:
            self._joint_pos_limits = wp.from_numpy(
                np_lim.reshape(self._num_instances, self._num_joints, 2), dtype=wp.vec2f, device=self.device
            )

        # Body inertia: [N, L, 9]
        np_iner = _read_cpu(TT.BODY_INERTIA)
        if np_iner is not None:
            self._body_inertia = wp.from_numpy(np_iner, dtype=wp.float32, device=self.device)

        # Friction: [N, D, 3] -> extract static friction (column 0)
        np_fric = _read_cpu(TT.DOF_FRICTION_PROPERTIES)
        if np_fric is not None:
            self._joint_friction_coeff = wp.from_numpy(
                np_fric[..., 0].copy(), dtype=wp.float32, device=self.device
            )

        # Fixed tendon properties (CPU-side, read once)
        T_fix = getattr(self, "_num_fixed_tendons", 0)
        if T_fix > 0:
            for tt, dst in [
                (TT.FIXED_TENDON_STIFFNESS, self._fixed_tendon_stiffness),
                (TT.FIXED_TENDON_DAMPING, self._fixed_tendon_damping),
                (TT.FIXED_TENDON_LIMIT_STIFFNESS, self._fixed_tendon_limit_stiffness),
                (TT.FIXED_TENDON_REST_LENGTH, self._fixed_tendon_rest_length),
                (TT.FIXED_TENDON_OFFSET, self._fixed_tendon_offset),
            ]:
                np_buf = _read_cpu(tt)
                if np_buf is not None and dst is not None:
                    wp.copy(dst, wp.from_numpy(np_buf, dtype=wp.float32, device=self.device))
            # Fixed tendon limits: [N, T, 2] -> (N, T) wp.vec2f
            np_tlim = _read_cpu(TT.FIXED_TENDON_LIMIT)
            if np_tlim is not None and self._fixed_tendon_pos_limits is not None:
                self._fixed_tendon_pos_limits = wp.from_numpy(
                    np_tlim.reshape(self._num_instances, T_fix, 2), dtype=wp.vec2f, device=self.device
                )

        # Spatial tendon properties (CPU-side, read once)
        T_spa = getattr(self, "_num_spatial_tendons", 0)
        if T_spa > 0:
            for tt, dst in [
                (TT.SPATIAL_TENDON_STIFFNESS, self._spatial_tendon_stiffness),
                (TT.SPATIAL_TENDON_DAMPING, self._spatial_tendon_damping),
                (TT.SPATIAL_TENDON_LIMIT_STIFFNESS, self._spatial_tendon_limit_stiffness),
                (TT.SPATIAL_TENDON_OFFSET, self._spatial_tendon_offset),
            ]:
                np_buf = _read_cpu(tt)
                if np_buf is not None and dst is not None:
                    wp.copy(dst, wp.from_numpy(np_buf, dtype=wp.float32, device=self.device))

    # ------------------------------------------------------------------
    # Binding read helpers
    # ------------------------------------------------------------------

    def _get_binding(self, tensor_type: int):
        """Return a binding, lazily creating it if a binding_getter was provided."""
        b = self._bindings.get(tensor_type)
        if b is not None:
            return b
        if self._binding_getter is not None:
            b = self._binding_getter(tensor_type)
            if b is not None:
                self._bindings[tensor_type] = b
            return b
        return None

    def _get_read_scratch(self, tensor_type: int) -> wp.array | None:
        """Return a pre-allocated flat float32 scratch buffer for a binding.

        Allocated once on first use, then reused every step.  CPU-only
        bindings (body properties, DOF properties) get CPU scratch; GPU
        bindings get GPU scratch.  wp.copy handles cross-device transfer
        when the destination buffer lives on a different device.
        """
        if tensor_type in self._read_scratch:
            return self._read_scratch[tensor_type]
        binding = self._get_binding(tensor_type)
        if binding is None:
            return None
        from isaaclab_ovphysx.tensor_types import _CPU_ONLY_TYPES
        dev = "cpu" if tensor_type in _CPU_ONLY_TYPES else self.device
        buf = wp.zeros(binding.shape, dtype=wp.float32, device=dev)
        self._read_scratch[tensor_type] = buf
        return buf

    def _read_binding_into_flat(self, tensor_type: int, wp_array: wp.array) -> None:
        """Read a flat binding (no structured dtype) into an existing warp array.

        Full GPU path: ovphysx reads via DLPack into the scratch buffer on
        the simulation device, then wp.copy stays on the same device.
        """
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        scratch = self._get_read_scratch(tensor_type)
        binding.read(scratch)
        wp.copy(wp_array, scratch)

    def _read_binding_into_buf(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read from an ovphysx binding into a TimestampedBuffer, skipping if fresh."""
        if buf.timestamp >= self._sim_timestamp:
            return
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        scratch = self._get_read_scratch(tensor_type)
        binding.read(scratch)
        wp.copy(buf.data, scratch)
        buf.timestamp = self._sim_timestamp

    def _read_transform_binding(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read a pose binding ([N, 7] or [N, L, 7]) into a wp.transformf buffer.

        GPU-native path: ovphysx reads via DLPack into a flat float32 scratch
        buffer on the sim device, then we copy the raw bytes directly into the
        transformf destination buffer (same memory layout: 7 contiguous floats
        per element).  No CPU roundtrip, no numpy.
        """
        if buf.timestamp >= self._sim_timestamp:
            return
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        scratch = self._get_read_scratch(tensor_type)
        binding.read(scratch)
        # scratch is [N, 7] or [N, L, 7] float32; buf.data is [N] or [N, L] transformf.
        # Both have identical byte layouts (7 contiguous float32 per element).
        # Use wp.copy with a flat float32 view of the destination to avoid
        # dtype mismatch.  The flat view aliases buf.data's memory.
        n_elements = 1
        for s in buf.data.shape:
            n_elements *= s
        dst_flat = wp.array(
            ptr=buf.data.ptr, shape=(n_elements * 7,),
            dtype=wp.float32, device=str(buf.data.device), copy=False,
        )
        src_flat = wp.array(
            ptr=scratch.ptr, shape=(n_elements * 7,),
            dtype=wp.float32, device=str(scratch.device), copy=False,
        )
        wp.copy(dst_flat, src_flat)
        buf.timestamp = self._sim_timestamp

    def _read_spatial_vector_binding(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read a velocity binding ([N, 6] or [N, L, 6]) into a spatial_vectorf buffer.

        Same byte-copy path as _read_transform_binding. wp.copy handles
        cross-device transfer when scratch is CPU and buf is GPU.
        """
        if buf.timestamp >= self._sim_timestamp:
            return
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        scratch = self._get_read_scratch(tensor_type)
        binding.read(scratch)
        n_elements = 1
        for s in buf.data.shape:
            n_elements *= s
        dst_flat = wp.array(
            ptr=buf.data.ptr, shape=(n_elements * 6,),
            dtype=wp.float32, device=str(buf.data.device), copy=False,
        )
        src_flat = wp.array(
            ptr=scratch.ptr, shape=(n_elements * 6,),
            dtype=wp.float32, device=str(scratch.device), copy=False,
        )
        wp.copy(dst_flat, src_flat)
        buf.timestamp = self._sim_timestamp

    # ------------------------------------------------------------------
    # Extraction helpers (slice pos/quat/lin_vel/ang_vel from structured)
    # ------------------------------------------------------------------

    def _get_pos_from_transform(self, transform: wp.array) -> wp.array:
        return wp.array(
            ptr=transform.ptr, shape=transform.shape, dtype=wp.vec3f,
            strides=transform.strides, device=self.device,
        )

    def _get_quat_from_transform(self, transform: wp.array) -> wp.array:
        return wp.array(
            ptr=transform.ptr + 3 * 4, shape=transform.shape, dtype=wp.quatf,
            strides=transform.strides, device=self.device,
        )

    def _get_lin_vel_from_spatial_vector(self, sv: wp.array) -> wp.array:
        return wp.array(
            ptr=sv.ptr, shape=sv.shape, dtype=wp.vec3f,
            strides=sv.strides, device=self.device,
        )

    def _get_ang_vel_from_spatial_vector(self, sv: wp.array) -> wp.array:
        return wp.array(
            ptr=sv.ptr + 3 * 4, shape=sv.shape, dtype=wp.vec3f,
            strides=sv.strides, device=self.device,
        )


# ======================================================================
# Warp kernels
# ======================================================================

@wp.kernel
def _fd_joint_acc(
    cur_vel: wp.array2d(dtype=wp.float32),
    prev_vel: wp.array2d(dtype=wp.float32),
    inv_dt: float,
    out: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    out[i, j] = (cur_vel[i, j] - prev_vel[i, j]) * inv_dt


@wp.kernel
def _copy_first_body(
    body_vel: wp.array(dtype=wp.spatial_vectorf, ndim=2),
    root_vel: wp.array(dtype=wp.spatial_vectorf),
):
    i = wp.tid()
    root_vel[i] = body_vel[i, 0]


@wp.kernel
def _compose_root_com_pose(
    link_pose: wp.array(dtype=wp.transformf),
    com_pose_b: wp.array(dtype=wp.transformf, ndim=2),
    com_pose_w: wp.array(dtype=wp.transformf),
):
    i = wp.tid()
    com_pose_w[i] = wp.transform_multiply(link_pose[i], com_pose_b[i, 0])


@wp.kernel
def _compose_body_com_poses(
    link_pose: wp.array(dtype=wp.transformf, ndim=2),
    com_pose_b: wp.array(dtype=wp.transformf, ndim=2),
    com_pose_w: wp.array(dtype=wp.transformf, ndim=2),
):
    i, j = wp.tid()
    com_pose_w[i, j] = wp.transform_multiply(link_pose[i, j], com_pose_b[i, j])


@wp.kernel
def _projected_gravity(
    root_pose: wp.array(dtype=wp.transformf),
    out: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    gravity_w = wp.vec3f(0.0, 0.0, -1.0)
    out[i] = wp.quat_rotate_inv(q, gravity_w)


@wp.kernel
def _compute_heading(
    root_pose: wp.array(dtype=wp.transformf),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    forward = wp.quat_rotate(q, wp.vec3f(1.0, 0.0, 0.0))
    out[i] = wp.atan2(forward[1], forward[0])


@wp.kernel
def _world_vel_to_body_lin(
    root_pose: wp.array(dtype=wp.transformf),
    vel_w: wp.array(dtype=wp.spatial_vectorf),
    out: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    lin = wp.spatial_top(vel_w[i])
    out[i] = wp.quat_rotate_inv(q, lin)


@wp.kernel
def _world_vel_to_body_ang(
    root_pose: wp.array(dtype=wp.transformf),
    vel_w: wp.array(dtype=wp.spatial_vectorf),
    out: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    ang = wp.spatial_bottom(vel_w[i])
    out[i] = wp.quat_rotate_inv(q, ang)
