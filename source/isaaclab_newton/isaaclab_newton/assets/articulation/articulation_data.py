# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import logging
import warnings
import weakref
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.math import normalize
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.utils import capture_unsafe

from isaaclab_newton.assets import kernels as shared_kernels
from isaaclab_newton.assets.articulation import kernels as articulation_kernels
from isaaclab_newton.physics import NewtonManager as SimulationManager

if TYPE_CHECKING:
    from newton.selection import ArticulationView

# import logger
logger = logging.getLogger(__name__)

_LAZY_CAPTURE_REASON = (
    "This is a lazily-computed derived property guarded by a Python timestamp check "
    "that is invisible during graph replay.  Use Tier 1 base data (root_link_pose_w, "
    "root_com_vel_w, body_link_pose_w, body_com_vel_w, joint_pos, joint_vel) and "
    "inline the computation in your warp kernel.  See GRAPH_CAPTURE_MIGRATION.md."
)


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
    """

    __backend_name__: str = "newton"
    """The name of the backend for the articulation data."""

    def __init__(self, root_view: ArticulationView, device: str):
        """Initializes the articulation data.

        Args:
            root_view: The root articulation view.
            device: The device used for processing.
        """
        super().__init__(root_view, device)
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_view: ArticulationView = weakref.proxy(root_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0
        self._is_primed = False
        self._fk_timestamp = 0.0

        # Convert to direction vector
        gravity = wp.to_torch(SimulationManager.get_model().gravity)[0]
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        gravity_dir = gravity_dir.repeat(self._root_view.count, 1)
        forward_vec = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_view.count, 1)

        # Initialize constants
        self.GRAVITY_VEC_W = ProxyArray(wp.from_torch(gravity_dir, dtype=wp.vec3f))
        self.FORWARD_VEC_B = ProxyArray(wp.from_torch(forward_vec, dtype=wp.vec3f))

        self._create_simulation_bindings()
        self._create_buffers()

    @property
    def is_primed(self) -> bool:
        """Whether the articulation data is fully instantiated and ready to use."""
        return self._is_primed

    @is_primed.setter
    def is_primed(self, value: bool) -> None:
        """Set whether the articulation data is fully instantiated and ready to use.

        .. note::
            Once this quantity is set to True, it cannot be changed.

        Args:
            value: The primed state.

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self._is_primed:
            raise ValueError("The articulation data is already primed.")
        self._is_primed = True

    def update(self, dt: float) -> None:
        """Updates the data for the articulation.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt
        # FK is current after a sim step — keep fk_timestamp in sync unless it was explicitly invalidated
        if self._fk_timestamp >= 0.0:
            self._fk_timestamp = self._sim_timestamp
        # Trigger an update of the joint and body com acceleration buffers at a higher frequency
        # since we do finite differencing.
        self.joint_acc
        self.body_com_acc_w

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
    def default_root_pose(self) -> ProxyArray:
        """Default root pose ``[pos, quat]`` in the local environment frame.

        The position and quaternion are of the articulation root's actor frame. Shape is (num_instances),
        dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        """
        return self._default_root_pose_ta

    @default_root_pose.setter
    def default_root_pose(self, value: wp.array) -> None:
        """Set the default root pose.

        Args:
            value: The default root pose. Shape is (num_instances, 7).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self.is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_root_pose.assign(value)

    @property
    def default_root_vel(self) -> ProxyArray:
        """Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame.

        The linear and angular velocities are of the articulation root's center of mass frame.
        Shape is (num_instances), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).
        """
        return self._default_root_vel_ta

    @default_root_vel.setter
    def default_root_vel(self, value: wp.array) -> None:
        """Set the default root velocity.

        Args:
            value: The default root velocity. Shape is (num_instances, 6).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self.is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_root_vel.assign(value)

    @property
    def default_joint_pos(self) -> ProxyArray:
        """Default joint positions of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_joint_pos_ta

    @default_joint_pos.setter
    def default_joint_pos(self, value: wp.array) -> None:
        """Set the default joint positions.

        Args:
            value: The default joint positions. Shape is (num_instances, num_joints).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self.is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_joint_pos.assign(value)

    @property
    def default_joint_vel(self) -> ProxyArray:
        """Default joint velocities of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_joint_vel_ta

    @default_joint_vel.setter
    def default_joint_vel(self, value: wp.array) -> None:
        """Set the default joint velocities.

        Args:
            value: The default joint velocities. Shape is (num_instances, num_joints).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self.is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_joint_vel.assign(value)

    """
    Joint commands -- Set into simulation.
    """

    @property
    def joint_pos_target(self) -> ProxyArray:
        """Joint position targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return self._joint_pos_target_ta

    @property
    def joint_vel_target(self) -> ProxyArray:
        """Joint velocity targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return self._joint_vel_target_ta

    @property
    def joint_effort_target(self) -> ProxyArray:
        """Joint effort targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return self._joint_effort_target_ta

    """
    Joint commands -- Explicit actuators.
    """

    @property
    def computed_torque(self) -> ProxyArray:
        """Joint torques computed from the actuator model (before clipping).

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        This quantity is the raw torque output from the actuator mode, before any clipping is applied.
        It is exposed for users who want to inspect the computations inside the actuator model.
        For instance, to penalize the learning agent for a difference between the computed and applied torques.
        """
        return self._computed_torque_ta

    @property
    def applied_torque(self) -> ProxyArray:
        """Joint torques applied from the actuator model (after clipping).

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
        actuator model.
        """
        return self._applied_torque_ta

    """
    Joint properties
    """

    @property
    def joint_stiffness(self) -> ProxyArray:
        """Joint stiffness provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._joint_stiffness_ta

    @property
    def joint_damping(self) -> ProxyArray:
        """Joint damping provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._joint_damping_ta

    @property
    def joint_armature(self) -> ProxyArray:
        """Joint armature provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        return self._joint_armature_ta

    @property
    def joint_friction_coeff(self) -> ProxyArray:
        """Joint static friction coefficient provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        return self._joint_friction_coeff_ta

    @property
    def joint_pos_limits_lower(self) -> ProxyArray:
        """Joint position limits lower provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_pos_limits_lower_ta

    @property
    def joint_pos_limits_upper(self) -> ProxyArray:
        """Joint position limits upper provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_pos_limits_upper_ta

    @property
    def joint_pos_limits(self) -> ProxyArray:
        """Joint position limits provided to the simulation.

        Shape is (num_instances, num_joints, 2), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        if self._joint_pos_limits is None:
            self._joint_pos_limits = wp.zeros(
                (self._num_instances, self._num_joints), dtype=wp.vec2f, device=self.device
            )
            self._joint_pos_limits_ta = ProxyArray(self._joint_pos_limits)
        wp.launch(
            articulation_kernels.concat_joint_pos_limits_lower_and_upper,
            dim=(self._num_instances, self._num_joints),
            inputs=[
                self._sim_bind_joint_pos_limits_lower,
                self._sim_bind_joint_pos_limits_upper,
            ],
            outputs=[
                self._joint_pos_limits,
            ],
            device=self.device,
        )
        return self._joint_pos_limits_ta

    @property
    def joint_vel_limits(self) -> ProxyArray:
        """Joint maximum velocity provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        return self._joint_vel_limits_ta

    @property
    def joint_effort_limits(self) -> ProxyArray:
        """Joint maximum effort provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        return self._joint_effort_limits_ta

    """
    Joint properties - Custom.
    """

    @property
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
        return self._soft_joint_pos_limits_ta

    @property
    def soft_joint_vel_limits(self) -> ProxyArray:
        """Soft joint velocity limits for all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
        has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
        """
        return self._soft_joint_vel_limits_ta

    @property
    def gear_ratio(self) -> ProxyArray:
        """Gear ratio for relating motor torques to applied Joint torques.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        return self._gear_ratio_ta

    """
    Fixed tendon properties.
    """

    @property
    def fixed_tendon_stiffness(self) -> ProxyArray:
        """Fixed tendon stiffness provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    def fixed_tendon_damping(self) -> ProxyArray:
        """Fixed tendon damping provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    def fixed_tendon_limit_stiffness(self) -> ProxyArray:
        """Fixed tendon limit stiffness provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    def fixed_tendon_rest_length(self) -> ProxyArray:
        """Fixed tendon rest length provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    def fixed_tendon_offset(self) -> ProxyArray:
        """Fixed tendon offset provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    def fixed_tendon_pos_limits(self) -> ProxyArray:
        """Fixed tendon position limits provided to the simulation.

        Shape is (num_instances, num_fixed_tendons, 2), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_fixed_tendons, 2).
        """
        raise NotImplementedError

    """
    Spatial tendon properties.
    """

    @property
    def spatial_tendon_stiffness(self) -> ProxyArray:
        """Spatial tendon stiffness provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    @property
    def spatial_tendon_damping(self) -> ProxyArray:
        """Spatial tendon damping provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    @property
    def spatial_tendon_limit_stiffness(self) -> ProxyArray:
        """Spatial tendon limit stiffness provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    @property
    def spatial_tendon_offset(self) -> ProxyArray:
        """Spatial tendon offset provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    """
    Root state properties.
    """

    @property
    def root_link_pose_w(self) -> ProxyArray:
        """Root link pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).

        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return self._root_link_pose_w_ta

    @property
    @capture_unsafe(_LAZY_CAPTURE_REASON)
    def root_link_vel_w(self) -> ProxyArray:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_root_link_vel_from_root_com_vel,
                dim=self._num_instances,
                inputs=[
                    self.root_com_vel_w.warp,
                    self.root_link_pose_w.warp,
                    self.body_com_pos_b.warp,
                ],
                outputs=[
                    self._root_link_vel_w.data,
                ],
                device=self.device,
            )
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w_ta

    @property
    @capture_unsafe(_LAZY_CAPTURE_REASON)
    def root_com_pose_w(self) -> ProxyArray:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).

        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            wp.launch(
                shared_kernels.get_root_com_pose_from_root_link_pose,
                dim=self._num_instances,
                inputs=[
                    self.root_link_pose_w.warp,
                    self.body_com_pos_b.warp,
                ],
                outputs=[
                    self._root_com_pose_w.data,
                ],
                device=self.device,
            )
            self._root_com_pose_w.timestamp = self._sim_timestamp

        return self._root_com_pose_w_ta

    @property
    def root_com_vel_w(self) -> ProxyArray:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        return self._root_com_vel_w_ta

    """
    Body state properties.
    """

    @property
    def body_mass(self) -> ProxyArray:
        """Body mass ``wp.float32`` in the world frame.

        Shape is (num_instances, num_bodies), dtype = wp.float32. In torch this resolves to (num_instances, num_bodies).
        """
        return self._body_mass_ta

    @property
    def body_inertia(self) -> ProxyArray:
        """Flattened body inertia in the world frame.

        Shape is (num_instances, num_bodies, 9), dtype = wp.float32. In torch this resolves to
        (num_instances, num_bodies, 9).
        """
        return self._body_inertia_ta

    @property
    def body_link_pose_w(self) -> ProxyArray:
        """Body link pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._fk_timestamp < self._sim_timestamp:
            SimulationManager.forward()
            self._fk_timestamp = self._sim_timestamp
        return self._body_link_pose_w_ta

    @property
    @capture_unsafe(_LAZY_CAPTURE_REASON)
    def body_link_vel_w(self) -> ProxyArray:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        if self._body_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_body_link_vel_from_body_com_vel,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_com_vel_w.warp,
                    self.body_link_pose_w.warp,
                    self.body_com_pos_b.warp,
                ],
                outputs=[
                    self._body_link_vel_w.data,
                ],
                device=self.device,
            )
            self._body_link_vel_w.timestamp = self._sim_timestamp

        return self._body_link_vel_w_ta

    @property
    @capture_unsafe(_LAZY_CAPTURE_REASON)
    def body_com_pose_w(self) -> ProxyArray:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_body_com_pose_from_body_link_pose,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_link_pose_w.warp,
                    self.body_com_pos_b.warp,
                ],
                outputs=[
                    self._body_com_pose_w.data,
                ],
                device=self.device,
            )
            self._body_com_pose_w.timestamp = self._sim_timestamp

        return self._body_com_pose_w_ta

    @property
    def body_com_vel_w(self) -> ProxyArray:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        return self._body_com_vel_w_ta

    @property
    def body_com_acc_w(self) -> ProxyArray:
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]``.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.derive_body_acceleration_from_body_com_velocities,
                dim=(self._num_instances, self._num_bodies),
                device=self.device,
                inputs=[
                    self._sim_bind_body_com_vel_w,
                    SimulationManager.get_dt(),
                    self._previous_body_com_vel,
                ],
                outputs=[
                    self._body_com_acc_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._body_com_acc_w.timestamp = self._sim_timestamp
            # update the previous velocity
        return self._body_com_acc_w_ta

    @property
    def body_com_pos_b(self) -> ProxyArray:
        """Center of mass position of all of the bodies in their respective link frames.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body's link frame.
        """
        return self._body_com_pos_b_ta

    @property
    def body_com_pose_b(self) -> ProxyArray:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        warnings.warn(
            "In Newton, body com pose always has unit quaternion. Consider using body_com_pos_b instead."
            "Querying this property requires appending a unit quaternion to the position which is expensive.",
            category=UserWarning,
            stacklevel=2,
        )
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            wp.launch(
                shared_kernels.make_dummy_body_com_pose_b,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_com_pos_b.warp,
                ],
                outputs=[
                    self._body_com_pose_b.data,
                ],
                device=self.device,
            )
            self._body_com_pose_b.timestamp = self._sim_timestamp
        return self._body_com_pose_b_ta

    """
    Joint state properties.
    """

    @property
    def joint_pos(self) -> ProxyArray:
        """Joint positions of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        return self._joint_pos_ta

    @property
    def joint_vel(self) -> ProxyArray:
        """Joint velocities of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        return self._joint_vel_ta

    @property
    def joint_acc(self) -> ProxyArray:
        """Joint acceleration of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        if self._joint_acc.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            time_elapsed = self._sim_timestamp - self._joint_acc.timestamp
            wp.launch(
                articulation_kernels.get_joint_acc_from_joint_vel,
                dim=(self._num_instances, self._num_joints),
                inputs=[
                    self.joint_vel.warp,
                    self._previous_joint_vel,
                    time_elapsed,
                ],
                outputs=[
                    self._joint_acc.data,
                ],
                device=self.device,
            )
            self._joint_acc.timestamp = self._sim_timestamp
        return self._joint_acc_ta

    """
    Derived Properties.
    """

    @property
    @capture_unsafe(_LAZY_CAPTURE_REASON)
    def projected_gravity_b(self) -> ProxyArray:
        """Projection of the gravity direction on base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.GRAVITY_VEC_W.warp, self.root_link_quat_w.warp],
                outputs=[self._projected_gravity_b.data],
                device=self.device,
            )
            self._projected_gravity_b.timestamp = self._sim_timestamp
        return self._projected_gravity_b_ta

    @property
    @capture_unsafe(_LAZY_CAPTURE_REASON)
    def heading_w(self) -> ProxyArray:
        """Yaw heading of the base frame (in radians).

        Shape is (num_instances), dtype = wp.float32. In torch this resolves to (num_instances,).

        .. note::
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        if self._heading_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.root_heading_w,
                dim=self._num_instances,
                inputs=[self.FORWARD_VEC_B.warp, self.root_link_quat_w.warp],
                outputs=[self._heading_w.data],
                device=self.device,
            )
            self._heading_w.timestamp = self._sim_timestamp
        return self._heading_w_ta

    @property
    @capture_unsafe(_LAZY_CAPTURE_REASON)
    def root_link_lin_vel_b(self) -> ProxyArray:
        """Root link linear velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to
        its actor frame.
        """
        if self._root_link_lin_vel_b is None:
            self._root_link_lin_vel_b = TimestampedBuffer(
                shape=(self._num_instances,), dtype=wp.vec3f, device=self.device
            )
            self._root_link_lin_vel_b_ta = ProxyArray(self._root_link_lin_vel_b.data)
        if self._root_link_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_link_lin_vel_w.warp, self.root_link_quat_w.warp],
                outputs=[self._root_link_lin_vel_b.data],
                device=self.device,
            )
            self._root_link_lin_vel_b.timestamp = self._sim_timestamp
        return self._root_link_lin_vel_b_ta

    @property
    def root_link_ang_vel_b(self) -> ProxyArray:
        """Root link angular velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to
        its actor frame.
        """
        if self._root_link_ang_vel_b is None:
            self._root_link_ang_vel_b = TimestampedBuffer(
                shape=(self._num_instances,), dtype=wp.vec3f, device=self.device
            )
            self._root_link_ang_vel_b_ta = ProxyArray(self._root_link_ang_vel_b.data)
        if self._root_link_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_link_ang_vel_w.warp, self.root_link_quat_w.warp],
                outputs=[self._root_link_ang_vel_b.data],
                device=self.device,
            )
            self._root_link_ang_vel_b.timestamp = self._sim_timestamp
        return self._root_link_ang_vel_b_ta

    @property
    def root_com_lin_vel_b(self) -> ProxyArray:
        """Root center of mass linear velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to
        its actor frame.
        """
        if self._root_com_lin_vel_b is None:
            self._root_com_lin_vel_b = TimestampedBuffer(
                shape=(self._num_instances,), dtype=wp.vec3f, device=self.device
            )
            self._root_com_lin_vel_b_ta = ProxyArray(self._root_com_lin_vel_b.data)
        if self._root_com_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_com_lin_vel_w.warp, self.root_link_quat_w.warp],
                outputs=[self._root_com_lin_vel_b.data],
                device=self.device,
            )
            self._root_com_lin_vel_b.timestamp = self._sim_timestamp
        return self._root_com_lin_vel_b_ta

    @property
    def root_com_ang_vel_b(self) -> ProxyArray:
        """Root center of mass angular velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to
        its actor frame.
        """
        if self._root_com_ang_vel_b is None:
            self._root_com_ang_vel_b = TimestampedBuffer(
                shape=(self._num_instances,), dtype=wp.vec3f, device=self.device
            )
            self._root_com_ang_vel_b_ta = ProxyArray(self._root_com_ang_vel_b.data)
        if self._root_com_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_com_ang_vel_w.warp, self.root_link_quat_w.warp],
                outputs=[self._root_com_ang_vel_b.data],
                device=self.device,
            )
            self._root_com_ang_vel_b.timestamp = self._sim_timestamp
        return self._root_com_ang_vel_b_ta

    """
    Sliced properties.
    """

    @property
    def root_link_pos_w(self) -> ProxyArray:
        """Root link position in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        self._root_link_pos_w = self._get_pos_from_transform(self._root_link_pos_w, self.root_link_pose_w.warp)
        if self._root_link_pos_w_ta is None:
            self._root_link_pos_w_ta = ProxyArray(self._root_link_pos_w)
        return self._root_link_pos_w_ta

    @property
    def root_link_quat_w(self) -> ProxyArray:
        """Root link orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances), dtype = wp.quatf. In torch this resolves to (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        self._root_link_quat_w = self._get_quat_from_transform(self._root_link_quat_w, self.root_link_pose_w.warp)
        if self._root_link_quat_w_ta is None:
            self._root_link_quat_w_ta = ProxyArray(self._root_link_quat_w)
        return self._root_link_quat_w_ta

    @property
    def root_link_lin_vel_w(self) -> ProxyArray:
        """Root linear velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        self._root_link_lin_vel_w = self._get_top_from_spatial_vector(
            self._root_link_lin_vel_w, self.root_link_vel_w.warp
        )
        if self._root_link_lin_vel_w_ta is None:
            self._root_link_lin_vel_w_ta = ProxyArray(self._root_link_lin_vel_w)
        return self._root_link_lin_vel_w_ta

    @property
    def root_link_ang_vel_w(self) -> ProxyArray:
        """Root link angular velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        self._root_link_ang_vel_w = self._get_bottom_from_spatial_vector(
            self._root_link_ang_vel_w, self.root_link_vel_w.warp
        )
        if self._root_link_ang_vel_w_ta is None:
            self._root_link_ang_vel_w_ta = ProxyArray(self._root_link_ang_vel_w)
        return self._root_link_ang_vel_w_ta

    @property
    def root_com_pos_w(self) -> ProxyArray:
        """Root center of mass position in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the position of the center of mass frame of the root rigid body relative to the world.
        """
        self._root_com_pos_w = self._get_pos_from_transform(self._root_com_pos_w, self.root_com_pose_w.warp)
        if self._root_com_pos_w_ta is None:
            self._root_com_pos_w_ta = ProxyArray(self._root_com_pos_w)
        return self._root_com_pos_w_ta

    @property
    def root_com_quat_w(self) -> ProxyArray:
        """Root center of mass orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances), dtype = wp.quatf. In torch this resolves to (num_instances, 4).

        This quantity is the orientation of the principal axes of inertia of the root rigid body relative to the world.
        """
        self._root_com_quat_w = self._get_quat_from_transform(self._root_com_quat_w, self.root_com_pose_w.warp)
        if self._root_com_quat_w_ta is None:
            self._root_com_quat_w_ta = ProxyArray(self._root_com_quat_w)
        return self._root_com_quat_w_ta

    @property
    def root_com_lin_vel_w(self) -> ProxyArray:
        """Root center of mass linear velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        self._root_com_lin_vel_w = self._get_top_from_spatial_vector(self._root_com_lin_vel_w, self.root_com_vel_w.warp)
        if self._root_com_lin_vel_w_ta is None:
            self._root_com_lin_vel_w_ta = ProxyArray(self._root_com_lin_vel_w)
        return self._root_com_lin_vel_w_ta

    @property
    def root_com_ang_vel_w(self) -> ProxyArray:
        """Root center of mass angular velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        self._root_com_ang_vel_w = self._get_bottom_from_spatial_vector(
            self._root_com_ang_vel_w, self.root_com_vel_w.warp
        )
        if self._root_com_ang_vel_w_ta is None:
            self._root_com_ang_vel_w_ta = ProxyArray(self._root_com_ang_vel_w)
        return self._root_com_ang_vel_w_ta

    @property
    def body_link_pos_w(self) -> ProxyArray:
        """Positions of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        self._body_link_pos_w = self._get_pos_from_transform(self._body_link_pos_w, self.body_link_pose_w.warp)
        if self._body_link_pos_w_ta is None:
            self._body_link_pos_w_ta = ProxyArray(self._body_link_pos_w)
        return self._body_link_pos_w_ta

    @property
    def body_link_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        self._body_link_quat_w = self._get_quat_from_transform(self._body_link_quat_w, self.body_link_pose_w.warp)
        if self._body_link_quat_w_ta is None:
            self._body_link_quat_w_ta = ProxyArray(self._body_link_quat_w)
        return self._body_link_quat_w_ta

    @property
    def body_link_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' actor frame relative to the world.
        """
        self._body_link_lin_vel_w = self._get_top_from_spatial_vector(
            self._body_link_lin_vel_w, self.body_link_vel_w.warp
        )
        if self._body_link_lin_vel_w_ta is None:
            self._body_link_lin_vel_w_ta = ProxyArray(self._body_link_lin_vel_w)
        return self._body_link_lin_vel_w_ta

    @property
    def body_link_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' actor frame relative to the world.
        """
        self._body_link_ang_vel_w = self._get_bottom_from_spatial_vector(
            self._body_link_ang_vel_w, self.body_link_vel_w.warp
        )
        if self._body_link_ang_vel_w_ta is None:
            self._body_link_ang_vel_w_ta = ProxyArray(self._body_link_ang_vel_w)
        return self._body_link_ang_vel_w_ta

    @property
    def body_com_pos_w(self) -> ProxyArray:
        """Positions of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' center of mass frame.
        """
        self._body_com_pos_w = self._get_pos_from_transform(self._body_com_pos_w, self.body_com_pose_w.warp)
        if self._body_com_pos_w_ta is None:
            self._body_com_pos_w_ta = ProxyArray(self._body_com_pos_w)
        return self._body_com_pos_w_ta

    @property
    def body_com_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the principal axes of inertia of the articulation bodies.
        """
        self._body_com_quat_w = self._get_quat_from_transform(self._body_com_quat_w, self.body_com_pose_w.warp)
        if self._body_com_quat_w_ta is None:
            self._body_com_quat_w_ta = ProxyArray(self._body_com_quat_w)
        return self._body_com_quat_w_ta

    @property
    def body_com_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        self._body_com_lin_vel_w = self._get_top_from_spatial_vector(self._body_com_lin_vel_w, self.body_com_vel_w.warp)
        if self._body_com_lin_vel_w_ta is None:
            self._body_com_lin_vel_w_ta = ProxyArray(self._body_com_lin_vel_w)
        return self._body_com_lin_vel_w_ta

    @property
    def body_com_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        self._body_com_ang_vel_w = self._get_bottom_from_spatial_vector(
            self._body_com_ang_vel_w, self.body_com_vel_w.warp
        )
        if self._body_com_ang_vel_w_ta is None:
            self._body_com_ang_vel_w_ta = ProxyArray(self._body_com_ang_vel_w)
        return self._body_com_ang_vel_w_ta

    @property
    def body_com_lin_acc_w(self) -> ProxyArray:
        """Linear acceleration of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        self._body_com_lin_acc_w = self._get_top_from_spatial_vector(self._body_com_lin_acc_w, self.body_com_acc_w.warp)
        if self._body_com_lin_acc_w_ta is None:
            self._body_com_lin_acc_w_ta = ProxyArray(self._body_com_lin_acc_w)
        return self._body_com_lin_acc_w_ta

    @property
    def body_com_ang_acc_w(self) -> ProxyArray:
        """Angular acceleration of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        self._body_com_ang_acc_w = self._get_bottom_from_spatial_vector(
            self._body_com_ang_acc_w, self.body_com_acc_w.warp
        )
        if self._body_com_ang_acc_w_ta is None:
            self._body_com_ang_acc_w_ta = ProxyArray(self._body_com_ang_acc_w)
        return self._body_com_ang_acc_w_ta

    @property
    def body_com_quat_b(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all of the bodies in their respective link
        frames.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the principal axes of inertia relative to its body's link frame.
        """
        self._body_com_quat_b = self._get_quat_from_transform(self._body_com_quat_b, self.body_com_pose_b.warp)
        if self._body_com_quat_b_ta is None:
            self._body_com_quat_b_ta = ProxyArray(self._body_com_quat_b)
        return self._body_com_quat_b_ta

    def _create_simulation_bindings(self) -> None:
        """Create simulation bindings for the root data.

        Direct simulation bindings are pointers to the simulation data, their data is not copied, and should
        only be updated using warp kernels. Any modifications made to the bindings will be reflected in the simulation.
        Hence we encourage users to carefully think about the data they modify and in which order it should be updated.

        .. caution:: This is possible if and only if the properties that we access are strided from newton and not
        indexed. Newton willing this is the case all the time, but we should pay attention to this if things look off.
        """
        # Short-hand for the number of instances, number of links, and number of joints.
        self._num_instances = self._root_view.count
        self._num_joints = self._root_view.joint_dof_count
        self._num_bodies = self._root_view.link_count
        self._num_fixed_tendons = 0  # self._root_view.max_fixed_tendons
        self._num_spatial_tendons = 0  # self._root_view.max_spatial_tendons

        # -- root properties
        self._sim_bind_root_link_pose_w = self._root_view.get_root_transforms(SimulationManager.get_state_0())[:, 0]
        self._sim_bind_root_com_vel_w = self._root_view.get_root_velocities(SimulationManager.get_state_0())
        if self._sim_bind_root_com_vel_w is not None:
            if self._root_view.is_fixed_base:
                self._sim_bind_root_com_vel_w = self._sim_bind_root_com_vel_w[:, 0, 0]
            else:
                self._sim_bind_root_com_vel_w = self._sim_bind_root_com_vel_w[:, 0]
        # -- body properties
        self._sim_bind_body_com_pos_b = self._root_view.get_attribute("body_com", SimulationManager.get_model())[:, 0]
        self._sim_bind_body_link_pose_w = self._root_view.get_link_transforms(SimulationManager.get_state_0())[:, 0]
        self._sim_bind_body_com_vel_w = self._root_view.get_link_velocities(SimulationManager.get_state_0())
        if self._sim_bind_body_com_vel_w is not None:
            self._sim_bind_body_com_vel_w = self._sim_bind_body_com_vel_w[:, 0]
        self._sim_bind_body_mass = self._root_view.get_attribute("body_mass", SimulationManager.get_model())[:, 0]
        # Newton stores body_inertia as (N, 1, B) mat33f — the [:, 0] removes the padding dim
        # giving (N, B) mat33f. Reinterpret as (N, B, 9) float32 via pointer aliasing.
        # Each mat33f element is 9 contiguous float32 values (36 bytes), so the inner stride is 4.
        # The slice may be non-contiguous in the outer dims, so we preserve those strides.
        _body_inertia_raw = self._root_view.get_attribute("body_inertia", SimulationManager.get_model())[:, 0]
        self._sim_bind_body_inertia = wp.array(
            ptr=_body_inertia_raw.ptr,
            dtype=wp.float32,
            shape=(self._num_instances, self._num_bodies, 9),
            strides=(_body_inertia_raw.strides[0], _body_inertia_raw.strides[1], 4),
            device=_body_inertia_raw.device,
            copy=False,
        )
        self._sim_bind_body_external_wrench = self._root_view.get_attribute("body_f", SimulationManager.get_state_0())[
            :, 0
        ]
        try:
            self._sim_bind_body_parent_f = self._root_view.get_attribute(
                "body_parent_f", SimulationManager.get_state_0()
            )[:, 0]
        except Exception:
            self._sim_bind_body_parent_f = None
        # -- joint properties
        if self._num_joints > 0:
            self._sim_bind_joint_pos_limits_lower = self._root_view.get_attribute(
                "joint_limit_lower", SimulationManager.get_model()
            )[:, 0]
            self._sim_bind_joint_pos_limits_upper = self._root_view.get_attribute(
                "joint_limit_upper", SimulationManager.get_model()
            )[:, 0]
            self._sim_bind_joint_stiffness_sim = self._root_view.get_attribute(
                "joint_target_ke", SimulationManager.get_model()
            )[:, 0]
            self._sim_bind_joint_damping_sim = self._root_view.get_attribute(
                "joint_target_kd", SimulationManager.get_model()
            )[:, 0]
            self._sim_bind_joint_armature = self._root_view.get_attribute(
                "joint_armature", SimulationManager.get_model()
            )[:, 0]
            self._sim_bind_joint_friction_coeff = self._root_view.get_attribute(
                "joint_friction", SimulationManager.get_model()
            )[:, 0]
            self._sim_bind_joint_vel_limits_sim = self._root_view.get_attribute(
                "joint_velocity_limit", SimulationManager.get_model()
            )[:, 0]
            self._sim_bind_joint_effort_limits_sim = self._root_view.get_attribute(
                "joint_effort_limit", SimulationManager.get_model()
            )[:, 0]
            # -- joint states
            self._sim_bind_joint_pos = self._root_view.get_dof_positions(SimulationManager.get_state_0())[:, 0]
            self._sim_bind_joint_vel = self._root_view.get_dof_velocities(SimulationManager.get_state_0())[:, 0]
            # -- joint commands (sent to the simulation)
            self._sim_bind_joint_effort = self._root_view.get_attribute("joint_f", SimulationManager.get_control())[
                :, 0
            ]
            self._sim_bind_joint_position_target = self._root_view.get_attribute(
                "joint_target_pos", SimulationManager.get_control()
            )[:, 0]
            self._sim_bind_joint_velocity_target = self._root_view.get_attribute(
                "joint_target_vel", SimulationManager.get_control()
            )[:, 0]
        else:
            # No joints (e.g., free-floating rigid body) - set bindings to empty arrays
            self._sim_bind_joint_pos_limits_lower = wp.zeros(
                (self._num_instances, 0), dtype=wp.float32, device=self.device
            )
            self._sim_bind_joint_pos_limits_upper = wp.zeros(
                (self._num_instances, 0), dtype=wp.float32, device=self.device
            )
            self._sim_bind_joint_stiffness_sim = wp.zeros(
                (self._num_instances, 0), dtype=wp.float32, device=self.device
            )
            self._sim_bind_joint_damping_sim = wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_armature = wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_friction_coeff = wp.zeros(
                (self._num_instances, 0), dtype=wp.float32, device=self.device
            )
            self._sim_bind_joint_vel_limits_sim = wp.zeros(
                (self._num_instances, 0), dtype=wp.float32, device=self.device
            )
            self._sim_bind_joint_effort_limits_sim = wp.zeros(
                (self._num_instances, 0), dtype=wp.float32, device=self.device
            )
            self._sim_bind_joint_pos = wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_vel = wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_effort = wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_position_target = wp.zeros(
                (self._num_instances, 0), dtype=wp.float32, device=self.device
            )
            self._sim_bind_joint_velocity_target = wp.zeros(
                (self._num_instances, 0), dtype=wp.float32, device=self.device
            )

        # Re-pin ProxyArray wrappers to the newly created sim bindings.
        # On first init, _create_buffers() handles this after all buffers exist.
        if hasattr(self, "_root_link_pose_w_ta"):
            self._pin_proxy_arrays()

    def _create_buffers(self) -> None:
        """Create buffers for the root data."""
        super()._create_buffers()

        # Initialize history for finite differencing. If the articulation is fixed, the root com velocity is not
        # available, so we use zeros.
        if self._root_view.get_root_velocities(SimulationManager.get_state_0()) is None:
            logger.warning(
                "Failed to get root com velocity. If the articulation is fixed, this is expected. "
                "Setting root com velocity to zeros."
            )
            self._sim_bind_root_com_vel_w = wp.zeros(
                (self._num_instances), dtype=wp.spatial_vectorf, device=self.device
            )
            self._sim_bind_body_com_vel_w = wp.zeros(
                (self._num_instances, self._num_bodies), dtype=wp.spatial_vectorf, device=self.device
            )
        # -- default root pose and velocity
        self._default_root_pose = wp.zeros((self._num_instances,), dtype=wp.transformf, device=self.device)
        self._default_root_vel = wp.zeros((self._num_instances,), dtype=wp.spatial_vectorf, device=self.device)
        # -- default joint positions and velocities
        self._default_joint_pos = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        self._default_joint_vel = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        # -- joint commands (sent to the actuator from the user)
        self._joint_pos_target = wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        self._joint_vel_target = wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        self._joint_effort_target = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        # -- computed joint efforts from the actuator models
        self._computed_torque = wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        self._applied_torque = wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        # -- joint properties for the actuator models
        if self._num_joints > 0:
            self._actuator_stiffness = wp.clone(self._sim_bind_joint_stiffness_sim)
            self._actuator_damping = wp.clone(self._sim_bind_joint_damping_sim)
        else:
            self._actuator_stiffness = wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
            self._actuator_damping = wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
        # -- other data that are filled based on explicit actuator models
        self._joint_dynamic_friction = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        self._joint_viscous_friction = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        self._soft_joint_vel_limits = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        self._gear_ratio = wp.ones((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        # -- update the soft joint position limits
        self._soft_joint_pos_limits = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.vec2f, device=self.device
        )

        # Initialize history for finite differencing
        if self._num_joints > 0:
            self._previous_joint_vel = wp.clone(
                self._root_view.get_dof_velocities(SimulationManager.get_state_0())[:, 0]
            )
        else:
            self._previous_joint_vel = wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
        self._previous_body_com_vel = wp.clone(self._sim_bind_body_com_vel_w)

        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._root_link_vel_w = TimestampedBuffer(
            shape=(self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
        )
        self._root_link_vel_b = TimestampedBuffer(
            shape=(self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
        )
        self._body_link_vel_w = TimestampedBuffer(
            shape=(self._num_instances, self._num_bodies), dtype=wp.spatial_vectorf, device=self.device
        )
        # -- com frame w.r.t. link frame
        self._body_com_pose_b = TimestampedBuffer(
            shape=(self._num_instances, self._num_bodies), dtype=wp.transformf, device=self.device
        )
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedBuffer(shape=(self._num_instances,), dtype=wp.transformf, device=self.device)
        self._root_com_vel_b = TimestampedBuffer(
            shape=(self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
        )
        self._root_com_acc_w = TimestampedBuffer(
            shape=(self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
        )
        self._body_com_pose_w = TimestampedBuffer(
            shape=(self._num_instances, self._num_bodies), dtype=wp.transformf, device=self.device
        )
        self._body_com_acc_w = TimestampedBuffer(
            shape=(self._num_instances, self._num_bodies), dtype=wp.spatial_vectorf, device=self.device
        )
        # -- derived properties (these are cached to avoid repeated memory allocations)
        self._projected_gravity_b = TimestampedBuffer(shape=(self._num_instances,), dtype=wp.vec3f, device=self.device)
        self._heading_w = TimestampedBuffer(shape=(self._num_instances,), dtype=wp.float32, device=self.device)
        # -- joint state
        self._joint_acc = TimestampedBuffer(
            shape=(self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        # Empty memory pre-allocations
        self._root_link_lin_vel_b = None
        self._root_link_ang_vel_b = None
        self._root_com_lin_vel_b = None
        self._root_com_ang_vel_b = None
        self._joint_pos_limits = None
        self._root_state_w = None
        self._root_link_state_w = None
        self._root_com_state_w = None
        self._body_com_quat_b = None
        self._root_link_pos_w = None
        self._root_link_quat_w = None
        self._root_link_lin_vel_w = None
        self._root_link_ang_vel_w = None
        self._root_com_pos_w = None
        self._root_com_quat_w = None
        self._root_com_lin_vel_w = None
        self._root_com_ang_vel_w = None
        self._body_state_w = None
        self._body_link_state_w = None
        self._body_com_state_w = None
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
        self._default_root_state = None

        # Pin all ProxyArray wrappers to current buffers.
        self._pin_proxy_arrays()

    def _pin_proxy_arrays(self) -> None:
        """Create or rebind all pinned ProxyArray wrappers.

        Called from :meth:`_create_buffers` on first initialization and from
        :meth:`_create_simulation_bindings` after a full simulation reset when
        the solver recreates its internal arrays.
        """
        is_rebind = hasattr(self, "_root_link_pose_w_ta")

        if is_rebind:
            # Rebind sim-bound ProxyArrays to new solver arrays
            self._root_link_pose_w_ta = ProxyArray(self._sim_bind_root_link_pose_w)
            self._root_com_vel_w_ta = ProxyArray(self._sim_bind_root_com_vel_w)
            self._body_link_pose_w_ta = ProxyArray(self._sim_bind_body_link_pose_w)
            self._body_com_vel_w_ta = ProxyArray(self._sim_bind_body_com_vel_w)
            self._joint_pos_ta = ProxyArray(self._sim_bind_joint_pos)
            self._joint_vel_ta = ProxyArray(self._sim_bind_joint_vel)
            self._joint_stiffness_ta = ProxyArray(self._sim_bind_joint_stiffness_sim)
            self._joint_damping_ta = ProxyArray(self._sim_bind_joint_damping_sim)
            self._joint_armature_ta = ProxyArray(self._sim_bind_joint_armature)
            self._joint_friction_coeff_ta = ProxyArray(self._sim_bind_joint_friction_coeff)
            self._joint_pos_limits_lower_ta = ProxyArray(self._sim_bind_joint_pos_limits_lower)
            self._joint_pos_limits_upper_ta = ProxyArray(self._sim_bind_joint_pos_limits_upper)
            self._joint_vel_limits_ta = ProxyArray(self._sim_bind_joint_vel_limits_sim)
            self._joint_effort_limits_ta = ProxyArray(self._sim_bind_joint_effort_limits_sim)
            self._body_mass_ta = ProxyArray(self._sim_bind_body_mass)
            self._body_inertia_ta = ProxyArray(self._sim_bind_body_inertia)
            self._body_com_pos_b_ta = ProxyArray(self._sim_bind_body_com_pos_b)
        else:
            # First-time creation: pin ProxyArrays to current buffers
            # Category 1: sim-bound and pre-allocated buffers
            # Sim-bound pointers are re-created on full reset; _create_simulation_bindings()
            # calls rebind() on each ProxyArray to keep them in sync.
            self._root_link_pose_w_ta = ProxyArray(self._sim_bind_root_link_pose_w)
            self._root_com_vel_w_ta = ProxyArray(self._sim_bind_root_com_vel_w)
            self._body_link_pose_w_ta = ProxyArray(self._sim_bind_body_link_pose_w)
            self._body_com_vel_w_ta = ProxyArray(self._sim_bind_body_com_vel_w)
            self._joint_pos_ta = ProxyArray(self._sim_bind_joint_pos)
            self._joint_vel_ta = ProxyArray(self._sim_bind_joint_vel)
            self._default_root_pose_ta = ProxyArray(self._default_root_pose)
            self._default_root_vel_ta = ProxyArray(self._default_root_vel)
            self._default_joint_pos_ta = ProxyArray(self._default_joint_pos)
            self._default_joint_vel_ta = ProxyArray(self._default_joint_vel)
            self._joint_pos_target_ta = ProxyArray(self._joint_pos_target)
            self._joint_vel_target_ta = ProxyArray(self._joint_vel_target)
            self._joint_effort_target_ta = ProxyArray(self._joint_effort_target)
            self._computed_torque_ta = ProxyArray(self._computed_torque)
            self._applied_torque_ta = ProxyArray(self._applied_torque)
            self._joint_stiffness_ta = ProxyArray(self._sim_bind_joint_stiffness_sim)
            self._joint_damping_ta = ProxyArray(self._sim_bind_joint_damping_sim)
            self._joint_armature_ta = ProxyArray(self._sim_bind_joint_armature)
            self._joint_friction_coeff_ta = ProxyArray(self._sim_bind_joint_friction_coeff)
            self._joint_pos_limits_lower_ta = ProxyArray(self._sim_bind_joint_pos_limits_lower)
            self._joint_pos_limits_upper_ta = ProxyArray(self._sim_bind_joint_pos_limits_upper)
            self._joint_vel_limits_ta = ProxyArray(self._sim_bind_joint_vel_limits_sim)
            self._joint_effort_limits_ta = ProxyArray(self._sim_bind_joint_effort_limits_sim)
            self._soft_joint_pos_limits_ta = ProxyArray(self._soft_joint_pos_limits)
            self._soft_joint_vel_limits_ta = ProxyArray(self._soft_joint_vel_limits)
            self._gear_ratio_ta = ProxyArray(self._gear_ratio)
            self._body_mass_ta = ProxyArray(self._sim_bind_body_mass)
            self._body_inertia_ta = ProxyArray(self._sim_bind_body_inertia)
            self._body_com_pos_b_ta = ProxyArray(self._sim_bind_body_com_pos_b)

            # Category 2: TimestampedBuffer properties
            self._root_link_vel_w_ta = ProxyArray(self._root_link_vel_w.data)
            self._body_link_vel_w_ta = ProxyArray(self._body_link_vel_w.data)
            self._root_com_pose_w_ta = ProxyArray(self._root_com_pose_w.data)
            self._body_com_pose_w_ta = ProxyArray(self._body_com_pose_w.data)
            self._body_com_acc_w_ta = ProxyArray(self._body_com_acc_w.data)
            self._body_com_pose_b_ta = ProxyArray(self._body_com_pose_b.data)
            self._projected_gravity_b_ta = ProxyArray(self._projected_gravity_b.data)
            self._heading_w_ta = ProxyArray(self._heading_w.data)
            self._joint_acc_ta = ProxyArray(self._joint_acc.data)

            # -- deprecated state properties (lazy); type annotations declared once here
            self._root_state_w_ta: ProxyArray | None = None
            self._root_link_state_w_ta: ProxyArray | None = None
            self._root_com_state_w_ta: ProxyArray | None = None
            self._default_root_state_ta: ProxyArray | None = None
            self._body_state_w_ta: ProxyArray | None = None
            self._body_link_state_w_ta: ProxyArray | None = None
            self._body_com_state_w_ta: ProxyArray | None = None

        # Invalidate lazy sliced ProxyArrays AND their backing wp.arrays so they are
        # re-created from fresh data on next access.  On first init the backing fields
        # are already None (set by _create_buffers), so the assignments below are
        # harmless no-ops.  On rebind they reset stale pointers into freed transform
        # memory after a sim reset.
        self._root_link_pos_w_ta: ProxyArray | None = None
        self._root_link_pos_w = None
        self._root_link_quat_w_ta: ProxyArray | None = None
        self._root_link_quat_w = None
        self._root_link_lin_vel_w_ta: ProxyArray | None = None
        self._root_link_lin_vel_w = None
        self._root_link_ang_vel_w_ta: ProxyArray | None = None
        self._root_link_ang_vel_w = None
        self._root_com_pos_w_ta: ProxyArray | None = None
        self._root_com_pos_w = None
        self._root_com_quat_w_ta: ProxyArray | None = None
        self._root_com_quat_w = None
        self._root_com_lin_vel_w_ta: ProxyArray | None = None
        self._root_com_lin_vel_w = None
        self._root_com_ang_vel_w_ta: ProxyArray | None = None
        self._root_com_ang_vel_w = None
        self._body_link_pos_w_ta: ProxyArray | None = None
        self._body_link_pos_w = None
        self._body_link_quat_w_ta: ProxyArray | None = None
        self._body_link_quat_w = None
        self._body_link_lin_vel_w_ta: ProxyArray | None = None
        self._body_link_lin_vel_w = None
        self._body_link_ang_vel_w_ta: ProxyArray | None = None
        self._body_link_ang_vel_w = None
        self._body_com_pos_w_ta: ProxyArray | None = None
        self._body_com_pos_w = None
        self._body_com_quat_w_ta: ProxyArray | None = None
        self._body_com_quat_w = None
        self._body_com_lin_vel_w_ta: ProxyArray | None = None
        self._body_com_lin_vel_w = None
        self._body_com_ang_vel_w_ta: ProxyArray | None = None
        self._body_com_ang_vel_w = None
        self._body_com_lin_acc_w_ta: ProxyArray | None = None
        self._body_com_lin_acc_w = None
        self._body_com_ang_acc_w_ta: ProxyArray | None = None
        self._body_com_ang_acc_w = None
        self._body_com_quat_b_ta: ProxyArray | None = None
        self._body_com_quat_b = None
        self._joint_pos_limits_ta: ProxyArray | None = None
        self._joint_pos_limits = None
        self._root_link_lin_vel_b_ta: ProxyArray | None = None
        self._root_link_lin_vel_b = None
        self._root_link_ang_vel_b_ta: ProxyArray | None = None
        self._root_link_ang_vel_b = None
        self._root_com_lin_vel_b_ta: ProxyArray | None = None
        self._root_com_lin_vel_b = None
        self._root_com_ang_vel_b_ta: ProxyArray | None = None
        self._root_com_ang_vel_b = None

    """
    Internal helpers.
    """

    def _get_pos_from_transform(self, source: wp.array | None, transform: wp.array) -> wp.array:
        """Generates a position array from a transform array.

        Args:
            transform: The transform array. Shape is (N) dtype=wp.transformf.

        Returns:
            The position array. Shape is (N) dtype=wp.vec3f.
        """
        # Check if we already created the lazy buffer.
        if source is None:
            if transform.is_contiguous:
                # Check if the array is contiguous. If so, we can just return a strided array.
                # Then this update becomes a no-op.
                return wp.array(
                    ptr=transform.ptr,
                    shape=transform.shape,
                    dtype=wp.vec3f,
                    strides=transform.strides,
                    device=self.device,
                )
            else:
                # If the array is not contiguous, we need to create a new array to write to.
                # Shape matches transform.shape since each element is vec3f (already contains 3 floats)
                source = wp.zeros(transform.shape, dtype=wp.vec3f, device=self.device)

        # If the array is not contiguous, we need to launch the kernel to get the position part of the transform.
        if not transform.is_contiguous:
            # Launch the right kernel based on the shape of the transform array.
            if len(transform.shape) > 1:
                wp.launch(
                    shared_kernels.split_transform_to_pos_2d,
                    dim=transform.shape,
                    inputs=[transform],
                    outputs=[source],
                    device=self.device,
                )
            else:
                wp.launch(
                    shared_kernels.split_transform_to_pos_1d,
                    dim=transform.shape,
                    inputs=[transform],
                    outputs=[source],
                    device=self.device,
                )
        return source

    def _get_quat_from_transform(self, source: wp.array | None, transform: wp.array) -> wp.array:
        """Generates a quaternion array from a transform array.

        Args:
            transform: The transform array. Shape is (N) dtype=wp.transformf.

        Returns:
            The quaternion array. Shape is (N) dtype=wp.quatf.
        """
        # Check if we already created the lazy buffer.
        if source is None:
            if transform.is_contiguous:
                # Check if the array is contiguous. If so, we can just return a strided array.
                # Then this update becomes a no-op.
                return wp.array(
                    ptr=transform.ptr + 3 * 4,
                    shape=transform.shape,
                    dtype=wp.quatf,
                    strides=transform.strides,
                    device=self.device,
                )
            else:
                # If the array is not contiguous, we need to create a new array to write to.
                # Shape matches transform.shape since each element is quatf (already contains 4 floats)
                source = wp.zeros(transform.shape, dtype=wp.quatf, device=self.device)

        # If the array is not contiguous, we need to launch the kernel to get the quaternion part of the transform.
        if not transform.is_contiguous:
            # Launch the right kernel based on the shape of the transform array.
            if len(transform.shape) > 1:
                wp.launch(
                    shared_kernels.split_transform_to_quat_2d,
                    dim=transform.shape,
                    inputs=[transform],
                    outputs=[source],
                    device=self.device,
                )
            else:
                wp.launch(
                    shared_kernels.split_transform_to_quat_1d,
                    dim=transform.shape,
                    inputs=[transform],
                    outputs=[source],
                    device=self.device,
                )
        # Return the source array. (no-op if the array is contiguous.)
        return source

    def _get_top_from_spatial_vector(self, source: wp.array | None, spatial_vector: wp.array) -> wp.array:
        """Gets the top part of a spatial vector array.

        For instance the linear velocity is the top part of a velocity vector.

        Args:
            spatial_vector: The spatial vector array. Shape is (N) dtype=wp.spatial_vectorf.

        Returns:
            The top part of the spatial vector array. Shape is (N) dtype=wp.vec3f.
        """
        # Check if we already created the lazy buffer.
        if source is None:
            if spatial_vector.is_contiguous:
                # Check if the array is contiguous. If so, we can just return a strided array.
                # Then this update becomes a no-op.
                return wp.array(
                    ptr=spatial_vector.ptr,
                    shape=spatial_vector.shape,
                    dtype=wp.vec3f,
                    strides=spatial_vector.strides,
                    device=self.device,
                )
            else:
                # If the array is not contiguous, we need to create a new array to write to.
                # Shape matches spatial_vector.shape since each element is vec3f (already contains 3 floats)
                source = wp.zeros(spatial_vector.shape, dtype=wp.vec3f, device=self.device)

        # If the array is not contiguous, we need to launch the kernel to get the top part of the spatial vector.
        if not spatial_vector.is_contiguous:
            # Launch the right kernel based on the shape of the spatial_vector array.
            if len(spatial_vector.shape) > 1:
                wp.launch(
                    shared_kernels.split_spatial_vector_to_top_2d,
                    dim=spatial_vector.shape,
                    inputs=[spatial_vector],
                    outputs=[source],
                    device=self.device,
                )
            else:
                wp.launch(
                    shared_kernels.split_spatial_vector_to_top_1d,
                    dim=spatial_vector.shape,
                    inputs=[spatial_vector],
                    outputs=[source],
                    device=self.device,
                )
        # Return the source array. (no-op if the array is contiguous.)
        return source

    def _get_bottom_from_spatial_vector(self, source: wp.array | None, spatial_vector: wp.array) -> wp.array:
        """Gets the bottom part of a spatial vector array.

        For instance the angular velocity is the bottom part of a velocity vector.

        Args:
            spatial_vector: The spatial vector array. Shape is (N) dtype=wp.spatial_vectorf.

        Returns:
            The bottom part of the spatial vector array. Shape is (N) dtype=wp.vec3f.
        """
        # Check if we already created the lazy buffer.
        if source is None:
            if spatial_vector.is_contiguous:
                # Check if the array is contiguous. If so, we can just return a strided array.
                # Then this update becomes a no-op.
                return wp.array(
                    ptr=spatial_vector.ptr + 3 * 4,
                    shape=spatial_vector.shape,
                    dtype=wp.vec3f,
                    strides=spatial_vector.strides,
                    device=self.device,
                )
            else:
                # If the array is not contiguous, we need to create a new array to write to.
                # Shape matches spatial_vector.shape since each element is vec3f (already contains 3 floats)
                source = wp.zeros(spatial_vector.shape, dtype=wp.vec3f, device=self.device)

        # If the array is not contiguous, we need to launch the kernel to get the bottom part of the spatial vector.
        if not spatial_vector.is_contiguous:
            # Launch the right kernel based on the shape of the spatial_vector array.
            if len(spatial_vector.shape) > 1:
                wp.launch(
                    shared_kernels.split_spatial_vector_to_bottom_2d,
                    dim=spatial_vector.shape,
                    inputs=[spatial_vector],
                    outputs=[source],
                    device=self.device,
                )
            else:
                wp.launch(
                    shared_kernels.split_spatial_vector_to_bottom_1d,
                    dim=spatial_vector.shape,
                    inputs=[spatial_vector],
                    outputs=[source],
                    device=self.device,
                )
        # Return the source array. (no-op if the array is contiguous.)
        return source

    """
    Deprecated properties.
    """

    @property
    def root_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`root_link_pose_w` and :attr:`root_com_vel_w`."""
        warnings.warn(
            "The `root_state_w` property will be deprecated in a IsaacLab 4.0. Please use `root_link_pose_w` and "
            "`root_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._root_state_w is None:
            self._root_state_w = TimestampedBuffer(
                shape=(self._num_instances,), dtype=shared_kernels.vec13f, device=self.device
            )
            self._root_state_w_ta = ProxyArray(self._root_state_w.data)
        if self._root_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=(self._num_instances),
                inputs=[
                    self.root_link_pose_w.warp,
                    self.root_com_vel_w.warp,
                ],
                outputs=[
                    self._root_state_w.data,
                ],
                device=self.device,
            )
            self._root_state_w.timestamp = self._sim_timestamp

        return self._root_state_w_ta

    @property
    def root_link_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`root_link_pose_w` and :attr:`root_link_vel_w`."""
        warnings.warn(
            "The `root_link_state_w` property will be deprecated in a IsaacLab 4.0. Please use `root_link_pose_w` and "
            "`root_link_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._root_link_state_w is None:
            self._root_link_state_w = TimestampedBuffer(
                shape=(self._num_instances,), dtype=shared_kernels.vec13f, device=self.device
            )
            self._root_link_state_w_ta = ProxyArray(self._root_link_state_w.data)
        if self._root_link_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=self._num_instances,
                inputs=[
                    self.root_link_pose_w.warp,
                    self.root_link_vel_w.warp,
                ],
                outputs=[
                    self._root_link_state_w.data,
                ],
                device=self.device,
            )
            self._root_link_state_w.timestamp = self._sim_timestamp

        return self._root_link_state_w_ta

    @property
    def root_com_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`root_com_pose_w` and :attr:`root_com_vel_w`."""
        warnings.warn(
            "The `root_com_state_w` property will be deprecated in a IsaacLab 4.0. Please use `root_com_pose_w` and "
            "`root_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._root_com_state_w is None:
            self._root_com_state_w = TimestampedBuffer(
                shape=(self._num_instances,), dtype=shared_kernels.vec13f, device=self.device
            )
            self._root_com_state_w_ta = ProxyArray(self._root_com_state_w.data)
        if self._root_com_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=self._num_instances,
                inputs=[
                    self.root_com_pose_w.warp,
                    self.root_com_vel_w.warp,
                ],
                outputs=[
                    self._root_com_state_w.data,
                ],
                device=self.device,
            )
            self._root_com_state_w.timestamp = self._sim_timestamp

        return self._root_com_state_w_ta

    @property
    def default_root_state(self) -> ProxyArray:
        """Default root state ``[pos, quat, lin_vel, ang_vel]`` in the local environment frame.


        The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
        velocities are of its center of mass frame. Shape is (num_instances, 13).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        warnings.warn(
            "Reading the root state directly is deprecated since IsaacLab 3.0 and will be removed in a future version. "
            "Please use the default_root_pose and default_root_vel properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_root_state is None:
            self._default_root_state = wp.zeros((self._num_instances), dtype=shared_kernels.vec13f, device=self.device)
            self._default_root_state_ta = ProxyArray(self._default_root_state)
        wp.launch(
            shared_kernels.concat_root_pose_and_vel_to_state,
            dim=self._num_instances,
            inputs=[
                self._default_root_pose,
                self._default_root_vel,
            ],
            outputs=[
                self._default_root_state,
            ],
            device=self.device,
        )
        return self._default_root_state_ta

    @property
    def body_state_w(self) -> ProxyArray:
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position and quaternion are of all the articulation links' actor frame. Meanwhile, the linear and angular
        velocities are of the articulation links's center of mass frame.
        """
        warnings.warn(
            "The `body_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_link_pose_w` and "
            "`body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._body_state_w is None:
            self._body_state_w = TimestampedBuffer(
                (self._num_instances, self._num_bodies), self.device, shared_kernels.vec13f
            )
            self._body_state_w_ta = ProxyArray(self._body_state_w.data)
        if self._body_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_link_pose_w.warp,
                    self.body_com_vel_w.warp,
                ],
                outputs=[
                    self._body_state_w.data,
                ],
                device=self.device,
            )
            self._body_state_w.timestamp = self._sim_timestamp

        return self._body_state_w_ta

    @property
    def body_link_state_w(self) -> ProxyArray:
        """State of all bodies' link frame`[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """
        warnings.warn(
            "The `body_link_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_link_pose_w` and "
            "`body_link_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._body_link_state_w is None:
            self._body_link_state_w = TimestampedBuffer(
                (self._num_instances, self._num_bodies), self.device, shared_kernels.vec13f
            )
            self._body_link_state_w_ta = ProxyArray(self._body_link_state_w.data)
        if self._body_link_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_link_pose_w.warp,
                    self.body_link_vel_w.warp,
                ],
                outputs=[
                    self._body_link_state_w.data,
                ],
                device=self.device,
            )
            self._body_link_state_w.timestamp = self._sim_timestamp

        return self._body_link_state_w_ta

    @property
    def body_com_state_w(self) -> ProxyArray:
        """State of all bodies center of mass `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principal inertia.
        """
        warnings.warn(
            "The `body_com_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_com_pose_w` and "
            "`body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._body_com_state_w is None:
            self._body_com_state_w = TimestampedBuffer(
                (self._num_instances, self._num_bodies), self.device, shared_kernels.vec13f
            )
            self._body_com_state_w_ta = ProxyArray(self._body_com_state_w.data)
        if self._body_com_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_com_pose_w.warp,
                    self.body_com_vel_w.warp,
                ],
                outputs=[
                    self._body_com_state_w.data,
                ],
                device=self.device,
            )
            self._body_com_state_w.timestamp = self._sim_timestamp

        return self._body_com_state_w_ta
