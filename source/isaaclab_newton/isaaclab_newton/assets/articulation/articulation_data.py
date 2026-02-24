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

from isaaclab_newton.assets import kernels as shared_kernels
from isaaclab_newton.assets.articulation import kernels as articulation_kernels
from isaaclab_newton.physics import NewtonManager as SimulationManager

if TYPE_CHECKING:
    from newton.selection import ArticulationView

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

        # Convert to direction vector
        gravity = wp.to_torch(SimulationManager.get_model().gravity)[0]
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        gravity_dir = gravity_dir.repeat(self._root_view.count, 1)
        forward_vec = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_view.count, 1)

        # Initialize constants
        self.GRAVITY_VEC_W = wp.from_torch(gravity_dir, dtype=wp.vec3f)
        self.FORWARD_VEC_B = wp.from_torch(forward_vec, dtype=wp.vec3f)

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
    def default_root_pose(self) -> wp.array:
        """Default root pose ``[pos, quat]`` in the local environment frame.

        The position and quaternion are of the articulation root's actor frame. Shape is (num_instances),
        dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        """
        return self._default_root_pose

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
    def default_root_vel(self) -> wp.array:
        """Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame.

        The linear and angular velocities are of the articulation root's center of mass frame.
        Shape is (num_instances), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).
        """
        return self._default_root_vel

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
    def default_joint_pos(self) -> wp.array:
        """Default joint positions of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_joint_pos

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
    def default_joint_vel(self) -> wp.array:
        """Default joint velocities of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_joint_vel

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
    def joint_pos_target(self) -> wp.array:
        """Joint position targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return self._joint_pos_target

    @property
    def joint_vel_target(self) -> wp.array:
        """Joint velocity targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return self._joint_vel_target

    @property
    def joint_effort_target(self) -> wp.array:
        """Joint effort targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return self._joint_effort_target

    """
    Joint commands -- Explicit actuators.
    """

    @property
    def computed_torque(self) -> wp.array:
        """Joint torques computed from the actuator model (before clipping).

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        This quantity is the raw torque output from the actuator mode, before any clipping is applied.
        It is exposed for users who want to inspect the computations inside the actuator model.
        For instance, to penalize the learning agent for a difference between the computed and applied torques.
        """
        return self._computed_torque

    @property
    def applied_torque(self) -> wp.array:
        """Joint torques applied from the actuator model (after clipping).

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

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

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._sim_bind_joint_stiffness_sim

    @property
    def joint_damping(self) -> wp.array:
        """Joint damping provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._sim_bind_joint_damping_sim

    @property
    def joint_armature(self) -> wp.array:
        """Joint armature provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        return self._sim_bind_joint_armature

    @property
    def joint_friction_coeff(self) -> wp.array:
        """Joint static friction coefficient provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        return self._sim_bind_joint_friction_coeff

    @property
    def joint_pos_limits_lower(self) -> wp.array:
        """Joint position limits lower provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_pos_limits_lower

    @property
    def joint_pos_limits_upper(self) -> wp.array:
        """Joint position limits upper provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._sim_bind_joint_pos_limits_upper

    @property
    def joint_pos_limits(self) -> wp.array:
        """Joint position limits provided to the simulation.

        Shape is (num_instances, num_joints, 2), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        if self._joint_pos_limits is None:
            self._joint_pos_limits = wp.zeros(
                (self._num_instances, self._num_joints), dtype=wp.vec2f, device=self.device
            )
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
        return self._joint_pos_limits

    @property
    def joint_vel_limits(self) -> wp.array:
        """Joint maximum velocity provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        return self._sim_bind_joint_vel_limits_sim

    @property
    def joint_effort_limits(self) -> wp.array:
        """Joint maximum effort provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        return self._sim_bind_joint_effort_limits_sim

    """
    Joint properties - Custom.
    """

    @property
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
        return self._soft_joint_pos_limits

    @property
    def soft_joint_vel_limits(self) -> wp.array:
        """Soft joint velocity limits for all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
        has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
        """
        return self._soft_joint_vel_limits

    @property
    def gear_ratio(self) -> wp.array:
        """Gear ratio for relating motor torques to applied Joint torques.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
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
        raise NotImplementedError

    @property
    def fixed_tendon_damping(self) -> wp.array:
        """Fixed tendon damping provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    def fixed_tendon_limit_stiffness(self) -> wp.array:
        """Fixed tendon limit stiffness provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    def fixed_tendon_rest_length(self) -> wp.array:
        """Fixed tendon rest length provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    def fixed_tendon_offset(self) -> wp.array:
        """Fixed tendon offset provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        raise NotImplementedError

    @property
    def fixed_tendon_pos_limits(self) -> wp.array:
        """Fixed tendon position limits provided to the simulation.

        Shape is (num_instances, num_fixed_tendons, 2), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_fixed_tendons, 2).
        """
        raise NotImplementedError

    """
    Spatial tendon properties.
    """

    @property
    def spatial_tendon_stiffness(self) -> wp.array:
        """Spatial tendon stiffness provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    @property
    def spatial_tendon_damping(self) -> wp.array:
        """Spatial tendon damping provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    @property
    def spatial_tendon_limit_stiffness(self) -> wp.array:
        """Spatial tendon limit stiffness provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

    @property
    def spatial_tendon_offset(self) -> wp.array:
        """Spatial tendon offset provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        raise NotImplementedError

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
        return self._sim_bind_root_link_pose_w

    @property
    def root_link_vel_w(self) -> wp.array:
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
                    self.root_com_vel_w,
                    self.root_link_quat_w,
                    self.body_com_pos_b,
                ],
                outputs=[
                    self._root_link_vel_w.data,
                ],
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
            # apply local transform to center of mass frame
            wp.launch(
                shared_kernels.get_root_com_pose_from_root_link_pose,
                dim=self._num_instances,
                inputs=[
                    self.root_link_pose_w,
                    self.body_com_pos_b,
                ],
                outputs=[
                    self._root_com_pose_w.data,
                ],
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
        return self._sim_bind_root_com_vel_w

    """
    Body state properties.
    """

    @property
    def body_mass(self) -> wp.array:
        """Body mass ``wp.float32`` in the world frame.

        Shape is (num_instances, num_bodies), dtype = wp.float32. In torch this resolves to (num_instances, num_bodies).
        """
        return self._sim_bind_body_mass

    @property
    def body_inertia(self) -> wp.array:
        """Flattened body inertia in the world frame.

        Shape is (num_instances, num_bodies, 9), dtype = wp.float32. In torch this resolves to
        (num_instances, num_bodies, 9).
        """
        return self._sim_bind_body_inertia

    @property
    def body_link_pose_w(self) -> wp.array:
        """Body link pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return self._sim_bind_body_link_pose_w

    @property
    def body_link_vel_w(self) -> wp.array:
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
                    self.body_com_vel_w,
                    self.body_link_pose_w,
                    self.body_com_pos_b,
                ],
                outputs=[
                    self._body_link_vel_w.data,
                ],
                device=self.device,
            )
            self._body_link_vel_w.timestamp = self._sim_timestamp

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
                shared_kernels.get_body_com_pose_from_body_link_pose,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_link_pose_w,
                    self.body_com_pos_b,
                ],
                outputs=[
                    self._body_com_pose_w.data,
                ],
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
        """
        return self._sim_bind_body_com_vel_w

    @property
    def body_com_acc_w(self):
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
        return self._body_com_acc_w.data

    @property
    def body_com_pos_b(self) -> wp.array:
        """Center of mass position of all of the bodies in their respective link frames.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body's link frame.
        """
        return self._sim_bind_body_com_pos_b

    @property
    def body_com_pose_b(self) -> wp.array:
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
                    self.body_com_pos_b,
                ],
                outputs=[
                    self._body_com_pose_b.data,
                ],
                device=self.device,
            )
            self._body_com_pose_b.timestamp = self._sim_timestamp
        return self._body_com_pose_b.data

    @property
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

    """
    Joint state properties.
    """

    @property
    def joint_pos(self) -> wp.array:
        """Joint positions of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        return self._sim_bind_joint_pos

    @property
    def joint_vel(self) -> wp.array:
        """Joint velocities of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to
        (num_instances, num_joints).
        """
        return self._sim_bind_joint_vel

    @property
    def joint_acc(self) -> wp.array:
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
                    self.joint_vel,
                    self._previous_joint_vel,
                    time_elapsed,
                ],
                outputs=[
                    self._joint_acc.data,
                ],
                device=self.device,
            )
            self._joint_acc.timestamp = self._sim_timestamp
        return self._joint_acc.data

    """
    Derived Properties.
    """

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.GRAVITY_VEC_W, self.root_link_quat_w],
                outputs=[self._projected_gravity_b.data],
                device=self.device,
            )
            self._projected_gravity_b.timestamp = self._sim_timestamp
        return self._projected_gravity_b.data

    @property
    def heading_w(self):
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
                inputs=[self.FORWARD_VEC_B, self.root_link_quat_w],
                outputs=[self._heading_w.data],
                device=self.device,
            )
            self._heading_w.timestamp = self._sim_timestamp
        return self._heading_w.data

    @property
    def root_link_lin_vel_b(self) -> wp.array:
        """Root link linear velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to
        its actor frame.
        """
        if self._root_link_lin_vel_b is None:
            self._root_link_lin_vel_b = TimestampedBuffer(
                shape=(self._num_instances,), dtype=wp.vec3f, device=self.device
            )
        if self._root_link_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_link_lin_vel_w, self.root_link_quat_w],
                outputs=[self._root_link_lin_vel_b.data],
                device=self.device,
            )
            self._root_link_lin_vel_b.timestamp = self._sim_timestamp
        return self._root_link_lin_vel_b.data

    @property
    def root_link_ang_vel_b(self) -> wp.array:
        """Root link angular velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to
        its actor frame.
        """
        if self._root_link_ang_vel_b is None:
            self._root_link_ang_vel_b = TimestampedBuffer(
                shape=(self._num_instances,), dtype=wp.vec3f, device=self.device
            )
        if self._root_link_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_link_ang_vel_w, self.root_link_quat_w],
                outputs=[self._root_link_ang_vel_b.data],
                device=self.device,
            )
            self._root_link_ang_vel_b.timestamp = self._sim_timestamp
        return self._root_link_ang_vel_b.data

    @property
    def root_com_lin_vel_b(self) -> wp.array:
        """Root center of mass linear velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to
        its actor frame.
        """
        if self._root_com_lin_vel_b is None:
            self._root_com_lin_vel_b = TimestampedBuffer(
                shape=(self._num_instances,), dtype=wp.vec3f, device=self.device
            )
        if self._root_com_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_com_lin_vel_w, self.root_link_quat_w],
                outputs=[self._root_com_lin_vel_b.data],
                device=self.device,
            )
            self._root_com_lin_vel_b.timestamp = self._sim_timestamp
        return self._root_com_lin_vel_b.data

    @property
    def root_com_ang_vel_b(self) -> wp.array:
        """Root center of mass angular velocity in base frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to
        its actor frame.
        """
        if self._root_com_ang_vel_b is None:
            self._root_com_ang_vel_b = TimestampedBuffer(
                shape=(self._num_instances,), dtype=wp.vec3f, device=self.device
            )
        if self._root_com_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_com_ang_vel_w, self.root_link_quat_w],
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

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._get_pos_from_transform(self._root_link_pos_w, self.root_link_pose_w)

    @property
    def root_link_quat_w(self) -> wp.array:
        """Root link orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances), dtype = wp.quatf. In torch this resolves to (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self._get_quat_from_transform(self._root_link_quat_w, self.root_link_pose_w)

    @property
    def root_link_lin_vel_w(self) -> wp.array:
        """Root linear velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._get_top_from_spatial_vector(self._root_link_lin_vel_w, self.root_link_vel_w)

    @property
    def root_link_ang_vel_w(self) -> wp.array:
        """Root link angular velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self._get_bottom_from_spatial_vector(self._root_link_ang_vel_w, self.root_link_vel_w)

    @property
    def root_com_pos_w(self) -> wp.array:
        """Root center of mass position in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the position of the center of mass frame of the root rigid body relative to the world.
        """
        return self._get_pos_from_transform(self._root_com_pos_w, self.root_com_pose_w)

    @property
    def root_com_quat_w(self) -> wp.array:
        """Root center of mass orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances), dtype = wp.quatf. In torch this resolves to (num_instances, 4).

        This quantity is the orientation of the principal axes of inertia of the root rigid body relative to the world.
        """
        return self._get_quat_from_transform(self._root_com_quat_w, self.root_com_pose_w)

    @property
    def root_com_lin_vel_w(self) -> wp.array:
        """Root center of mass linear velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._get_top_from_spatial_vector(self._root_com_lin_vel_w, self.root_com_vel_w)

    @property
    def root_com_ang_vel_w(self) -> wp.array:
        """Root center of mass angular velocity in simulation world frame.

        Shape is (num_instances), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._get_bottom_from_spatial_vector(self._root_com_ang_vel_w, self.root_com_vel_w)

    @property
    def body_link_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        return self._get_pos_from_transform(self._body_link_pos_w, self.body_link_pose_w)

    @property
    def body_link_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        return self._get_quat_from_transform(self._body_link_quat_w, self.body_link_pose_w)

    @property
    def body_link_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' actor frame relative to the world.
        """
        return self._get_top_from_spatial_vector(self._body_link_lin_vel_w, self.body_link_vel_w)

    @property
    def body_link_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' actor frame relative to the world.
        """
        return self._get_bottom_from_spatial_vector(self._body_link_ang_vel_w, self.body_link_vel_w)

    @property
    def body_com_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' center of mass frame.
        """
        return self._get_pos_from_transform(self._body_com_pos_w, self.body_com_pose_w)

    @property
    def body_com_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the principal axes of inertia of the articulation bodies.
        """
        return self._get_quat_from_transform(self._body_com_quat_w, self.body_com_pose_w)

    @property
    def body_com_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        return self._get_top_from_spatial_vector(self._body_com_lin_vel_w, self.body_com_vel_w)

    @property
    def body_com_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        return self._get_bottom_from_spatial_vector(self._body_com_ang_vel_w, self.body_com_vel_w)

    @property
    def body_com_lin_acc_w(self) -> wp.array:
        """Linear acceleration of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        return self._get_top_from_spatial_vector(self._body_com_lin_acc_w, self.body_com_acc_w)

    @property
    def body_com_ang_acc_w(self) -> wp.array:
        """Angular acceleration of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        return self._get_bottom_from_spatial_vector(self._body_com_ang_acc_w, self.body_com_acc_w)

    @property
    def body_com_quat_b(self) -> wp.array:
        """Orientation (x, y, z, w) of the principal axes of inertia of all of the bodies in their respective link
        frames.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the principal axes of inertia relative to its body's link frame.
        """
        return self._get_quat_from_transform(self._body_com_quat_b, self.body_com_pose_b)

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
            self._sim_bind_root_link_pose_w = self._root_view.get_root_transforms(SimulationManager.get_state_0())[
                :, 0, 0
            ]
        else:
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
        self._sim_bind_body_inertia = self._root_view.get_attribute("body_inertia", SimulationManager.get_model())[:, 0]
        self._sim_bind_body_external_wrench = self._root_view.get_attribute("body_f", SimulationManager.get_state_0())[
            :, 0
        ]
        # -- joint properties
        if n_dof > 0:
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
        super()._create_buffers()

        # Short-hand for the number of instances, number of links, and number of joints.
        self._num_instances = self._root_view.count
        self._num_joints = self._root_view.joint_dof_count
        self._num_bodies = self._root_view.link_count
        self._num_fixed_tendons = 0  # self._root_view.max_fixed_tendons
        self._num_spatial_tendons = 0  # self._root_view.max_spatial_tendons

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
        # self._body_incoming_joint_wrench_b = TimestampedWarpBuffer(
        #     shape=(self._num_instances, self._num_joints), dtype=wp.spatial_vectorf, device=self.device
        # )
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
    def root_state_w(self) -> wp.array:
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
        if self._root_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=(self._num_instances),
                inputs=[
                    self.root_link_pose_w,
                    self.root_com_vel_w,
                ],
                outputs=[
                    self._root_state_w.data,
                ],
                device=self.device,
            )
            self._root_state_w.timestamp = self._sim_timestamp

        return self._root_state_w.data

    @property
    def root_link_state_w(self) -> wp.array:
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
        if self._root_link_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=self._num_instances,
                inputs=[
                    self.root_link_pose_w,
                    self.root_link_vel_w,
                ],
                outputs=[
                    self._root_link_state_w.data,
                ],
                device=self.device,
            )
            self._root_link_state_w.timestamp = self._sim_timestamp

        return self._root_link_state_w.data

    @property
    def root_com_state_w(self) -> wp.array:
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
        if self._root_com_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=self._num_instances,
                inputs=[
                    self.root_com_pose_w,
                    self.root_com_vel_w,
                ],
                outputs=[
                    self._root_com_state_w.data,
                ],
                device=self.device,
            )
            self._root_com_state_w.timestamp = self._sim_timestamp

        return self._root_com_state_w.data

    @property
    def default_root_state(self) -> wp.array:
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
        return self._default_root_state

    @property
    def body_state_w(self):
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
        if self._body_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_link_pose_w,
                    self.body_com_vel_w,
                ],
                outputs=[
                    self._body_state_w.data,
                ],
                device=self.device,
            )
            self._body_state_w.timestamp = self._sim_timestamp

        return self._body_state_w.data

    @property
    def body_link_state_w(self):
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
        if self._body_link_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_link_pose_w,
                    self.body_link_vel_w,
                ],
                outputs=[
                    self._body_link_state_w.data,
                ],
                device=self.device,
            )
            self._body_link_state_w.timestamp = self._sim_timestamp

        return self._body_link_state_w.data

    @property
    def body_com_state_w(self):
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
        if self._body_com_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_com_pose_w,
                    self.body_com_vel_w,
                ],
                outputs=[
                    self._body_com_state_w.data,
                ],
                device=self.device,
            )
            self._body_com_state_w.timestamp = self._sim_timestamp

        return self._body_com_state_w.data
