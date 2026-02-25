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

from isaaclab_physx.assets import kernels as shared_kernels
from isaaclab_physx.assets.articulation import kernels as articulation_kernels
from isaaclab_physx.physics import PhysxManager as SimulationManager

if TYPE_CHECKING:
    from isaaclab.assets.articulation.articulation_view import ArticulationView

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

    __backend_name__: str = "physx"
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

        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        gravity = self._physics_sim_view.get_gravity()
        # Convert to direction vector
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        gravity_dir = gravity_dir.repeat(self._root_view.count, 1)
        forward_vec = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_view.count, 1)

        # Initialize constants
        self.GRAVITY_VEC_W = wp.from_torch(gravity_dir, dtype=wp.vec3f)
        self.FORWARD_VEC_B = wp.from_torch(forward_vec, dtype=wp.vec3f)

        self._create_buffers()

    @property
    def is_primed(self) -> bool:
        """Whether the articulation data is fully instantiated and ready to use."""
        return self._is_primed

    @is_primed.setter
    def is_primed(self, value: bool) -> None:
        """Set whether the articulation data is fully instantiated and ready to use.

        .. note:: Once this quantity is set to True, it cannot be changed.

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
        # Trigger an update of the joint acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.joint_acc

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

        The position and quaternion are of the articulation root's actor frame. Shape is (num_instances, 7).
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
        Shape is (num_instances, 6).
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
    def default_joint_pos(self) -> wp.array:
        """Default joint positions of all joints. Shape is (num_instances, num_joints).

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
        """Default joint velocities of all joints. Shape is (num_instances, num_joints).

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
        """Joint position targets commanded by the user. Shape is (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return self._joint_pos_target

    @property
    def joint_vel_target(self) -> wp.array:
        """Joint velocity targets commanded by the user. Shape is (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return self._joint_vel_target

    @property
    def joint_effort_target(self) -> wp.array:
        """Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

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
        """Joint torques computed from the actuator model (before clipping). Shape is (num_instances, num_joints).

        This quantity is the raw torque output from the actuator mode, before any clipping is applied.
        It is exposed for users who want to inspect the computations inside the actuator model.
        For instance, to penalize the learning agent for a difference between the computed and applied torques.
        """
        return self._computed_torque

    @property
    def applied_torque(self) -> wp.array:
        """Joint torques applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

        These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
        actuator model.
        """
        return self._applied_torque

    """
    Joint properties
    """

    @property
    def joint_stiffness(self) -> wp.array:
        """Joint stiffness provided to the simulation. Shape is (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._joint_stiffness

    @property
    def joint_damping(self) -> wp.array:
        """Joint damping provided to the simulation. Shape is (num_instances, num_joints)

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._joint_damping

    @property
    def joint_armature(self) -> wp.array:
        """Joint armature provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_armature

    @property
    def joint_friction_coeff(self) -> wp.array:
        """Joint static friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_friction_coeff

    @property
    def joint_dynamic_friction_coeff(self) -> wp.array:
        """Joint dynamic friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_dynamic_friction_coeff

    @property
    def joint_viscous_friction_coeff(self) -> wp.array:
        """Joint viscous friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_viscous_friction_coeff

    @property
    def joint_pos_limits(self) -> wp.array:
        """Joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        return self._joint_pos_limits

    @property
    def joint_vel_limits(self) -> wp.array:
        """Joint maximum velocity provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_vel_limits

    @property
    def joint_effort_limits(self) -> wp.array:
        """Joint maximum effort provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_effort_limits

    """
    Joint properties - Custom.
    """

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

    @property
    def soft_joint_vel_limits(self) -> wp.array:
        """Soft joint velocity limits for all joints. Shape is (num_instances, num_joints).

        These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
        has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
        """
        return self._soft_joint_vel_limits

    @property
    def gear_ratio(self) -> wp.array:
        """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""
        return self._gear_ratio

    """
    Fixed tendon properties.
    """

    @property
    def fixed_tendon_stiffness(self) -> wp.array:
        """Fixed tendon stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        return self._fixed_tendon_stiffness

    @property
    def fixed_tendon_damping(self) -> wp.array:
        """Fixed tendon damping provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        return self._fixed_tendon_damping

    @property
    def fixed_tendon_limit_stiffness(self) -> wp.array:
        """Fixed tendon limit stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        return self._fixed_tendon_limit_stiffness

    @property
    def fixed_tendon_rest_length(self) -> wp.array:
        """Fixed tendon rest length provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        return self._fixed_tendon_rest_length

    @property
    def fixed_tendon_offset(self) -> wp.array:
        """Fixed tendon offset provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        return self._fixed_tendon_offset

    @property
    def fixed_tendon_pos_limits(self) -> wp.array:
        """Fixed tendon position limits provided to the simulation. Shape is (num_instances, num_fixed_tendons, 2)."""
        return self._fixed_tendon_pos_limits

    """
    Spatial tendon properties.
    """

    @property
    def spatial_tendon_stiffness(self) -> wp.array:
        """Spatial tendon stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        return self._spatial_tendon_stiffness

    @property
    def spatial_tendon_damping(self) -> wp.array:
        """Spatial tendon damping provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        return self._spatial_tendon_damping

    @property
    def spatial_tendon_limit_stiffness(self) -> wp.array:
        """Spatial tendon limit stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        return self._spatial_tendon_limit_stiffness

    @property
    def spatial_tendon_offset(self) -> wp.array:
        """Spatial tendon offset provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        return self._spatial_tendon_offset

    """
    Root state properties.
    """

    @property
    def root_link_pose_w(self) -> wp.array:
        """Root link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._root_link_pose_w.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            self._root_link_pose_w.data = self._root_view.get_root_transforms().view(wp.transformf)
            self._root_link_pose_w.timestamp = self._sim_timestamp

        return self._root_link_pose_w.data

    @property
    def root_link_vel_w(self) -> wp.array:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_root_link_vel_from_root_com_vel,
                dim=self._num_instances,
                inputs=[
                    self.root_com_vel_w,
                    self.root_link_pose_w,
                    self.body_com_pose_b,
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
        """Root center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

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
                    self.body_com_pose_b,
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
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        if self._root_com_vel_w.timestamp < self._sim_timestamp:
            self._root_com_vel_w.data = self._root_view.get_root_velocities().view(wp.spatial_vectorf)
            self._root_com_vel_w.timestamp = self._sim_timestamp

        return self._root_com_vel_w.data

    """
    Body state properties.
    """

    @property
    def body_mass(self) -> wp.array:
        """Body mass ``wp.float32`` in the world frame. Shape is (num_instances, num_bodies)."""
        self._body_mass.assign(self._root_view.get_masses())
        return self._body_mass

    @property
    def body_inertia(self) -> wp.array:
        """Body inertia ``wp.mat33`` in the world frame. Shape is (num_instances, num_bodies, 3, 3)."""
        self._body_inertia.assign(self._root_view.get_inertias())
        return self._body_inertia

    @property
    def body_link_pose_w(self) -> wp.array:
        """Body link pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_link_pose_w.timestamp < self._sim_timestamp:
            # perform forward kinematics (shouldn't cause overhead if it happened already)
            self._physics_sim_view.update_articulations_kinematic()
            # set the buffer data and timestamp
            self._body_link_pose_w.data = self._root_view.get_link_transforms().view(wp.transformf)
            self._body_link_pose_w.timestamp = self._sim_timestamp

        return self._body_link_pose_w.data

    @property
    def body_link_vel_w(self) -> wp.array:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

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
                    self.body_com_pose_b,
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
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_body_com_pose_from_body_link_pose,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_link_pose_w,
                    self.body_com_pose_b,
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
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        if self._body_com_vel_w.timestamp < self._sim_timestamp:
            self._body_com_vel_w.data = self._root_view.get_link_velocities().view(wp.spatial_vectorf)
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
        principle inertia.
        """
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

    @property
    def body_com_acc_w(self):
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]``.
        Shape is (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._body_com_acc_w.data = self._root_view.get_link_accelerations().view(wp.spatial_vectorf)
            self._body_com_acc_w.timestamp = self._sim_timestamp

        return self._body_com_acc_w.data

    @property
    def body_com_pose_b(self) -> wp.array:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.
        Shape is (num_instances, 1, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            self._body_com_pose_b.data.assign(self._root_view.get_coms().view(wp.transformf))
            self._body_com_pose_b.timestamp = self._sim_timestamp

        return self._body_com_pose_b.data

    @property
    def body_incoming_joint_wrench_b(self) -> wp.array:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.

        For more information on joint wrenches, please check the`PhysX documentation`_ and the underlying
        `PhysX Tensor API`_.

        .. _`PhysX documentation`: https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force
        .. _`PhysX Tensor API`: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force
        """

        if self._body_incoming_joint_wrench_b.timestamp < self._sim_timestamp:
            self._body_incoming_joint_wrench_b.data = self._root_view.get_link_incoming_joint_force().view(
                wp.spatial_vectorf
            )
            self._body_incoming_joint_wrench_b.timestamp = self._sim_timestamp
        return self._body_incoming_joint_wrench_b.data

    """
    Joint state properties.
    """

    @property
    def joint_pos(self) -> wp.array:
        """Joint positions of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_pos.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_pos.data = self._root_view.get_dof_positions()
            self._joint_pos.timestamp = self._sim_timestamp
        return self._joint_pos.data

    @property
    def joint_vel(self) -> wp.array:
        """Joint velocities of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_vel.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_vel.data = self._root_view.get_dof_velocities()
            self._joint_vel.timestamp = self._sim_timestamp
        return self._joint_vel.data

    @property
    def joint_acc(self) -> wp.array:
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
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
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
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
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

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
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
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
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
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
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
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
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
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
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._get_pos_from_transform(self.root_link_pose_w)

    @property
    def root_link_quat_w(self) -> wp.array:
        """Root link orientation (x, y, z, w) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self._get_quat_from_transform(self.root_link_pose_w)

    @property
    def root_link_lin_vel_w(self) -> wp.array:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._get_lin_vel_from_spatial_vector(self.root_link_vel_w)

    @property
    def root_link_ang_vel_w(self) -> wp.array:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self._get_ang_vel_from_spatial_vector(self.root_link_vel_w)

    @property
    def root_com_pos_w(self) -> wp.array:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._get_pos_from_transform(self.root_com_pose_w)

    @property
    def root_com_quat_w(self) -> wp.array:
        """Root center of mass orientation (x, y, z, w) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        return self._get_quat_from_transform(self.root_com_pose_w)

    @property
    def root_com_lin_vel_w(self) -> wp.array:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._get_lin_vel_from_spatial_vector(self.root_com_vel_w)

    @property
    def root_com_ang_vel_w(self) -> wp.array:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._get_ang_vel_from_spatial_vector(self.root_com_vel_w)

    @property
    def body_link_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        return self._get_pos_from_transform(self.body_link_pose_w)

    @property
    def body_link_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        return self._get_quat_from_transform(self.body_link_pose_w)

    @property
    def body_link_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return self._get_lin_vel_from_spatial_vector(self.body_link_vel_w)

    @property
    def body_link_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return self._get_ang_vel_from_spatial_vector(self.body_link_vel_w)

    @property
    def body_com_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame.
        """
        return self._get_pos_from_transform(self.body_com_pose_w)

    @property
    def body_com_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of the principle axis of inertia of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame.
        """
        return self._get_quat_from_transform(self.body_com_pose_w)

    @property
    def body_com_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        return self._get_lin_vel_from_spatial_vector(self.body_com_vel_w)

    @property
    def body_com_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        return self._get_ang_vel_from_spatial_vector(self.body_com_vel_w)

    @property
    def body_com_lin_acc_w(self) -> wp.array:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        return self._get_lin_vel_from_spatial_vector(self.body_com_acc_w)

    @property
    def body_com_ang_acc_w(self) -> wp.array:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        return self._get_ang_vel_from_spatial_vector(self.body_com_acc_w)

    @property
    def body_com_pos_b(self) -> wp.array:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body'slink frame.
        """
        return self._get_pos_from_transform(self.body_com_pose_b)

    @property
    def body_com_quat_b(self) -> wp.array:
        """Orientation (x, y, z, w) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        return self._get_quat_from_transform(self.body_com_pose_b)

    def _create_buffers(self) -> None:
        super()._create_buffers()
        # Initialize the lazy buffers.
        self._num_instances = self._root_view.count
        self._num_joints = self._root_view.shared_metatype.dof_count
        self._num_bodies = self._root_view.shared_metatype.link_count
        self._num_fixed_tendons = self._root_view.max_fixed_tendons
        self._num_spatial_tendons = self._root_view.max_spatial_tendons

        # -- link frame w.r.t. world frame
        self._root_link_pose_w = TimestampedBuffer((self._num_instances), self.device, wp.transformf)
        self._root_link_vel_w = TimestampedBuffer((self._num_instances), self.device, wp.spatial_vectorf)
        self._body_link_pose_w = TimestampedBuffer((self._num_instances, self._num_bodies), self.device, wp.transformf)
        self._body_link_vel_w = TimestampedBuffer(
            (self._num_instances, self._num_bodies), self.device, wp.spatial_vectorf
        )
        # -- com frame w.r.t. link frame
        self._body_com_pose_b = TimestampedBuffer((self._num_instances, self._num_bodies), self.device, wp.transformf)
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedBuffer((self._num_instances), self.device, wp.transformf)
        self._root_com_vel_w = TimestampedBuffer((self._num_instances), self.device, wp.spatial_vectorf)
        self._body_com_pose_w = TimestampedBuffer((self._num_instances, self._num_bodies), self.device, wp.transformf)
        self._body_com_vel_w = TimestampedBuffer(
            (self._num_instances, self._num_bodies), self.device, wp.spatial_vectorf
        )
        self._body_com_acc_w = TimestampedBuffer(
            (self._num_instances, self._num_bodies), self.device, wp.spatial_vectorf
        )
        # -- combined state (these are cached as they concatenate)
        self._root_state_w = TimestampedBuffer((self._num_instances), self.device, shared_kernels.vec13f)
        self._root_link_state_w = TimestampedBuffer((self._num_instances), self.device, shared_kernels.vec13f)
        self._root_com_state_w = TimestampedBuffer((self._num_instances), self.device, shared_kernels.vec13f)
        self._body_state_w = TimestampedBuffer(
            (self._num_instances, self._num_bodies), self.device, shared_kernels.vec13f
        )
        self._body_link_state_w = TimestampedBuffer(
            (self._num_instances, self._num_bodies), self.device, shared_kernels.vec13f
        )
        self._body_com_state_w = TimestampedBuffer(
            (self._num_instances, self._num_bodies), self.device, shared_kernels.vec13f
        )
        # -- joint state
        self._joint_pos = TimestampedBuffer((self._num_instances, self._num_joints), self.device, wp.float32)
        self._joint_vel = TimestampedBuffer((self._num_instances, self._num_joints), self.device, wp.float32)
        self._joint_acc = TimestampedBuffer((self._num_instances, self._num_joints), self.device, wp.float32)
        self._body_incoming_joint_wrench_b = TimestampedBuffer(
            (self._num_instances, self._num_bodies, self._num_joints), self.device, wp.spatial_vectorf
        )
        # -- derived properties (these are cached to avoid repeated memory allocations)
        self._projected_gravity_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)
        self._heading_w = TimestampedBuffer((self._num_instances), self.device, wp.float32)
        self._root_link_lin_vel_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)
        self._root_link_ang_vel_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)
        self._root_com_lin_vel_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)
        self._root_com_ang_vel_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)

        # Default root pose and velocity
        self._default_root_pose = wp.zeros((self._num_instances), dtype=wp.transformf, device=self.device)
        self._default_root_vel = wp.zeros((self._num_instances), dtype=wp.spatial_vectorf, device=self.device)
        self._default_joint_pos = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        self._default_joint_vel = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )

        # Initialize history for finite differencing
        self._previous_joint_vel = wp.clone(self._root_view.get_dof_velocities(), device=self.device)

        # Pre-allocated buffers
        # -- Joint commands (set into simulation)
        self._joint_pos_target = wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        self._joint_vel_target = wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        self._joint_effort_target = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        # -- Joint commands (explicit actuator model)
        self._computed_torque = wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        self._applied_torque = wp.zeros((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        # -- Joint properties
        self._joint_stiffness = wp.clone(self._root_view.get_dof_stiffnesses(), device=self.device)
        self._joint_damping = wp.clone(self._root_view.get_dof_dampings(), device=self.device)
        self._joint_armature = wp.clone(self._root_view.get_dof_armatures(), device=self.device)
        friction_props = wp.clone(self._root_view.get_dof_friction_properties(), device=self.device)
        # Initialize output arrays
        self._joint_friction_coeff = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        self._joint_dynamic_friction_coeff = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        self._joint_viscous_friction_coeff = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        # Extract friction properties using kernel
        wp.launch(
            articulation_kernels.extract_friction_properties,
            dim=(self._num_instances, self._num_joints),
            inputs=[friction_props],
            outputs=[
                self._joint_friction_coeff,
                self._joint_dynamic_friction_coeff,
                self._joint_viscous_friction_coeff,
            ],
            device=self.device,
        )
        self._joint_pos_limits = wp.zeros((self._num_instances, self._num_joints), dtype=wp.vec2f, device=self.device)
        self._joint_pos_limits.assign(self._root_view.get_dof_limits().view(wp.vec2f))
        self._joint_vel_limits = wp.clone(self._root_view.get_dof_max_velocities(), device=self.device)
        self._joint_effort_limits = wp.clone(self._root_view.get_dof_max_forces(), device=self.device)
        # -- Joint properties (custom)
        self._soft_joint_pos_limits = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.vec2f, device=self.device
        )
        self._soft_joint_vel_limits = wp.zeros(
            (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        self._gear_ratio = wp.ones((self._num_instances, self._num_joints), dtype=wp.float32, device=self.device)
        # -- Fixed tendon properties
        if self._num_fixed_tendons > 0:
            self._fixed_tendon_stiffness = wp.clone(self._root_view.get_fixed_tendon_stiffnesses(), device=self.device)
            self._fixed_tendon_damping = wp.clone(self._root_view.get_fixed_tendon_dampings(), device=self.device)
            self._fixed_tendon_limit_stiffness = wp.clone(
                self._root_view.get_fixed_tendon_limit_stiffnesses(), device=self.device
            )
            self._fixed_tendon_rest_length = wp.clone(
                self._root_view.get_fixed_tendon_rest_lengths(), device=self.device
            )
            self._fixed_tendon_offset = wp.clone(self._root_view.get_fixed_tendon_offsets(), device=self.device)
            self._fixed_tendon_pos_limits = wp.clone(self._root_view.get_fixed_tendon_limits(), device=self.device)
        else:
            self._fixed_tendon_stiffness = None
            self._fixed_tendon_damping = None
            self._fixed_tendon_limit_stiffness = None
            self._fixed_tendon_rest_length = None
            self._fixed_tendon_offset = None
            self._fixed_tendon_pos_limits = None
        # -- Spatial tendon properties
        if self._num_spatial_tendons > 0:
            self._spatial_tendon_stiffness = wp.clone(
                self._root_view.get_spatial_tendon_stiffnesses(), device=self.device
            )
            self._spatial_tendon_damping = wp.clone(self._root_view.get_spatial_tendon_dampings(), device=self.device)
            self._spatial_tendon_limit_stiffness = wp.clone(
                self._root_view.get_spatial_tendon_limit_stiffnesses(), device=self.device
            )
            self._spatial_tendon_offset = wp.clone(self._root_view.get_spatial_tendon_offsets(), device=self.device)
        else:
            self._spatial_tendon_stiffness = None
            self._spatial_tendon_damping = None
            self._spatial_tendon_limit_stiffness = None
            self._spatial_tendon_offset = None
        # -- Body properties
        self._body_mass = wp.clone(self._root_view.get_masses(), device=self.device)
        self._body_inertia = wp.clone(self._root_view.get_inertias(), device=self.device)
        self._default_root_state = None

    """
    Internal helpers.
    """

    def _get_pos_from_transform(self, transform: wp.array) -> wp.array:
        """Generates a position array from a transform array.

        Args:
            transform: The transform array. Shape is (N, 7).

        Returns:
            The position array. Shape is (N, 3).
        """
        return wp.array(
            ptr=transform.ptr,
            shape=transform.shape,
            dtype=wp.vec3f,
            strides=transform.strides,
            device=self.device,
        )

    def _get_quat_from_transform(self, transform: wp.array) -> wp.array:
        """Generates a quaternion array from a transform array.

        Args:
            transform: The transform array. Shape is (N, 7).

        Returns:
            The quaternion array. Shape is (N, 4).
        """
        return wp.array(
            ptr=transform.ptr + 3 * 4,
            shape=transform.shape,
            dtype=wp.quatf,
            strides=transform.strides,
            device=self.device,
        )

    def _get_lin_vel_from_spatial_vector(self, spatial_vector: wp.array) -> wp.array:
        """Generates a linear velocity array from a spatial vector array.

        Args:
            spatial_vector: The spatial vector array. Shape is (N, 6).

        Returns:
            The linear velocity array. Shape is (N, 3).
        """
        return wp.array(
            ptr=spatial_vector.ptr,
            shape=spatial_vector.shape,
            dtype=wp.vec3f,
            strides=spatial_vector.strides,
            device=self.device,
        )

    def _get_ang_vel_from_spatial_vector(self, spatial_vector: wp.array) -> wp.array:
        """Generates an angular velocity array from a spatial vector array.

        Args:
            spatial_vector: The spatial vector array. Shape is (N, 6).

        Returns:
            The angular velocity array. Shape is (N, 3).
        """
        return wp.array(
            ptr=spatial_vector.ptr + 3 * 4,
            shape=spatial_vector.shape,
            dtype=wp.vec3f,
            strides=spatial_vector.strides,
            device=self.device,
        )

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
