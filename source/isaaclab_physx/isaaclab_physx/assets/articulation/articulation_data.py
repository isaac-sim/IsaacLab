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

from isaaclab_physx.assets import kernels as shared_kernels
from isaaclab_physx.assets.articulation import kernels as articulation_kernels
from isaaclab_physx.physics import PhysxManager as SimulationManager

if TYPE_CHECKING:
    import omni.physics.tensors.api as physx

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

    .. note::
        **Pull-to-refresh model.** PhysX state properties are *not* automatically updated each
        simulation step. Each property getter pulls fresh data from the PhysX tensor API on first
        access per timestamp, then caches the result until the next step. This differs from the
        Newton backend, where buffers are refreshed automatically by the simulation.

    .. note::
        **ProxyArray pointer stability.** Each :class:`ProxyArray` wrapper is created once on the
        first property access and reused thereafter. This is safe because the PhysX tensor API
        returns views into stable, pre-allocated GPU buffers whose device pointer does not change
        across simulation steps. The ``wp.array`` Python objects returned by getters like
        ``get_root_transforms()`` are new wrappers each call, but they alias the same underlying
        GPU memory. Sub-view properties (``root_pos_w``, ``root_quat_w``, etc.) similarly wrap
        pointer offsets into these stable buffers and are therefore also safe to cache.
    """

    __backend_name__: str = "physx"
    """The name of the backend for the articulation data."""

    def __init__(self, root_view: physx.ArticulationView, device: str):
        """Initializes the articulation data.

        Args:
            root_view: The root articulation view.
            device: The device used for processing.
        """
        super().__init__(root_view, device)
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_view: physx.ArticulationView = weakref.proxy(root_view)

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
        self.GRAVITY_VEC_W = ProxyArray(wp.from_torch(gravity_dir, dtype=wp.vec3f))
        self.FORWARD_VEC_B = ProxyArray(wp.from_torch(forward_vec, dtype=wp.vec3f))

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
    def default_root_pose(self) -> ProxyArray:
        """Default root pose ``[pos, quat]`` in the local environment frame.

        The position and quaternion are of the articulation root's actor frame.
        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        """
        if self._default_root_pose_ta is None:
            self._default_root_pose_ta = ProxyArray(self._default_root_pose)
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
        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).
        """
        if self._default_root_vel_ta is None:
            self._default_root_vel_ta = ProxyArray(self._default_root_vel)
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
        if self._default_joint_pos_ta is None:
            self._default_joint_pos_ta = ProxyArray(self._default_joint_pos)
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
        if self._default_joint_vel_ta is None:
            self._default_joint_vel_ta = ProxyArray(self._default_joint_vel)
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
        if self._joint_pos_target_ta is None:
            self._joint_pos_target_ta = ProxyArray(self._joint_pos_target)
        return self._joint_pos_target_ta

    @property
    def joint_vel_target(self) -> ProxyArray:
        """Joint velocity targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        if self._joint_vel_target_ta is None:
            self._joint_vel_target_ta = ProxyArray(self._joint_vel_target)
        return self._joint_vel_target_ta

    @property
    def joint_effort_target(self) -> ProxyArray:
        """Joint effort targets commanded by the user.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        if self._joint_effort_target_ta is None:
            self._joint_effort_target_ta = ProxyArray(self._joint_effort_target)
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
        if self._computed_torque_ta is None:
            self._computed_torque_ta = ProxyArray(self._computed_torque)
        return self._computed_torque_ta

    @property
    def applied_torque(self) -> ProxyArray:
        """Joint torques applied from the actuator model (after clipping).

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
        actuator model.
        """
        if self._applied_torque_ta is None:
            self._applied_torque_ta = ProxyArray(self._applied_torque)
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
        if self._joint_stiffness_ta is None:
            self._joint_stiffness_ta = ProxyArray(self._joint_stiffness)
        return self._joint_stiffness_ta

    @property
    def joint_damping(self) -> ProxyArray:
        """Joint damping provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        if self._joint_damping_ta is None:
            self._joint_damping_ta = ProxyArray(self._joint_damping)
        return self._joint_damping_ta

    @property
    def joint_armature(self) -> ProxyArray:
        """Joint armature provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        if self._joint_armature_ta is None:
            self._joint_armature_ta = ProxyArray(self._joint_armature)
        return self._joint_armature_ta

    @property
    def joint_friction_coeff(self) -> ProxyArray:
        """Joint static friction coefficient provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        if self._joint_friction_coeff_ta is None:
            self._joint_friction_coeff_ta = ProxyArray(self._joint_friction_coeff)
        return self._joint_friction_coeff_ta

    @property
    def joint_dynamic_friction_coeff(self) -> ProxyArray:
        """Joint dynamic friction coefficient provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        if self._joint_dynamic_friction_coeff_ta is None:
            self._joint_dynamic_friction_coeff_ta = ProxyArray(self._joint_dynamic_friction_coeff)
        return self._joint_dynamic_friction_coeff_ta

    @property
    def joint_viscous_friction_coeff(self) -> ProxyArray:
        """Joint viscous friction coefficient provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        if self._joint_viscous_friction_coeff_ta is None:
            self._joint_viscous_friction_coeff_ta = ProxyArray(self._joint_viscous_friction_coeff)
        return self._joint_viscous_friction_coeff_ta

    @property
    def joint_pos_limits(self) -> ProxyArray:
        """Joint position limits provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        if self._joint_pos_limits_ta is None:
            self._joint_pos_limits_ta = ProxyArray(self._joint_pos_limits)
        return self._joint_pos_limits_ta

    @property
    def joint_vel_limits(self) -> ProxyArray:
        """Joint maximum velocity provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        if self._joint_vel_limits_ta is None:
            self._joint_vel_limits_ta = ProxyArray(self._joint_vel_limits)
        return self._joint_vel_limits_ta

    @property
    def joint_effort_limits(self) -> ProxyArray:
        """Joint maximum effort provided to the simulation.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        if self._joint_effort_limits_ta is None:
            self._joint_effort_limits_ta = ProxyArray(self._joint_effort_limits)
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
        if self._soft_joint_pos_limits_ta is None:
            self._soft_joint_pos_limits_ta = ProxyArray(self._soft_joint_pos_limits)
        return self._soft_joint_pos_limits_ta

    @property
    def soft_joint_vel_limits(self) -> ProxyArray:
        """Soft joint velocity limits for all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).

        These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
        has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
        """
        if self._soft_joint_vel_limits_ta is None:
            self._soft_joint_vel_limits_ta = ProxyArray(self._soft_joint_vel_limits)
        return self._soft_joint_vel_limits_ta

    @property
    def gear_ratio(self) -> ProxyArray:
        """Gear ratio for relating motor torques to applied Joint torques.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        if self._gear_ratio_ta is None:
            self._gear_ratio_ta = ProxyArray(self._gear_ratio)
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
        if self._fixed_tendon_stiffness_ta is None:
            self._fixed_tendon_stiffness_ta = ProxyArray(self._fixed_tendon_stiffness)
        return self._fixed_tendon_stiffness_ta

    @property
    def fixed_tendon_damping(self) -> ProxyArray:
        """Fixed tendon damping provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        if self._fixed_tendon_damping_ta is None:
            self._fixed_tendon_damping_ta = ProxyArray(self._fixed_tendon_damping)
        return self._fixed_tendon_damping_ta

    @property
    def fixed_tendon_limit_stiffness(self) -> ProxyArray:
        """Fixed tendon limit stiffness provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        if self._fixed_tendon_limit_stiffness_ta is None:
            self._fixed_tendon_limit_stiffness_ta = ProxyArray(self._fixed_tendon_limit_stiffness)
        return self._fixed_tendon_limit_stiffness_ta

    @property
    def fixed_tendon_rest_length(self) -> ProxyArray:
        """Fixed tendon rest length provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        if self._fixed_tendon_rest_length_ta is None:
            self._fixed_tendon_rest_length_ta = ProxyArray(self._fixed_tendon_rest_length)
        return self._fixed_tendon_rest_length_ta

    @property
    def fixed_tendon_offset(self) -> ProxyArray:
        """Fixed tendon offset provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        if self._fixed_tendon_offset_ta is None:
            self._fixed_tendon_offset_ta = ProxyArray(self._fixed_tendon_offset)
        return self._fixed_tendon_offset_ta

    @property
    def fixed_tendon_pos_limits(self) -> ProxyArray:
        """Fixed tendon position limits provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.vec2f. In torch this resolves to
        (num_instances, num_fixed_tendons, 2).
        """
        if self._fixed_tendon_pos_limits_ta is None:
            self._fixed_tendon_pos_limits_ta = ProxyArray(self._fixed_tendon_pos_limits)
        return self._fixed_tendon_pos_limits_ta

    """
    Spatial tendon properties.
    """

    @property
    def spatial_tendon_stiffness(self) -> ProxyArray:
        """Spatial tendon stiffness provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        if self._spatial_tendon_stiffness_ta is None:
            self._spatial_tendon_stiffness_ta = ProxyArray(self._spatial_tendon_stiffness)
        return self._spatial_tendon_stiffness_ta

    @property
    def spatial_tendon_damping(self) -> ProxyArray:
        """Spatial tendon damping provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        if self._spatial_tendon_damping_ta is None:
            self._spatial_tendon_damping_ta = ProxyArray(self._spatial_tendon_damping)
        return self._spatial_tendon_damping_ta

    @property
    def spatial_tendon_limit_stiffness(self) -> ProxyArray:
        """Spatial tendon limit stiffness provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        if self._spatial_tendon_limit_stiffness_ta is None:
            self._spatial_tendon_limit_stiffness_ta = ProxyArray(self._spatial_tendon_limit_stiffness)
        return self._spatial_tendon_limit_stiffness_ta

    @property
    def spatial_tendon_offset(self) -> ProxyArray:
        """Spatial tendon offset provided to the simulation.

        Shape is (num_instances, num_spatial_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_spatial_tendons).
        """
        if self._spatial_tendon_offset_ta is None:
            self._spatial_tendon_offset_ta = ProxyArray(self._spatial_tendon_offset)
        return self._spatial_tendon_offset_ta

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
        if self._root_link_pose_w.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            self._root_link_pose_w.data = self._root_view.get_root_transforms().view(wp.transformf)
            self._root_link_pose_w.timestamp = self._sim_timestamp

        if self._root_link_pose_w_ta is None:
            self._root_link_pose_w_ta = ProxyArray(self._root_link_pose_w.data)
        return self._root_link_pose_w_ta

    @property
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

        if self._root_link_vel_w_ta is None:
            self._root_link_vel_w_ta = ProxyArray(self._root_link_vel_w.data)
        return self._root_link_vel_w_ta

    @property
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
                    self.root_link_pose_w,
                    self.body_com_pose_b,
                ],
                outputs=[
                    self._root_com_pose_w.data,
                ],
                device=self.device,
            )
            self._root_com_pose_w.timestamp = self._sim_timestamp

        if self._root_com_pose_w_ta is None:
            self._root_com_pose_w_ta = ProxyArray(self._root_com_pose_w.data)
        return self._root_com_pose_w_ta

    @property
    def root_com_vel_w(self) -> ProxyArray:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        if self._root_com_vel_w.timestamp < self._sim_timestamp:
            self._root_com_vel_w.data = self._root_view.get_root_velocities().view(wp.spatial_vectorf)
            self._root_com_vel_w.timestamp = self._sim_timestamp

        if self._root_com_vel_w_ta is None:
            self._root_com_vel_w_ta = ProxyArray(self._root_com_vel_w.data)
        return self._root_com_vel_w_ta

    """
    Body state properties.
    """

    @property
    def body_mass(self) -> ProxyArray:
        """Body mass in the world frame.

        Shape is (num_instances, num_bodies), dtype = wp.float32. In torch this resolves to (num_instances, num_bodies).
        """
        self._body_mass.assign(self._root_view.get_masses())
        if self._body_mass_ta is None:
            self._body_mass_ta = ProxyArray(self._body_mass)
        return self._body_mass_ta

    @property
    def body_inertia(self) -> ProxyArray:
        """Flattened body inertia in the world frame.

        Shape is (num_instances, num_bodies, 9), dtype = wp.float32. In torch this resolves to
        (num_instances, num_bodies, 9).
        """
        self._body_inertia.assign(self._root_view.get_inertias())
        if self._body_inertia_ta is None:
            self._body_inertia_ta = ProxyArray(self._body_inertia)
        return self._body_inertia_ta

    @property
    def body_link_pose_w(self) -> ProxyArray:
        """Body link pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_link_pose_w.timestamp < self._sim_timestamp:
            # perform forward kinematics (shouldn't cause overhead if it happened already)
            self._physics_sim_view.update_articulations_kinematic()
            # set the buffer data and timestamp
            self._body_link_pose_w.data = self._root_view.get_link_transforms().view(wp.transformf)
            self._body_link_pose_w.timestamp = self._sim_timestamp

        if self._body_link_pose_w_ta is None:
            self._body_link_pose_w_ta = ProxyArray(self._body_link_pose_w.data)
        return self._body_link_pose_w_ta

    @property
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

        if self._body_link_vel_w_ta is None:
            self._body_link_vel_w_ta = ProxyArray(self._body_link_vel_w.data)
        return self._body_link_vel_w_ta

    @property
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
                    self.body_link_pose_w,
                    self.body_com_pose_b,
                ],
                outputs=[
                    self._body_com_pose_w.data,
                ],
                device=self.device,
            )
            self._body_com_pose_w.timestamp = self._sim_timestamp

        if self._body_com_pose_w_ta is None:
            self._body_com_pose_w_ta = ProxyArray(self._body_com_pose_w.data)
        return self._body_com_pose_w_ta

    @property
    def body_com_vel_w(self) -> ProxyArray:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        if self._body_com_vel_w.timestamp < self._sim_timestamp:
            self._body_com_vel_w.data = self._root_view.get_link_velocities().view(wp.spatial_vectorf)
            self._body_com_vel_w.timestamp = self._sim_timestamp

        if self._body_com_vel_w_ta is None:
            self._body_com_vel_w_ta = ProxyArray(self._body_com_vel_w.data)
        return self._body_com_vel_w_ta

    @property
    def body_com_acc_w(self) -> ProxyArray:
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]``.
        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._body_com_acc_w.data = self._root_view.get_link_accelerations().view(wp.spatial_vectorf)
            self._body_com_acc_w.timestamp = self._sim_timestamp

        if self._body_com_acc_w_ta is None:
            self._body_com_acc_w_ta = ProxyArray(self._body_com_acc_w.data)
        return self._body_com_acc_w_ta

    @property
    def body_com_pose_b(self) -> ProxyArray:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.
        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            self._body_com_pose_b.data.assign(self._root_view.get_coms().view(wp.transformf))
            self._body_com_pose_b.timestamp = self._sim_timestamp

        if self._body_com_pose_b_ta is None:
            self._body_com_pose_b_ta = ProxyArray(self._body_com_pose_b.data)
        return self._body_com_pose_b_ta

    """
    Joint state properties.
    """

    @property
    def joint_pos(self) -> ProxyArray:
        """Joint positions of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        if self._joint_pos.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_pos.data = self._root_view.get_dof_positions()
            self._joint_pos.timestamp = self._sim_timestamp
        if self._joint_pos_ta is None:
            self._joint_pos_ta = ProxyArray(self._joint_pos.data)
        return self._joint_pos_ta

    @property
    def joint_vel(self) -> ProxyArray:
        """Joint velocities of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        if self._joint_vel.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_vel.data = self._root_view.get_dof_velocities()
            self._joint_vel.timestamp = self._sim_timestamp
        if self._joint_vel_ta is None:
            self._joint_vel_ta = ProxyArray(self._joint_vel.data)
        return self._joint_vel_ta

    @property
    def joint_acc(self) -> ProxyArray:
        """Joint acceleration of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
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
        if self._joint_acc_ta is None:
            self._joint_acc_ta = ProxyArray(self._joint_acc.data)
        return self._joint_acc_ta

    """
    Derived Properties.
    """

    @property
    def projected_gravity_b(self) -> ProxyArray:
        """Projection of the gravity direction on base frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3)."""
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.GRAVITY_VEC_W, self.root_link_quat_w],
                outputs=[self._projected_gravity_b.data],
                device=self.device,
            )
            self._projected_gravity_b.timestamp = self._sim_timestamp
        if self._projected_gravity_b_ta is None:
            self._projected_gravity_b_ta = ProxyArray(self._projected_gravity_b.data)
        return self._projected_gravity_b_ta

    @property
    def heading_w(self) -> ProxyArray:
        """Yaw heading of the base frame (in radians). Shape is (num_instances,), dtype = wp.float32.

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
        if self._heading_w_ta is None:
            self._heading_w_ta = ProxyArray(self._heading_w.data)
        return self._heading_w_ta

    @property
    def root_link_lin_vel_b(self) -> ProxyArray:
        """Root link linear velocity in base frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to its actor frame.
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
        if self._root_link_lin_vel_b_ta is None:
            self._root_link_lin_vel_b_ta = ProxyArray(self._root_link_lin_vel_b.data)
        return self._root_link_lin_vel_b_ta

    @property
    def root_link_ang_vel_b(self) -> ProxyArray:
        """Root link angular velocity in base frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to its actor frame.
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
        if self._root_link_ang_vel_b_ta is None:
            self._root_link_ang_vel_b_ta = ProxyArray(self._root_link_ang_vel_b.data)
        return self._root_link_ang_vel_b_ta

    @property
    def root_com_lin_vel_b(self) -> ProxyArray:
        """Root center of mass linear velocity in base frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame
        with respect to its actor frame.
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
        if self._root_com_lin_vel_b_ta is None:
            self._root_com_lin_vel_b_ta = ProxyArray(self._root_com_lin_vel_b.data)
        return self._root_com_lin_vel_b_ta

    @property
    def root_com_ang_vel_b(self) -> ProxyArray:
        """Root center of mass angular velocity in base frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame
        with respect to its actor frame.
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
        if self._root_com_ang_vel_b_ta is None:
            self._root_com_ang_vel_b_ta = ProxyArray(self._root_com_ang_vel_b.data)
        return self._root_com_ang_vel_b_ta

    """
    Sliced properties.
    """

    @property
    def root_link_pos_w(self) -> ProxyArray:
        """Root link position in simulation world frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.root_link_pose_w
        if self._root_link_pos_w_ta is None:
            self._root_link_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._root_link_pos_w_ta

    @property
    def root_link_quat_w(self) -> ProxyArray:
        """Root link orientation (x, y, z, w) in simulation world frame.
        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.root_link_pose_w
        if self._root_link_quat_w_ta is None:
            self._root_link_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._root_link_quat_w_ta

    @property
    def root_link_lin_vel_w(self) -> ProxyArray:
        """Root linear velocity in simulation world frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.root_link_vel_w
        if self._root_link_lin_vel_w_ta is None:
            self._root_link_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._root_link_lin_vel_w_ta

    @property
    def root_link_ang_vel_w(self) -> ProxyArray:
        """Root link angular velocity in simulation world frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.root_link_vel_w
        if self._root_link_ang_vel_w_ta is None:
            self._root_link_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._root_link_ang_vel_w_ta

    @property
    def root_com_pos_w(self) -> ProxyArray:
        """Root center of mass position in simulation world frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the position of the center of mass frame of the root rigid body relative to the world.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.root_com_pose_w
        if self._root_com_pos_w_ta is None:
            self._root_com_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._root_com_pos_w_ta

    @property
    def root_com_quat_w(self) -> ProxyArray:
        """Root center of mass orientation (x, y, z, w) in simulation world frame.
        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).

        This quantity is the orientation of the principal axes of inertia of the root rigid body relative to the world.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.root_com_pose_w
        if self._root_com_quat_w_ta is None:
            self._root_com_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._root_com_quat_w_ta

    @property
    def root_com_lin_vel_w(self) -> ProxyArray:
        """Root center of mass linear velocity in simulation world frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.root_com_vel_w
        if self._root_com_lin_vel_w_ta is None:
            self._root_com_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._root_com_lin_vel_w_ta

    @property
    def root_com_ang_vel_w(self) -> ProxyArray:
        """Root center of mass angular velocity in simulation world frame.
        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.root_com_vel_w
        if self._root_com_ang_vel_w_ta is None:
            self._root_com_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._root_com_ang_vel_w_ta

    @property
    def body_link_pos_w(self) -> ProxyArray:
        """Positions of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.body_link_pose_w
        if self._body_link_pos_w_ta is None:
            self._body_link_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_link_pos_w_ta

    @property
    def body_link_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.body_link_pose_w
        if self._body_link_quat_w_ta is None:
            self._body_link_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_link_quat_w_ta

    @property
    def body_link_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' actor frame relative to the world.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.body_link_vel_w
        if self._body_link_lin_vel_w_ta is None:
            self._body_link_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_link_lin_vel_w_ta

    @property
    def body_link_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' actor frame relative to the world.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.body_link_vel_w
        if self._body_link_ang_vel_w_ta is None:
            self._body_link_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_link_ang_vel_w_ta

    @property
    def body_com_pos_w(self) -> ProxyArray:
        """Positions of all bodies' center of mass in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' center of mass frame.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.body_com_pose_w
        if self._body_com_pos_w_ta is None:
            self._body_com_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_w_ta

    @property
    def body_com_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' principal axes of inertia.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.body_com_pose_w
        if self._body_com_quat_w_ta is None:
            self._body_com_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_w_ta

    @property
    def body_com_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.body_com_vel_w
        if self._body_com_lin_vel_w_ta is None:
            self._body_com_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_vel_w_ta

    @property
    def body_com_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.body_com_vel_w
        if self._body_com_ang_vel_w_ta is None:
            self._body_com_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_vel_w_ta

    @property
    def body_com_lin_acc_w(self) -> ProxyArray:
        """Linear acceleration of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.body_com_acc_w
        if self._body_com_lin_acc_w_ta is None:
            self._body_com_lin_acc_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_acc_w_ta

    @property
    def body_com_ang_acc_w(self) -> ProxyArray:
        """Angular acceleration of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.body_com_acc_w
        if self._body_com_ang_acc_w_ta is None:
            self._body_com_ang_acc_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_acc_w_ta

    @property
    def body_com_pos_b(self) -> ProxyArray:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body's link frame.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.body_com_pose_b
        if self._body_com_pos_b_ta is None:
            self._body_com_pos_b_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_b_ta

    @property
    def body_com_quat_b(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all of the bodies in their
        respective link frames.
        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the principal axes of inertia relative to its body's link frame.
        """
        # Access parent property to trigger its getter call (PhysX is pull-on-demand)
        parent = self.body_com_pose_b
        if self._body_com_quat_b_ta is None:
            self._body_com_quat_b_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_b_ta

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

        # Initialize ProxyArray wrappers
        self._pin_proxy_arrays()

    def _pin_proxy_arrays(self) -> None:
        """Create pinned ProxyArray wrappers for all data buffers.

        This is called once from :meth:`_create_buffers` during initialization.
        PhysX tensor API buffers have stable GPU pointers across simulation steps,
        so no rebinding is needed (unlike Newton).
        """
        # -- Pinned ProxyArray cache (one per read property, lazily created on first access)
        # Defaults
        self._default_root_pose_ta: ProxyArray | None = None
        self._default_root_vel_ta: ProxyArray | None = None
        self._default_joint_pos_ta: ProxyArray | None = None
        self._default_joint_vel_ta: ProxyArray | None = None
        # Joint commands (set into simulation)
        self._joint_pos_target_ta: ProxyArray | None = None
        self._joint_vel_target_ta: ProxyArray | None = None
        self._joint_effort_target_ta: ProxyArray | None = None
        # Joint commands (explicit actuator model)
        self._computed_torque_ta: ProxyArray | None = None
        self._applied_torque_ta: ProxyArray | None = None
        # Joint properties
        self._joint_stiffness_ta: ProxyArray | None = None
        self._joint_damping_ta: ProxyArray | None = None
        self._joint_armature_ta: ProxyArray | None = None
        self._joint_friction_coeff_ta: ProxyArray | None = None
        self._joint_dynamic_friction_coeff_ta: ProxyArray | None = None
        self._joint_viscous_friction_coeff_ta: ProxyArray | None = None
        self._joint_pos_limits_ta: ProxyArray | None = None
        self._joint_vel_limits_ta: ProxyArray | None = None
        self._joint_effort_limits_ta: ProxyArray | None = None
        # Joint properties (custom)
        self._soft_joint_pos_limits_ta: ProxyArray | None = None
        self._soft_joint_vel_limits_ta: ProxyArray | None = None
        self._gear_ratio_ta: ProxyArray | None = None
        # Fixed tendon properties
        self._fixed_tendon_stiffness_ta: ProxyArray | None = None
        self._fixed_tendon_damping_ta: ProxyArray | None = None
        self._fixed_tendon_limit_stiffness_ta: ProxyArray | None = None
        self._fixed_tendon_rest_length_ta: ProxyArray | None = None
        self._fixed_tendon_offset_ta: ProxyArray | None = None
        self._fixed_tendon_pos_limits_ta: ProxyArray | None = None
        # Spatial tendon properties
        self._spatial_tendon_stiffness_ta: ProxyArray | None = None
        self._spatial_tendon_damping_ta: ProxyArray | None = None
        self._spatial_tendon_limit_stiffness_ta: ProxyArray | None = None
        self._spatial_tendon_offset_ta: ProxyArray | None = None
        # Root state (timestamped)
        self._root_link_pose_w_ta: ProxyArray | None = None
        self._root_link_vel_w_ta: ProxyArray | None = None
        self._root_com_pose_w_ta: ProxyArray | None = None
        self._root_com_vel_w_ta: ProxyArray | None = None
        # Body state (timestamped)
        self._body_link_pose_w_ta: ProxyArray | None = None
        self._body_link_vel_w_ta: ProxyArray | None = None
        self._body_com_pose_w_ta: ProxyArray | None = None
        self._body_com_vel_w_ta: ProxyArray | None = None
        self._body_com_acc_w_ta: ProxyArray | None = None
        self._body_com_pose_b_ta: ProxyArray | None = None
        # Body properties
        self._body_mass_ta: ProxyArray | None = None
        self._body_inertia_ta: ProxyArray | None = None
        # Joint state (timestamped)
        self._joint_pos_ta: ProxyArray | None = None
        self._joint_vel_ta: ProxyArray | None = None
        self._joint_acc_ta: ProxyArray | None = None
        # Derived properties (timestamped)
        self._projected_gravity_b_ta: ProxyArray | None = None
        self._heading_w_ta: ProxyArray | None = None
        self._root_link_lin_vel_b_ta: ProxyArray | None = None
        self._root_link_ang_vel_b_ta: ProxyArray | None = None
        self._root_com_lin_vel_b_ta: ProxyArray | None = None
        self._root_com_ang_vel_b_ta: ProxyArray | None = None
        # Sliced properties (root link)
        self._root_link_pos_w_ta: ProxyArray | None = None
        self._root_link_quat_w_ta: ProxyArray | None = None
        self._root_link_lin_vel_w_ta: ProxyArray | None = None
        self._root_link_ang_vel_w_ta: ProxyArray | None = None
        # Sliced properties (root com)
        self._root_com_pos_w_ta: ProxyArray | None = None
        self._root_com_quat_w_ta: ProxyArray | None = None
        self._root_com_lin_vel_w_ta: ProxyArray | None = None
        self._root_com_ang_vel_w_ta: ProxyArray | None = None
        # Sliced properties (body link)
        self._body_link_pos_w_ta: ProxyArray | None = None
        self._body_link_quat_w_ta: ProxyArray | None = None
        self._body_link_lin_vel_w_ta: ProxyArray | None = None
        self._body_link_ang_vel_w_ta: ProxyArray | None = None
        # Sliced properties (body com)
        self._body_com_pos_w_ta: ProxyArray | None = None
        self._body_com_quat_w_ta: ProxyArray | None = None
        self._body_com_lin_vel_w_ta: ProxyArray | None = None
        self._body_com_ang_vel_w_ta: ProxyArray | None = None
        self._body_com_lin_acc_w_ta: ProxyArray | None = None
        self._body_com_ang_acc_w_ta: ProxyArray | None = None
        # Sliced properties (body com in body frame)
        self._body_com_pos_b_ta: ProxyArray | None = None
        self._body_com_quat_b_ta: ProxyArray | None = None
        # Deprecated state-concat properties
        self._default_root_state_ta: ProxyArray | None = None
        self._root_state_w_ta: ProxyArray | None = None
        self._root_link_state_w_ta: ProxyArray | None = None
        self._root_com_state_w_ta: ProxyArray | None = None
        self._body_state_w_ta: ProxyArray | None = None
        self._body_link_state_w_ta: ProxyArray | None = None
        self._body_com_state_w_ta: ProxyArray | None = None

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
        if self._default_root_state_ta is None:
            self._default_root_state_ta = ProxyArray(self._default_root_state)
        return self._default_root_state_ta

    @property
    def root_state_w(self) -> ProxyArray:
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

        if self._root_state_w_ta is None:
            self._root_state_w_ta = ProxyArray(self._root_state_w.data)
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

        if self._root_link_state_w_ta is None:
            self._root_link_state_w_ta = ProxyArray(self._root_link_state_w.data)
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

        if self._root_com_state_w_ta is None:
            self._root_com_state_w_ta = ProxyArray(self._root_com_state_w.data)
        return self._root_com_state_w_ta

    @property
    def body_state_w(self) -> ProxyArray:
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position and quaternion are of all the articulation links' actor frame. Meanwhile, the linear and angular
        velocities are of the articulation links' center of mass frame.
        """
        warnings.warn(
            "The `body_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_link_pose_w` and "
            "`body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
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

        if self._body_state_w_ta is None:
            self._body_state_w_ta = ProxyArray(self._body_state_w.data)
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

        if self._body_link_state_w_ta is None:
            self._body_link_state_w_ta = ProxyArray(self._body_link_state_w.data)
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

        if self._body_com_state_w_ta is None:
            self._body_com_state_w_ta = ProxyArray(self._body_com_state_w.data)
        return self._body_com_state_w_ta
