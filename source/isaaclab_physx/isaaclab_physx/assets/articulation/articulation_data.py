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

from isaaclab.assets.articulation import ordering_kernels
from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.buffers import reset_timestamps
from isaaclab.utils.math import normalize
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache

from isaaclab_physx.assets import kernels as shared_kernels
from isaaclab_physx.assets.articulation import kernels as articulation_kernels
from isaaclab_physx.physics import PhysxManager as SimulationManager

if TYPE_CHECKING:
    from collections.abc import Callable

    import omni.physics.tensors as physx

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
        access per timestamp, then caches the result until the next step. This differs from Newton,
        where solver-owned buffers and nonidentity public-order shadows are refreshed automatically
        by the simulation.

    .. note::
        **ProxyArray pointer stability.** Each :class:`ProxyArray` wrapper is created once on the
        first property access and reused thereafter. Without ordering or joint-direction correction,
        direct properties alias stable, pre-allocated PhysX GPU buffers whose device pointer does
        not change across simulation steps. The ``wp.array`` Python objects returned by getters such
        as ``get_root_transforms()`` are new wrappers each call, but they alias the same underlying
        GPU memory. With nonidentity ordering, ordering-sensitive properties instead use stable,
        owned public-order shadows populated lazily from backend data. Sub-view properties
        (``root_pos_w``, ``root_quat_w``, etc.) wrap pointer offsets into their stable parent
        buffers and are therefore also safe to cache.
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
        self._fk_timestamp = 0.0
        self._read_launch_cache = _WarpLaunchCache(device)
        self._joint_dof_signs = wp.ones(root_view.max_dofs, dtype=wp.int32, device=device)
        self._has_reversed_joints = False

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
        # FK is current after a sim step. Keep fk_timestamp in sync unless it was explicitly invalidated.
        if self._fk_timestamp >= 0.0:
            self._fk_timestamp = self._sim_timestamp
        # Trigger an update of the joint acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.joint_acc

    def _ensure_fk_fresh(self) -> None:
        """Run forward kinematics if the joint / body state has changed since the last FK update.

        Cheap to call repeatedly: the ``_fk_timestamp`` guard skips the recomputation when the
        kinematic state is already up to date.
        """
        if self._fk_timestamp < self._sim_timestamp:
            self._physics_sim_view.update_articulations_kinematic()
            self._fk_timestamp = self._sim_timestamp

    def _reset_pose(self, from_link: bool = True) -> None:
        """Reset pose-dependent cached articulation properties.

        Args:
            from_link: Set ``True`` when the root link pose was written so the derived root
                center-of-mass pose (:attr:`root_com_pose_w`) is also invalidated; set ``False`` when
                the center-of-mass pose was written directly so it is not clobbered. Defaults to True.
        """
        # Invalidate the derived root com pose only when the root link pose was the quantity just written.
        reset_timestamps(
            [
                self._root_com_pose_w if from_link else None,
                self._body_link_pose_w,
                self._body_com_pose_w,
                self._root_link_vel_w,
                self._body_link_vel_w,
                self._body_com_vel_w,
                self._projected_gravity_b,
                self._heading_w,
                self._root_link_lin_vel_b,
                self._root_link_ang_vel_b,
                self._root_com_lin_vel_b,
                self._root_com_ang_vel_b,
                self._root_state_w,
                self._root_link_state_w,
                self._root_com_state_w,
                self._body_state_w,
                self._body_link_state_w,
                self._body_com_state_w,
                self._body_com_jacobian_w,
                self._gravity_compensation_forces,
                self._mass_matrix,
            ]
        )
        self._fk_timestamp = -1.0

    def _reset_velocity(self, from_com: bool = True) -> None:
        """Reset velocity-dependent cached articulation properties.

        Args:
            from_com: Set ``True`` when the root center-of-mass velocity was written so the derived root
                link velocity (:attr:`root_link_vel_w`) is also invalidated; set ``False`` when the link
                velocity was written directly so it is not clobbered. Defaults to True.
        """
        # Invalidate the derived root link velocity only when the root com velocity was the quantity just written.
        reset_timestamps(
            [
                self._root_link_vel_w if from_com else None,
                self._body_com_vel_w,
                self._body_link_vel_w,
                self._root_link_lin_vel_b,
                self._root_link_ang_vel_b,
                self._root_com_lin_vel_b,
                self._root_com_ang_vel_b,
                self._root_state_w,
                self._root_link_state_w,
                self._root_com_state_w,
                self._body_state_w,
                self._body_link_state_w,
                self._body_com_state_w,
            ]
        )
        self._fk_timestamp = -1.0

    def _reset_body_com_pose_b_dependents(self) -> None:
        """Reset cached properties derived from body-frame center-of-mass offsets."""
        reset_timestamps(
            [
                self._root_com_pose_w,
                self._root_com_vel_w,
                self._root_link_vel_w,
                self._body_com_pose_w,
                self._body_com_vel_w,
                self._body_link_vel_w,
                self._root_link_lin_vel_b,
                self._root_link_ang_vel_b,
                self._root_com_lin_vel_b,
                self._root_com_ang_vel_b,
                self._root_state_w,
                self._root_link_state_w,
                self._root_com_state_w,
                self._body_state_w,
                self._body_link_state_w,
                self._body_com_state_w,
                self._body_com_jacobian_w,
                self._mass_matrix,
                self._gravity_compensation_forces,
            ]
        )

    def _reset_dynamics(
        self, *, body_com_jacobian: bool = False, mass_matrix: bool = False, gravity_compensation: bool = False
    ) -> None:
        """Reset selected computed-dynamics caches after same-timestamp model writes."""
        reset_timestamps(
            [
                self._body_com_jacobian_w if body_com_jacobian else None,
                self._mass_matrix if mass_matrix else None,
                self._gravity_compensation_forces if gravity_compensation else None,
            ]
        )

    """
    Names.
    """

    body_names: list[str] = None
    """Body names in public order (configured ordering when set, otherwise backend order)."""

    joint_names: list[str] = None
    """Joint names in public order (configured ordering when set, otherwise backend order)."""

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
        """PhysX joint static friction value provided to the simulation.

        For Isaac Sim 5.0 and later, this is the static friction effort [N or N·m, depending on joint type].
        For earlier Isaac Sim versions, this is the legacy unitless joint friction coefficient.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        if self._joint_friction_coeff_ta is None:
            self._joint_friction_coeff_ta = ProxyArray(self._joint_friction_coeff)
        return self._joint_friction_coeff_ta

    @property
    def joint_dynamic_friction_coeff(self) -> ProxyArray:
        """PhysX joint dynamic friction effort provided to the simulation.

        The effort is [N or N·m, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        if self._joint_dynamic_friction_coeff_ta is None:
            self._joint_dynamic_friction_coeff_ta = ProxyArray(self._joint_dynamic_friction_coeff)
        return self._joint_dynamic_friction_coeff_ta

    @property
    def joint_viscous_friction_coeff(self) -> ProxyArray:
        """Joint viscous friction coefficient provided to the simulation.

        The coefficient is [N·s/m or N·m·s/rad, depending on joint type].

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
            body_com_pose_b = self._backend_body_com_pose_b
            self._read_launch_cache.launch(
                "root_link_vel_w",
                shared_kernels.get_root_link_vel_from_root_com_vel,
                dim=self._num_instances,
                inputs=[
                    self.root_com_vel_w,
                    self.root_link_pose_w,
                    body_com_pose_b,
                ],
                outputs=[
                    self._root_link_vel_w.data,
                ],
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
            body_com_pose_b = self._backend_body_com_pose_b
            self._read_launch_cache.launch(
                "root_com_pose_w",
                shared_kernels.get_root_com_pose_from_root_link_pose,
                dim=self._num_instances,
                inputs=[
                    self.root_link_pose_w,
                    body_com_pose_b,
                ],
                outputs=[
                    self._root_com_pose_w.data,
                ],
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

    def _fetch_body_com_pose_b_backend(self, buf: TimestampedBuffer) -> None:
        """Assign the current backend-order body COM pose from the tensor view when stale.

        Backend fetch for the shared :meth:`_ensure_body_com_pose_b_current` /
        :attr:`_backend_body_com_pose_b`. ``get_coms()`` is CPU-only, so this stages
        host-to-device on GPU pipelines; the guard keeps it to at most one fetch per
        invalidation.
        """
        if buf.timestamp < 0.0:
            buf.data.assign(self._root_view.get_coms().view(wp.transformf))
            buf.timestamp = 0.0

    """
    Body state properties.
    """

    def _refresh_reordered_body_buffer(
        self,
        buf: TimestampedBuffer,
        backend_buffer: wp.array | None,
        view_getter: Callable[[], wp.array],
        *,
        component_count: int | None = None,
    ) -> None:
        """Refresh a timestamp-lazy static body buffer from a CPU-only tensor view.

        When stale for the current step, copies the view into the public buffer under
        identity ordering or into backend-order staging before gathering into public
        order. The CPU-only view is copied host-to-device on GPU pipelines.

        Args:
            buf: Owned public-order buffer to refresh in place.
            backend_buffer: Backend-order staging array used under body ordering.
            view_getter: Zero-argument callable returning the backend-order tensor view.
            component_count: Trailing components per body for a three-dimensional buffer,
                or ``None`` for a two-dimensional buffer.
        """
        if buf.timestamp >= self._sim_timestamp:
            return
        if not self.has_body_ordering:
            buf.data.assign(view_getter())
        else:
            # Stage the backend-order view on-device, then gather it into public order.
            backend_buffer.assign(view_getter())
            if component_count is None:
                self._read_launch_cache.launch(
                    (id(buf), "body_2d"),
                    ordering_kernels.reorder_2d_backend_to_user,
                    dim=(self._num_instances, self._num_bodies),
                    inputs=[backend_buffer, self.body_ordering.user_to_backend],
                    outputs=[buf.data],
                )
            else:
                self._read_launch_cache.launch(
                    (id(buf), "body_3d"),
                    ordering_kernels.reorder_3d_backend_to_user,
                    dim=(self._num_instances, self._num_bodies, component_count),
                    inputs=[backend_buffer, self.body_ordering.user_to_backend],
                    outputs=[buf.data],
                )
        buf.timestamp = self._sim_timestamp

    def _refresh_body_state_user(
        self,
        buf: TimestampedBuffer,
        view_getter: Callable[[], wp.array],
    ) -> None:
        """Refresh a public-order body-state buffer from its backend view.

        Args:
            buf: Public-order body-state buffer to refresh in place.
            view_getter: Zero-argument callable returning the typed backend-order view.
        """
        if buf.timestamp >= self._sim_timestamp:
            return
        backend_source = view_getter()
        if self.has_body_ordering:
            self._read_launch_cache.launch(
                (id(buf), "body_state_2d"),
                ordering_kernels.reorder_2d_backend_to_user,
                dim=(self._num_instances, self._num_bodies),
                inputs=[backend_source, self.body_ordering.user_to_backend],
                outputs=[buf.data],
            )
        else:
            buf.data = backend_source
        buf.timestamp = self._sim_timestamp

    @property
    def body_mass(self) -> ProxyArray:
        """Body mass [kg].

        Shape is (num_instances, num_bodies), dtype = wp.float32. In torch this resolves to (num_instances, num_bodies).

        The buffer refreshes from the tensor view at most once per simulation timestamp. Direct tensor-view writes
        after that refresh become visible after the next update; use
        :meth:`isaaclab.assets.Articulation.set_masses_index` for immediate coherence.
        """
        self._refresh_reordered_body_buffer(self._body_mass, self._body_mass_backend, self._root_view.get_masses)
        if self._body_mass_ta is None:
            self._body_mass_ta = ProxyArray(self._body_mass.data)
        return self._body_mass_ta

    @property
    def body_inertia(self) -> ProxyArray:
        """Flattened body inertia [kg*m^2].

        Shape is (num_instances, num_bodies, 9), dtype = wp.float32. In torch this resolves to
        (num_instances, num_bodies, 9).

        The buffer refreshes from the tensor view at most once per simulation timestamp. Direct tensor-view writes
        after that refresh become visible after the next update; use
        :meth:`isaaclab.assets.Articulation.set_inertias_index` for immediate coherence.
        """
        self._refresh_reordered_body_buffer(
            self._body_inertia, self._body_inertia_backend, self._root_view.get_inertias, component_count=9
        )
        if self._body_inertia_ta is None:
            self._body_inertia_ta = ProxyArray(self._body_inertia.data)
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
            self._ensure_fk_fresh()
        self._refresh_body_state_user(
            self._body_link_pose_w, lambda: self._root_view.get_link_transforms().view(wp.transformf)
        )

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
            self._read_launch_cache.launch(
                "body_link_vel_w",
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
            self._read_launch_cache.launch(
                "body_com_pose_w",
                shared_kernels.get_body_com_pose_from_body_link_pose,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_link_pose_w,
                    self.body_com_pose_b,
                ],
                outputs=[
                    self._body_com_pose_w.data,
                ],
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
            self._ensure_fk_fresh()
        self._refresh_body_state_user(
            self._body_com_vel_w, lambda: self._root_view.get_link_velocities().view(wp.spatial_vectorf)
        )

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
        self._refresh_body_state_user(
            self._body_com_acc_w, lambda: self._root_view.get_link_accelerations().view(wp.spatial_vectorf)
        )

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
        self._ensure_body_com_pose_b_current()

        if self._body_com_pose_b_ta is None:
            self._body_com_pose_b_ta = ProxyArray(self._body_com_pose_b.data)
        return self._body_com_pose_b_ta

    """
    Dynamics quantities (task-space controllers).
    """

    @property
    def body_com_jacobian_w(self) -> ProxyArray:
        """See :attr:`isaaclab.assets.BaseArticulationData.body_com_jacobian_w`.

        PhysX provides a natively center-of-mass-referenced Jacobian. The view refreshes
        once per simulation timestamp and normalizes body and joint axes when needed.
        No explicit :meth:`_ensure_fk_fresh` call is needed because PhysX
        recomputes the Jacobian from the current joint state on query.
        """
        if self._body_com_jacobian_w.timestamp < self._sim_timestamp:
            backend_jacobian = self._root_view.get_jacobians()
            has_body_ordering = self.has_body_ordering
            has_joint_ordering = self.has_joint_ordering
            if has_body_ordering or has_joint_ordering or self._has_reversed_joints:
                joint_user_to_backend = (
                    self._jacobian_joint_user_to_backend
                    if self._jacobian_joint_user_to_backend is not None
                    else self._joint_dof_signs
                )
                self._read_launch_cache.launch(
                    "body_com_jacobian_w_ordering",
                    ordering_kernels.reorder_jacobian_backend_to_user,
                    dim=self._body_com_jacobian_w.data.shape,
                    inputs=[
                        backend_jacobian,
                        self._jacobian_body_user_to_backend,
                        joint_user_to_backend,
                        self._joint_dof_signs,
                        self._num_base_dofs,
                        has_body_ordering,
                        has_joint_ordering,
                    ],
                    outputs=[self._body_com_jacobian_w.data],
                )
            else:
                self._body_com_jacobian_w.data = backend_jacobian
            self._body_com_jacobian_w.timestamp = self._sim_timestamp
        if self._body_com_jacobian_w_ta is None:
            self._body_com_jacobian_w_ta = ProxyArray(self._body_com_jacobian_w.data)
        return self._body_com_jacobian_w_ta

    @property
    def body_link_jacobian_w(self) -> ProxyArray:
        """See :attr:`isaaclab.assets.BaseArticulationData.body_link_jacobian_w`.

        PhysX implementation: applies the COM→origin shift kernel to
        :attr:`body_com_jacobian_w` (PhysX's engine output is COM-referenced).
        """
        self._read_launch_cache.launch(
            "body_link_jacobian_w",
            articulation_kernels.shift_jacobian_com_to_origin,
            dim=self._body_link_jacobian_w_buf.shape[:2] + (self._body_link_jacobian_w_buf.shape[3],),
            inputs=[
                self.body_link_pose_w.warp,
                self.body_com_pos_b.warp,
                self._jacobian_link_offset,
                self.body_com_jacobian_w.warp,
            ],
            outputs=[self._body_link_jacobian_w_buf],
        )
        return self._body_link_jacobian_w_ta

    def _refresh_generalized_joint_buffer(
        self,
        buf: TimestampedBuffer,
        view_getter: Callable[[], wp.array],
        reorder_kernel: wp.Kernel,
    ) -> None:
        """Refresh a timestamp-lazy generalized joint-axis buffer from its backend view.

        Reads the backend view once when stale for the current step and either aliases
        it or normalizes its joint axes into the owned public-order buffer with
        :paramref:`reorder_kernel`. Unlike the world-frame pose
        buffers this needs no explicit :meth:`_ensure_fk_fresh`, because PhysX recomputes
        the quantity from the current joint state on query.

        Args:
            buf: Owned public-order buffer to refresh in place.
            view_getter: Zero-argument callable returning the backend-order view.
            reorder_kernel: Warp kernel that normalizes the joint axes.
        """
        if buf.timestamp >= self._sim_timestamp:
            return
        backend_source = view_getter()
        has_joint_ordering = self.has_joint_ordering
        if has_joint_ordering or self._has_reversed_joints:
            joint_user_to_backend = self.joint_ordering.user_to_backend if has_joint_ordering else self._joint_dof_signs
            self._read_launch_cache.launch(
                id(buf),
                reorder_kernel,
                dim=buf.data.shape,
                inputs=[
                    backend_source,
                    joint_user_to_backend,
                    self._joint_dof_signs,
                    self._num_base_dofs,
                    has_joint_ordering,
                ],
                outputs=[buf.data],
            )
        else:
            buf.data = backend_source
        buf.timestamp = self._sim_timestamp

    @property
    def mass_matrix(self) -> ProxyArray:
        """See :attr:`isaaclab.assets.BaseArticulationData.mass_matrix`.

        Uses :meth:`_refresh_generalized_joint_buffer` for timestamped refresh and
        joint-axis reordering.
        """
        self._refresh_generalized_joint_buffer(
            self._mass_matrix,
            self._root_view.get_generalized_mass_matrices,
            ordering_kernels.reorder_mass_matrix_backend_to_user,
        )
        if self._mass_matrix_ta is None:
            self._mass_matrix_ta = ProxyArray(self._mass_matrix.data)
        return self._mass_matrix_ta

    @property
    def gravity_compensation_forces(self) -> ProxyArray:
        """See :attr:`isaaclab.assets.BaseArticulationData.gravity_compensation_forces`.

        Uses :meth:`_refresh_generalized_joint_buffer` for timestamped refresh and
        joint-axis reordering.
        """
        self._refresh_generalized_joint_buffer(
            self._gravity_compensation_forces,
            self._root_view.get_gravity_compensation_forces,
            ordering_kernels.reorder_generalized_vector_backend_to_user,
        )
        if self._gravity_compensation_forces_ta is None:
            self._gravity_compensation_forces_ta = ProxyArray(self._gravity_compensation_forces.data)
        return self._gravity_compensation_forces_ta

    """
    Joint state properties.
    """

    def _refresh_joint_state_user(self, user_buffer: TimestampedBuffer, view_getter: Callable[[], wp.array]) -> None:
        """Refresh a public-order joint-state buffer directly from its backend view.

        Reads the backend view once and, when stale for the current timestamp, either aliases it
        (identity ordering) or gathers it into the owned public-order buffer. The PhysX tensor API
        returns views into stable pre-allocated buffers, so the gather can read the view directly
        without an intermediate full-buffer staging copy.

        Args:
            user_buffer: The public-order joint-state buffer to refresh in place.
            view_getter: Zero-argument callable returning the backend-order view.
        """
        if user_buffer.timestamp >= self._sim_timestamp:
            return
        if not self.has_joint_ordering:
            user_buffer.data = view_getter()
        else:
            self._read_launch_cache.launch(
                id(user_buffer),
                ordering_kernels.reorder_2d_backend_to_user,
                dim=(self._num_instances, self._num_joints),
                inputs=[view_getter(), self.joint_ordering.user_to_backend],
                outputs=[user_buffer.data],
            )
        user_buffer.timestamp = self._sim_timestamp

    def _get_joint_state_write_buffer(
        self,
        user_buffer: TimestampedBuffer,
        backend_buffer: TimestampedBuffer | None,
        view_getter: Callable[[], wp.array],
        require_current: bool,
    ) -> wp.array:
        """Return the complete backend-order joint-state rows used by PhysX setters.

        Partial writes scatter into this full-image buffer, so when ``require_current`` is set the
        buffer is first refreshed from the backend view to keep the unwritten rows current. This is
        the only path that stages the backend-order image from the view; ordered reads gather from
        the view directly in :meth:`_refresh_joint_state_user`.

        Args:
            user_buffer: The public-order joint-state buffer, also the write target under identity
                ordering.
            backend_buffer: The owned backend-order staging buffer, or ``None`` under identity
                ordering.
            view_getter: Zero-argument callable returning the backend-order view.
            require_current: Whether the unwritten rows must reflect the current state, i.e. the
                write covers only a subset of joints.

        Returns:
            The backend-order buffer that setters scatter into and push to PhysX.
        """
        if not self.has_joint_ordering:
            if require_current:
                self._refresh_joint_state_user(user_buffer, view_getter)
            return user_buffer.data
        if require_current and backend_buffer.timestamp < self._sim_timestamp:
            backend_buffer.data.assign(view_getter())
            backend_buffer.timestamp = self._sim_timestamp
        return backend_buffer.data

    @property
    def joint_pos(self) -> ProxyArray:
        """Joint positions of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        self._refresh_joint_pos()
        if self._joint_pos_ta is None:
            self._joint_pos_ta = ProxyArray(self._joint_pos.data)
        return self._joint_pos_ta

    def _refresh_joint_pos(self) -> None:
        """Refresh the public-order joint-position buffer when stale."""
        self._refresh_joint_state_user(self._joint_pos, self._root_view.get_dof_positions)

    def _get_joint_pos_write_buffer(self, require_current: bool) -> wp.array:
        """Return the complete backend-order position rows used by PhysX setters."""
        return self._get_joint_state_write_buffer(
            self._joint_pos, self._joint_pos_backend, self._root_view.get_dof_positions, require_current
        )

    @property
    def joint_vel(self) -> ProxyArray:
        """Joint velocities of all joints.

        Shape is (num_instances, num_joints), dtype = wp.float32. In torch this resolves to (num_instances, num_joints).
        """
        self._refresh_joint_vel()
        if self._joint_vel_ta is None:
            self._joint_vel_ta = ProxyArray(self._joint_vel.data)
        return self._joint_vel_ta

    def _refresh_joint_vel(self) -> None:
        """Refresh the public-order joint-velocity buffer when stale."""
        self._refresh_joint_state_user(self._joint_vel, self._root_view.get_dof_velocities)

    def _get_joint_vel_write_buffer(self, require_current: bool) -> wp.array:
        """Return the complete backend-order velocity rows used by PhysX setters."""
        return self._get_joint_state_write_buffer(
            self._joint_vel, self._joint_vel_backend, self._root_view.get_dof_velocities, require_current
        )

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
            self._read_launch_cache.launch(
                "projected_gravity_b",
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.GRAVITY_VEC_W, self.root_link_quat_w],
                outputs=[self._projected_gravity_b.data],
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
            self._read_launch_cache.launch(
                "heading_w",
                shared_kernels.root_heading_w,
                dim=self._num_instances,
                inputs=[self.FORWARD_VEC_B, self.root_link_quat_w],
                outputs=[self._heading_w.data],
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
            self._read_launch_cache.launch(
                "root_link_lin_vel_b",
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_link_lin_vel_w, self.root_link_quat_w],
                outputs=[self._root_link_lin_vel_b.data],
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
            self._read_launch_cache.launch(
                "root_link_ang_vel_b",
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_link_ang_vel_w, self.root_link_quat_w],
                outputs=[self._root_link_ang_vel_b.data],
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
            self._read_launch_cache.launch(
                "root_com_lin_vel_b",
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_com_lin_vel_w, self.root_link_quat_w],
                outputs=[self._root_com_lin_vel_b.data],
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
            self._read_launch_cache.launch(
                "root_com_ang_vel_b",
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_com_ang_vel_w, self.root_link_quat_w],
                outputs=[self._root_com_ang_vel_b.data],
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
        self._body_com_pose_b_backend: TimestampedBuffer | None = None
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
        self._joint_pos_backend: TimestampedBuffer | None = None
        self._joint_vel_backend: TimestampedBuffer | None = None
        self._joint_acc = TimestampedBuffer((self._num_instances, self._num_joints), self.device, wp.float32)
        # -- derived properties (these are cached to avoid repeated memory allocations)
        self._projected_gravity_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)
        self._heading_w = TimestampedBuffer((self._num_instances), self.device, wp.float32)
        self._root_link_lin_vel_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)
        self._root_link_ang_vel_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)
        self._root_com_lin_vel_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)
        self._root_com_ang_vel_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)

        # -- dynamics quantities for task-space controllers
        # PhysX Jacobians exclude only the fixed root body and prepend six base-DoF columns
        # for floating-base articulations. Preserve that engine-native layout, including in
        # Newton's matching ``eval_jacobian`` wrapper. Default ordering returns engine views;
        # nonidentity ordering gathers into owned public buffers. The link-origin Jacobian
        # always remains owned because it is the COM-to-origin shift-kernel output.
        is_fixed_base = self._root_view.shared_metatype.fixed_base
        self._jacobian_link_offset = 1 if is_fixed_base else 0
        num_jacobi_bodies = self._num_bodies - self._jacobian_link_offset
        num_base_dofs = 0 if is_fixed_base else 6
        self._num_base_dofs = num_base_dofs
        self._jacobian_body_user_to_backend: wp.array | None = None
        self._jacobian_joint_user_to_backend: wp.array | None = None
        self._body_link_jacobian_w_buf = wp.zeros(
            (self._num_instances, num_jacobi_bodies, 6, self._num_joints + num_base_dofs),
            dtype=wp.float32,
            device=self.device,
        )
        # Under default or identity ordering, these placeholder allocations are replaced on the
        # first read by views returned from ``_root_view.get_*()``. Under nonidentity ordering,
        # they remain owned gather destinations. Timestamps advance on each refresh and are
        # invalidated by write paths.
        self._body_com_jacobian_w = TimestampedBuffer(
            (self._num_instances, num_jacobi_bodies, 6, self._num_joints + num_base_dofs),
            self.device,
            wp.float32,
        )
        self._mass_matrix = TimestampedBuffer(
            (self._num_instances, self._num_joints + num_base_dofs, self._num_joints + num_base_dofs),
            self.device,
            wp.float32,
        )
        self._gravity_compensation_forces = TimestampedBuffer(
            (self._num_instances, self._num_joints + num_base_dofs),
            self.device,
            wp.float32,
        )

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
        self._joint_stiffness_backend: wp.array | None = None
        self._joint_damping_backend: wp.array | None = None
        self._joint_armature_backend: wp.array | None = None
        self._joint_pos_limits_backend: wp.array | None = None
        self._joint_vel_limits_backend: wp.array | None = None
        self._joint_effort_limits_backend: wp.array | None = None
        self._joint_friction_props_user: wp.array | None = None
        self._joint_friction_props_backend: wp.array | None = None
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
        # Timestamp-lazy model-property buffers (initial timestamp -1.0 so the first read always
        # refreshes). Direct tensor-view writes (``root_view.set_masses`` / ``set_inertias``) thus
        # become visible on the first read after the next simulation update, matching the OVPhysX
        # articulation. ``get_masses()`` / ``get_inertias()`` are CPU-only; the refresh copies
        # host-to-device on GPU pipelines. Under body ordering the backend-order staging buffers
        # below are gathered into these public-order buffers on read.
        _masses = self._root_view.get_masses()
        self._body_mass = TimestampedBuffer(_masses.shape, self.device, _masses.dtype)
        _inertias = self._root_view.get_inertias()
        self._body_inertia = TimestampedBuffer(_inertias.shape, self.device, _inertias.dtype)
        self._body_mass_backend: wp.array | None = None
        self._body_inertia_backend: wp.array | None = None
        self._default_root_state = None

        # Initialize ProxyArray wrappers
        self._pin_proxy_arrays()

    def _configure_ordering_buffers(self) -> None:
        """Allocate and seed buffers owned only by nonidentity ordering."""
        if self.has_joint_ordering:
            if self._joint_pos_backend is None:
                self._joint_pos_backend = TimestampedBuffer(
                    (self._num_instances, self._num_joints), self.device, wp.float32
                )
            if self._joint_vel_backend is None:
                self._joint_vel_backend = TimestampedBuffer(
                    (self._num_instances, self._num_joints), self.device, wp.float32
                )
            joint_property_specs = (
                ("_joint_stiffness_backend", self._joint_stiffness),
                ("_joint_damping_backend", self._joint_damping),
                ("_joint_armature_backend", self._joint_armature),
                ("_joint_pos_limits_backend", self._joint_pos_limits),
                ("_joint_vel_limits_backend", self._joint_vel_limits),
                ("_joint_effort_limits_backend", self._joint_effort_limits),
            )
            for backend_name, user_buffer in joint_property_specs:
                if getattr(self, backend_name) is None:
                    setattr(self, backend_name, wp.clone(user_buffer, device=self.device))
            if self._joint_friction_props_user is None:
                self._joint_friction_props_user = wp.zeros(
                    (self._num_instances, self._num_joints, 3), dtype=wp.float32, device=self.device
                )
            if self._joint_friction_props_backend is None:
                self._joint_friction_props_backend = wp.clone(
                    self._root_view.get_dof_friction_properties(), device=self.device
                )

            self._joint_pos.data = wp.zeros(
                (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
            )
            self._joint_vel.data = wp.zeros(
                (self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
            )
            reset_timestamps(
                [self._joint_pos, self._joint_vel, self._joint_acc, self._joint_pos_backend, self._joint_vel_backend]
            )

            previous_joint_vel_backend = wp.clone(self._root_view.get_dof_velocities(), device=self.device)
            wp.launch(
                ordering_kernels.reorder_2d_backend_to_user,
                dim=(self._num_instances, self._num_joints),
                inputs=[previous_joint_vel_backend, self.joint_ordering.user_to_backend],
                outputs=[self._previous_joint_vel],
                device=self.device,
            )
            for backend_name, user_buffer in joint_property_specs:
                wp.launch(
                    ordering_kernels.reorder_2d_backend_to_user,
                    dim=(self._num_instances, self._num_joints),
                    inputs=[getattr(self, backend_name), self.joint_ordering.user_to_backend],
                    outputs=[user_buffer],
                    device=self.device,
                )
            wp.launch(
                ordering_kernels.reorder_3d_backend_to_user,
                dim=(self._num_instances, self._num_joints, 3),
                inputs=[self._joint_friction_props_backend, self.joint_ordering.user_to_backend],
                outputs=[self._joint_friction_props_user],
                device=self.device,
            )
            wp.launch(
                articulation_kernels.extract_friction_properties,
                dim=(self._num_instances, self._num_joints),
                inputs=[self._joint_friction_props_user],
                outputs=[
                    self._joint_friction_coeff,
                    self._joint_dynamic_friction_coeff,
                    self._joint_viscous_friction_coeff,
                ],
                device=self.device,
            )
        if self.has_body_ordering:
            # Invariant: from seeding onward, each backend staging must stay the backend-order
            # image of its public buffer. Partial body-property setters scatter only the
            # selected cells into both buffers and push full backend rows to the simulation,
            # so a stale or divergent staging silently corrupts the unselected cells.
            if self._body_com_pose_b_backend is None:
                self._body_com_pose_b_backend = TimestampedBuffer(
                    (self._num_instances, self._num_bodies), self.device, wp.transformf
                )
            if self._body_mass_backend is None:
                self._body_mass_backend = wp.clone(self._body_mass.data, device=self.device)
            if self._body_inertia_backend is None:
                self._body_inertia_backend = wp.clone(self._body_inertia.data, device=self.device)
            # The public-order mass/inertia buffers are refreshed lazily (gathered from the backend
            # staging on the next read), so only reset their timestamps here.
            reset_timestamps(
                [self._body_com_pose_b, self._body_com_pose_b_backend, self._body_mass, self._body_inertia]
            )

    def _apply_ordering_maps_after_resolve(self) -> None:
        """Configure public-order buffers after articulation ordering maps are installed."""
        self._read_launch_cache.clear()
        joint_ordering = self.joint_ordering
        body_ordering = self.body_ordering
        self._configure_ordering_buffers()

        if body_ordering is not None:
            self._jacobian_body_user_to_backend = self._make_jacobian_body_user_to_backend()
        else:
            self._jacobian_body_user_to_backend = None
        if joint_ordering is not None:
            self._jacobian_joint_user_to_backend = joint_ordering.user_to_backend
        else:
            self._jacobian_joint_user_to_backend = None

        if self.has_body_ordering:
            self._body_link_pose_w.data = wp.zeros(
                (self._num_instances, self._num_bodies), dtype=wp.transformf, device=self.device
            )
            self._body_com_vel_w.data = wp.zeros(
                (self._num_instances, self._num_bodies), dtype=wp.spatial_vectorf, device=self.device
            )
            self._body_com_acc_w.data = wp.zeros(
                (self._num_instances, self._num_bodies), dtype=wp.spatial_vectorf, device=self.device
            )
        reset_timestamps(
            [
                self._body_link_pose_w,
                self._body_link_vel_w,
                self._body_com_pose_w,
                self._body_com_vel_w,
                self._body_com_acc_w,
                self._body_com_pose_b,
                self._body_state_w,
                self._body_link_state_w,
                self._body_com_state_w,
            ]
        )

        if self.has_body_ordering or self.has_joint_ordering or self._has_reversed_joints:
            self._body_com_jacobian_w.data = wp.zeros(
                self._body_com_jacobian_w.data.shape, dtype=wp.float32, device=self.device
            )
        if self.has_joint_ordering or self._has_reversed_joints:
            self._mass_matrix.data = wp.zeros(self._mass_matrix.data.shape, dtype=wp.float32, device=self.device)
            self._gravity_compensation_forces.data = wp.zeros(
                self._gravity_compensation_forces.data.shape, dtype=wp.float32, device=self.device
            )
        reset_timestamps([self._body_com_jacobian_w, self._mass_matrix, self._gravity_compensation_forces])
        self._pin_proxy_arrays()

    def _pin_proxy_arrays(self) -> None:
        """Create pinned ProxyArray wrappers for all data buffers.

        This is called from :meth:`_create_buffers` and after ordering maps install owned
        public-order buffers. PhysX tensor API buffers have stable GPU pointers across
        simulation steps, so no per-step rebinding is needed.
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
        # Dynamics quantities (task-space controllers). ``_body_link_jacobian_w`` wraps our
        # own pre-allocated buffer (pointer-stable, eager wrap). The other three wrappers are
        # initialized lazily inside their property bodies. They wrap direct engine aliases for
        # default or identity ordering and owned public-order buffers for nonidentity ordering,
        # matching the ``TimestampedBuffer`` + ``ProxyArray`` cache pattern used by
        # ``body_link_pose_w``, ``joint_pos``, and the rest of this file. Refresh is gated by
        # ``_sim_timestamp`` and invalidated by ``write_*_to_sim_index`` setting
        # ``timestamp = -1.0``.
        self._body_link_jacobian_w_ta = ProxyArray(self._body_link_jacobian_w_buf)
        self._body_com_jacobian_w_ta: ProxyArray | None = None
        self._mass_matrix_ta: ProxyArray | None = None
        self._gravity_compensation_forces_ta: ProxyArray | None = None
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
        self._read_launch_cache.launch(
            "default_root_state",
            shared_kernels.concat_root_pose_and_vel_to_state,
            dim=self._num_instances,
            inputs=[
                self._default_root_pose,
                self._default_root_vel,
            ],
            outputs=[
                self._default_root_state,
            ],
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
            self._read_launch_cache.launch(
                "root_state_w",
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=(self._num_instances),
                inputs=[
                    self.root_link_pose_w,
                    self.root_com_vel_w,
                ],
                outputs=[
                    self._root_state_w.data,
                ],
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
            self._read_launch_cache.launch(
                "root_link_state_w",
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=self._num_instances,
                inputs=[
                    self.root_link_pose_w,
                    self.root_link_vel_w,
                ],
                outputs=[
                    self._root_link_state_w.data,
                ],
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
            self._read_launch_cache.launch(
                "root_com_state_w",
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=self._num_instances,
                inputs=[
                    self.root_com_pose_w,
                    self.root_com_vel_w,
                ],
                outputs=[
                    self._root_com_state_w.data,
                ],
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
            self._read_launch_cache.launch(
                "body_state_w",
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_link_pose_w,
                    self.body_com_vel_w,
                ],
                outputs=[
                    self._body_state_w.data,
                ],
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
            self._read_launch_cache.launch(
                "body_link_state_w",
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_link_pose_w,
                    self.body_link_vel_w,
                ],
                outputs=[
                    self._body_link_state_w.data,
                ],
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
            self._read_launch_cache.launch(
                "body_com_state_w",
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_com_pose_w,
                    self.body_com_vel_w,
                ],
                outputs=[
                    self._body_com_state_w.data,
                ],
            )
            self._body_com_state_w.timestamp = self._sim_timestamp

        if self._body_com_state_w_ta is None:
            self._body_com_state_w_ta = ProxyArray(self._body_com_state_w.data)
        return self._body_com_state_w_ta
