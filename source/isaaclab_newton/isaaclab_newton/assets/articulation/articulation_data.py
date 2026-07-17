# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import logging
import warnings
import weakref
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from isaaclab.assets.articulation import ordering_kernels
from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.buffers import reset_timestamps
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

        # Bind ``GRAVITY_VEC_W`` to Newton's per-env ``model.gravity`` (m/s^2) so
        # per-env gravity randomization stays live; consumers normalize on read.
        self.GRAVITY_VEC_W = ProxyArray(SimulationManager.get_model().gravity)
        forward_vec = np.full((self._root_view.count, 3), (1.0, 0.0, 0.0), dtype=np.float32)
        self.FORWARD_VEC_B = ProxyArray(wp.array(forward_vec, dtype=wp.vec3f, device=self.device))

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

    def _ensure_fk_fresh(self) -> None:
        """Run forward kinematics if joint state has changed since the last FK update.

        Newton's ``state.body_q`` (per-body world transforms) is updated by the active
        solver manager's ``forward()``, which calls a solver-specialized FK hook.
        After a manual joint or root write that bypassed the sim step (``write_*_to_sim_*``),
        ``_fk_timestamp`` is set to ``-1.0`` to force a refresh on the next read of any
        property that depends on body poses (``body_link_pose_w``, the Jacobian properties,
        ``mass_matrix``).

        This out-of-band FK path also republishes the user-order body-state shadows via
        :meth:`_refresh_user_order_body_state`: the post-step callback only fires inside a
        sim step, so a manual write followed by an FK refresh would otherwise leave the
        passthrough ``body_link_pose_w`` / ``body_com_vel_w`` shadows stale.
        """
        if self._fk_timestamp < self._sim_timestamp:
            SimulationManager.forward()
            self._refresh_user_order_body_state()
            self._fk_timestamp = self._sim_timestamp

    def _reset_pose(
        self, from_link: bool = True, *, env_ids: wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Reset the pose of the articulation.

        This will mark all the pose related properties as stale, and trigger a FK refresh.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            from_link: Set ``True`` when the root link pose was written so the derived root
                center-of-mass pose (:attr:`root_com_pose_w`) is also invalidated; set ``False`` when
                the center-of-mass pose was written directly so it is not clobbered. Defaults to True.
        """
        # Invalidate the derived root com pose only when it was not the quantity just written.
        reset_timestamps(
            [
                self._root_com_pose_w if from_link else None,
                self._body_com_pose_w,
                # root states
                self._root_state_w,
                self._root_link_state_w,
                self._root_com_state_w,
                # body com states
                self._body_state_w,
                self._body_link_state_w,
                self._body_com_state_w,
            ]
        )
        # NOTE: _fk_timestamp and invalidate_fk serve two distinct roles. _fk_timestamp is on the
        # data side and forces a refresh on the next outdated read. invalidate_fk is on the
        # simulation-manager side and lets the solver know state changed before its next step.
        self._fk_timestamp = -1.0
        SimulationManager.invalidate_fk(
            env_mask=env_mask, env_ids=env_ids, articulation_ids=self._root_view.articulation_ids
        )

    def _reset_velocity(
        self, from_com: bool = True, *, env_ids: wp.array | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Reset the velocity of the articulation.

        This will mark all the velocity related properties as stale, and trigger a FK refresh.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
            from_com: Set ``True`` when the root center-of-mass velocity was written so the derived root
                link velocity (:attr:`root_link_vel_w`) is also invalidated; set ``False`` when the link
                velocity was written directly so it is not clobbered. Defaults to True.
        """
        # Invalidate the derived root link velocity only when it was not the quantity just written.
        reset_timestamps(
            [
                self._root_link_vel_w if from_com else None,
                self._body_link_vel_w,
                # root states
                self._root_state_w,
                self._root_link_state_w,
                self._root_com_state_w,
                # body com states
                self._body_state_w,
                self._body_link_state_w,
                self._body_com_state_w,
            ]
        )
        # NOTE: _fk_timestamp and invalidate_fk serve two distinct roles. _fk_timestamp is on the
        # data side and forces a refresh on the next outdated read. invalidate_fk is on the
        # simulation-manager side and lets the solver know state changed before its next step.
        self._fk_timestamp = -1.0
        SimulationManager.invalidate_fk(
            env_mask=env_mask, env_ids=env_ids, articulation_ids=self._root_view.articulation_ids
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
        """Newton joint friction force/torque provided to the simulation.

        Despite the ``coeff`` suffix in the Isaac Lab API name, Newton stores this as an absolute joint friction
        force/torque [N or N·m, depending on joint type].

        For example, the MJWarp solver copies this value into MuJoCo Warp's ``dof_frictionloss``. Setting
        ``joint_friction_coeff`` to 0.2 configures a dry-friction loss limit of 0.2 N·m on a revolute joint DOF,
        or 0.2 N on a prismatic joint DOF.

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
        if self._joint_pos_limits_timestamp < self._sim_timestamp:
            joint_pos_limits_lower = (
                self._joint_pos_limits_lower_user if self.has_joint_ordering else self._sim_bind_joint_pos_limits_lower
            )
            joint_pos_limits_upper = (
                self._joint_pos_limits_upper_user if self.has_joint_ordering else self._sim_bind_joint_pos_limits_upper
            )
            wp.launch(
                articulation_kernels.concat_joint_pos_limits_lower_and_upper,
                dim=(self._num_instances, self._num_joints),
                inputs=[
                    joint_pos_limits_lower,
                    joint_pos_limits_upper,
                ],
                outputs=[
                    self._joint_pos_limits,
                ],
                device=self.device,
            )
            self._joint_pos_limits_timestamp = self._sim_timestamp
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
        return self._fixed_tendon_stiffness_ta

    @property
    def fixed_tendon_damping(self) -> ProxyArray:
        """Fixed tendon damping provided to the simulation.

        Shape is (num_instances, num_fixed_tendons), dtype = wp.float32. In torch this resolves to
        (num_instances, num_fixed_tendons).
        """
        return self._fixed_tendon_damping_ta

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
                    self._sim_bind_body_com_pos_b,
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
                    self._sim_bind_body_com_pos_b,
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

        With body ordering active, direct writes to the Newton model arrays bypass the public-order
        buffers; use the asset setters instead.
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
        self._ensure_fk_fresh()
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
        self._ensure_fk_fresh()
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
        self._ensure_fk_fresh()
        return self._body_com_vel_w_ta

    @property
    def body_com_acc_w(self) -> ProxyArray:
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]``.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            body_ordering = self.body_ordering
            wp.launch(
                articulation_kernels.get_body_com_acc_from_body_com_vel_ordered,
                dim=(self._num_instances, self._num_bodies),
                device=self.device,
                inputs=[
                    self._sim_bind_body_com_vel_w,
                    self._previous_body_com_vel,
                    body_ordering.user_to_backend if body_ordering is not None else None,
                    body_ordering is not None,
                    SimulationManager.get_dt(),
                ],
                outputs=[self._body_com_acc_w.data],
            )
            self._body_com_acc_w.timestamp = self._sim_timestamp
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
    Dynamics quantities (task-space controllers).
    """

    @property
    def body_com_jacobian_w(self) -> ProxyArray:
        """See :attr:`isaaclab.assets.BaseArticulationData.body_com_jacobian_w`.

        Newton implementation: ``eval_jacobian`` (writes the model-wide buffer) then a
        gather kernel extracts this view's rows. ``link_offset`` drops Newton's fixed-
        root row for fixed-base; the DoF axis is preserved in full.
        """
        # Newton's eval_jacobian reads ``state.body_q`` (link poses); refresh FK if stale.
        # Matches the convention in ``body_link_pose_w`` — Python-guarded lazy refresh.
        self._ensure_fk_fresh()
        # eval_jacobian writes every articulation in the model; gather kernel extracts this
        # view's rows. ``link_offset`` skips Newton's fixed-root row for fixed-base; the DoF
        # axis is preserved in full (free-root joint's 6 columns up front for floating-base),
        # matching the PhysX layout and the cross-library industry convention.
        self._root_view.eval_jacobian(
            SimulationManager.get_state_0(),
            J=self._jacobian_buf_flat,
            joint_S_s=self._joint_S_s_buf,
        )
        joint_ordering = self.joint_ordering
        wp.launch(
            articulation_kernels.gather_jacobian_rows,
            dim=self._body_com_jacobian_w_buf.shape,
            inputs=[
                self._jacobian_buf,
                self._jacobian_view_art_ids,
                self._jacobian_body_user_to_backend,
                joint_ordering.user_to_backend if joint_ordering is not None else None,
                self._num_base_dofs,
                joint_ordering is not None,
            ],
            outputs=[self._body_com_jacobian_w_buf],
            device=self.device,
        )
        return self._body_com_jacobian_w_ta

    @property
    def body_link_jacobian_w(self) -> ProxyArray:
        """See :attr:`isaaclab.assets.BaseArticulationData.body_link_jacobian_w`.

        Newton implementation: applies the COM→origin shift kernel to
        :attr:`body_com_jacobian_w` (Newton's ``eval_jacobian`` is COM-referenced).
        """
        # ``body_link_pose_w`` accessor triggers ``SimulationManager.forward()`` if FK is
        # stale (after a manual joint / root write that bypassed the sim step). Reading the
        # property here — not ``_sim_bind_body_link_pose_w`` directly — keeps the shift
        # kernel from using stale link rotations during reset / IK-warm-start paths.
        link_pose_w = self.body_link_pose_w.warp
        com_jac = self.body_com_jacobian_w
        wp.launch(
            articulation_kernels.shift_jacobian_com_to_origin,
            dim=self._body_link_jacobian_w_buf.shape[:2] + (self._body_link_jacobian_w_buf.shape[3],),
            inputs=[
                link_pose_w,
                self.body_com_pos_b.warp,
                self._jacobian_link_offset,
                com_jac.warp,
            ],
            outputs=[self._body_link_jacobian_w_buf],
            device=self.device,
        )
        return self._body_link_jacobian_w_ta

    @property
    def mass_matrix(self) -> ProxyArray:
        """See :attr:`isaaclab.assets.BaseArticulationData.mass_matrix`.

        Newton implementation: ``eval_mass_matrix`` (writes the model-wide buffer) then a
        gather kernel extracts this view's rows.
        """
        # eval_jacobian / eval_mass_matrix read ``state.body_q``; refresh FK if stale.
        # Matches the convention in ``body_link_pose_w`` — Python-guarded lazy refresh.
        self._ensure_fk_fresh()
        # eval_mass_matrix treats ``J`` as an input (skips its own jacobian compute when
        # provided), so we must populate the scratch first via eval_jacobian. Reusing
        # ``_jacobian_buf_flat`` (same shape) avoids a second allocation. All scratch buffers
        # are pre-allocated for CUDA-graph capture safety.
        state = SimulationManager.get_state_0()
        self._root_view.eval_jacobian(
            state,
            J=self._jacobian_buf_flat,
            joint_S_s=self._joint_S_s_buf,
        )
        self._root_view.eval_mass_matrix(
            state,
            H=self._mass_matrix_full_buf,
            J=self._jacobian_buf_flat,
            body_I_s=self._mass_matrix_body_I_s_buf,
            joint_S_s=self._joint_S_s_buf,
        )
        joint_ordering = self.joint_ordering
        wp.launch(
            articulation_kernels.gather_mass_matrix_rows,
            dim=self._mass_matrix_buf.shape,
            inputs=[
                self._mass_matrix_full_buf,
                self._jacobian_view_art_ids,
                joint_ordering.user_to_backend if joint_ordering is not None else None,
                self._num_base_dofs,
                joint_ordering is not None,
            ],
            outputs=[self._mass_matrix_buf],
            device=self.device,
        )
        return self._mass_matrix_ta

    @property
    def gravity_compensation_forces(self) -> ProxyArray:
        """See :attr:`isaaclab.assets.BaseArticulationData.gravity_compensation_forces`.

        Newton implementation: raises :class:`NotImplementedError` — Newton's
        ``ArticulationView`` exposes only ``eval_fk`` / ``eval_jacobian`` /
        ``eval_mass_matrix``. Use PhysX, or set the controller's
        ``gravity_compensation=False`` until upstream Newton adds the primitive.
        Tracking upstream: `newton#2497 <https://github.com/newton-physics/newton/issues/2497>`_,
        `newton#2529 <https://github.com/newton-physics/newton/issues/2529>`_,
        `newton#2625 <https://github.com/newton-physics/newton/issues/2625>`_.
        """
        raise NotImplementedError(
            "Newton has no gravity-compensation primitive. Use PhysX, or set the controller's"
            " ``gravity_compensation=False`` until upstream Newton adds an"
            " ``eval_gravity_compensation`` API. Tracking upstream:"
            " https://github.com/newton-physics/newton/issues/2497,"
            " https://github.com/newton-physics/newton/issues/2529,"
            " https://github.com/newton-physics/newton/issues/2625."
        )

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
                shared_kernels.projected_gravity_b_kernel,
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
        self._num_fixed_tendons = self._root_view.tendon_count
        self._num_spatial_tendons = 0  # spatial tendons not supported

        # -- root properties
        self._sim_bind_root_link_pose_w = self._root_view.get_root_transforms(SimulationManager.get_state_0())[:, 0]
        # ``get_root_velocities`` returns ``None`` for fixed-base articulations; the
        # ``wp.zeros`` fallback set by :meth:`_create_buffers` must survive subsequent
        # resets, so only overwrite when the solver actually exposes the binding.
        root_vel_w = self._root_view.get_root_velocities(SimulationManager.get_state_0())
        if root_vel_w is not None:
            if self._root_view.is_fixed_base:
                self._sim_bind_root_com_vel_w = root_vel_w[:, 0, 0]
            else:
                self._sim_bind_root_com_vel_w = root_vel_w[:, 0]
        # -- body properties
        self._sim_bind_body_com_pos_b = self._root_view.get_attribute("body_com", SimulationManager.get_model())[:, 0]
        self._sim_bind_body_link_pose_w = self._root_view.get_link_transforms(SimulationManager.get_state_0())[:, 0]
        body_com_vel_w = self._root_view.get_link_velocities(SimulationManager.get_state_0())
        if body_com_vel_w is not None:
            self._sim_bind_body_com_vel_w = body_com_vel_w[:, 0]
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
            self._sim_bind_joint_act = self._root_view.get_attribute("joint_act", SimulationManager.get_control())[:, 0]
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
            self._sim_bind_joint_act = wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
            self._sim_bind_joint_position_target = wp.zeros(
                (self._num_instances, 0), dtype=wp.float32, device=self.device
            )
            self._sim_bind_joint_velocity_target = wp.zeros(
                (self._num_instances, 0), dtype=wp.float32, device=self.device
            )

        # assumes all tendons are fixed and only one arti in scene
        if self._root_view.tendon_count > 0:
            self._sim_bind_fixed_tendon_stiffness = self._root_view.get_attribute(
                "mujoco.tendon_stiffness", SimulationManager.get_model()
            )[:, 0]
            self._sim_bind_fixed_tendon_damping = self._root_view.get_attribute(
                "mujoco.tendon_damping",
                SimulationManager.get_model(),
            )[:, 0]
        else:
            self._sim_bind_fixed_tendon_stiffness = wp.zeros(
                (self._num_instances, 0), dtype=wp.float32, device=self.device
            )
            self._sim_bind_fixed_tendon_damping = wp.zeros(
                (self._num_instances, 0), dtype=wp.float32, device=self.device
            )

        # Re-pin ProxyArray wrappers to the newly created sim bindings.
        # On first init, _create_buffers() handles this after all buffers exist.
        if hasattr(self, "_root_link_pose_w_ta"):
            self._pin_proxy_arrays()
            if self.has_joint_ordering:
                wp.launch(
                    ordering_kernels.reorder_2d_backend_to_user,
                    dim=(self._num_instances, self._num_joints),
                    inputs=[self._sim_bind_joint_vel, self.joint_ordering.user_to_backend],
                    outputs=[self._previous_joint_vel],
                    device=self.device,
                )
            else:
                self._previous_joint_vel.assign(self._sim_bind_joint_vel)
            self._previous_body_com_vel.assign(self._sim_bind_body_com_vel_w)
            reset_timestamps([self._joint_acc, self._body_com_acc_w])

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
        # Body velocities are well-defined regardless of the base type (fixed-base articulations
        # still report link velocities); fall back to zeros only when the view genuinely cannot
        # provide them. Zeroing this binding together with the root velocity silently zeroes
        # every body-velocity read for fixed-base robots.
        if self._root_view.get_link_velocities(SimulationManager.get_state_0()) is None:
            logger.warning("Failed to get body com velocities. Setting body com velocities to zeros.")
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

        # staging buffers to write all tendon params to sim at once
        if self._num_fixed_tendons > 0:
            self._fixed_tendon_stiffness = wp.clone(self._sim_bind_fixed_tendon_stiffness)
            self._fixed_tendon_damping = wp.clone(self._sim_bind_fixed_tendon_damping)
        else:
            self._fixed_tendon_stiffness = wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)
            self._fixed_tendon_damping = wp.zeros((self._num_instances, 0), dtype=wp.float32, device=self.device)

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
        self._body_link_pose_w_user: wp.array | None = None
        self._body_com_vel_w_user: wp.array | None = None
        self._body_mass_user: wp.array | None = None
        self._body_inertia_user: wp.array | None = None
        self._body_com_pos_b_user: wp.array | None = None
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
        self._joint_pos_user: wp.array | None = None
        self._joint_vel_user: wp.array | None = None
        self._joint_stiffness_user: wp.array | None = None
        self._joint_damping_user: wp.array | None = None
        self._joint_armature_user: wp.array | None = None
        self._joint_friction_coeff_user: wp.array | None = None
        self._joint_pos_limits_lower_user: wp.array | None = None
        self._joint_pos_limits_upper_user: wp.array | None = None
        self._joint_vel_limits_user: wp.array | None = None
        self._joint_effort_limits_user: wp.array | None = None
        # -- dynamics quantities for task-space controllers
        self._create_jacobian_buffers(SimulationManager.get_model())
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

    def _create_jacobian_buffers(self, model) -> None:
        """Allocate the scratch + view-sized buffers used by task-space accessors.

        Newton's :meth:`eval_jacobian` / :meth:`eval_mass_matrix` write into model-sized
        scratch buffers spanning every articulation in the model; the gather kernels in
        :attr:`body_com_jacobian_w` / :attr:`mass_matrix` extract this view's rows. The
        output buffers are sized using THIS articulation's body / DoF counts (not the
        model-wide ``max_*``) so heterogeneous scenes do not leak zero-padded rows / cols
        into the returned tensor. The DoF axis includes ``num_base_dofs`` floating-base
        columns up front (0 for fixed-base, 6 for floating-base), matching the cross-
        library industry convention (PhysX, Pinocchio, Drake, MuJoCo, RBDL, OCS2, iDynTree).

        Args:
            model: Newton ``Model`` from :meth:`SimulationManager.get_model`. Read for
                ``articulation_count``, ``max_joints_per_articulation``,
                ``max_dofs_per_articulation``, ``joint_dof_count``, ``body_count``.
        """
        max_links = model.max_joints_per_articulation
        max_dofs = model.max_dofs_per_articulation

        # -- shared scratch (eval_jacobian outputs; consumed by ``body_com_jacobian_w``
        #    and reused as ``eval_mass_matrix``'s ``J`` input to skip a re-compute)
        self._jacobian_buf_flat = wp.zeros(
            (model.articulation_count, max_links * 6, max_dofs), dtype=wp.float32, device=self.device
        )
        # Motion subspace (Featherstone ``S``, spatial frame); produced by eval_jacobian,
        # also consumed by eval_mass_matrix.
        self._joint_S_s_buf = wp.zeros(model.joint_dof_count, dtype=wp.spatial_vector, device=self.device)

        # -- per-view gather config (shared by every gather/shift kernel below)
        # Link-row offset: fixed-base skips Newton's row-0 fixed-root row; floating-base keeps it.
        self._jacobian_link_offset = 1 if self._root_view.is_fixed_base else 0
        num_jacobi_bodies = self._num_bodies - self._jacobian_link_offset
        # Free-root DoF columns Newton fills for floating-base (0 fixed-base, 6 floating-base);
        # included in the DoF axis to match the cross-library industry convention.
        num_base_dofs = 0 if self._root_view.is_fixed_base else 6
        self._num_base_dofs = num_base_dofs
        self._jacobian_body_user_to_backend: wp.array | None = None
        # Flattened (num_worlds*num_per_view,) view-to-model index map for the gather kernels.
        self._jacobian_view_art_ids = self._root_view.articulation_ids.reshape((-1,))

        # -- ``body_com_jacobian_w``: 4-D reshape view of the shared scratch (kernel input
        #    to the gather) and the per-view output buffer (gather output)
        self._jacobian_buf = self._jacobian_buf_flat.reshape((model.articulation_count, max_links, 6, max_dofs))
        self._body_com_jacobian_w_buf = wp.zeros(
            (self._num_instances, num_jacobi_bodies, 6, self._num_joints + num_base_dofs),
            dtype=wp.float32,
            device=self.device,
        )

        # -- ``body_link_jacobian_w``: output of the COM→origin shift kernel applied to
        #    the COM-referenced Jacobian above; same shape, link-origin reference
        self._body_link_jacobian_w_buf = wp.zeros(
            (self._num_instances, num_jacobi_bodies, 6, self._num_joints + num_base_dofs),
            dtype=wp.float32,
            device=self.device,
        )

        # -- ``mass_matrix``: model-wide ``H`` scratch (eval_mass_matrix output), per-body
        #    spatial-inertia aux (Featherstone ``I``), and per-view output (gather output)
        self._mass_matrix_full_buf = wp.zeros(
            (model.articulation_count, max_dofs, max_dofs), dtype=wp.float32, device=self.device
        )
        self._mass_matrix_body_I_s_buf = wp.zeros(model.body_count, dtype=wp.spatial_matrix, device=self.device)
        self._mass_matrix_buf = wp.zeros(
            (self._num_instances, self._num_joints + num_base_dofs, self._num_joints + num_base_dofs),
            dtype=wp.float32,
            device=self.device,
        )

    def _validate_joint_ordering_buffers(self) -> None:
        """Validate buffers required while nonidentity joint ordering is active."""
        required_fields = (
            "_joint_pos_user",
            "_joint_vel_user",
            "_joint_stiffness_user",
            "_joint_damping_user",
            "_joint_armature_user",
            "_joint_friction_coeff_user",
            "_joint_pos_limits_lower_user",
            "_joint_pos_limits_upper_user",
            "_joint_vel_limits_user",
            "_joint_effort_limits_user",
        )
        missing_fields = [name for name in required_fields if getattr(self, name) is None]
        if missing_fields:
            raise RuntimeError(
                "Newton nonidentity joint ordering requires initialized public-order buffers; "
                f"missing {', '.join(missing_fields)}."
            )

    def _validate_body_ordering_buffers(self) -> None:
        """Validate buffers required while nonidentity body ordering is active."""
        required_fields = (
            "_jacobian_body_user_to_backend",
            "_body_link_pose_w_user",
            "_body_com_vel_w_user",
            "_body_mass_user",
            "_body_inertia_user",
            "_body_com_pos_b_user",
        )
        missing_fields = [name for name in required_fields if getattr(self, name) is None]
        if missing_fields:
            raise RuntimeError(
                "Newton nonidentity body ordering requires initialized public-order buffers; "
                f"missing {', '.join(missing_fields)}."
            )

    def _configure_joint_ordering_buffers(self) -> None:
        """Allocate or release buffers owned only by nonidentity joint ordering."""
        if self.has_joint_ordering:
            shape = (self._num_instances, self._num_joints)
            if self._joint_pos_user is None:
                self._joint_pos_user = wp.zeros(shape, dtype=wp.float32, device=self.device)
            if self._joint_vel_user is None:
                self._joint_vel_user = wp.zeros(shape, dtype=wp.float32, device=self.device)
            for field_name in (
                "_joint_stiffness_user",
                "_joint_damping_user",
                "_joint_armature_user",
                "_joint_friction_coeff_user",
                "_joint_pos_limits_lower_user",
                "_joint_pos_limits_upper_user",
                "_joint_vel_limits_user",
                "_joint_effort_limits_user",
            ):
                if getattr(self, field_name) is None:
                    setattr(self, field_name, wp.zeros(shape, dtype=wp.float32, device=self.device))
            self._validate_joint_ordering_buffers()
            # Seed the Lab-owned actuator gain records from the solver's initial
            # gains at ordering-resolve time, before actuator processing overwrites
            # the actuator-covered DOFs. These records are Lab-owned afterwards
            # (sim gains are deliberately zeroed for explicit DOFs), so they must
            # never be resynced from the solver on rebind -- unlike the sim-owned
            # mirrors in :meth:`_sync_user_ordered_joint_property_buffers`.
            for backend_data, user_data in (
                (self._sim_bind_joint_stiffness_sim, self._actuator_stiffness),
                (self._sim_bind_joint_damping_sim, self._actuator_damping),
            ):
                wp.launch(
                    ordering_kernels.reorder_2d_backend_to_user,
                    dim=shape,
                    inputs=[backend_data, self.joint_ordering.user_to_backend],
                    outputs=[user_data],
                    device=self.device,
                )
            wp.launch(
                ordering_kernels.reorder_2d_backend_to_user,
                dim=shape,
                inputs=[self._sim_bind_joint_vel, self.joint_ordering.user_to_backend],
                outputs=[self._previous_joint_vel],
                device=self.device,
            )
        else:
            self._joint_pos_user = None
            self._joint_vel_user = None
            self._joint_stiffness_user = None
            self._joint_damping_user = None
            self._joint_armature_user = None
            self._joint_friction_coeff_user = None
            self._joint_pos_limits_lower_user = None
            self._joint_pos_limits_upper_user = None
            self._joint_vel_limits_user = None
            self._joint_effort_limits_user = None
            self._previous_joint_vel.assign(self._sim_bind_joint_vel)
            self._actuator_stiffness.assign(self._sim_bind_joint_stiffness_sim)
            self._actuator_damping.assign(self._sim_bind_joint_damping_sim)
        self._joint_pos_limits_timestamp = -1.0
        reset_timestamps([self._joint_acc])

    def _configure_body_ordering_buffers(self) -> None:
        """Allocate or release buffers owned only by nonidentity body ordering."""
        if self.has_body_ordering:
            shape = (self._num_instances, self._num_bodies)
            if self._body_link_pose_w_user is None:
                self._body_link_pose_w_user = wp.zeros(shape, dtype=wp.transformf, device=self.device)
            if self._body_com_vel_w_user is None:
                self._body_com_vel_w_user = wp.zeros(shape, dtype=wp.spatial_vectorf, device=self.device)
            if self._body_mass_user is None:
                self._body_mass_user = wp.zeros(shape, dtype=wp.float32, device=self.device)
            if self._body_inertia_user is None:
                self._body_inertia_user = wp.zeros(
                    (self._num_instances, self._num_bodies, 9), dtype=wp.float32, device=self.device
                )
            if self._body_com_pos_b_user is None:
                self._body_com_pos_b_user = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
            self._validate_body_ordering_buffers()
        else:
            self._body_link_pose_w_user = None
            self._body_com_vel_w_user = None
            self._body_mass_user = None
            self._body_inertia_user = None
            self._body_com_pos_b_user = None

        reset_timestamps(
            [
                self._body_link_vel_w,
                self._body_com_pose_b,
                self._body_com_pose_w,
                self._body_com_acc_w,
                self._body_state_w,
                self._body_link_state_w,
                self._body_com_state_w,
            ]
        )

    def _make_jacobian_body_user_to_backend(self) -> wp.array:
        """Build the Newton-layout user-to-backend row map for Jacobian body axes.

        Newton's ``eval_jacobian`` keeps every link row in its output, including the
        fixed-base root row at index 0, so this map indexes that full-row layout
        directly with no shift. It therefore differs from the PhysX-style base
        implementation ``BaseArticulationData._make_jacobian_body_user_to_backend``,
        whose source omits the root row and shifts the remaining rows down by one.
        Fixed-base articulations drop the root row here (Newton pins the root at
        public index 0), so the map already encodes the link offset: the gather
        kernel needs neither a runtime link offset nor a body-ordering branch. The
        map is built even under identity ordering so the kernel can index it
        unconditionally.

        Returns:
            One-dimensional ``wp.int32`` device array of backend Jacobian rows in
            public body order.
        """
        body_ordering = self.body_ordering
        body_user_to_backend = (
            body_ordering.user_to_backend_indices if body_ordering is not None else range(self._num_bodies)
        )
        if self._jacobian_link_offset == 0:
            backend_rows = tuple(int(backend_id) for backend_id in body_user_to_backend)
        else:
            # Fixed-base: drop the root row (backend id 0, pinned at public index 0); no shift.
            backend_rows = tuple(int(backend_id) for backend_id in body_user_to_backend if int(backend_id) != 0)
        return wp.array(backend_rows, dtype=wp.int32, device=self.device)

    def _apply_ordering_maps_after_resolve(self) -> None:
        """Configure and re-pin public buffers after ordering maps are installed.

        Ordering state lives only on :attr:`joint_ordering` / :attr:`body_ordering`;
        a non-``None`` map denotes an active permutation. The staging
        (re)allocation below reconciles from those maps read directly, so an
        ordering that was cleared on rebind releases its buffers.
        """
        # Always build the row map (even under identity ordering) so the gather kernel can
        # index the body axis unconditionally -- the map owns the entire body-axis encoding.
        self._jacobian_body_user_to_backend = self._make_jacobian_body_user_to_backend()

        self._configure_joint_ordering_buffers()
        if self.has_joint_ordering:
            self._sync_user_ordered_joint_property_buffers()
        self._configure_body_ordering_buffers()
        if self.has_body_ordering:
            self._sync_user_ordered_body_property_buffers()
        self._pin_proxy_arrays()

    def _refresh_user_order_joint_state(self) -> None:
        """Reorder the live backend joint state into the user-order shadows.

        Launches :func:`ordering_kernels.reorder_2d_backend_to_user` for
        ``_sim_bind_joint_pos`` -> ``_joint_pos_user`` and ``_sim_bind_joint_vel``
        -> ``_joint_vel_user``. No-op under identity ordering (the shadows are the
        sim-bound arrays themselves).
        """
        if not self.has_joint_ordering:
            return
        for backend_data, user_data in (
            (self._sim_bind_joint_pos, self._joint_pos_user),
            (self._sim_bind_joint_vel, self._joint_vel_user),
        ):
            wp.launch(
                ordering_kernels.reorder_2d_backend_to_user,
                dim=(self._num_instances, self._num_joints),
                inputs=[backend_data, self.joint_ordering.user_to_backend],
                outputs=[user_data],
                device=self.device,
            )

    def _refresh_user_order_body_state(self) -> None:
        """Reorder the live backend body state into the user-order shadows.

        Launches :func:`ordering_kernels.reorder_2d_backend_to_user` for
        ``_sim_bind_body_link_pose_w`` -> ``_body_link_pose_w_user`` and
        ``_sim_bind_body_com_vel_w`` -> ``_body_com_vel_w_user``. No-op under
        identity ordering (the shadows are the sim-bound arrays themselves).
        """
        if not self.has_body_ordering:
            return
        for backend_data, user_data in (
            (self._sim_bind_body_link_pose_w, self._body_link_pose_w_user),
            (self._sim_bind_body_com_vel_w, self._body_com_vel_w_user),
        ):
            wp.launch(
                ordering_kernels.reorder_2d_backend_to_user,
                dim=(self._num_instances, self._num_bodies),
                inputs=[backend_data, self.body_ordering.user_to_backend],
                outputs=[user_data],
                device=self.device,
            )

    def _refresh_user_order_state(self) -> None:
        """Republish all Tier-1 user-order state shadows from live backend state.

        Registered as a post-step callback (see
        :meth:`isaaclab_newton.physics.NewtonManager.register_post_step_callback`)
        so the reorder launches land inside the stepped/captured region right after
        the last solver substep. With no Python freshness guard the launches are
        recorded into every captured graph and replayed on each tick, so the
        passthrough ``joint_pos`` / ``joint_vel`` / ``body_link_pose_w`` /
        ``body_com_vel_w`` shadows behave exactly like sim-bound memory.
        """
        self._refresh_user_order_joint_state()
        self._refresh_user_order_body_state()

    def _sync_user_ordered_joint_property_buffers(self) -> None:
        """Refresh user-order joint property buffers from sim-bound backend-order arrays.

        Only sim-owned mirrors belong here: the solver is authoritative for these
        arrays, so re-gathering them on init and rebind is always correct. This
        includes the Tier-1 joint-state shadows (``_joint_pos_user`` /
        ``_joint_vel_user``), which the solver owns; seeding them here at resolve
        and rebind is what makes the first pre-step reads valid, since the
        post-step publish in :meth:`_refresh_user_order_state` only runs inside a
        sim step. The Lab-owned actuator gain records (``_actuator_stiffness`` /
        ``_actuator_damping``) are deliberately excluded -- they are seeded once in
        :meth:`_configure_joint_ordering_buffers` and would be clobbered here on
        every rebind.

        The required public-order buffers are allocated and validated by
        :meth:`_configure_joint_ordering_buffers`, which always runs before this.
        """
        for backend_data, user_data in (
            (self._sim_bind_joint_pos, self._joint_pos_user),
            (self._sim_bind_joint_vel, self._joint_vel_user),
            (self._sim_bind_joint_stiffness_sim, self._joint_stiffness_user),
            (self._sim_bind_joint_damping_sim, self._joint_damping_user),
            (self._sim_bind_joint_armature, self._joint_armature_user),
            (self._sim_bind_joint_friction_coeff, self._joint_friction_coeff_user),
            (self._sim_bind_joint_pos_limits_lower, self._joint_pos_limits_lower_user),
            (self._sim_bind_joint_pos_limits_upper, self._joint_pos_limits_upper_user),
            (self._sim_bind_joint_vel_limits_sim, self._joint_vel_limits_user),
            (self._sim_bind_joint_effort_limits_sim, self._joint_effort_limits_user),
        ):
            wp.launch(
                ordering_kernels.reorder_2d_backend_to_user,
                dim=(self._num_instances, self._num_joints),
                inputs=[backend_data, self.joint_ordering.user_to_backend],
                outputs=[user_data],
                device=self.device,
            )

    def _sync_user_ordered_body_property_buffers(self) -> None:
        """Refresh user-order body property buffers from sim-bound backend-order arrays.

        Only sim-owned mirrors belong here; the solver is authoritative for all of
        them. The Tier-1 body-state shadows (``_body_link_pose_w_user`` /
        ``_body_com_vel_w_user``) are republished via
        :meth:`_refresh_user_order_body_state`.

        This sync also seeds the invariant the partial body-property setters rely
        on: the user buffer must stay the public-order image of the sim-bound
        backend buffer, since the setters scatter only the selected cells into
        both. A divergent pair silently corrupts the unselected cells.

        The required public-order buffers are allocated and validated by
        :meth:`_configure_body_ordering_buffers`, which always runs before this.
        """
        self._refresh_user_order_body_state()
        wp.launch(
            ordering_kernels.reorder_2d_backend_to_user,
            dim=(self._num_instances, self._num_bodies),
            inputs=[self._sim_bind_body_mass, self.body_ordering.user_to_backend],
            outputs=[self._body_mass_user],
            device=self.device,
        )
        wp.launch(
            ordering_kernels.reorder_3d_backend_to_user,
            dim=(self._num_instances, self._num_bodies, 9),
            inputs=[self._sim_bind_body_inertia, self.body_ordering.user_to_backend],
            outputs=[self._body_inertia_user],
            device=self.device,
        )
        wp.launch(
            ordering_kernels.reorder_2d_backend_to_user,
            dim=(self._num_instances, self._num_bodies),
            inputs=[self._sim_bind_body_com_pos_b, self.body_ordering.user_to_backend],
            outputs=[self._body_com_pos_b_user],
            device=self.device,
        )

    def _pin_proxy_arrays(self) -> None:
        """Create or rebind all pinned ProxyArray wrappers.

        Called from :meth:`_create_buffers` on first initialization and from
        :meth:`_create_simulation_bindings` after a full simulation reset when
        the solver recreates its internal arrays. The public-order buffers this
        pins are allocated and validated by the ``_configure_*_ordering_buffers``
        methods, which run at ordering-resolve time before any pin or rebind.
        """
        is_rebind = hasattr(self, "_root_link_pose_w_ta")
        if is_rebind and self.has_joint_ordering:
            self._sync_user_ordered_joint_property_buffers()
        if is_rebind and self.has_body_ordering:
            self._sync_user_ordered_body_property_buffers()
        if is_rebind:
            self._joint_pos_limits_timestamp = -1.0

        joint_stiffness = self._joint_stiffness_user if self.has_joint_ordering else self._sim_bind_joint_stiffness_sim
        joint_damping = self._joint_damping_user if self.has_joint_ordering else self._sim_bind_joint_damping_sim
        joint_armature = self._joint_armature_user if self.has_joint_ordering else self._sim_bind_joint_armature
        joint_friction_coeff = (
            self._joint_friction_coeff_user if self.has_joint_ordering else self._sim_bind_joint_friction_coeff
        )
        joint_pos_limits_lower = (
            self._joint_pos_limits_lower_user if self.has_joint_ordering else self._sim_bind_joint_pos_limits_lower
        )
        joint_pos_limits_upper = (
            self._joint_pos_limits_upper_user if self.has_joint_ordering else self._sim_bind_joint_pos_limits_upper
        )
        joint_vel_limits = (
            self._joint_vel_limits_user if self.has_joint_ordering else self._sim_bind_joint_vel_limits_sim
        )
        joint_effort_limits = (
            self._joint_effort_limits_user if self.has_joint_ordering else self._sim_bind_joint_effort_limits_sim
        )

        if is_rebind:
            # Rebind sim-bound ProxyArrays to new solver arrays
            self._root_link_pose_w_ta = ProxyArray(self._sim_bind_root_link_pose_w)
            self._root_com_vel_w_ta = ProxyArray(self._sim_bind_root_com_vel_w)
            body_link_pose_w = (
                self._body_link_pose_w_user if self.has_body_ordering else self._sim_bind_body_link_pose_w
            )
            body_com_vel_w = self._body_com_vel_w_user if self.has_body_ordering else self._sim_bind_body_com_vel_w
            joint_pos = self._joint_pos_user if self.has_joint_ordering else self._sim_bind_joint_pos
            joint_vel = self._joint_vel_user if self.has_joint_ordering else self._sim_bind_joint_vel
            self._body_link_pose_w_ta = ProxyArray(body_link_pose_w)
            self._body_com_vel_w_ta = ProxyArray(body_com_vel_w)
            self._joint_pos_ta = ProxyArray(joint_pos)
            self._joint_vel_ta = ProxyArray(joint_vel)
            self._joint_stiffness_ta = ProxyArray(joint_stiffness)
            self._joint_damping_ta = ProxyArray(joint_damping)
            self._joint_armature_ta = ProxyArray(joint_armature)
            self._joint_friction_coeff_ta = ProxyArray(joint_friction_coeff)
            self._joint_pos_limits_lower_ta = ProxyArray(joint_pos_limits_lower)
            self._joint_pos_limits_upper_ta = ProxyArray(joint_pos_limits_upper)
            self._joint_vel_limits_ta = ProxyArray(joint_vel_limits)
            self._joint_effort_limits_ta = ProxyArray(joint_effort_limits)
            body_mass = self._body_mass_user if self.has_body_ordering else self._sim_bind_body_mass
            body_inertia = self._body_inertia_user if self.has_body_ordering else self._sim_bind_body_inertia
            body_com_pos_b = self._body_com_pos_b_user if self.has_body_ordering else self._sim_bind_body_com_pos_b
            self._body_mass_ta = ProxyArray(body_mass)
            self._body_inertia_ta = ProxyArray(body_inertia)
            self._body_com_pos_b_ta = ProxyArray(body_com_pos_b)
        else:
            # First-time creation: pin ProxyArrays to current buffers
            # Category 1: sim-bound and pre-allocated buffers
            # Sim-bound pointers are re-created on full reset; _create_simulation_bindings()
            # calls rebind() on each ProxyArray to keep them in sync.
            self._root_link_pose_w_ta = ProxyArray(self._sim_bind_root_link_pose_w)
            self._root_com_vel_w_ta = ProxyArray(self._sim_bind_root_com_vel_w)
            body_link_pose_w = (
                self._body_link_pose_w_user if self.has_body_ordering else self._sim_bind_body_link_pose_w
            )
            body_com_vel_w = self._body_com_vel_w_user if self.has_body_ordering else self._sim_bind_body_com_vel_w
            joint_pos = self._joint_pos_user if self.has_joint_ordering else self._sim_bind_joint_pos
            joint_vel = self._joint_vel_user if self.has_joint_ordering else self._sim_bind_joint_vel
            self._body_link_pose_w_ta = ProxyArray(body_link_pose_w)
            self._body_com_vel_w_ta = ProxyArray(body_com_vel_w)
            self._joint_pos_ta = ProxyArray(joint_pos)
            self._joint_vel_ta = ProxyArray(joint_vel)
            self._default_root_pose_ta = ProxyArray(self._default_root_pose)
            self._default_root_vel_ta = ProxyArray(self._default_root_vel)
            self._default_joint_pos_ta = ProxyArray(self._default_joint_pos)
            self._default_joint_vel_ta = ProxyArray(self._default_joint_vel)
            self._joint_pos_target_ta = ProxyArray(self._joint_pos_target)
            self._joint_vel_target_ta = ProxyArray(self._joint_vel_target)
            self._joint_effort_target_ta = ProxyArray(self._joint_effort_target)
            self._computed_torque_ta = ProxyArray(self._computed_torque)
            self._applied_torque_ta = ProxyArray(self._applied_torque)
            self._joint_stiffness_ta = ProxyArray(joint_stiffness)
            self._joint_damping_ta = ProxyArray(joint_damping)
            self._joint_armature_ta = ProxyArray(joint_armature)
            self._joint_friction_coeff_ta = ProxyArray(joint_friction_coeff)
            self._joint_pos_limits_lower_ta = ProxyArray(joint_pos_limits_lower)
            self._joint_pos_limits_upper_ta = ProxyArray(joint_pos_limits_upper)
            self._joint_vel_limits_ta = ProxyArray(joint_vel_limits)
            self._joint_effort_limits_ta = ProxyArray(joint_effort_limits)
            self._soft_joint_pos_limits_ta = ProxyArray(self._soft_joint_pos_limits)
            self._soft_joint_vel_limits_ta = ProxyArray(self._soft_joint_vel_limits)
            self._gear_ratio_ta = ProxyArray(self._gear_ratio)
            body_mass = self._body_mass_user if self.has_body_ordering else self._sim_bind_body_mass
            body_inertia = self._body_inertia_user if self.has_body_ordering else self._sim_bind_body_inertia
            body_com_pos_b = self._body_com_pos_b_user if self.has_body_ordering else self._sim_bind_body_com_pos_b
            self._body_mass_ta = ProxyArray(body_mass)
            self._body_inertia_ta = ProxyArray(body_inertia)
            self._body_com_pos_b_ta = ProxyArray(body_com_pos_b)
            self._fixed_tendon_stiffness_ta = ProxyArray(self._sim_bind_fixed_tendon_stiffness)
            self._fixed_tendon_damping_ta = ProxyArray(self._sim_bind_fixed_tendon_damping)

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
            self._body_com_jacobian_w_ta = ProxyArray(self._body_com_jacobian_w_buf)
            self._body_link_jacobian_w_ta = ProxyArray(self._body_link_jacobian_w_buf)
            self._mass_matrix_ta = ProxyArray(self._mass_matrix_buf)

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
        self._joint_pos_limits_timestamp = -1.0
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
