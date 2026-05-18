# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import warp as wp

from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.warp import ProxyArray

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.assets.kernels import (
    _compose_root_com_pose,
    _compute_heading,
    _copy_first_body,
    _projected_gravity,
    _world_vel_to_body_ang,
    _world_vel_to_body_lin,
    concat_body_pose_and_vel_to_state,
    concat_root_pose_and_vel_to_state,
    get_body_com_pose_from_body_link_pose,
    get_body_link_vel_from_body_com_vel,
    vec13f,
)
from isaaclab_ovphysx.physics import OvPhysxManager

from .kernels import _fd_joint_acc

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
        **Pull-to-refresh model.** OVPhysX state properties are *not* automatically updated each
        simulation step. Each property getter pulls fresh data from the OVPhysX ``TensorBinding``
        on first access per timestamp, then caches the result until the next step. This differs
        from the Newton backend, where buffers are refreshed automatically by the simulation.

    .. note::
        **CPU-only bindings.** OVPhysX exposes a subset of bindings (``BODY_MASS``, ``BODY_COM_POSE``,
        ``BODY_INERTIA``, and most ``DOF_*`` property bindings) on CPU only. These are routed through
        pinned-host staging buffers via :meth:`_binding_read` so that GPU-resident consumers see the
        data without per-step host allocations.
    """

    __backend_name__: str = "ovphysx"
    """The name of the backend for the articulation data."""

    def __init__(self, bindings: dict[int, Any], device: str) -> None:
        """Initialize the articulation data container.

        Args:
            bindings: Dictionary of OVPhysX :class:`TensorBinding` objects keyed
                by :class:`isaaclab_ovphysx.tensor_types.TensorType`. All counts
                (instances, bodies, DOFs, fixed/spatial tendons) are derived
                from the binding metadata. Name lists are assigned by
                :meth:`~isaaclab_ovphysx.assets.Articulation._initialize_impl`
                after construction.
            device: Simulation device string (e.g., ``"cuda:0"`` or ``"cpu"``).
        """
        super().__init__(root_view=None, device=device)
        self._bindings = bindings

        # Every OVPhysX TensorBinding carries the articulation metadata
        # (instance count, dof_count, body_count, fixed/spatial tendon counts);
        # any binding will do for the read.
        sample = next(iter(bindings.values()))
        self.num_instances = sample.count
        self.num_bodies = sample.body_count
        self.num_joints = sample.dof_count
        self.num_fixed_tendons = getattr(sample, "fixed_tendon_count", 0)
        self.num_spatial_tendons = getattr(sample, "spatial_tendon_count", 0)
        # private aliases used throughout _create_buffers and property bodies
        self._num_instances = self.num_instances
        self._num_bodies = self.num_bodies
        self._num_joints = self.num_joints
        self._num_fixed_tendons = self.num_fixed_tendons
        self._num_spatial_tendons = self.num_spatial_tendons

        # Set initial time stamp
        self._sim_timestamp: float = 0.0
        self._is_primed: bool = False
        # pinned-host staging buffers for CPU-only bindings (keyed by tensor_type)
        self._cpu_staging_buffers: dict[int, wp.array] = {}
        # scratch buffers for _get_read_view cache (keyed by (tensor_type, ptr))
        self._read_scratch: dict = {}

        # obtain gravity from the simulation configuration (fall back to standard
        # gravity when the simulation has not been configured yet, e.g. in unit tests)
        gravity = (0.0, 0.0, -9.81)
        from isaaclab.physics import PhysicsManager

        if PhysicsManager._sim is not None and hasattr(PhysicsManager._sim, "cfg"):
            gravity = PhysicsManager._sim.cfg.gravity
        gravity_np = np.array(gravity, dtype=np.float32)
        gravity_mag = float(np.linalg.norm(gravity_np))
        if gravity_mag == 0.0:
            gravity_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            gravity_dir = gravity_np / gravity_mag
        gravity_dir_tiled = np.tile(gravity_dir, (self._num_instances, 1))
        forward_tiled = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (self._num_instances, 1))

        # Initialize constants
        self.GRAVITY_VEC_W = ProxyArray(wp.from_numpy(gravity_dir_tiled, dtype=wp.vec3f, device=device))
        self.FORWARD_VEC_B = ProxyArray(wp.from_numpy(forward_tiled, dtype=wp.vec3f, device=device))

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
        if not self._is_primed:
            return
        # trigger an update of the joint acceleration buffer via finite differencing
        if dt > 0.0 and self._previous_joint_vel is not None:
            cur_vel_buf = self._joint_vel_buf
            # ensure joint vel buffer is fresh before differencing
            self._read_binding_into_buf(TT.DOF_VELOCITY, cur_vel_buf)
            wp.launch(
                _fd_joint_acc,
                dim=(self._num_instances, self._num_joints),
                inputs=[cur_vel_buf.data, self._previous_joint_vel, 1.0 / dt],
                outputs=[self._joint_acc.data],
                device=self.device,
            )
            self._joint_acc.timestamp = self._sim_timestamp
            wp.copy(self._previous_joint_vel, cur_vel_buf.data)

    """
    Names.
    """

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    joint_names: list[str] = None
    """Joint names in the order parsed by the simulation view."""

    fixed_tendon_names: list[str] = None
    """Fixed tendon names in the order parsed by USD."""

    spatial_tendon_names: list[str] = None
    """Spatial tendon names in the order parsed by USD."""

    """
    Defaults - Initial state.
    """

    @property
    def default_root_pose(self) -> ProxyArray:
        """Default root pose ``[pos, quat]`` in local environment frame [m, -].

        Shape is (num_instances,), dtype = wp.transformf.
        In torch this resolves to (num_instances, 7).

        Populated from :attr:`ArticulationCfg.init_state` during initialisation.
        """
        if self._default_root_pose_ta is None:
            self._default_root_pose_ta = ProxyArray(self._default_root_pose)
        return self._default_root_pose_ta

    @default_root_pose.setter
    def default_root_pose(self, value: wp.array) -> None:
        """Set the default root pose.

        Args:
            value: The default root pose, shape (num_instances, 7).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self._is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_root_pose.assign(value)

    @property
    def default_root_vel(self) -> ProxyArray:
        """Default root velocity ``[lin_vel, ang_vel]`` in local environment frame [m/s, rad/s].

        Shape is (num_instances,), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, 6).

        Populated from :attr:`ArticulationCfg.init_state` during initialisation.
        """
        if self._default_root_vel_ta is None:
            self._default_root_vel_ta = ProxyArray(self._default_root_vel)
        return self._default_root_vel_ta

    @default_root_vel.setter
    def default_root_vel(self, value: wp.array) -> None:
        """Set the default root velocity.

        Args:
            value: The default root velocity, shape (num_instances, 6).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self._is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_root_vel.assign(value)

    @property
    def default_joint_pos(self) -> ProxyArray:
        """Default joint positions of all joints [m or rad, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        """
        if self._default_joint_pos_ta is None:
            self._default_joint_pos_ta = ProxyArray(self._default_joint_pos)
        return self._default_joint_pos_ta

    @default_joint_pos.setter
    def default_joint_pos(self, value: wp.array) -> None:
        """Set the default joint positions.

        Args:
            value: The default joint positions, shape (num_instances, num_joints).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self._is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_joint_pos.assign(value)

    @property
    def default_joint_vel(self) -> ProxyArray:
        """Default joint velocities of all joints [m/s or rad/s, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        """
        if self._default_joint_vel_ta is None:
            self._default_joint_vel_ta = ProxyArray(self._default_joint_vel)
        return self._default_joint_vel_ta

    @default_joint_vel.setter
    def default_joint_vel(self, value: wp.array) -> None:
        """Set the default joint velocities.

        Args:
            value: The default joint velocities, shape (num_instances, num_joints).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self._is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_joint_vel.assign(value)

    """
    Joint commands -- Set into simulation.
    """

    @property
    def joint_pos_target(self) -> ProxyArray:
        """Joint position targets commanded by the user [m or rad, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        """
        if self._joint_pos_target_ta is None:
            self._joint_pos_target_ta = ProxyArray(self._joint_pos_target)
        return self._joint_pos_target_ta

    @property
    def joint_vel_target(self) -> ProxyArray:
        """Joint velocity targets commanded by the user [m/s or rad/s, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        """
        if self._joint_vel_target_ta is None:
            self._joint_vel_target_ta = ProxyArray(self._joint_vel_target)
        return self._joint_vel_target_ta

    @property
    def joint_effort_target(self) -> ProxyArray:
        """Joint effort targets commanded by the user [N or N*m, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        """
        if self._joint_effort_target_ta is None:
            self._joint_effort_target_ta = ProxyArray(self._joint_effort_target)
        return self._joint_effort_target_ta

    """
    Joint commands -- Explicit actuators.
    """

    @property
    def computed_torque(self) -> ProxyArray:
        """Joint torques computed from the actuator model (before clipping) [N*m].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        """
        if self._computed_torque_ta is None:
            self._computed_torque_ta = ProxyArray(self._computed_torque)
        return self._computed_torque_ta

    @property
    def applied_torque(self) -> ProxyArray:
        """Joint torques applied from the actuator model (after clipping) [N*m].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        """
        if self._applied_torque_ta is None:
            self._applied_torque_ta = ProxyArray(self._applied_torque)
        return self._applied_torque_ta

    """
    Joint properties
    """

    @property
    def joint_stiffness(self) -> ProxyArray:
        """Joint stiffness provided to the simulation [N*m/rad or N/m, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.

        Routed through pinned-host staging because ``DOF_STIFFNESS`` is a
        CPU-only OVPhysX binding.
        """
        self._read_scalar_binding(TT.DOF_STIFFNESS, self._joint_stiffness)
        if self._joint_stiffness_ta is None:
            self._joint_stiffness_ta = ProxyArray(self._joint_stiffness.data)
        return self._joint_stiffness_ta

    @property
    def joint_damping(self) -> ProxyArray:
        """Joint damping provided to the simulation [N*m*s/rad or N*s/m, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.

        Routed through pinned-host staging because ``DOF_DAMPING`` is a
        CPU-only OVPhysX binding.
        """
        self._read_scalar_binding(TT.DOF_DAMPING, self._joint_damping)
        if self._joint_damping_ta is None:
            self._joint_damping_ta = ProxyArray(self._joint_damping.data)
        return self._joint_damping_ta

    @property
    def joint_armature(self) -> ProxyArray:
        """Joint armature provided to the simulation [kg*m^2].

        Shape is (num_instances, num_joints), dtype = wp.float32.

        Routed through pinned-host staging because ``DOF_ARMATURE`` is a
        CPU-only OVPhysX binding.
        """
        self._read_scalar_binding(TT.DOF_ARMATURE, self._joint_armature)
        if self._joint_armature_ta is None:
            self._joint_armature_ta = ProxyArray(self._joint_armature.data)
        return self._joint_armature_ta

    @property
    def joint_friction_coeff(self) -> ProxyArray:
        """Joint static friction coefficient [dimensionless].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        Component ``[..., 0]`` of the ``DOF_FRICTION_PROPERTIES`` binding.

        Routed through pinned-host staging because ``DOF_FRICTION_PROPERTIES``
        is a CPU-only OVPhysX binding.
        """
        self._read_scalar_binding(TT.DOF_FRICTION_PROPERTIES, self._joint_friction_props_buf)
        if self._joint_friction_coeff_ta is None:
            self._joint_friction_coeff_ta = ProxyArray(self._joint_friction_coeff)
        return self._joint_friction_coeff_ta

    @property
    def joint_dynamic_friction_coeff(self) -> ProxyArray:
        """Joint dynamic friction coefficient [dimensionless].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        Component ``[..., 1]`` of the ``DOF_FRICTION_PROPERTIES`` binding.

        Routed through pinned-host staging because ``DOF_FRICTION_PROPERTIES``
        is a CPU-only OVPhysX binding.
        """
        self._read_scalar_binding(TT.DOF_FRICTION_PROPERTIES, self._joint_friction_props_buf)
        if self._joint_dynamic_friction_coeff_ta is None:
            self._joint_dynamic_friction_coeff_ta = ProxyArray(self._joint_dynamic_friction_coeff)
        return self._joint_dynamic_friction_coeff_ta

    @property
    def joint_viscous_friction_coeff(self) -> ProxyArray:
        """Joint viscous friction coefficient [N*m*s/rad or N*s/m, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        Component ``[..., 2]`` of the ``DOF_FRICTION_PROPERTIES`` binding.

        Routed through pinned-host staging because ``DOF_FRICTION_PROPERTIES``
        is a CPU-only OVPhysX binding.
        """
        self._read_scalar_binding(TT.DOF_FRICTION_PROPERTIES, self._joint_friction_props_buf)
        if self._joint_viscous_friction_coeff_ta is None:
            self._joint_viscous_friction_coeff_ta = ProxyArray(self._joint_viscous_friction_coeff)
        return self._joint_viscous_friction_coeff_ta

    @property
    def joint_pos_limits(self) -> ProxyArray:
        """Joint position limits provided to the simulation [m or rad, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.vec2f.
        In torch this resolves to (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.  Routed through
        pinned-host staging because ``DOF_LIMIT`` is a CPU-only OVPhysX binding.
        """
        self._read_scalar_binding(TT.DOF_LIMIT, self._joint_pos_limits)
        if self._joint_pos_limits_ta is None:
            self._joint_pos_limits_ta = ProxyArray(self._joint_pos_limits.data)
        return self._joint_pos_limits_ta

    @property
    def joint_vel_limits(self) -> ProxyArray:
        """Joint maximum velocity provided to the simulation [m/s or rad/s, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.

        Routed through pinned-host staging because ``DOF_MAX_VELOCITY`` is a
        CPU-only OVPhysX binding.
        """
        self._read_scalar_binding(TT.DOF_MAX_VELOCITY, self._joint_vel_limits)
        if self._joint_vel_limits_ta is None:
            self._joint_vel_limits_ta = ProxyArray(self._joint_vel_limits.data)
        return self._joint_vel_limits_ta

    @property
    def joint_effort_limits(self) -> ProxyArray:
        """Joint maximum effort provided to the simulation [N or N*m, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.

        Routed through pinned-host staging because ``DOF_MAX_FORCE`` is a
        CPU-only OVPhysX binding.
        """
        self._read_scalar_binding(TT.DOF_MAX_FORCE, self._joint_effort_limits)
        if self._joint_effort_limits_ta is None:
            self._joint_effort_limits_ta = ProxyArray(self._joint_effort_limits.data)
        return self._joint_effort_limits_ta

    """
    Joint properties - Custom.
    """

    @property
    def soft_joint_pos_limits(self) -> ProxyArray:
        r"""Soft joint position limits for all joints [m or rad, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.vec2f.
        In torch this resolves to (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        if self._soft_joint_pos_limits_ta is None:
            self._soft_joint_pos_limits_ta = ProxyArray(self._soft_joint_pos_limits)
        return self._soft_joint_pos_limits_ta

    @property
    def soft_joint_vel_limits(self) -> ProxyArray:
        """Soft joint velocity limits for all joints [m/s or rad/s, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        """
        if self._soft_joint_vel_limits_ta is None:
            self._soft_joint_vel_limits_ta = ProxyArray(self._soft_joint_vel_limits)
        return self._soft_joint_vel_limits_ta

    @property
    def gear_ratio(self) -> ProxyArray:
        """Gear ratio for relating motor torques to applied joint torques.

        Shape is (num_instances, num_joints), dtype = wp.float32.
        """
        if self._gear_ratio_ta is None:
            self._gear_ratio_ta = ProxyArray(self._gear_ratio)
        return self._gear_ratio_ta

    """
    Fixed tendon properties.
    """

    @property
    def fixed_tendon_stiffness(self) -> ProxyArray:
        """Fixed-tendon stiffness gains [N*m/rad].

        Shape is (num_instances, num_fixed_tendons), dtype = ``wp.float32``.

        Routed through pinned-host staging (CPU-only binding).
        """
        self._read_scalar_binding(TT.FIXED_TENDON_STIFFNESS, self._fixed_tendon_stiffness)
        if self._fixed_tendon_stiffness_ta is None:
            self._fixed_tendon_stiffness_ta = ProxyArray(self._fixed_tendon_stiffness.data)
        return self._fixed_tendon_stiffness_ta

    @property
    def fixed_tendon_damping(self) -> ProxyArray:
        """Fixed-tendon damping coefficients [N*m*s/rad].

        Shape is (num_instances, num_fixed_tendons), dtype = ``wp.float32``.

        Routed through pinned-host staging (CPU-only binding).
        """
        self._read_scalar_binding(TT.FIXED_TENDON_DAMPING, self._fixed_tendon_damping)
        if self._fixed_tendon_damping_ta is None:
            self._fixed_tendon_damping_ta = ProxyArray(self._fixed_tendon_damping.data)
        return self._fixed_tendon_damping_ta

    @property
    def fixed_tendon_limit_stiffness(self) -> ProxyArray:
        """Fixed-tendon limit stiffness [N*m/rad].

        Shape is (num_instances, num_fixed_tendons), dtype = ``wp.float32``.

        Routed through pinned-host staging (CPU-only binding).
        """
        self._read_scalar_binding(TT.FIXED_TENDON_LIMIT_STIFFNESS, self._fixed_tendon_limit_stiffness)
        if self._fixed_tendon_limit_stiffness_ta is None:
            self._fixed_tendon_limit_stiffness_ta = ProxyArray(self._fixed_tendon_limit_stiffness.data)
        return self._fixed_tendon_limit_stiffness_ta

    @property
    def fixed_tendon_rest_length(self) -> ProxyArray:
        """Fixed-tendon rest lengths [m].

        Shape is (num_instances, num_fixed_tendons), dtype = ``wp.float32``.

        Routed through pinned-host staging (CPU-only binding).
        """
        self._read_scalar_binding(TT.FIXED_TENDON_REST_LENGTH, self._fixed_tendon_rest_length)
        if self._fixed_tendon_rest_length_ta is None:
            self._fixed_tendon_rest_length_ta = ProxyArray(self._fixed_tendon_rest_length.data)
        return self._fixed_tendon_rest_length_ta

    @property
    def fixed_tendon_offset(self) -> ProxyArray:
        """Fixed-tendon offsets [m].

        Shape is (num_instances, num_fixed_tendons), dtype = ``wp.float32``.

        Routed through pinned-host staging (CPU-only binding).
        """
        self._read_scalar_binding(TT.FIXED_TENDON_OFFSET, self._fixed_tendon_offset)
        if self._fixed_tendon_offset_ta is None:
            self._fixed_tendon_offset_ta = ProxyArray(self._fixed_tendon_offset.data)
        return self._fixed_tendon_offset_ta

    @property
    def fixed_tendon_pos_limits(self) -> ProxyArray:
        """Fixed tendon position limits provided to the simulation [m or rad].

        Shape is (num_instances, num_fixed_tendons), dtype = ``wp.vec2f``.
        In torch this resolves to (num_instances, num_fixed_tendons, 2).

        .. deprecated::
            Use :attr:`fixed_tendon_limit` (shape ``(N, T, 2)``, dtype
            ``wp.float32``) instead.  This alias is kept for backwards
            compatibility and reads the same underlying data.
        """
        self._read_scalar_binding(TT.FIXED_TENDON_LIMIT, self._fixed_tendon_pos_limits)
        if self._fixed_tendon_pos_limits_ta is None:
            self._fixed_tendon_pos_limits_ta = ProxyArray(self._fixed_tendon_pos_limits.data)
        return self._fixed_tendon_pos_limits_ta

    """
    Spatial tendon properties.
    """

    @property
    def spatial_tendon_stiffness(self) -> ProxyArray:
        """Spatial-tendon stiffness gains [N/m].

        Shape is (num_instances, num_spatial_tendons), dtype = ``wp.float32``.

        Routed through pinned-host staging (CPU-only binding).
        """
        self._read_scalar_binding(TT.SPATIAL_TENDON_STIFFNESS, self._spatial_tendon_stiffness)
        if self._spatial_tendon_stiffness_ta is None:
            self._spatial_tendon_stiffness_ta = ProxyArray(self._spatial_tendon_stiffness.data)
        return self._spatial_tendon_stiffness_ta

    @property
    def spatial_tendon_damping(self) -> ProxyArray:
        """Spatial-tendon damping coefficients [N*s/m].

        Shape is (num_instances, num_spatial_tendons), dtype = ``wp.float32``.

        Routed through pinned-host staging (CPU-only binding).
        """
        self._read_scalar_binding(TT.SPATIAL_TENDON_DAMPING, self._spatial_tendon_damping)
        if self._spatial_tendon_damping_ta is None:
            self._spatial_tendon_damping_ta = ProxyArray(self._spatial_tendon_damping.data)
        return self._spatial_tendon_damping_ta

    @property
    def spatial_tendon_limit_stiffness(self) -> ProxyArray:
        """Spatial-tendon limit stiffness [N/m].

        Shape is (num_instances, num_spatial_tendons), dtype = ``wp.float32``.

        Routed through pinned-host staging (CPU-only binding).
        """
        self._read_scalar_binding(TT.SPATIAL_TENDON_LIMIT_STIFFNESS, self._spatial_tendon_limit_stiffness)
        if self._spatial_tendon_limit_stiffness_ta is None:
            self._spatial_tendon_limit_stiffness_ta = ProxyArray(self._spatial_tendon_limit_stiffness.data)
        return self._spatial_tendon_limit_stiffness_ta

    @property
    def spatial_tendon_offset(self) -> ProxyArray:
        """Spatial-tendon offsets [m].

        Shape is (num_instances, num_spatial_tendons), dtype = ``wp.float32``.

        Routed through pinned-host staging (CPU-only binding).
        """
        self._read_scalar_binding(TT.SPATIAL_TENDON_OFFSET, self._spatial_tendon_offset)
        if self._spatial_tendon_offset_ta is None:
            self._spatial_tendon_offset_ta = ProxyArray(self._spatial_tendon_offset.data)
        return self._spatial_tendon_offset_ta

    """
    Root state properties.
    """

    @property
    def root_link_pose_w(self) -> ProxyArray:
        """Root link pose ``[pos, quat]`` in simulation world frame [m, -].

        Shape is (num_instances,), dtype = wp.transformf.
        In torch this resolves to (num_instances, 7).

        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        self._read_transform_binding(TT.ROOT_POSE, self._root_link_pose_w)
        if self._root_link_pose_w_ta is None:
            self._root_link_pose_w_ta = ProxyArray(self._root_link_pose_w.data)
        return self._root_link_pose_w_ta

    @property
    def root_pose_w(self) -> ProxyArray:
        """Alias for :attr:`root_link_pose_w` matching Newton's convention.

        Shape is (num_instances,), dtype = wp.transformf.
        In torch this resolves to (num_instances, 7).
        """
        return self.root_link_pose_w

    @property
    def root_link_vel_w(self) -> ProxyArray:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].

        Shape is (num_instances,), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        # ovphysx ROOT_VELOCITY is COM velocity; link velocity comes from the first
        # element of the per-link velocity tensor.
        self._read_spatial_vector_binding(TT.LINK_VELOCITY, self._body_link_vel_w)
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                _copy_first_body,
                dim=self.num_instances,
                inputs=[self._body_link_vel_w.data],
                outputs=[self._root_link_vel_w.data],
                device=self.device,
            )
            self._root_link_vel_w.timestamp = self._sim_timestamp
        if self._root_link_vel_w_ta is None:
            self._root_link_vel_w_ta = ProxyArray(self._root_link_vel_w.data)
        return self._root_link_vel_w_ta

    @property
    def root_com_pose_w(self) -> ProxyArray:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame [m, -].

        Shape is (num_instances,), dtype = wp.transformf.
        In torch this resolves to (num_instances, 7).

        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            wp.launch(
                _compose_root_com_pose,
                dim=self.num_instances,
                inputs=[self.root_link_pose_w, self.body_com_pose_b],
                outputs=[self._root_com_pose_w.data],
                device=self.device,
            )
            self._root_com_pose_w.timestamp = self._sim_timestamp
        if self._root_com_pose_w_ta is None:
            self._root_com_pose_w_ta = ProxyArray(self._root_com_pose_w.data)
        return self._root_com_pose_w_ta

    @property
    def root_com_vel_w(self) -> ProxyArray:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].

        Shape is (num_instances,), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        self._read_spatial_vector_binding(TT.ROOT_VELOCITY, self._root_com_vel_w)
        if self._root_com_vel_w_ta is None:
            self._root_com_vel_w_ta = ProxyArray(self._root_com_vel_w.data)
        return self._root_com_vel_w_ta

    """
    Body state properties.
    """

    @property
    def body_mass(self) -> ProxyArray:
        """Body masses [kg].

        Shape is (num_instances, num_bodies), dtype = ``wp.float32``.

        Routed through pinned-host staging because the underlying OVPhysX
        binding is CPU-only (``ARTICULATION_BODY_MASS``).
        """
        self._read_scalar_binding(TT.BODY_MASS, self._body_mass)
        if self._body_mass_ta is None:
            self._body_mass_ta = ProxyArray(self._body_mass.data)
        return self._body_mass_ta

    @property
    def body_inertia(self) -> ProxyArray:
        """Body inertia tensors [kg*m^2].

        Shape is (num_instances, num_bodies, 9), dtype = ``wp.float32``; the
        trailing 9 is the row-major 3×3 inertia tensor.

        Routed through pinned-host staging (``ARTICULATION_BODY_INERTIA`` is
        a CPU-only binding).
        """
        self._read_scalar_binding(TT.BODY_INERTIA, self._body_inertia)
        if self._body_inertia_ta is None:
            self._body_inertia_ta = ProxyArray(self._body_inertia.data)
        return self._body_inertia_ta

    @property
    def body_link_pose_w(self) -> ProxyArray:
        """Body link pose ``[pos, quat]`` in simulation world frame [m, -].

        Shape is (num_instances, num_bodies), dtype = wp.transformf.
        In torch this resolves to (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_link_pose_w.timestamp < self._sim_timestamp:
            # perform forward kinematics (shouldn't cause overhead if it happened already);
            # skip when no physics instance is bound (mocked iface tests)
            physx_instance = OvPhysxManager.get_physx_instance()
            if physx_instance is not None:
                physx_instance.update_articulations_kinematic()
        self._read_transform_binding(TT.LINK_POSE, self._body_link_pose_w)
        if self._body_link_pose_w_ta is None:
            self._body_link_pose_w_ta = ProxyArray(self._body_link_pose_w.data)
        return self._body_link_pose_w_ta

    @property
    def body_com_vel_w(self) -> ProxyArray:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, num_bodies, 6).
        """
        self._read_spatial_vector_binding(TT.LINK_VELOCITY, self._body_com_vel_w)
        if self._body_com_vel_w_ta is None:
            self._body_com_vel_w_ta = ProxyArray(self._body_com_vel_w.data)
        return self._body_com_vel_w_ta

    @property
    def body_link_vel_w(self) -> ProxyArray:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, num_bodies, 6).

        Derived from :attr:`body_com_vel_w` and :attr:`body_com_pose_b` via
        :func:`~isaaclab_ovphysx.assets.kernels.get_body_link_vel_from_body_com_vel`.
        """
        if self._body_link_vel_w.timestamp >= self._sim_timestamp:
            if self._body_link_vel_w_ta is None:
                self._body_link_vel_w_ta = ProxyArray(self._body_link_vel_w.data)
            return self._body_link_vel_w_ta
        _ = self.body_com_vel_w
        _ = self.body_link_pose_w
        _ = self.body_com_pose_b
        wp.launch(
            get_body_link_vel_from_body_com_vel,
            dim=(self.num_instances, self.num_bodies),
            inputs=[self._body_com_vel_w.data, self._body_link_pose_w.data, self._body_com_pose_b.data],
            outputs=[self._body_link_vel_w.data],
            device=self.device,
        )
        self._body_link_vel_w.timestamp = self._sim_timestamp
        if self._body_link_vel_w_ta is None:
            self._body_link_vel_w_ta = ProxyArray(self._body_link_vel_w.data)
        return self._body_link_vel_w_ta

    @property
    def body_com_pose_w(self) -> ProxyArray:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame [m, -].

        Shape is (num_instances, num_bodies), dtype = wp.transformf.
        In torch this resolves to (num_instances, num_bodies, 7).

        Derived from :attr:`body_link_pose_w` and :attr:`body_com_pose_b` via
        :func:`~isaaclab_ovphysx.assets.kernels.get_body_com_pose_from_body_link_pose`.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_w.timestamp >= self._sim_timestamp:
            if self._body_com_pose_w_ta is None:
                self._body_com_pose_w_ta = ProxyArray(self._body_com_pose_w.data)
            return self._body_com_pose_w_ta
        _ = self.body_link_pose_w
        _ = self.body_com_pose_b
        wp.launch(
            get_body_com_pose_from_body_link_pose,
            dim=(self.num_instances, self.num_bodies),
            inputs=[self._body_link_pose_w.data, self._body_com_pose_b.data],
            outputs=[self._body_com_pose_w.data],
            device=self.device,
        )
        self._body_com_pose_w.timestamp = self._sim_timestamp
        if self._body_com_pose_w_ta is None:
            self._body_com_pose_w_ta = ProxyArray(self._body_com_pose_w.data)
        return self._body_com_pose_w_ta

    @property
    def body_com_acc_w(self) -> ProxyArray:
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]`` [m/s^2, rad/s^2].

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        self._read_spatial_vector_binding(TT.LINK_ACCELERATION, self._body_com_acc_w)
        if self._body_com_acc_w_ta is None:
            self._body_com_acc_w_ta = ProxyArray(self._body_com_acc_w.data)
        return self._body_com_acc_w_ta

    @property
    def body_com_pose_b(self) -> ProxyArray:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames [m, -].

        Shape is (num_instances, num_bodies), dtype = wp.transformf.
        In torch this resolves to (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        self._read_transform_binding(TT.BODY_COM_POSE, self._body_com_pose_b)
        if self._body_com_pose_b_ta is None:
            self._body_com_pose_b_ta = ProxyArray(self._body_com_pose_b.data)
        return self._body_com_pose_b_ta

    @property
    def body_incoming_joint_wrench_b(self) -> ProxyArray:
        """Incoming joint wrenches on each body in the body frame [N, N*m].

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, num_bodies, 6).

        All body reaction wrenches are provided including the root body to the world of an articulation.
        """
        self._read_spatial_vector_binding(
            TT.LINK_INCOMING_JOINT_FORCE,
            self._body_incoming_joint_wrench_buf,
        )
        if self._body_incoming_joint_wrench_b_ta is None:
            self._body_incoming_joint_wrench_b_ta = ProxyArray(self._body_incoming_joint_wrench_buf.data)
        return self._body_incoming_joint_wrench_b_ta

    """
    Joint state properties.
    """

    @property
    def joint_pos(self) -> ProxyArray:
        """Joint positions of all joints [m or rad, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        """
        self._read_binding_into_buf(TT.DOF_POSITION, self._joint_pos_buf)
        if self._joint_pos_ta is None:
            self._joint_pos_ta = ProxyArray(self._joint_pos_buf.data)
        return self._joint_pos_ta

    @property
    def joint_vel(self) -> ProxyArray:
        """Joint velocities of all joints [m/s or rad/s, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.
        """
        self._read_binding_into_buf(TT.DOF_VELOCITY, self._joint_vel_buf)
        if self._joint_vel_ta is None:
            self._joint_vel_ta = ProxyArray(self._joint_vel_buf.data)
        return self._joint_vel_ta

    @property
    def joint_acc(self) -> ProxyArray:
        """Joint acceleration of all joints [m/s^2 or rad/s^2, depending on joint type].

        Shape is (num_instances, num_joints), dtype = wp.float32.

        .. note::
            This quantity is computed via finite differencing of joint velocities.
        """
        if self._joint_acc_ta is None:
            self._joint_acc_ta = ProxyArray(self._joint_acc.data)
        return self._joint_acc_ta

    """
    Derived Properties.
    """

    @property
    def projected_gravity_b(self) -> ProxyArray:
        """Projection of the gravity direction on base frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            wp.launch(
                _projected_gravity,
                dim=self.num_instances,
                inputs=[self.GRAVITY_VEC_W, self.root_link_pose_w],
                outputs=[self._projected_gravity_b.data],
                device=self.device,
            )
            self._projected_gravity_b.timestamp = self._sim_timestamp
        if self._projected_gravity_b_ta is None:
            self._projected_gravity_b_ta = ProxyArray(self._projected_gravity_b.data)
        return self._projected_gravity_b_ta

    @property
    def heading_w(self) -> ProxyArray:
        """Yaw heading of the base frame (in radians) [rad].

        Shape is (num_instances,), dtype = wp.float32.

        .. note::
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        if self._heading_w.timestamp < self._sim_timestamp:
            wp.launch(
                _compute_heading,
                dim=self.num_instances,
                inputs=[self.FORWARD_VEC_B, self.root_link_pose_w],
                outputs=[self._heading_w.data],
                device=self.device,
            )
            self._heading_w.timestamp = self._sim_timestamp
        if self._heading_w_ta is None:
            self._heading_w_ta = ProxyArray(self._heading_w.data)
        return self._heading_w_ta

    @property
    def root_link_lin_vel_b(self) -> ProxyArray:
        """Root link linear velocity in base frame [m/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to its actor frame.
        """
        if self._root_link_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                _world_vel_to_body_lin,
                dim=self.num_instances,
                inputs=[self.root_link_pose_w, self.root_link_vel_w],
                outputs=[self._root_link_lin_vel_b.data],
                device=self.device,
            )
            self._root_link_lin_vel_b.timestamp = self._sim_timestamp
        if self._root_link_lin_vel_b_ta is None:
            self._root_link_lin_vel_b_ta = ProxyArray(self._root_link_lin_vel_b.data)
        return self._root_link_lin_vel_b_ta

    @property
    def root_link_ang_vel_b(self) -> ProxyArray:
        """Root link angular velocity in base frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to its actor frame.
        """
        if self._root_link_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                _world_vel_to_body_ang,
                dim=self.num_instances,
                inputs=[self.root_link_pose_w, self.root_link_vel_w],
                outputs=[self._root_link_ang_vel_b.data],
                device=self.device,
            )
            self._root_link_ang_vel_b.timestamp = self._sim_timestamp
        if self._root_link_ang_vel_b_ta is None:
            self._root_link_ang_vel_b_ta = ProxyArray(self._root_link_ang_vel_b.data)
        return self._root_link_ang_vel_b_ta

    @property
    def root_com_lin_vel_b(self) -> ProxyArray:
        """Root center of mass linear velocity in base frame [m/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame
        with respect to its actor frame.
        """
        if self._root_com_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                _world_vel_to_body_lin,
                dim=self.num_instances,
                inputs=[self.root_link_pose_w, self.root_com_vel_w],
                outputs=[self._root_com_lin_vel_b.data],
                device=self.device,
            )
            self._root_com_lin_vel_b.timestamp = self._sim_timestamp
        if self._root_com_lin_vel_b_ta is None:
            self._root_com_lin_vel_b_ta = ProxyArray(self._root_com_lin_vel_b.data)
        return self._root_com_lin_vel_b_ta

    @property
    def root_com_ang_vel_b(self) -> ProxyArray:
        """Root center of mass angular velocity in base frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame
        with respect to its actor frame.
        """
        if self._root_com_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                _world_vel_to_body_ang,
                dim=self.num_instances,
                inputs=[self.root_link_pose_w, self.root_com_vel_w],
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
        """Root link position in simulation world frame [m].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        parent = self.root_link_pose_w
        if self._root_link_pos_w_ta is None:
            self._root_link_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._root_link_pos_w_ta

    @property
    def root_link_quat_w(self) -> ProxyArray:
        """Root link orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        """
        parent = self.root_link_pose_w
        if self._root_link_quat_w_ta is None:
            self._root_link_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._root_link_quat_w_ta

    @property
    def root_link_lin_vel_w(self) -> ProxyArray:
        """Root link linear velocity in simulation world frame [m/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        parent = self.root_link_vel_w
        if self._root_link_lin_vel_w_ta is None:
            self._root_link_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._root_link_lin_vel_w_ta

    @property
    def root_link_ang_vel_w(self) -> ProxyArray:
        """Root link angular velocity in simulation world frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        parent = self.root_link_vel_w
        if self._root_link_ang_vel_w_ta is None:
            self._root_link_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._root_link_ang_vel_w_ta

    @property
    def root_com_pos_w(self) -> ProxyArray:
        """Root center of mass position in simulation world frame [m].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        parent = self.root_com_pose_w
        if self._root_com_pos_w_ta is None:
            self._root_com_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._root_com_pos_w_ta

    @property
    def root_com_quat_w(self) -> ProxyArray:
        """Root center of mass orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        """
        parent = self.root_com_pose_w
        if self._root_com_quat_w_ta is None:
            self._root_com_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._root_com_quat_w_ta

    @property
    def root_com_lin_vel_w(self) -> ProxyArray:
        """Root center of mass linear velocity in simulation world frame [m/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        parent = self.root_com_vel_w
        if self._root_com_lin_vel_w_ta is None:
            self._root_com_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._root_com_lin_vel_w_ta

    @property
    def root_com_ang_vel_w(self) -> ProxyArray:
        """Root center of mass angular velocity in simulation world frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        parent = self.root_com_vel_w
        if self._root_com_ang_vel_w_ta is None:
            self._root_com_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._root_com_ang_vel_w_ta

    @property
    def body_link_pos_w(self) -> ProxyArray:
        """Positions of all bodies in simulation world frame [m].

        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_link_pose_w
        if self._body_link_pos_w_ta is None:
            self._body_link_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_link_pos_w_ta

    @property
    def body_link_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf.
        In torch this resolves to (num_instances, num_bodies, 4).
        """
        parent = self.body_link_pose_w
        if self._body_link_quat_w_ta is None:
            self._body_link_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_link_quat_w_ta

    @property
    def body_link_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame [m/s].

        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_link_vel_w
        if self._body_link_lin_vel_w_ta is None:
            self._body_link_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_link_lin_vel_w_ta

    @property
    def body_link_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame [rad/s].

        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_link_vel_w
        if self._body_link_ang_vel_w_ta is None:
            self._body_link_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_link_ang_vel_w_ta

    @property
    def body_com_pos_w(self) -> ProxyArray:
        """Positions of all bodies' center of mass in simulation world frame [m].

        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_pose_w
        if self._body_com_pos_w_ta is None:
            self._body_com_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_w_ta

    @property
    def body_com_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf.
        In torch this resolves to (num_instances, num_bodies, 4).
        """
        parent = self.body_com_pose_w
        if self._body_com_quat_w_ta is None:
            self._body_com_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_w_ta

    @property
    def body_com_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame [m/s].

        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_vel_w
        if self._body_com_lin_vel_w_ta is None:
            self._body_com_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_vel_w_ta

    @property
    def body_com_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame [rad/s].

        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_vel_w
        if self._body_com_ang_vel_w_ta is None:
            self._body_com_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_vel_w_ta

    @property
    def body_com_lin_acc_w(self) -> ProxyArray:
        """Linear acceleration of all bodies in simulation world frame [m/s^2].

        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_acc_w
        if self._body_com_lin_acc_w_ta is None:
            self._body_com_lin_acc_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_acc_w_ta

    @property
    def body_com_ang_acc_w(self) -> ProxyArray:
        """Angular acceleration of all bodies in simulation world frame [rad/s^2].

        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_acc_w
        if self._body_com_ang_acc_w_ta is None:
            self._body_com_ang_acc_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_acc_w_ta

    @property
    def body_com_pos_b(self) -> ProxyArray:
        """Center of mass position of all of the bodies in their respective link frames [m].

        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_pose_b
        if self._body_com_pos_b_ta is None:
            self._body_com_pos_b_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_b_ta

    @property
    def body_com_quat_b(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all of the bodies in their
        respective link frames.

        Shape is (num_instances, num_bodies), dtype = wp.quatf.
        In torch this resolves to (num_instances, num_bodies, 4).
        """
        parent = self.body_com_pose_b
        if self._body_com_quat_b_ta is None:
            self._body_com_quat_b_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_b_ta

    """
    Internal helpers.
    """

    def _create_buffers(self) -> None:  # noqa: C901
        """Eagerly allocate every TimestampedBuffer and pinned CPU staging buffer."""
        super()._create_buffers()

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

        # -- Joint properties (CPU-only; timestamped so they can be re-read after writes)
        self._joint_stiffness = TimestampedBuffer((N, D), dev, wp.float32)
        self._joint_damping = TimestampedBuffer((N, D), dev, wp.float32)
        self._joint_armature = TimestampedBuffer((N, D), dev, wp.float32)
        self._joint_pos_limits = TimestampedBuffer((N, D), dev, wp.vec2f)
        self._joint_vel_limits = TimestampedBuffer((N, D), dev, wp.float32)
        self._joint_effort_limits = TimestampedBuffer((N, D), dev, wp.float32)
        # Friction: single (N, D, 3) TimestampedBuffer; per-component views are created lazily.
        self._joint_friction_props_buf = TimestampedBuffer((N, D, 3), dev, wp.float32)
        # These are strided wp.array views into _joint_friction_props_buf.data; created in
        # _pin_proxy_arrays after the buffer exists.
        self._joint_friction_coeff: wp.array | None = None
        self._joint_dynamic_friction_coeff: wp.array | None = None
        self._joint_viscous_friction_coeff: wp.array | None = None

        # -- Body properties (CPU-only; read once at init, re-read via _read_scalar_binding)
        self._body_mass = TimestampedBuffer((N, L), dev, wp.float32)
        self._body_inertia = TimestampedBuffer((N, L, 9), dev, wp.float32)

        # -- Soft limits / custom joint properties
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

        # -- Deprecated combined state buffers (TimestampedBuffer; lazily filled on first access)
        self._root_state_w_buf = TimestampedBuffer(N, dev, vec13f)
        self._root_link_state_w_buf = TimestampedBuffer(N, dev, vec13f)
        self._root_com_state_w_buf = TimestampedBuffer(N, dev, vec13f)
        self._default_root_state_buf = wp.zeros(N, dtype=vec13f, device=dev)
        # -- Deprecated body combined state buffers (TimestampedBuffer; lazily filled on first access)
        self._body_state_w_buf = TimestampedBuffer((N, L), dev, vec13f)
        self._body_link_state_w_buf = TimestampedBuffer((N, L), dev, vec13f)
        self._body_com_state_w_buf = TimestampedBuffer((N, L), dev, vec13f)

        # -- Tendon property buffers (always allocated; empty shape when T==0 so
        #    properties never return None).  Routed through _read_scalar_binding.
        T_fix = self._num_fixed_tendons
        T_spa = self._num_spatial_tendons
        self._fixed_tendon_stiffness = TimestampedBuffer((N, T_fix), dev, wp.float32)
        self._fixed_tendon_damping = TimestampedBuffer((N, T_fix), dev, wp.float32)
        self._fixed_tendon_limit_stiffness = TimestampedBuffer((N, T_fix), dev, wp.float32)
        self._fixed_tendon_rest_length = TimestampedBuffer((N, T_fix), dev, wp.float32)
        self._fixed_tendon_offset = TimestampedBuffer((N, T_fix), dev, wp.float32)
        # Legacy alias kept for any internal callers that used the old vec2f buffer.
        self._fixed_tendon_pos_limits = TimestampedBuffer((N, T_fix), dev, wp.vec2f)
        self._spatial_tendon_stiffness = TimestampedBuffer((N, T_spa), dev, wp.float32)
        self._spatial_tendon_damping = TimestampedBuffer((N, T_spa), dev, wp.float32)
        self._spatial_tendon_limit_stiffness = TimestampedBuffer((N, T_spa), dev, wp.float32)
        self._spatial_tendon_offset = TimestampedBuffer((N, T_spa), dev, wp.float32)

        # -- CPU staging buffers for CPU-only bindings.
        # Pre-allocate all of them so there is no per-step allocation on the hot path.
        # These are keyed by tensor_type in self._cpu_staging_buffers; _binding_read
        # selects the right one at read time.  The sizes must match the binding shapes
        # (flat float32).  On a GPU sim the buffers are pinned-host (page-locked) so
        # the wheel can dispatch async copies; on a CPU sim the staging copy is
        # functionally redundant but the buffer must still exist for the write
        # helpers, so we allocate unpinned and pay only the intra-CPU memcpy.
        pinned = dev != "cpu"
        self._cpu_body_mass = wp.zeros((N, L), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_body_coms = wp.zeros((N, L, 7), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_body_inertia = wp.zeros((N, L, 9), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_joint_stiffness = wp.zeros((N, D), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_joint_damping = wp.zeros((N, D), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_joint_position_limit = wp.zeros((N, D, 2), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_joint_velocity_limit = wp.zeros((N, D), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_joint_effort_limit = wp.zeros((N, D), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_joint_armature = wp.zeros((N, D), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_joint_friction_coeff = wp.zeros((N, D), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_joint_dynamic_friction_coeff = wp.zeros((N, D), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_joint_viscous_friction_coeff = wp.zeros((N, D), dtype=wp.float32, device="cpu", pinned=pinned)
        if T_fix > 0:
            self._cpu_fixed_tendon_stiffness = wp.zeros((N, T_fix), dtype=wp.float32, device="cpu", pinned=pinned)
            self._cpu_fixed_tendon_damping = wp.zeros((N, T_fix), dtype=wp.float32, device="cpu", pinned=pinned)
            self._cpu_fixed_tendon_limit_stiffness = wp.zeros((N, T_fix), dtype=wp.float32, device="cpu", pinned=pinned)
            self._cpu_fixed_tendon_rest_length = wp.zeros((N, T_fix), dtype=wp.float32, device="cpu", pinned=pinned)
            self._cpu_fixed_tendon_offset = wp.zeros((N, T_fix), dtype=wp.float32, device="cpu", pinned=pinned)
            self._cpu_fixed_tendon_pos_limits = wp.zeros((N, T_fix, 2), dtype=wp.float32, device="cpu", pinned=pinned)
        if T_spa > 0:
            self._cpu_spatial_tendon_stiffness = wp.zeros((N, T_spa), dtype=wp.float32, device="cpu", pinned=pinned)
            self._cpu_spatial_tendon_damping = wp.zeros((N, T_spa), dtype=wp.float32, device="cpu", pinned=pinned)
            self._cpu_spatial_tendon_limit_stiffness = wp.zeros(
                (N, T_spa), dtype=wp.float32, device="cpu", pinned=pinned
            )
            self._cpu_spatial_tendon_offset = wp.zeros((N, T_spa), dtype=wp.float32, device="cpu", pinned=pinned)

        # Read initial joint/body properties from bindings (one-time CPU reads).
        self._read_initial_properties()
        # Initialize ProxyArray wrappers (lazily created on first property access).
        self._pin_proxy_arrays()

    def _binding_read(self, tensor_type: int, binding: Any, dst: wp.array) -> None:
        """Read *binding* into *dst*, staging through a pinned-host buffer for CPU-only bindings.

        For GPU-resident state bindings (pose, velocity, etc.) the read goes directly
        into the destination array.  For CPU-only property bindings (mass, COM, limits,
        stiffness, …) the wheel writes into a pinned-host staging buffer first, then
        :func:`wp.copy` moves the data to the simulation device asynchronously.

        Args:
            tensor_type: TensorType key identifying the binding.
            binding: OVPhysX TensorBinding whose ``read`` method is called.
            dst: Destination :class:`wp.array` on the simulation device.
        """
        if tensor_type not in TT._CPU_ONLY_TYPES or self.device == "cpu":
            binding.read(dst)
            return
        # Route through a lazily-allocated pinned-host staging buffer.
        staging = self._cpu_staging_buffers.get(tensor_type)
        if staging is None:
            staging = wp.zeros(binding.shape, dtype=wp.float32, device="cpu", pinned=True)
            self._cpu_staging_buffers[tensor_type] = staging
        binding.read(staging)
        # Build a flat float32 view of dst matching the binding's flat shape.
        if dst.dtype == wp.float32:
            view = dst
        else:
            view = wp.array(
                ptr=dst.ptr,
                shape=binding.shape,
                dtype=wp.float32,
                device=str(dst.device),
                copy=False,
            )
        wp.copy(view, staging)

    def _binding_write(
        self,
        tensor_type: int,
        binding: Any,
        src: wp.array,
        *,
        indices: wp.array | None = None,
        mask: wp.array | None = None,
    ) -> None:
        """Write *src* to *binding*, staging through pinned-host buffers for CPU-only bindings.

        Args:
            tensor_type: TensorType key identifying the binding.
            binding: OVPhysX TensorBinding whose ``write`` method is called.
            src: Source :class:`wp.array` on the simulation device.
            indices: Optional environment indices for partial writes.
            mask: Optional boolean mask for partial writes.
        """
        if tensor_type not in TT._CPU_ONLY_TYPES or self.device == "cpu":
            binding.write(src, indices=indices, mask=mask)
            return
        # Stage through a pinned-host buffer.
        staging = self._cpu_staging_buffers.get(tensor_type)
        if staging is None:
            staging = wp.zeros(binding.shape, dtype=wp.float32, device="cpu", pinned=True)
            self._cpu_staging_buffers[tensor_type] = staging
        if src.dtype == wp.float32:
            src_view = src
        else:
            src_view = wp.array(
                ptr=src.ptr,
                shape=binding.shape,
                dtype=wp.float32,
                device=str(src.device),
                copy=False,
            )
        wp.copy(staging, src_view)
        binding.write(staging, indices=indices, mask=mask)

    def _stage_to_pinned_cpu(self, tensor_type: int, role: str, src: wp.array) -> wp.array:
        """Copy *src* into a lazily-allocated pinned-host :class:`wp.array`.

        Keyed on *(tensor_type, role)* so the same pair always reuses the same
        buffer, avoiding per-call allocation on the hot path.

        Args:
            tensor_type: TensorType identifying the binding.
            role: Disambiguating string when the same tensor_type may serve
                multiple purposes (e.g. ``"read"`` vs ``"write"``).
            src: Source array on the simulation device.

        Returns:
            Pinned-host wp.array containing a copy of *src*.
        """
        key = (tensor_type, role)
        staging = self._cpu_staging_buffers.get(key)  # type: ignore[call-overload]
        if staging is None:
            if src.dtype == wp.float32:
                shape = src.shape
            else:
                # Flatten to float32 shape matching the element byte size.
                elem_floats = src.dtype.size // 4
                shape = src.shape + (elem_floats,)
            staging = wp.zeros(shape, dtype=wp.float32, device="cpu", pinned=True)
            self._cpu_staging_buffers[key] = staging  # type: ignore[index]
        if src.dtype == wp.float32:
            wp.copy(staging, src)
        else:
            flat_src = wp.array(ptr=src.ptr, shape=staging.shape, dtype=wp.float32, device=str(src.device), copy=False)
            wp.copy(staging, flat_src)
        return staging

    def _read_initial_properties(self) -> None:
        """Read static/initial joint and body properties from ovphysx bindings.

        These are one-time reads at init.  Property tensors (stiffness,
        damping, limits, mass, etc.) are CPU-resident in PhysX even in GPU
        mode, so we read them via CPU numpy buffers and then copy to the
        simulation device.
        """

        def _read_cpu(tensor_type):
            binding = self._get_binding(tensor_type)
            if binding is None:
                return None
            np_buf = np.zeros(binding.shape, dtype=np.float32)
            binding.read(np_buf)
            return np_buf

        # Joint scalar properties — write to .data since buffers are now TimestampedBuffer.
        for tt, buf in [
            (TT.DOF_STIFFNESS, self._joint_stiffness),
            (TT.DOF_DAMPING, self._joint_damping),
            (TT.DOF_ARMATURE, self._joint_armature),
            (TT.DOF_MAX_VELOCITY, self._joint_vel_limits),
            (TT.DOF_MAX_FORCE, self._joint_effort_limits),
        ]:
            np_buf = _read_cpu(tt)
            if np_buf is not None:
                wp.copy(buf.data, wp.from_numpy(np_buf, dtype=wp.float32, device=self.device))
                buf.timestamp = self._sim_timestamp

        # Body mass (now a TimestampedBuffer).
        np_buf = _read_cpu(TT.BODY_MASS)
        if np_buf is not None:
            wp.copy(self._body_mass.data, wp.from_numpy(np_buf, dtype=wp.float32, device=self.device))
            self._body_mass.timestamp = self._sim_timestamp

        # Joint position limits: [N, D, 2] -> (N, D) wp.vec2f stored in TimestampedBuffer.data
        np_lim = _read_cpu(TT.DOF_LIMIT)
        if np_lim is not None:
            src = wp.from_numpy(
                np_lim.reshape(self._num_instances, self._num_joints, 2), dtype=wp.vec2f, device=self.device
            )
            wp.copy(self._joint_pos_limits.data, src)
            self._joint_pos_limits.timestamp = self._sim_timestamp

        # Body inertia (now a TimestampedBuffer): [N, L, 9]
        np_iner = _read_cpu(TT.BODY_INERTIA)
        if np_iner is not None:
            wp.copy(
                self._body_inertia.data,
                wp.from_numpy(np_iner, dtype=wp.float32, device=self.device),
            )
            self._body_inertia.timestamp = self._sim_timestamp

        # Friction: [N, D, 3] -> load directly into the combined TimestampedBuffer.
        # The strided per-component views (_joint_friction_coeff/dynamic/viscous) are
        # created later in _pin_proxy_arrays, so we write to the combined buffer here.
        np_fric = _read_cpu(TT.DOF_FRICTION_PROPERTIES)
        if np_fric is not None:
            fric_contiguous = np.ascontiguousarray(np_fric.reshape(self._num_instances, self._num_joints, 3))
            wp.copy(
                self._joint_friction_props_buf.data,
                wp.from_numpy(fric_contiguous, dtype=wp.float32, device=self.device),
            )
            self._joint_friction_props_buf.timestamp = self._sim_timestamp

        # Fixed tendon properties.  PhysX exposes tendons on the simulation
        # device (no ``device="cpu"`` clone in its ``set_fixed_tendon_properties``
        # call); the OVPhysX wheel mirrors that, so we read directly into the
        # sim-device buffer rather than via a numpy round-trip.
        T_fix = self._num_fixed_tendons
        if T_fix > 0:
            for tt, buf in [
                (TT.FIXED_TENDON_STIFFNESS, self._fixed_tendon_stiffness),
                (TT.FIXED_TENDON_DAMPING, self._fixed_tendon_damping),
                (TT.FIXED_TENDON_LIMIT_STIFFNESS, self._fixed_tendon_limit_stiffness),
                (TT.FIXED_TENDON_REST_LENGTH, self._fixed_tendon_rest_length),
                (TT.FIXED_TENDON_OFFSET, self._fixed_tendon_offset),
            ]:
                binding = self._get_binding(tt)
                if binding is not None:
                    self._binding_read(tt, binding, buf.data)
                    buf.timestamp = self._sim_timestamp
            binding = self._get_binding(TT.FIXED_TENDON_LIMIT)
            if binding is not None:
                self._binding_read(TT.FIXED_TENDON_LIMIT, binding, self._fixed_tendon_pos_limits.data)
                self._fixed_tendon_pos_limits.timestamp = self._sim_timestamp

        # Spatial tendon properties (sim-device, see fixed-tendon comment above).
        T_spa = self._num_spatial_tendons
        if T_spa > 0:
            for tt, buf in [
                (TT.SPATIAL_TENDON_STIFFNESS, self._spatial_tendon_stiffness),
                (TT.SPATIAL_TENDON_DAMPING, self._spatial_tendon_damping),
                (TT.SPATIAL_TENDON_LIMIT_STIFFNESS, self._spatial_tendon_limit_stiffness),
                (TT.SPATIAL_TENDON_OFFSET, self._spatial_tendon_offset),
            ]:
                binding = self._get_binding(tt)
                if binding is not None:
                    self._binding_read(tt, binding, buf.data)
                    buf.timestamp = self._sim_timestamp

    def _pin_proxy_arrays(self) -> None:
        """Create pinned ProxyArray wrappers for all data buffers.

        Called once from :meth:`_create_buffers` during initialization.
        All ``_ta`` fields are lazily populated on first property access.
        """
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
        self._body_incoming_joint_wrench_b_ta: ProxyArray | None = None
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
        # Deprecated body state-concat properties
        self._body_state_w_ta: ProxyArray | None = None
        self._body_link_state_w_ta: ProxyArray | None = None
        self._body_com_state_w_ta: ProxyArray | None = None

        # Create strided wp.array views into _joint_friction_props_buf.data so that
        # each friction component is accessible without copying data.  The combined
        # buffer has shape (N, D, 3) and contiguous float32 storage, so component k
        # lives at byte offset k*4 with strides (D*3*4, 3*4).
        N = self._num_instances
        D = self._num_joints
        _fp = self._joint_friction_props_buf.data
        _float_bytes = 4  # sizeof(float32)
        _stride_row = D * 3 * _float_bytes  # bytes between rows
        _stride_col = 3 * _float_bytes  # bytes between columns (elements)
        _dev = str(_fp.device)
        self._joint_friction_coeff = wp.array(
            ptr=_fp.ptr,
            shape=(N, D),
            strides=(_stride_row, _stride_col),
            dtype=wp.float32,
            device=_dev,
            copy=False,
        )
        self._joint_dynamic_friction_coeff = wp.array(
            ptr=_fp.ptr + _float_bytes,
            shape=(N, D),
            strides=(_stride_row, _stride_col),
            dtype=wp.float32,
            device=_dev,
            copy=False,
        )
        self._joint_viscous_friction_coeff = wp.array(
            ptr=_fp.ptr + 2 * _float_bytes,
            shape=(N, D),
            strides=(_stride_row, _stride_col),
            dtype=wp.float32,
            device=_dev,
            copy=False,
        )

    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidate cached buffers when the simulation is reinitialized.

        Args:
            event: Simulation event (unused).
        """
        self._is_primed = False
        self._sim_timestamp = 0.0
        # Reset every TimestampedBuffer timestamp so the next property access
        # triggers a fresh pull from the binding.
        for attr_name in dir(self):
            if attr_name.startswith("_") and not attr_name.startswith("__"):
                val = getattr(self, attr_name, None)
                if isinstance(val, TimestampedBuffer):
                    val.timestamp = -1.0

    def _get_binding(self, tensor_type: int):
        """Return the cached binding for :paramref:`tensor_type`, or ``None`` if absent.

        Args:
            tensor_type: TensorType key.

        Returns:
            The TensorBinding, or ``None`` if not present in the binding dict.
        """
        return self._bindings.get(tensor_type)

    def _get_read_view(self, tensor_type: int, wp_array: wp.array, floats_per_elem: int = 0) -> wp.array | None:
        """Return a stable float32 view of a warp buffer for reading from a binding.

        For structured-dtype buffers (transformf, spatial_vectorf), the view
        reinterprets the same GPU memory as a flat float32 array matching the
        binding's shape.  For plain float32 buffers, returns the array as-is.

        The returned view is cached so that ``binding.read(view)`` sees the
        same object on every call, enabling the binding's internal read cache.

        Args:
            tensor_type: TensorType key.
            wp_array: Destination warp array.
            floats_per_elem: Number of float32 elements per logical element
                (e.g. 7 for transformf, 6 for spatial_vectorf).  Pass 0 to
                return the array as-is.

        Returns:
            Float32 view suitable for ``binding.read()``, or ``None``.
        """
        if not hasattr(self, "_read_view_cache"):
            self._read_view_cache = {}
        cache_key = (tensor_type, wp_array.ptr)
        cached = self._read_view_cache.get(cache_key)
        if cached is not None:
            return cached

        binding = self._get_binding(tensor_type)
        if binding is None:
            self._read_view_cache[cache_key] = None
            return None

        if floats_per_elem > 0:
            view = wp.array(
                ptr=wp_array.ptr,
                shape=binding.shape,
                dtype=wp.float32,
                device=str(wp_array.device),
                copy=False,
            )
        else:
            view = wp_array

        self._read_view_cache[cache_key] = view
        return view

    def _read_binding_into_buf(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read from an ovphysx binding into a :class:`TimestampedBuffer`, skipping if fresh.

        Args:
            tensor_type: TensorType key.
            buf: Timestamped buffer to refresh.
        """
        if buf.timestamp >= self._sim_timestamp:
            return
        view = self._get_read_view(tensor_type, buf.data)
        if view is None:
            return
        self._get_binding(tensor_type).read(view)
        buf.timestamp = self._sim_timestamp

    def _read_transform_binding(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read a pose binding (float32 view of transformf buffer), skipping if fresh.

        CPU-only bindings (e.g. ``BODY_COM_POSE``) are routed through a
        pinned-host staging buffer via :meth:`_binding_read` so the wheel's
        device-match requirement is satisfied even on a GPU sim.

        Args:
            tensor_type: TensorType key.
            buf: Timestamped :class:`wp.transformf` buffer to refresh.
        """
        if buf.timestamp >= self._sim_timestamp:
            return
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        view = self._get_read_view(tensor_type, buf.data, 7)
        if view is None:
            return
        self._binding_read(tensor_type, binding, view)
        buf.timestamp = self._sim_timestamp

    def _read_spatial_vector_binding(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read a velocity binding (float32 view of spatial_vectorf buffer), skipping if fresh.

        Args:
            tensor_type: TensorType key.
            buf: Timestamped :class:`wp.spatial_vectorf` buffer to refresh.
        """
        if buf.timestamp >= self._sim_timestamp:
            return
        view = self._get_read_view(tensor_type, buf.data, 6)
        if view is None:
            return
        self._get_binding(tensor_type).read(view)
        buf.timestamp = self._sim_timestamp

    def _read_scalar_binding(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Refresh a scalar or flat float32 buffer from the matching binding if stale.

        Identical timestamp-gating contract as :meth:`_read_transform_binding`
        but without a structured-dtype reinterpret cast.  CPU-only bindings
        (e.g. ``DOF_STIFFNESS``, ``DOF_LIMIT``) are routed through a
        pre-allocated pinned-host staging buffer via :meth:`_binding_read` so
        the wheel's device-match requirement is satisfied even on a GPU sim.

        Args:
            tensor_type: TensorType key identifying the binding.
            buf: Timestamped buffer whose :attr:`~TimestampedBuffer.data` field
                will be refreshed.
        """
        if buf.timestamp >= self._sim_timestamp:
            return
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        self._binding_read(tensor_type, binding, buf.data)
        buf.timestamp = self._sim_timestamp

    def _get_pos_from_transform(self, transform: wp.array) -> wp.array:
        """Return a position view aliased into a transform array.

        Args:
            transform: Source transform array.

        Returns:
            vec3f view into the position component.
        """
        return wp.array(
            ptr=transform.ptr,
            shape=transform.shape,
            dtype=wp.vec3f,
            strides=transform.strides,
            device=self.device,
        )

    def _get_quat_from_transform(self, transform: wp.array) -> wp.array:
        """Return a quaternion view aliased into a transform array.

        Args:
            transform: Source transform array.

        Returns:
            quatf view into the quaternion component (offset 3 floats = 12 bytes).
        """
        return wp.array(
            ptr=transform.ptr + 3 * 4,
            shape=transform.shape,
            dtype=wp.quatf,
            strides=transform.strides,
            device=self.device,
        )

    def _get_lin_vel_from_spatial_vector(self, sv: wp.array) -> wp.array:
        """Return a linear velocity view aliased into a spatial vector array.

        Args:
            sv: Source spatial vector array.

        Returns:
            vec3f view into the linear velocity component.
        """
        return wp.array(
            ptr=sv.ptr,
            shape=sv.shape,
            dtype=wp.vec3f,
            strides=sv.strides,
            device=self.device,
        )

    def _get_ang_vel_from_spatial_vector(self, sv: wp.array) -> wp.array:
        """Return an angular velocity view aliased into a spatial vector array.

        Args:
            sv: Source spatial vector array.

        Returns:
            vec3f view into the angular velocity component (offset 3 floats = 12 bytes).
        """
        return wp.array(
            ptr=sv.ptr + 3 * 4,
            shape=sv.shape,
            dtype=wp.vec3f,
            strides=sv.strides,
            device=self.device,
        )

    """
    Deprecated properties.
    """

    @property
    def default_root_state(self) -> ProxyArray:
        """Deprecated. Use :attr:`default_root_pose` and :attr:`default_root_vel` instead.

        Shape is (num_instances,), dtype = ``vec13f``. In torch this resolves to (num_instances, 13).
        """
        warnings.warn(
            "default_root_state is deprecated. Use default_root_pose and default_root_vel.",
            DeprecationWarning,
            stacklevel=2,
        )
        wp.launch(
            concat_root_pose_and_vel_to_state,
            dim=self.num_instances,
            inputs=[self._default_root_pose, self._default_root_vel],
            outputs=[self._default_root_state_buf],
            device=self.device,
        )
        if self._default_root_state_ta is None:
            self._default_root_state_ta = ProxyArray(self._default_root_state_buf)
        return self._default_root_state_ta

    @property
    def root_state_w(self) -> ProxyArray:
        """Deprecated. Use :attr:`root_link_pose_w` and :attr:`root_com_vel_w` instead.

        Shape is (num_instances,), dtype = ``vec13f``. In torch this resolves to (num_instances, 13).
        """
        warnings.warn(
            "The `root_state_w` property will be deprecated in IsaacLab 4.0. Please use `root_link_pose_w` and "
            "`root_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._root_state_w_buf.timestamp < self._sim_timestamp:
            wp.launch(
                concat_root_pose_and_vel_to_state,
                dim=self.num_instances,
                inputs=[self.root_link_pose_w, self.root_com_vel_w],
                outputs=[self._root_state_w_buf.data],
                device=self.device,
            )
            self._root_state_w_buf.timestamp = self._sim_timestamp
        if self._root_state_w_ta is None:
            self._root_state_w_ta = ProxyArray(self._root_state_w_buf.data)
        return self._root_state_w_ta

    @property
    def root_link_state_w(self) -> ProxyArray:
        """Deprecated. Use :attr:`root_link_pose_w` and :attr:`root_link_vel_w` instead.

        Shape is (num_instances,), dtype = ``vec13f``. In torch this resolves to (num_instances, 13).
        """
        warnings.warn(
            "The `root_link_state_w` property will be deprecated in IsaacLab 4.0. Please use `root_link_pose_w` and "
            "`root_link_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._root_link_state_w_buf.timestamp < self._sim_timestamp:
            wp.launch(
                concat_root_pose_and_vel_to_state,
                dim=self.num_instances,
                inputs=[self.root_link_pose_w, self.root_link_vel_w],
                outputs=[self._root_link_state_w_buf.data],
                device=self.device,
            )
            self._root_link_state_w_buf.timestamp = self._sim_timestamp
        if self._root_link_state_w_ta is None:
            self._root_link_state_w_ta = ProxyArray(self._root_link_state_w_buf.data)
        return self._root_link_state_w_ta

    @property
    def root_com_state_w(self) -> ProxyArray:
        """Deprecated. Use :attr:`root_com_pose_w` and :attr:`root_com_vel_w` instead.

        Shape is (num_instances,), dtype = ``vec13f``. In torch this resolves to (num_instances, 13).
        """
        warnings.warn(
            "The `root_com_state_w` property will be deprecated in IsaacLab 4.0. Please use `root_com_pose_w` and "
            "`root_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._root_com_state_w_buf.timestamp < self._sim_timestamp:
            wp.launch(
                concat_root_pose_and_vel_to_state,
                dim=self.num_instances,
                inputs=[self.root_com_pose_w, self.root_com_vel_w],
                outputs=[self._root_com_state_w_buf.data],
                device=self.device,
            )
            self._root_com_state_w_buf.timestamp = self._sim_timestamp
        if self._root_com_state_w_ta is None:
            self._root_com_state_w_ta = ProxyArray(self._root_com_state_w_buf.data)
        return self._root_com_state_w_ta

    @property
    def body_state_w(self) -> ProxyArray:
        """Deprecated. Use :attr:`body_link_pose_w` and :attr:`body_com_vel_w` instead.

        Shape is (num_instances, num_bodies), dtype = ``vec13f``.
        In torch this resolves to (num_instances, num_bodies, 13).
        """
        warnings.warn(
            "The `body_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_link_pose_w` and "
            "`body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._body_state_w_buf.timestamp >= self._sim_timestamp:
            if self._body_state_w_ta is None:
                self._body_state_w_ta = ProxyArray(self._body_state_w_buf.data)
            return self._body_state_w_ta
        _ = self.body_link_pose_w
        _ = self.body_com_vel_w
        wp.launch(
            concat_body_pose_and_vel_to_state,
            dim=(self.num_instances, self.num_bodies),
            inputs=[self._body_link_pose_w.data, self._body_com_vel_w.data],
            outputs=[self._body_state_w_buf.data],
            device=self.device,
        )
        self._body_state_w_buf.timestamp = self._sim_timestamp
        if self._body_state_w_ta is None:
            self._body_state_w_ta = ProxyArray(self._body_state_w_buf.data)
        return self._body_state_w_ta

    @property
    def body_link_state_w(self) -> ProxyArray:
        """Deprecated. Use :attr:`body_link_pose_w` and :attr:`body_link_vel_w` instead.

        Shape is (num_instances, num_bodies), dtype = ``vec13f``.
        In torch this resolves to (num_instances, num_bodies, 13).
        """
        warnings.warn(
            "The `body_link_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_link_pose_w` and "
            "`body_link_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._body_link_state_w_buf.timestamp >= self._sim_timestamp:
            if self._body_link_state_w_ta is None:
                self._body_link_state_w_ta = ProxyArray(self._body_link_state_w_buf.data)
            return self._body_link_state_w_ta
        _ = self.body_link_pose_w
        _ = self.body_link_vel_w
        wp.launch(
            concat_body_pose_and_vel_to_state,
            dim=(self.num_instances, self.num_bodies),
            inputs=[self._body_link_pose_w.data, self._body_link_vel_w.data],
            outputs=[self._body_link_state_w_buf.data],
            device=self.device,
        )
        self._body_link_state_w_buf.timestamp = self._sim_timestamp
        if self._body_link_state_w_ta is None:
            self._body_link_state_w_ta = ProxyArray(self._body_link_state_w_buf.data)
        return self._body_link_state_w_ta

    @property
    def body_com_state_w(self) -> ProxyArray:
        """Deprecated. Use :attr:`body_com_pose_w` and :attr:`body_com_vel_w` instead.

        Shape is (num_instances, num_bodies), dtype = ``vec13f``.
        In torch this resolves to (num_instances, num_bodies, 13).
        """
        warnings.warn(
            "The `body_com_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_com_pose_w` and "
            "`body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._body_com_state_w_buf.timestamp >= self._sim_timestamp:
            if self._body_com_state_w_ta is None:
                self._body_com_state_w_ta = ProxyArray(self._body_com_state_w_buf.data)
            return self._body_com_state_w_ta
        _ = self.body_com_pose_w
        _ = self.body_com_vel_w
        wp.launch(
            concat_body_pose_and_vel_to_state,
            dim=(self.num_instances, self.num_bodies),
            inputs=[self._body_com_pose_w.data, self._body_com_vel_w.data],
            outputs=[self._body_com_state_w_buf.data],
            device=self.device,
        )
        self._body_com_state_w_buf.timestamp = self._sim_timestamp
        if self._body_com_state_w_ta is None:
            self._body_com_state_w_ta = ProxyArray(self._body_com_state_w_buf.data)
        return self._body_com_state_w_ta
