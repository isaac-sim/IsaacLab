# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.sensors.imu import BaseImu

from isaaclab_physx.physics import PhysxManager as SimulationManager

from .imu_data import ImuData
from .kernels import imu_reset_kernel, imu_update_kernel

if TYPE_CHECKING:
    from isaaclab.sensors.imu import ImuCfg

logger = logging.getLogger(__name__)


class Imu(BaseImu):
    """The PhysX Inertial Measurement Unit (IMU) sensor.

    This sensor models a real IMU that measures angular velocity (gyroscope) and
    linear acceleration (accelerometer) in the sensor's body frame. Unlike the PVA
    sensor, it does not provide pose, linear velocity, angular acceleration, or
    projected gravity.

    Like a real accelerometer, the linear acceleration readings always include the
    contribution of gravity. The gravity vector is queried from the simulation at
    initialization.

    The sensor can be attached to any prim path with a rigid ancestor in its tree.
    If the provided path is not a rigid body, the closest rigid-body ancestor is used
    for simulation queries. The fixed transform from that ancestor to the target prim
    is computed once during initialization and composed with the configured sensor offset.

    .. note::

        Linear acceleration is computed using numerical differentiation from velocities.
        Consequently, the IMU sensor accuracy depends on the chosen physics timestep.
        For sufficient accuracy, we recommend keeping the timestep at least 200 Hz.
    """

    cfg: ImuCfg
    """The configuration parameters."""

    __backend_name__: str = "physx"
    """The name of the backend for the IMU sensor."""

    def __init__(self, cfg: ImuCfg):
        """Initializes the IMU sensor.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)
        self._data = ImuData()
        self._rigid_parent_expr: str | None = None
        self._raw_transforms: wp.array | None = None
        self._raw_velocities: wp.array | None = None
        self._raw_coms: wp.array | None = None
        self._update_cmd: wp.Launch | None = None
        self._update_env_mask: wp.array | None = None
        self._update_inv_dt: float | None = None
        self._use_recorded_launch: bool = False

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Imu sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type         : {self._view.__class__}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._view.count}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> ImuData:
        self._update_outdated_buffers()
        return self._data

    @property
    def num_instances(self) -> int:
        return self._view.count

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(None, env_mask)

        wp.launch(
            imu_reset_kernel,
            dim=self._num_envs,
            inputs=[
                env_mask,
                self._data._ang_vel_b,
                self._data._lin_acc_b,
                self._prev_lin_vel_w,
            ],
            device=self._device,
        )

    def update(self, dt: float, force_recompute: bool = False):
        self._dt = dt
        super().update(dt, force_recompute)

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        - If the target prim path is a rigid body, build the view directly on it.
        - Otherwise find the closest rigid-body ancestor, cache the fixed transform from that ancestor
          to the target prim, and build the view on the ancestor expression.
        """
        super()._initialize_impl()
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        self._rigid_parent_expr, fixed_pos_b, fixed_quat_b = self._resolve_rigid_body_ancestor_expr()
        self._view = self._physics_sim_view.create_rigid_body_view(self._rigid_parent_expr.replace(".*", "*"))

        # Query world gravity and compute accelerometer bias (real IMUs always measure gravity)
        gravity = self._physics_sim_view.get_gravity()
        gravity_bias = torch.tensor((-gravity[0], -gravity[1], -gravity[2]), device=self._device)
        gravity_bias_torch = gravity_bias.repeat(self._view.count, 1)
        self._gravity_bias_w = wp.from_torch(gravity_bias_torch.contiguous(), dtype=wp.vec3f)

        self._initialize_buffers_impl()

        # Compose the configured offset with the fixed ancestor->target transform (done once)
        if fixed_pos_b is not None and fixed_quat_b is not None:
            fixed_p = torch.tensor(fixed_pos_b, device=self._device).repeat(self._view.count, 1)
            fixed_q = torch.tensor(fixed_quat_b, device=self._device).repeat(self._view.count, 1)

            cfg_p = wp.to_torch(self._offset_pos_b).clone()
            cfg_q = wp.to_torch(self._offset_quat_b).clone()

            composed_p = fixed_p + math_utils.quat_apply(fixed_q, cfg_p)
            composed_q = math_utils.quat_mul(fixed_q, cfg_q)

            self._offset_pos_b = wp.from_torch(composed_p.contiguous(), dtype=wp.vec3f)
            self._offset_quat_b = wp.from_torch(composed_q.contiguous(), dtype=wp.quatf)

        self._use_recorded_launch = wp.get_device(self._device).is_cuda

    def _update_buffers_impl(self, env_mask: wp.array | None = None):
        """Fills the buffers of the sensor data."""
        env_mask = self._resolve_indices_and_mask(None, env_mask)

        # Refresh the PhysX buffers every update, but create their typed Warp views only once:
        # the getters lazily allocate their output buffers and refresh the same memory in place
        # on every call, so the cached views (and the recorded launch that consumes them) stay
        # valid. A re-backed buffer would silently freeze the sensor data, so fail loudly.
        transforms = self._view.get_transforms()
        velocities = self._view.get_velocities()
        coms = self._view.get_coms()
        if self._raw_transforms is None:
            self._raw_transforms = transforms.view(wp.transformf)
            self._raw_velocities = velocities.view(wp.spatial_vectorf)
            self._raw_coms = coms.view(wp.transformf)
        elif (
            transforms.ptr != self._raw_transforms.ptr
            or velocities.ptr != self._raw_velocities.ptr
            or coms.ptr != self._raw_coms.ptr
        ):
            raise RuntimeError(
                f"A PhysX rigid body buffer of the sensor at '{self.cfg.prim_path}' was re-allocated"
                " after its warp view was cached. The cached views and the recorded launch require"
                " pointer-stable buffers refreshed in place."
            )
        wp.copy(self._coms_buffer, self._raw_coms)

        inv_dt = 1.0 / self._dt
        if self._use_recorded_launch:
            if self._update_cmd is None:
                try:
                    self._update_cmd = self._launch_update(env_mask, inv_dt, record_cmd=True)
                    self._update_env_mask = env_mask
                    self._update_inv_dt = inv_dt
                except Exception as exc:
                    self._use_recorded_launch = False
                    logger.warning(
                        f"Failed to record the update of the IMU at '{self.cfg.prim_path}'."
                        f" Falling back to eager kernel launches. Reason: {exc}"
                    )
            if self._update_cmd is not None:
                if env_mask is not self._update_env_mask:
                    self._update_cmd.set_param_by_name("env_mask", env_mask)
                    self._update_env_mask = env_mask
                if inv_dt != self._update_inv_dt:
                    self._update_cmd.set_param_by_name("inv_dt", inv_dt)
                    self._update_inv_dt = inv_dt
                self._update_cmd.launch()
                return

        self._launch_update(env_mask, inv_dt)

    def _launch_update(self, env_mask: wp.array, inv_dt: float, record_cmd: bool = False) -> wp.Launch | None:
        """Launch or record the kernel that updates the IMU data."""

        return wp.launch(
            imu_update_kernel,
            dim=self._num_envs,
            inputs=[
                env_mask,
                self._raw_transforms,
                self._raw_velocities,
                self._coms_buffer,
                self._offset_pos_b,
                self._offset_quat_b,
                self._gravity_bias_w,
                inv_dt,
                self._timestamp,
                self._prev_lin_vel_w,
                self._data._ang_vel_b,
                self._data._lin_acc_b,
            ],
            device=self._device,
            record_cmd=record_cmd,
        )

    def _initialize_buffers_impl(self):
        """Create buffers for storing data."""
        self._data.create_buffers(num_envs=self._view.count, device=self._device)

        self._prev_lin_vel_w = wp.zeros(self._view.count, dtype=wp.vec3f, device=self._device)

        offset_pos_torch = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)
        offset_quat_torch = torch.tensor(list(self.cfg.offset.rot), device=self._device).repeat(self._view.count, 1)
        self._offset_pos_b = wp.from_torch(offset_pos_torch.contiguous(), dtype=wp.vec3f)
        self._offset_quat_b = wp.from_torch(offset_quat_torch.contiguous(), dtype=wp.quatf)

        self._coms_buffer = wp.zeros(self._view.count, dtype=wp.transformf, device=self._device)

    def _invalidate_initialize_callback(self, event):
        """Invalidate the sensor and release cached PhysX and launch state."""
        super()._invalidate_initialize_callback(event)
        self._view = None
        self._raw_transforms = None
        self._raw_velocities = None
        self._raw_coms = None
        self._update_cmd = None
        self._update_env_mask = None
        self._update_inv_dt = None
