# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.sensors.imu import BaseImu

import isaaclab_ovphysx.tensor_types as TT
from isaaclab_ovphysx.physics import OvPhysxManager as SimulationManager
from isaaclab_ovphysx.sim.views.ovphysx_view import OvPhysxView

from .imu_data import ImuData
from .kernels import imu_reset_kernel, imu_update_kernel

if TYPE_CHECKING:
    from isaaclab.sensors.imu import ImuCfg


class Imu(BaseImu):
    """The OVPhysX Inertial Measurement Unit (IMU) sensor.

    This sensor models a real IMU that measures angular velocity (gyroscope) and
    linear acceleration (accelerometer) in the sensor's body frame. Unlike the PVA
    sensor, it does not provide pose, linear velocity, angular acceleration, or
    projected gravity.

    Like a real accelerometer, the linear acceleration readings always include the
    contribution of gravity. The gravity vector is queried from the simulation at
    initialization.

    The sensor can be attached to any prim path with a rigid ancestor in its tree.
    If the provided path is not a rigid body, the closest rigid-body ancestor is
    used for simulation queries. The fixed transform from that ancestor to the
    target prim is computed once during initialization and composed with the
    configured sensor offset.

    .. note::

        Linear acceleration is computed using numerical differentiation from
        velocities. Consequently, the IMU sensor accuracy depends on the chosen
        physics timestep. For sufficient accuracy, we recommend keeping the
        timestep at least 200 Hz.
    """

    cfg: ImuCfg
    """The configuration parameters."""

    __backend_name__: str = "ovphysx"
    """The name of the backend for the IMU sensor."""

    def __init__(self, cfg: ImuCfg):
        """Initializes the IMU sensor.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)
        self._data = ImuData()
        self._rigid_parent_expr: str | None = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Imu sensor @ '{self.cfg.prim_path}': \n"
            f"\tbinding pattern   : {self._rigid_parent_expr}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._num_bodies}\n"
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
        return self._num_bodies

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

        - If the target prim path is a rigid body, bind directly to it.
        - Otherwise find the closest rigid-body ancestor, cache the fixed transform
          from that ancestor to the target prim, and bind to the ancestor pattern.
        """
        super()._initialize_impl()

        physx_instance = SimulationManager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")

        self._rigid_parent_expr, fixed_pos_b, fixed_quat_b = self._resolve_rigid_body_ancestor_expr()

        # Translate the regex-style path expression to an ovphysx fnmatch glob.
        pattern = self._rigid_parent_expr.replace(".*", "*")

        self._root_view = OvPhysxView(physx_instance, pattern=pattern, device=self._device)
        self._pose_binding = self._root_view.binding_for(TT.RIGID_BODY_POSE)
        self._vel_binding = self._root_view.binding_for(TT.RIGID_BODY_VELOCITY)
        self._com_binding = self._root_view.binding_for(TT.RIGID_BODY_COM_POSE)
        self._num_bodies = self._pose_binding.count

        if self._num_bodies != self._num_envs:
            raise ValueError(
                f"OvPhysx Imu: pattern '{pattern}' matched {self._num_bodies} rigid bodies; expected exactly one"
                f" body per environment (num_envs={self._num_envs}). Check that the prim path or its rigid-body"
                " ancestor is unique per env."
            )

        gravity = SimulationManager.get_gravity()
        gravity_bias = torch.tensor((-gravity[0], -gravity[1], -gravity[2]), device=self._device)
        gravity_bias_torch = gravity_bias.repeat(self._num_bodies, 1)
        self._gravity_bias_w = wp.from_torch(gravity_bias_torch.contiguous(), dtype=wp.vec3f)

        self._initialize_buffers_impl()

        # Compose the configured offset with the fixed ancestor->target transform (done once).
        if fixed_pos_b is not None and fixed_quat_b is not None:
            fixed_p = torch.tensor(fixed_pos_b, device=self._device).repeat(self._num_bodies, 1)
            fixed_q = torch.tensor(fixed_quat_b, device=self._device).repeat(self._num_bodies, 1)

            cfg_p = wp.to_torch(self._offset_pos_b).clone()
            cfg_q = wp.to_torch(self._offset_quat_b).clone()

            composed_p = fixed_p + math_utils.quat_apply(fixed_q, cfg_p)
            composed_q = math_utils.quat_mul(fixed_q, cfg_q)

            self._offset_pos_b = wp.from_torch(composed_p.contiguous(), dtype=wp.vec3f)
            self._offset_quat_b = wp.from_torch(composed_q.contiguous(), dtype=wp.quatf)

    def _invalidate_initialize_callback(self, event) -> None:
        """Drop the OVPhysX view and bindings when physics stops."""
        super()._invalidate_initialize_callback(event)
        # Drop the view (and the bindings it caches) so a stale/destroyed handle is not held
        # across the reset; ``_initialize_impl`` rebuilds a fresh view on the next play.
        self._root_view = None
        self._pose_binding = None
        self._vel_binding = None
        self._com_binding = None

    def _update_buffers_impl(self, env_mask: wp.array | None = None):
        """Fills the buffers of the sensor data."""
        env_mask = self._resolve_indices_and_mask(None, env_mask)

        # ``read_into`` fills the structured-dtype destination in place through a cached
        # float32 reinterpret of the binding's flat shape (no extra copy).
        self._root_view.read_into(TT.RIGID_BODY_POSE, self._transforms)
        self._root_view.read_into(TT.RIGID_BODY_VELOCITY, self._velocities)
        # RIGID_BODY_COM_POSE is a CPU tensor type in the OVPhysX wheel.
        # For GPU simulations, stage on a pinned CPU buffer then copy into the kernel buffer.
        self._root_view.read_into(TT.RIGID_BODY_COM_POSE, self._coms_read_view)
        if self._coms_read_view is not self._coms_gpu_view:
            wp.copy(self._coms_gpu_view, self._coms_read_view)

        wp.launch(
            imu_update_kernel,
            dim=self._num_envs,
            inputs=[
                env_mask,
                self._transforms,
                self._velocities,
                self._coms_buffer,
                self._offset_pos_b,
                self._offset_quat_b,
                self._gravity_bias_w,
                1.0 / self._dt,
                self._timestamp,
                self._prev_lin_vel_w,
                self._data._ang_vel_b,
                self._data._lin_acc_b,
            ],
            device=self._device,
        )

    def _initialize_buffers_impl(self):
        """Create buffers for storing data."""
        self._data.create_buffers(num_envs=self._num_bodies, device=self._device)

        self._prev_lin_vel_w = wp.zeros(self._num_bodies, dtype=wp.vec3f, device=self._device)

        offset_pos_torch = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._num_bodies, 1)
        offset_quat_torch = torch.tensor(list(self.cfg.offset.rot), device=self._device).repeat(self._num_bodies, 1)
        self._offset_pos_b = wp.from_torch(offset_pos_torch.contiguous(), dtype=wp.vec3f)
        self._offset_quat_b = wp.from_torch(offset_quat_torch.contiguous(), dtype=wp.quatf)

        # Structured-dtype buffers consumed by the kernel. ``read_into`` fills the GPU-resident
        # pose/velocity buffers directly, building and caching the float32 reinterpret itself.
        self._transforms = wp.zeros(self._num_bodies, dtype=wp.transformf, device=self._device)
        self._velocities = wp.zeros(self._num_bodies, dtype=wp.spatial_vectorf, device=self._device)
        self._coms_buffer = wp.zeros(self._num_bodies, dtype=wp.transformf, device=self._device)

        # RIGID_BODY_COM_POSE is CPU-only in the OVPhysX wheel. ``read_into`` requires the
        # destination on the binding's native device (cpu), so on a GPU sim we read into a pinned
        # CPU buffer and copy into the GPU kernel buffer; on a CPU sim the two alias and the copy
        # is skipped. ``_coms_gpu_view`` stays a flat float32 view so the copy dtype matches.
        self._coms_gpu_view = wp.array(
            ptr=self._coms_buffer.ptr,
            shape=self._com_binding.shape,
            dtype=wp.float32,
            device=self._device,
            copy=False,
        )
        if self._device == "cpu":
            self._coms_read_view = self._coms_gpu_view
        else:
            self._coms_read_view = wp.zeros(self._com_binding.shape, dtype=wp.float32, device="cpu", pinned=True)
