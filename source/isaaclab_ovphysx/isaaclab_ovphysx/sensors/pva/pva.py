# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from pxr import UsdGeom

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.pva import BasePva

import isaaclab_ovphysx.tensor_types as TT
from isaaclab_ovphysx.physics import OvPhysxManager as SimulationManager
from isaaclab_ovphysx.sim.views.ovphysx_view import OvPhysxView

from .kernels import pva_reset_kernel, pva_update_kernel
from .pva_data import PvaData

if TYPE_CHECKING:
    from isaaclab.sensors.pva import PvaCfg


class Pva(BasePva):
    """The OVPhysX Pose Velocity Acceleration (PVA) sensor.

    The sensor reports world-frame pose, body-frame linear and angular velocities,
    body-frame linear and angular accelerations, and projected gravity. Unlike the
    :class:`~isaaclab.sensors.imu.BaseImu` sensor, linear acceleration here is the
    coordinate acceleration of the sensor frame (zero at rest, ``-g`` in freefall)
    and does not include the gravity bias.

    The sensor can be attached to any prim path with a rigid ancestor in its tree.
    If the provided path is not a rigid body, the closest rigid-body ancestor is
    used for simulation queries. The fixed transform from that ancestor to the
    target prim is computed once during initialization and composed with the
    configured sensor offset.

    .. note::

        Linear and angular accelerations are computed using numerical differentiation
        of the corresponding velocities. Consequently, the PVA sensor accuracy
        depends on the chosen physics timestep. For sufficient accuracy, we
        recommend keeping the timestep at least 200 Hz.
    """

    cfg: PvaCfg
    """The configuration parameters."""

    __backend_name__: str = "ovphysx"
    """The name of the backend for the PVA sensor."""

    def __init__(self, cfg: PvaCfg):
        """Initializes the PVA sensor.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)
        self._data = PvaData()
        self._rigid_parent_expr: str | None = None
        # Sentinel — set in :meth:`_initialize_impl`; ``None`` means the sensor has not been bound yet
        # (used by :meth:`_debug_vis_callback` to safely no-op before init).
        self._root_view: OvPhysxView | None = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Pva sensor @ '{self.cfg.prim_path}': \n"
            f"\tbinding pattern   : {self._rigid_parent_expr}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._num_bodies}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> PvaData:
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
            pva_reset_kernel,
            dim=self._num_envs,
            inputs=[
                env_mask,
                self._data._pos_w,
                self._data._quat_w,
                self._data._lin_vel_b,
                self._data._ang_vel_b,
                self._data._lin_acc_b,
                self._data._ang_acc_b,
                self._data._projected_gravity_b,
                self._prev_lin_vel_w,
                self._prev_ang_vel_w,
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
        self._num_bodies = self._root_view.binding_for(TT.RIGID_BODY_POSE).count

        if self._num_bodies != self._num_envs:
            raise ValueError(
                f"OvPhysx Pva: pattern '{pattern}' matched {self._num_bodies} rigid bodies; expected exactly one"
                f" body per environment (num_envs={self._num_envs}). Check that the prim path or its rigid-body"
                " ancestor is unique per env."
            )

        # PVA reports projected gravity as the unit direction vector (not the bias the IMU uses).
        gravity = SimulationManager.get_gravity()
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self._device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        gravity_dir_repeated = gravity_dir.repeat(self._num_bodies, 1)
        self._gravity_vec_w = wp.from_torch(gravity_dir_repeated.contiguous(), dtype=wp.vec3f)

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
        """Drop the OVPhysX view when physics stops."""
        super()._invalidate_initialize_callback(event)
        # Drop the view (and the bindings it caches) so a stale/destroyed handle is not held
        # across the reset; ``_initialize_impl`` rebuilds a fresh view on the next play.
        self._root_view = None

    def _update_buffers_impl(self, env_mask: wp.array | None = None):
        """Fills the buffers of the sensor data."""
        env_mask = self._resolve_indices_and_mask(None, env_mask)

        # ``OvPhysxView.read_into`` fills the structured-dtype buffer in place via a
        # cached float32 reinterpret; no manual float32 alias is needed.
        self._root_view.read_into(TT.RIGID_BODY_POSE, self._transforms)
        self._root_view.read_into(TT.RIGID_BODY_VELOCITY, self._velocities)
        # RIGID_BODY_COM_POSE is a CPU tensor type in the OVPhysX wheel.
        # For GPU simulations, stage on CPU then copy into the kernel buffer.
        self._root_view.read_into(TT.RIGID_BODY_COM_POSE, self._coms_read_view)
        if self._coms_read_view is not self._coms_buffer:
            wp.copy(self._coms_buffer, self._coms_read_view)

        wp.launch(
            pva_update_kernel,
            dim=self._num_envs,
            inputs=[
                env_mask,
                self._transforms,
                self._velocities,
                self._coms_buffer,
                self._offset_pos_b,
                self._offset_quat_b,
                self._gravity_vec_w,
                1.0 / self._dt,
                self._timestamp,
                self._prev_lin_vel_w,
                self._prev_ang_vel_w,
                self._data._pos_w,
                self._data._quat_w,
                self._data._lin_vel_b,
                self._data._ang_vel_b,
                self._data._lin_acc_b,
                self._data._ang_acc_b,
                self._data._projected_gravity_b,
            ],
            device=self._device,
        )

    def _initialize_buffers_impl(self):
        """Create buffers for storing data."""
        self._data.create_buffers(num_envs=self._num_bodies, device=self._device)

        # Sensor-internal buffers for velocity tracking (not exposed via data).
        self._prev_lin_vel_w = wp.zeros(self._num_bodies, dtype=wp.vec3f, device=self._device)
        self._prev_ang_vel_w = wp.zeros(self._num_bodies, dtype=wp.vec3f, device=self._device)

        offset_pos_torch = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._num_bodies, 1)
        offset_quat_torch = torch.tensor(list(self.cfg.offset.rot), device=self._device).repeat(self._num_bodies, 1)
        self._offset_pos_b = wp.from_torch(offset_pos_torch.contiguous(), dtype=wp.vec3f)
        self._offset_quat_b = wp.from_torch(offset_quat_torch.contiguous(), dtype=wp.quatf)

        # Structured-dtype buffers filled in place by :meth:`OvPhysxView.read_into`.
        self._transforms = wp.zeros(self._num_bodies, dtype=wp.transformf, device=self._device)
        self._velocities = wp.zeros(self._num_bodies, dtype=wp.spatial_vectorf, device=self._device)
        self._coms_buffer = wp.zeros(self._num_bodies, dtype=wp.transformf, device=self._device)

        # RIGID_BODY_COM_POSE is CPU-resident even on a GPU sim, so its binding requires a
        # CPU destination. On a GPU sim, stage the read into a pinned CPU buffer and copy into
        # the kernel buffer; on a CPU sim, read straight into the kernel buffer.
        if self._device == "cpu":
            self._coms_read_view = self._coms_buffer
        else:
            self._coms_read_view = wp.zeros(self._num_bodies, dtype=wp.transformf, device="cpu", pinned=True)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "acceleration_visualizer"):
                self.acceleration_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.acceleration_visualizer.set_visibility(True)
        else:
            if hasattr(self, "acceleration_visualizer"):
                self.acceleration_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # safely return if the sensor has not been bound yet (matches the PhysX `_view is None` idiom)
        if self._root_view is None:
            return
        # get marker location
        # -- base state (convert warp -> torch for visualization)
        base_pos_w = self._data.pos_w.torch.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales
        default_scale = self.acceleration_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self._data.lin_acc_b.torch.shape[0], 1)
        # get up axis of current stage
        up_axis = UsdGeom.GetStageUpAxis(self.stage)
        # arrow-direction; filter out bodies with effectively zero accel (no defined direction)
        pos_w_torch = self._data.pos_w.torch
        accel_w = math_utils.quat_apply(self._data.quat_w.torch, self._data.lin_acc_b.torch)
        valid_indices = (torch.linalg.norm(accel_w, dim=-1) > 1e-5).nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            return
        pos_filtered = pos_w_torch.index_select(0, valid_indices)
        accel_filtered = accel_w.index_select(0, valid_indices)
        rotation_matrix = math_utils.create_rotation_matrix_from_view(
            pos_filtered,
            pos_filtered + accel_filtered,
            up_axis=up_axis,
            device=self._device,
        )
        quat_opengl = math_utils.quat_from_matrix(rotation_matrix)
        quat_w = math_utils.convert_camera_frame_orientation_convention(quat_opengl, "opengl", "world")
        # display markers
        self.acceleration_visualizer.visualize(
            base_pos_w.index_select(0, valid_indices),
            quat_w,
            arrow_scale.index_select(0, valid_indices),
        )
