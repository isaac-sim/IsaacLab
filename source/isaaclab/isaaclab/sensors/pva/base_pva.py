# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import torch
import warp as wp

from pxr import UsdGeom

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers

from ..sensor_base import SensorBase
from .base_pva_data import BasePvaData
from .kernels import pva_reset_kernel

if TYPE_CHECKING:
    from .pva_cfg import PvaCfg


class BasePva(SensorBase):
    """The Pose Velocity Acceleration (PVA) sensor.

    The sensor can be attached to any prim path with a rigid ancestor in its tree and produces world-frame
    pose, body-frame velocities, body-frame coordinate accelerations, and projected gravity. This differs
    from the :class:`~isaaclab.sensors.imu.BaseImu` sensor, which reports proper acceleration.

    If the provided path is not a rigid body, the closest rigid-body ancestor is used for simulation queries.
    The fixed transform from that ancestor to the target prim is computed once during initialization and
    composed with the configured sensor offset.

    .. note::

        When enabled, accelerations are read directly from the physics solver and transported
        from the body center of mass to the sensor frame, so they do not depend on the sensor
        update period.

    .. note::

        The user can configure the sensor offset in the configuration file. The offset is applied relative to the
        rigid source prim. If the target prim is not a rigid body, the offset is composed with the fixed transform
        from the rigid ancestor to the target prim. The offset is applied in the body frame of the rigid source prim.
        The offset is defined as a position vector and a quaternion rotation, which
        are applied in the order: position, then rotation. The position is applied as a translation
        in the body frame of the rigid source prim, and the rotation is applied as a rotation
        in the body frame of the rigid source prim.

    """

    cfg: PvaCfg
    """The configuration parameters."""

    __backend_name__: str = "base"
    """The name of the backend for the PVA sensor."""

    supported_data_types: ClassVar[frozenset[str]] = frozenset({"pose"})
    """Output groups implemented by the concrete backend; extensions may add groups."""

    def __init__(self, cfg: PvaCfg):
        # Validate before SensorBase registers callbacks or enables debug visualization.
        self._initialize_handle = self._invalidate_initialize_handle = self._prim_deletion_handle = None
        self._debug_vis_handle = None
        requested = frozenset(cfg.data_types)
        unsupported = requested - self.supported_data_types
        if unsupported:
            raise ValueError(f"PVA backend {self.__backend_name__!r} does not support outputs {sorted(unsupported)}")
        if "pose" not in requested:
            raise ValueError("PVA data_types must include 'pose'")
        if cfg.debug_vis and "acceleration" not in requested:
            raise ValueError("PVA debug visualization requires the 'acceleration' output")
        super().__init__(cfg)

    """
    Properties
    """

    @property
    def data(self) -> BasePvaData:
        """Refresh and return the enabled measurements."""
        self._update_outdated_buffers()
        return self._data

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        """Reset enabled measurements and timing for the selected environments."""
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(None, env_mask)
        wp.launch(
            pva_reset_kernel,
            self._num_envs,
            [
                env_mask,
                self._data.pose_w.warp,
                self._data._velocity_b,
                self._data._acceleration_b,
                self._data._gravity_b,
            ],
            device=self._device,
        )

    def set_debug_vis(self, debug_vis: bool) -> bool:
        if debug_vis and "acceleration" not in self.cfg.data_types:
            raise ValueError("PVA debug visualization requires the 'acceleration' output")
        return super().set_debug_vis(debug_vis)

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        self._data_dirty = False
        if self._data.pose_w is not None:
            wp.launch(
                pva_reset_kernel,
                self._num_envs,
                [
                    self._ALL_ENV_MASK,
                    self._data.pose_w.warp,
                    self._data._velocity_b,
                    self._data._acceleration_b,
                    self._data._gravity_b,
                ],
                device=self._device,
            )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "acceleration_visualizer"):
                self.acceleration_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.acceleration_visualizer.set_visibility(True)
        elif hasattr(self, "acceleration_visualizer"):
            self.acceleration_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._is_initialized:
            return
        self._update_outdated_buffers()
        pos_w = self._data.pos_w.torch
        accel_w = math_utils.quat_apply(self._data.quat_w.torch, self._data.lin_acc_b.torch)
        valid = (torch.linalg.norm(accel_w, dim=-1) > 1e-5).nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            return
        pos = pos_w.index_select(0, valid)
        accel = accel_w.index_select(0, valid)
        rotation = math_utils.create_rotation_matrix_from_view(
            pos, pos + accel, up_axis=UsdGeom.GetStageUpAxis(self.stage), device=self._device
        )
        quat = math_utils.convert_camera_frame_orientation_convention(
            math_utils.quat_from_matrix(rotation), "opengl", "world"
        )
        marker_pos = pos.clone()
        marker_pos[:, 2] += 0.5
        scale = torch.tensor(self.acceleration_visualizer.cfg.markers["arrow"].scale, device=self.device)
        self.acceleration_visualizer.visualize(marker_pos, quat, scale.repeat(valid.numel(), 1))
