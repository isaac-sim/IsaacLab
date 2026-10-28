# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

from ..sensor_base import SensorBase
from .kernels import pva_reset_kernel
from .pva_data import PvaData

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

    supported_data_types = frozenset({"pose"})
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

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        """Reset enabled measurements and timing for the selected environments."""
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(None, env_mask)
        wp.launch(pva_reset_kernel, self._num_envs, [env_mask, self._data.buffers], device=self._device)

    def set_debug_vis(self, debug_vis: bool) -> bool:
        if debug_vis and "acceleration" not in self.cfg.data_types:
            raise ValueError("PVA debug visualization requires the 'acceleration' output")
        return super().set_debug_vis(debug_vis)

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        self._data_dirty = False
        if self._data.buffers.pose_w:
            wp.launch(pva_reset_kernel, self._num_envs, [self._ALL_ENV_MASK, self._data.buffers], device=self._device)

    """
    Properties
    """

    @property
    def data(self) -> PvaData:
        """Refresh and return the enabled measurements."""
        self._update_outdated_buffers()
        return self._data
