# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.utils.leapp import (
    QUAT_XYZW_ELEMENT_NAMES,
    XYZ_ELEMENT_NAMES,
    leapp_tensor_semantics,
)
from isaaclab.utils.warp import ProxyArray


class RayCasterData:
    """Data container for the ray-cast sensor.

    Public properties return :class:`~isaaclab.utils.warp.ProxyArray` wrappers.
    Use ``.torch`` for a cached zero-copy :class:`torch.Tensor` view or
    ``.warp`` for the underlying :class:`warp.array`.
    """

    def __init__(self):
        self._pos_w: wp.array | None = None
        self._quat_w: wp.array | None = None
        self._ray_hits_w: wp.array | None = None

        # _pos_w_ta / _quat_w_ta / _ray_hits_w_ta are created in create_buffers().
        # Accessing the public properties before create_buffers() raises AttributeError.

    @property
    @leapp_tensor_semantics(kind="state/sensor/position", element_names=XYZ_ELEMENT_NAMES)
    def pos_w(self) -> ProxyArray:
        """Position of the sensor origin in world frame [m].

        Shape is (N,), dtype ``wp.vec3f``. In torch this resolves to (N, 3),
        where N is the number of sensors. Use ``.warp`` for the underlying
        ``wp.array`` or ``.torch`` for a cached zero-copy ``torch.Tensor`` view.
        """
        return self._pos_w_ta

    @property
    @leapp_tensor_semantics(kind="state/sensor/rotation", element_names=QUAT_XYZW_ELEMENT_NAMES)
    def quat_w(self) -> ProxyArray:
        """Orientation of the sensor origin in quaternion (x, y, z, w) in world frame.

        Shape is (N,), dtype ``wp.quatf``. In torch this resolves to (N, 4),
        where N is the number of sensors. Use ``.warp`` for the underlying
        ``wp.array`` or ``.torch`` for a cached zero-copy ``torch.Tensor`` view.
        """
        return self._quat_w_ta

    @property
    @leapp_tensor_semantics(kind="state/sensor/ray_hit_position")
    def ray_hits_w(self) -> ProxyArray:
        """The ray hit positions in the world frame [m].

        Shape is (N, B), dtype ``wp.vec3f``. In torch this resolves to (N, B, 3),
        where N is the number of sensors and B is the number of rays per sensor.
        Contains ``inf`` for missed hits. Use ``.warp`` for the underlying
        ``wp.array`` or ``.torch`` for a cached zero-copy ``torch.Tensor`` view.
        """
        return self._ray_hits_w_ta

    def create_buffers(self, num_envs: int, num_rays: int, device: str) -> None:
        """Create internal warp buffers and their :class:`ProxyArray` wrappers.

        Args:
            num_envs: Number of environments / sensors.
            num_rays: Number of rays per sensor.
            device: Device for tensor storage.
        """
        self._device = device

        self._pos_w = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._quat_w = wp.zeros(num_envs, dtype=wp.quatf, device=device)
        self._ray_hits_w = wp.zeros((num_envs, num_rays), dtype=wp.vec3f, device=device)

        self._pos_w_ta = ProxyArray(self._pos_w)
        self._quat_w_ta = ProxyArray(self._quat_w)
        self._ray_hits_w_ta = ProxyArray(self._ray_hits_w)
