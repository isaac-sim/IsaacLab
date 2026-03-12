# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp


class RayCasterData:
    """Data container for the ray-cast sensor.

    All public properties return :class:`wp.array` backed by device memory. Use
    :func:`wp.to_torch` at the call-site when a PyTorch tensor is needed.
    """

    def __init__(self):
        self._pos_w: wp.array | None = None
        self._quat_w: wp.array | None = None
        self._ray_hits_w: wp.array | None = None

        self._pos_w_torch = None
        self._quat_w_torch = None
        self._ray_hits_w_torch = None

    @property
    def pos_w(self) -> wp.array | None:
        """Position of the sensor origin in world frame [m].

        Shape is (N,), dtype ``wp.vec3f``. In torch this resolves to (N, 3),
        where N is the number of sensors.
        """
        return self._pos_w

    @property
    def quat_w(self) -> wp.array | None:
        """Orientation of the sensor origin in quaternion (x, y, z, w) in world frame.

        Shape is (N,), dtype ``wp.quatf``. In torch this resolves to (N, 4),
        where N is the number of sensors.
        """
        return self._quat_w

    @property
    def ray_hits_w(self) -> wp.array | None:
        """The ray hit positions in the world frame [m].

        Shape is (N, B), dtype ``wp.vec3f``. In torch this resolves to (N, B, 3),
        where N is the number of sensors, B is the number of rays per sensor.
        Contains ``inf`` for missed hits.
        """
        return self._ray_hits_w

    def create_buffers(self, num_envs: int, num_rays: int, device: str) -> None:
        """Create internal warp buffers and corresponding zero-copy torch views.

        Args:
            num_envs: Number of environments / sensors.
            num_rays: Number of rays per sensor.
            device: Device for tensor storage.
        """
        self._device = device

        self._pos_w = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._quat_w = wp.zeros(num_envs, dtype=wp.quatf, device=device)
        self._ray_hits_w = wp.zeros((num_envs, num_rays), dtype=wp.vec3f, device=device)

        self._pos_w_torch = wp.to_torch(self._pos_w)
        self._quat_w_torch = wp.to_torch(self._quat_w)
        self._ray_hits_w_torch = wp.to_torch(self._ray_hits_w)
