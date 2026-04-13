# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp


class RayCasterData:
    """Data container for the ray-cast sensor.

    Public properties return :class:`torch.Tensor` objects that are zero-copy views of the
    underlying Warp buffers. Internal code accesses the raw ``wp.array`` buffers via the
    private ``_pos_w``, ``_quat_w``, and ``_ray_hits_w`` attributes.
    """

    def __init__(self):
        self._pos_w: wp.array | None = None
        self._quat_w: wp.array | None = None
        self._ray_hits_w: wp.array | None = None

        self._pos_w_torch: torch.Tensor | None = None
        self._quat_w_torch: torch.Tensor | None = None
        self._ray_hits_w_torch: torch.Tensor | None = None

    @property
    def pos_w(self) -> torch.Tensor | None:
        """Position of the sensor origin in world frame [m].

        Shape is (N, 3) where N is the number of sensors.
        Returns a zero-copy torch view of the underlying Warp buffer.
        """
        return self._pos_w_torch

    @property
    def quat_w(self) -> torch.Tensor | None:
        """Orientation of the sensor origin in quaternion (x, y, z, w) in world frame.

        Shape is (N, 4) where N is the number of sensors.
        Returns a zero-copy torch view of the underlying Warp buffer.
        """
        return self._quat_w_torch

    @property
    def ray_hits_w(self) -> torch.Tensor | None:
        """The ray hit positions in the world frame [m].

        Shape is (N, B, 3) where N is the number of sensors and B is the number of rays per sensor.
        Contains ``inf`` for missed hits.
        Returns a zero-copy torch view of the underlying Warp buffer.
        """
        return self._ray_hits_w_torch

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
