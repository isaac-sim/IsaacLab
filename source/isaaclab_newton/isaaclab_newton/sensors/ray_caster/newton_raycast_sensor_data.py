# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.sensors.ray_caster.ray_caster_data import RayCasterData
from isaaclab.utils.leapp import leapp_tensor_semantics
from isaaclab.utils.warp import ProxyArray


class NewtonRaycastSensorData(RayCasterData):
    """Data container for the Newton BVH ray-cast sensor.

    Extends :class:`~isaaclab.sensors.RayCasterData` with surface normals,
    which the Newton BVH query provides at no extra cost.
    """

    @property
    @leapp_tensor_semantics(kind="state/sensor/ray_hit_normal")
    def ray_normals_w(self) -> ProxyArray:
        """Surface normals at the ray hit points in world frame.

        Shape is (N, B), dtype ``wp.vec3f``. In torch this resolves to (N, B, 3).
        Contains ``inf`` for missed hits.
        """
        return self._ray_normals_w_ta

    def create_buffers(self, num_envs: int, num_rays: int, device: str) -> None:
        super().create_buffers(num_envs, num_rays, device)
        self._ray_normals_w = wp.zeros((num_envs, num_rays), dtype=wp.vec3f, device=device)
        self._ray_normals_w_ta = ProxyArray(self._ray_normals_w)
