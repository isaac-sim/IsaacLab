# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.utils.warp import ProxyArray


class LidarData:
    """Batched lidar returns for one sensor.

    Buffers live on the simulation device. Ray ``r`` of environment ``i`` is ``points[i, r]``.
    A missed ray is non-finite, matching a ray-caster miss, and ``valid[i, r]`` is false.
    """

    def __init__(self, num_envs: int, num_rays: int, device: str):
        """Allocate one point slot for every ray in every environment.

        Args:
            num_envs: Number of environments.
            num_rays: Ray slots in each environment, including misses.
            device: Simulation device that stores the buffers.
        """
        points = wp.full(
            (num_envs, num_rays),
            wp.vec3f(float("inf"), float("inf"), float("inf")),
            dtype=wp.vec3f,
            device=device,
        )
        self._points = ProxyArray(points)
        self._intensity = ProxyArray(wp.zeros((num_envs, num_rays), dtype=wp.float32, device=device))
        self._valid = ProxyArray(wp.zeros((num_envs, num_rays), dtype=wp.bool, device=device))

    @property
    def points(self) -> ProxyArray:
        """World-frame returns as a ``vec3f`` array of shape ``(num_envs, num_rays)``.

        The Torch view has shape ``(num_envs, num_rays, 3)``. Misses are non-finite.
        """
        return self._points

    @property
    def intensity(self) -> ProxyArray:
        """Return intensity, shape ``(num_envs, num_rays)``. Misses are zero."""
        return self._intensity

    @property
    def valid(self) -> ProxyArray:
        """Whether each ray hit something, shape ``(num_envs, num_rays)``."""
        return self._valid
