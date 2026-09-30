# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep deforming Gaussians inside stable local bounds without changing world shape."""

import numpy as np
import warp as wp

INF = wp.constant(float("inf"))


@wp.kernel
def gaussian_bounds(x: wp.array[wp.vec3], s: wp.array[wp.vec3], bounds: wp.array2d[float]):
    i = wp.tid()
    reach = 3.0 * wp.max(s[i][0], wp.max(s[i][1], s[i][2]))
    for axis in range(3):
        wp.atomic_min(bounds, 0, axis, x[i][axis] - reach)
        wp.atomic_max(bounds, 1, axis, x[i][axis] + reach)


@wp.kernel
def gaussian_bounds_reduced(x: wp.array[wp.vec3], s: wp.array[wp.vec3], bounds: wp.array2d[float]):
    tid = wp.tid()
    threads = (x.shape[0] + 15) // 16
    lower = wp.vec3(INF)
    upper = wp.vec3(-INF)
    for j in range(16):
        i = tid + j * threads
        if i < x.shape[0]:
            reach = 3.0 * wp.max(s[i][0], wp.max(s[i][1], s[i][2]))
            for axis in range(3):
                lower[axis] = wp.min(lower[axis], x[i][axis] - reach)
                upper[axis] = wp.max(upper[axis], x[i][axis] + reach)
    for axis in range(3):
        wp.atomic_min(bounds, 0, axis, lower[axis])
        wp.atomic_max(bounds, 1, axis, upper[axis])


@wp.kernel
def normalize_gaussians(
    x: wp.array[wp.vec3],
    s: wp.array[wp.vec3],
    center: wp.vec3,
    initial_center: wp.vec3,
    factor: float,
    local_x: wp.array[wp.vec3],
    local_s: wp.array[wp.vec3],
):
    i = wp.tid()
    local_x[i] = (x[i] - center) / factor + initial_center
    local_s[i] = s[i] / factor


class GaussianLocalFrame:
    def __init__(self, xyz, scales, device):
        self.center = (xyz.min(0) + xyz.max(0)) / 2
        self.half = (xyz.max(0) - xyz.min(0)) / 2
        if not np.isfinite(self.half).all() or np.any(self.half <= 0):
            raise ValueError("Gaussian local frame needs finite nonzero initial bounds")
        reach = 3 * np.max(scales, axis=1)[:, None]
        self.extent = np.array([(xyz - reach).min(0), (xyz + reach).max(0)], np.float64)
        self.bounds = wp.empty((2, 3), dtype=float, device=device)
        self.positions = wp.empty(len(xyz), dtype=wp.vec3, device=device)
        self.scales = wp.empty_like(self.positions)
        self.transform = np.eye(4, dtype=np.float64)

    def evaluate(self, xyz, scales):
        self.bounds.assign(np.array([[np.inf] * 3, [-np.inf] * 3], np.float32))
        wp.launch(gaussian_bounds, dim=len(xyz), inputs=[xyz, scales, self.bounds], device=xyz.device)
        extent = self.bounds.numpy().astype(np.float64)
        center = extent.mean(0)
        factor = float(np.max((extent[1] - extent[0]) / 2 / self.half))
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid deformed Gaussian bounds")
        wp.launch(
            normalize_gaussians,
            dim=len(xyz),
            inputs=[xyz, scales, wp.vec3(*center), wp.vec3(*self.center), factor, self.positions, self.scales],
            device=xyz.device,
        )
        self.transform[:3, :3] = np.eye(3) * factor
        self.transform[3, :3] = center - factor * self.center
        return self.positions, self.scales, self.transform
