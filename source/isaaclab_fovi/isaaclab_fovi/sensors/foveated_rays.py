# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Foveated ray computation from SamplingCoords.

Converts fovi's SamplingCoords.cartesian (N, 2) in [-1, 1] into a Warp ray
array suitable for Newton's SensorTiledCamera.  The ray array has shape
(camera_count=1, height=N, width=1, 2) where [:, :, :, 0] is origin (always
zero) and [:, :, :, 1] is the normalized direction.

Supports two projection models:
- ``"pinhole"``: rectilinear projection ``ray_xy = coord * tan(hfov/2)``
- ``"fisheye_equidistant"``: equidistant fisheye ``θ = r * (hfov/2)``
"""

from __future__ import annotations

import math

import torch
import warp as wp


@wp.kernel(enable_backward=False)
def _fill_foveated_rays(
    coords_x: wp.array(dtype=wp.float32),
    coords_y: wp.array(dtype=wp.float32),
    h_tan: wp.float32,
    out_rays: wp.array(dtype=wp.vec3f, ndim=4),
):
    """Warp kernel: map fovi (x, y) to OpenGL camera-space ray directions.

    Standard pinhole projection in OpenGL frame (+X=right, +Y=up, -Z=forward):
      cart_x → ray_x (horizontal)
      cart_y → ray_y (vertical)
    """
    i = wp.tid()
    x = coords_x[i]
    y = coords_y[i]
    direction = wp.normalize(wp.vec3f(x * h_tan, y * h_tan, -1.0))
    out_rays[0, i, 0, 0] = wp.vec3f(0.0, 0.0, 0.0)  # origin
    out_rays[0, i, 0, 1] = direction


@wp.kernel(enable_backward=False)
def _fill_foveated_rays_fisheye_equidistant(
    coords_x: wp.array(dtype=wp.float32),
    coords_y: wp.array(dtype=wp.float32),
    half_fov: wp.float32,
    out_rays: wp.array(dtype=wp.vec3f, ndim=4),
):
    """Warp kernel: equidistant fisheye projection.

    Maps normalized coords to angles linearly: θ = r * half_fov, where
    r = sqrt(x² + y²).  Unlike pinhole, the z-component varies per ray
    (-cos(θ)) and the lateral scaling is sin(θ)/r instead of constant h_tan.
    """
    i = wp.tid()
    x = coords_x[i]
    y = coords_y[i]
    r = wp.sqrt(x * x + y * y)
    theta = r * half_fov
    # sin(theta)/r → half_fov as r → 0 (L'Hôpital)
    if r > 1.0e-7:
        scale = wp.sin(theta) / r
    else:
        scale = half_fov
    direction = wp.normalize(wp.vec3f(x * scale, y * scale, -wp.cos(theta)))
    out_rays[0, i, 0, 0] = wp.vec3f(0.0, 0.0, 0.0)  # origin
    out_rays[0, i, 0, 1] = direction


@wp.kernel(enable_backward=False)
def _fill_explicit_rays(
    directions: wp.array(dtype=wp.vec3f),
    out_rays: wp.array(dtype=wp.vec3f, ndim=4),
):
    i = wp.tid()
    out_rays[0, i, 0, 0] = wp.vec3f(0.0, 0.0, 0.0)
    out_rays[0, i, 0, 1] = directions[i]


def compute_explicit_ray_array(
    ray_directions: torch.Tensor,
    *,
    device: str,
) -> wp.array:
    """Build a Newton ray array from explicit normalized camera-space rays."""
    if ray_directions.ndim != 2 or ray_directions.shape[1] != 3:
        raise ValueError(f"ray_directions must have shape (N, 3), got {tuple(ray_directions.shape)}.")
    directions = wp.from_torch(
        ray_directions.contiguous().float(),
        dtype=wp.vec3f,
    )
    num_rays = ray_directions.shape[0]
    out_rays = wp.zeros((1, num_rays, 1, 2), dtype=wp.vec3f, device=device)
    wp.launch(
        kernel=_fill_explicit_rays,
        dim=num_rays,
        inputs=[directions, out_rays],
        device=device,
    )
    return out_rays


def compute_foveated_ray_array(
    sampling_coords_cartesian: torch.Tensor,
    camera_fov_rad: float,
    device: str,
    projection: str = "pinhole",
) -> wp.array:
    """Build a ``(1, N, 1, 2)`` Warp ray array from foveated sample coordinates.

    Args:
        sampling_coords_cartesian: ``(N, 2)`` tensor in ``[-1, 1]`` from
            :class:`fovi.sensing.coords.SamplingCoords`.  Column 0 is the
            horizontal (x) coordinate, column 1 is vertical (y).
        camera_fov_rad: Horizontal field-of-view of the camera in **radians**
            (derived from spawn optics / intrinsic matrix).
        device: Warp device string (e.g. ``"cuda:0"``).
        projection: Projection model — ``"pinhole"`` (rectilinear) or
            ``"fisheye_equidistant"`` (linear angle mapping).

    Returns:
        ``wp.array(dtype=vec3f, ndim=4)`` of shape ``(1, N, 1, 2)``.
    """
    N = sampling_coords_cartesian.shape[0]

    # Flatten coords to 1-D Warp arrays
    coords_x = wp.from_torch(sampling_coords_cartesian[:, 0].contiguous().float(), dtype=wp.float32)
    coords_y = wp.from_torch(sampling_coords_cartesian[:, 1].contiguous().float(), dtype=wp.float32)

    out_rays = wp.zeros((1, N, 1, 2), dtype=wp.vec3f, device=device)

    if projection == "pinhole":
        h_tan = math.tan(camera_fov_rad / 2.0)
        wp.launch(
            kernel=_fill_foveated_rays,
            dim=N,
            inputs=[coords_x, coords_y, h_tan, out_rays],
            device=device,
        )
    elif projection == "fisheye_equidistant":
        half_fov = camera_fov_rad / 2.0
        wp.launch(
            kernel=_fill_foveated_rays_fisheye_equidistant,
            dim=N,
            inputs=[coords_x, coords_y, half_fov, out_rays],
            device=device,
        )
    else:
        raise ValueError(f"Unknown projection {projection!r}. Supported: 'pinhole', 'fisheye_equidistant'.")

    return out_rays
