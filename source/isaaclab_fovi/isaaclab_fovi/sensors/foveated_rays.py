# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton ray arrays for foveated cameras.

Newton's Warp renderer takes a ray array of shape (camera_count=1, height, width, 2),
where [..., 0] is the ray origin (always zero) and [..., 1] its unit direction.
"""

from __future__ import annotations

import torch
import warp as wp


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
    """Build a ``(1, N, 1, 2)`` Newton ray array from unit OpenGL camera-frame directions, shape (N, 3)."""
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
