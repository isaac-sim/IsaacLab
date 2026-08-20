# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


def transform_to_vec_quat(
    t: wp.array,
) -> tuple[wp.array, wp.array]:
    """Split a wp.transformf array into position (vec3f) and quaternion (quatf) arrays.

    Zero-copy: returns views into the same underlying memory.

    Args:
        t: Array of transforms (dtype=wp.transformf). Shape ``(N,)``, ``(N, M)``, or ``(N, M, K)``.

    Returns:
        Tuple of (positions, quaternions) as warp array views with matching dimensionality.

    Raises:
        TypeError: If *t* does not have dtype ``wp.transformf``.
    """
    if t.dtype != wp.transformf:
        raise TypeError(f"Expected wp.transformf array, got dtype={t.dtype}")
    floats = t.view(wp.float32)
    if t.ndim == 1:
        return floats[:, :3].view(wp.vec3f), floats[:, 3:].view(wp.quatf)
    if t.ndim == 2:
        return floats[:, :, :3].view(wp.vec3f), floats[:, :, 3:].view(wp.quatf)
    if t.ndim == 3:
        return floats[:, :, :, :3].view(wp.vec3f), floats[:, :, :, 3:].view(wp.quatf)
    raise ValueError(f"Expected 1D, 2D, or 3D transform array, got ndim={t.ndim}")
