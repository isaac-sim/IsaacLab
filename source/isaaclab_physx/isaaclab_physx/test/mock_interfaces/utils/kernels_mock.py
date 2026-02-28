# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for mock PhysX views."""

from __future__ import annotations

import warp as wp


@wp.kernel
def init_identity_transforms_1d_flat(out: wp.array2d(dtype=wp.float32)):
    """Initialize (N, 7) float32 array: pos=0, quat=(0,0,0,1)."""
    i = wp.tid()
    out[i, 0] = 0.0
    out[i, 1] = 0.0
    out[i, 2] = 0.0
    out[i, 3] = 0.0
    out[i, 4] = 0.0
    out[i, 5] = 0.0
    out[i, 6] = 1.0


@wp.kernel
def init_identity_transforms_2d_flat(out: wp.array3d(dtype=wp.float32)):
    """Initialize (N, L, 7) float32 array: pos=0, quat=(0,0,0,1)."""
    i, j = wp.tid()
    out[i, j, 0] = 0.0
    out[i, j, 1] = 0.0
    out[i, j, 2] = 0.0
    out[i, j, 3] = 0.0
    out[i, j, 4] = 0.0
    out[i, j, 5] = 0.0
    out[i, j, 6] = 1.0


@wp.kernel
def scatter_floats_2d(
    src: wp.array2d(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
    dst: wp.array2d(dtype=wp.float32),
):
    """Scatter 2D float arrays from src to dst rows at specified indices.

    Args:
        src: Source array, shape (M, J).
        indices: Target row indices in dst, shape (M,).
        dst: Destination array, shape (N, J) where N >= max(indices).
    """
    i, j = wp.tid()
    dst[indices[i], j] = src[i, j]


@wp.kernel
def init_identity_inertias_1d(out: wp.array2d(dtype=wp.float32)):
    """Initialize 1D array of inertia tensors to identity (9 values per body).

    Sets diagonal elements to 1.0, off-diagonal to 0.0.
    Shape: (N, 9) where 9 represents flattened 3x3 matrix.

    Args:
        out: Output array of shape (N, 9) to initialize.
    """
    i = wp.tid()
    # Flattened row-major 3x3 identity: [1,0,0,0,1,0,0,0,1]
    out[i, 0] = 1.0  # [0,0]
    out[i, 1] = 0.0
    out[i, 2] = 0.0
    out[i, 3] = 0.0
    out[i, 4] = 1.0  # [1,1]
    out[i, 5] = 0.0
    out[i, 6] = 0.0
    out[i, 7] = 0.0
    out[i, 8] = 1.0  # [2,2]


@wp.kernel
def init_identity_inertias_2d(out: wp.array3d(dtype=wp.float32)):
    """Initialize 2D array of inertia tensors to identity (9 values per body per link).

    Sets diagonal elements to 1.0, off-diagonal to 0.0.
    Shape: (N, L, 9) where 9 represents flattened 3x3 matrix.

    Args:
        out: Output array of shape (N, L, 9) to initialize.
    """
    i, j = wp.tid()
    # Flattened row-major 3x3 identity: [1,0,0,0,1,0,0,0,1]
    out[i, j, 0] = 1.0  # [0,0]
    out[i, j, 1] = 0.0
    out[i, j, 2] = 0.0
    out[i, j, 3] = 0.0
    out[i, j, 4] = 1.0  # [1,1]
    out[i, j, 5] = 0.0
    out[i, j, 6] = 0.0
    out[i, j, 7] = 0.0
    out[i, j, 8] = 1.0  # [2,2]
