# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for mock PhysX views."""

from __future__ import annotations

import warp as wp


@wp.kernel
def init_identity_transforms_1d(out: wp.array(dtype=wp.transformf)):
    """Initialize 1D array of transforms to identity (zero position, identity quaternion).

    Args:
        out: Output array of shape (N,) to initialize.
    """
    i = wp.tid()
    out[i] = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))


@wp.kernel
def init_identity_transforms_2d(out: wp.array2d(dtype=wp.transformf)):
    """Initialize 2D array of transforms to identity (zero position, identity quaternion).

    Args:
        out: Output array of shape (N, L) to initialize.
    """
    i, j = wp.tid()
    out[i, j] = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))


@wp.kernel
def init_zero_spatial_vectors_1d(out: wp.array(dtype=wp.spatial_vectorf)):
    """Initialize 1D array of spatial vectors to zero.

    Args:
        out: Output array of shape (N,) to initialize.
    """
    i = wp.tid()
    out[i] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def init_zero_spatial_vectors_2d(out: wp.array2d(dtype=wp.spatial_vectorf)):
    """Initialize 2D array of spatial vectors to zero.

    Args:
        out: Output array of shape (N, L) to initialize.
    """
    i, j = wp.tid()
    out[i, j] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def scatter_transforms_1d(
    src: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.transformf),
):
    """Scatter transforms from src to dst at specified indices.

    Args:
        src: Source transforms, shape (M,).
        indices: Target indices in dst, shape (M,).
        dst: Destination transforms, shape (N,) where N >= max(indices).
    """
    i = wp.tid()
    dst[indices[i]] = src[i]


@wp.kernel
def scatter_spatial_vectors_1d(
    src: wp.array(dtype=wp.spatial_vectorf),
    indices: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.spatial_vectorf),
):
    """Scatter spatial vectors from src to dst at specified indices.

    Args:
        src: Source spatial vectors, shape (M,).
        indices: Target indices in dst, shape (M,).
        dst: Destination spatial vectors, shape (N,) where N >= max(indices).
    """
    i = wp.tid()
    dst[indices[i]] = src[i]


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
def copy_transforms_1d(src: wp.array(dtype=wp.transformf), dst: wp.array(dtype=wp.transformf)):
    """Copy transforms from src to dst.

    Args:
        src: Source transforms, shape (N,).
        dst: Destination transforms, shape (N,).
    """
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def copy_spatial_vectors_1d(src: wp.array(dtype=wp.spatial_vectorf), dst: wp.array(dtype=wp.spatial_vectorf)):
    """Copy spatial vectors from src to dst.

    Args:
        src: Source spatial vectors, shape (N,).
        dst: Destination spatial vectors, shape (N,).
    """
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def copy_transforms_2d(src: wp.array2d(dtype=wp.transformf), dst: wp.array2d(dtype=wp.transformf)):
    """Copy 2D transforms from src to dst.

    Args:
        src: Source transforms, shape (N, L).
        dst: Destination transforms, shape (N, L).
    """
    i, j = wp.tid()
    dst[i, j] = src[i, j]


@wp.kernel
def copy_spatial_vectors_2d(src: wp.array2d(dtype=wp.spatial_vectorf), dst: wp.array2d(dtype=wp.spatial_vectorf)):
    """Copy 2D spatial vectors from src to dst.

    Args:
        src: Source spatial vectors, shape (N, L).
        dst: Destination spatial vectors, shape (N, L).
    """
    i, j = wp.tid()
    dst[i, j] = src[i, j]


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
