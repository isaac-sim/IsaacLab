# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels and device constant for OVRTX renderer."""

import numpy as np
import warp as wp

DEVICE = "cuda:0"


@wp.kernel
def create_camera_transforms_kernel(
    positions: wp.array(dtype=wp.vec3),  # type: ignore
    orientations: wp.array(dtype=wp.quatf),  # type: ignore
    transforms: wp.array(dtype=wp.mat44d),  # type: ignore
):
    """Build camera 4x4 transforms from positions and quaternions (column-major for OVRTX)."""
    i = wp.tid()
    pos = positions[i]
    quat = orientations[i]
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qw * qz)
    r02 = 2.0 * (qx * qz + qw * qy)
    r10 = 2.0 * (qx * qy + qw * qz)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qw * qx)
    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    _0 = wp.float64(0.0)
    _1 = wp.float64(1.0)
    transforms[i] = wp.mat44d(  # type: ignore
        wp.float64(r00), wp.float64(r10), wp.float64(r20), _0,
        wp.float64(r01), wp.float64(r11), wp.float64(r21), _0,
        wp.float64(r02), wp.float64(r12), wp.float64(r22), _0,
        wp.float64(float(pos[0])), wp.float64(float(pos[1])), wp.float64(float(pos[2])), _1,
    )


@wp.kernel
def extract_tile_from_tiled_buffer_kernel(
    tiled_buffer: wp.array(dtype=wp.uint8, ndim=3),  # type: ignore
    tile_buffer: wp.array(dtype=wp.uint8, ndim=3),  # type: ignore
    tile_x: int,
    tile_y: int,
    tile_width: int,
    tile_height: int,
):
    """Extract one RGBA tile from a tiled buffer."""
    y, x = wp.tid()
    src_x = tile_x * tile_width + x
    src_y = tile_y * tile_height + y
    tile_buffer[y, x, 0] = tiled_buffer[src_y, src_x, 0]
    tile_buffer[y, x, 1] = tiled_buffer[src_y, src_x, 1]
    tile_buffer[y, x, 2] = tiled_buffer[src_y, src_x, 2]
    tile_buffer[y, x, 3] = tiled_buffer[src_y, src_x, 3]


@wp.kernel
def extract_depth_tile_from_tiled_buffer_kernel(
    tiled_buffer: wp.array(dtype=wp.float32, ndim=2),  # type: ignore
    tile_buffer: wp.array(dtype=wp.float32, ndim=3),  # type: ignore
    tile_x: int,
    tile_y: int,
    tile_width: int,
    tile_height: int,
):
    """Extract one depth tile from a tiled depth buffer."""
    y, x = wp.tid()
    src_x = tile_x * tile_width + x
    src_y = tile_y * tile_height + y
    tile_buffer[y, x, 0] = tiled_buffer[src_y, src_x]


@wp.kernel
def sync_newton_transforms_kernel(
    ovrtx_transforms: wp.array(dtype=wp.mat44d),  # type: ignore
    newton_body_indices: wp.array(dtype=wp.int32),  # type: ignore
    newton_body_q: wp.array(dtype=wp.transformf),  # type: ignore
):
    """Sync Newton physics body transforms to OVRTX 4x4 column-major matrices."""
    i = wp.tid()
    body_idx = newton_body_indices[i]
    transform = newton_body_q[body_idx]
    ovrtx_transforms[i] = wp.transpose(wp.mat44d(wp.math.transform_to_matrix(transform)))


def normalize_depth_to_uint8(
    depth_np: np.ndarray,
) -> tuple[np.ndarray, float | None, float | None]:
    """Normalize depth to uint8 [0, 255] for visualization; invalid (inf/nan) -> 0.
    Returns (normalized_uint8, depth_min, depth_max); min/max are None if no valid pixels.
    """
    depth_valid = np.isfinite(depth_np)
    depth_min = depth_max = None
    if np.any(depth_valid):
        depth_min = float(depth_np[depth_valid].min())
        depth_max = float(depth_np[depth_valid].max())
    depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)
    if depth_min is not None and depth_max is not None and depth_max > depth_min:
        depth_normalized[depth_valid] = (
            (depth_np[depth_valid] - depth_min) / (depth_max - depth_min) * 255
        ).astype(np.uint8)
    return depth_normalized, depth_min, depth_max
