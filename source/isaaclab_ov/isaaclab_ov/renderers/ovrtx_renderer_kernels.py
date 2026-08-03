# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for OVRTX rendering pipeline."""

from typing import Any

import warp as wp

# Segmentation colorization is shared across renderer backends to keep colors visually consistent.
from isaaclab.renderers.segmentation_colors import random_color_from_id_wp


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
        wp.float64(r00),
        wp.float64(r10),
        wp.float64(r20),
        _0,
        wp.float64(r01),
        wp.float64(r11),
        wp.float64(r21),
        _0,
        wp.float64(r02),
        wp.float64(r12),
        wp.float64(r22),
        _0,
        wp.float64(float(pos[0])),
        wp.float64(float(pos[1])),
        wp.float64(float(pos[2])),
        _1,
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
def extract_all_tiles_kernel(
    tiled_buffer: wp.array(dtype=Any, ndim=3),  # type: ignore
    output_buffer: wp.array(dtype=Any, ndim=4),  # type: ignore
    num_cols: int,
    tile_width: int,
    tile_height: int,
):
    """Extract ALL tiles from a tiled buffer into per-env tiles in a single kernel launch.

    Generic over the tile channel layout (RGB, RGBA, or single-channel depth) and dtype (e.g. uint8 color,
    float32 depth/HDR color, or float16 HDR color widened to float32 on output). The channel count is taken
    from ``output_buffer`` rather than passed explicitly, and each element is cast to the output dtype, so the
    same kernel body serves every tile-extraction case; see the :func:`warp.overload` registrations below for
    the concrete dtype pairs this is compiled for.

    Precondition:
        ``output_buffer``'s channel count (last dimension) must not exceed ``tiled_buffer``'s (the per-thread
        channel loop below indexes ``tiled_buffer`` up to that bound). In this package, always launch this
        kernel through ``OVRTXRenderer._launch_extract_all_tiles``, which validates this before every launch
        instead of relying on each call site to remember to check. Passing a wider output buffer than the
        tiled input reads out of bounds on the GPU.

    Args:
        tiled_buffer: 3D array of shape (H, W, C) holding all tiles packed into one buffer.
        output_buffer: 4D array of shape (num_envs, H, W, C) to receive the per-env tiles, with C no greater
            than ``tiled_buffer``'s channel count.
        num_cols: number of columns in the tiled buffer.
        tile_width: width of each tile.
        tile_height: height of each tile.
    """
    env_idx, y, x = wp.tid()
    tile_x = env_idx % num_cols
    tile_y = env_idx // num_cols
    src_x = tile_x * tile_width + x
    src_y = tile_y * tile_height + y
    for channel in range(output_buffer.shape[3]):
        output_buffer[env_idx, y, x, channel] = output_buffer.dtype(tiled_buffer[src_y, src_x, channel])


# uint8 color tiles (e.g. RGB/RGBA, semantic segmentation).
wp.overload(
    extract_all_tiles_kernel,
    [wp.array(dtype=wp.uint8, ndim=3), wp.array(dtype=wp.uint8, ndim=4), int, int, int],
)
# float32 tiles (e.g. depth, normals, HDR color).
wp.overload(
    extract_all_tiles_kernel,
    [wp.array(dtype=wp.float32, ndim=3), wp.array(dtype=wp.float32, ndim=4), int, int, int],
)
# float16 tiles (e.g. HDR color), widened to a float32 output buffer.
wp.overload(
    extract_all_tiles_kernel,
    [wp.array(dtype=wp.float16, ndim=3), wp.array(dtype=wp.float32, ndim=4), int, int, int],
)
# uint32 tiles (e.g. raw instance segmentation IDs).
wp.overload(
    extract_all_tiles_kernel,
    [wp.array(dtype=wp.uint32, ndim=3), wp.array(dtype=wp.uint32, ndim=4), int, int, int],
)
# uint32 tiles cast to an int32 output buffer (raw semantic segmentation IDs; matches Isaac RTX's int32
# non-colorized semantic output, whose per-pixel value is the semantic ID).
wp.overload(
    extract_all_tiles_kernel,
    [wp.array(dtype=wp.uint32, ndim=3), wp.array(dtype=wp.int32, ndim=4), int, int, int],
)


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
def generate_random_colors_from_ids_kernel(
    input_ids: wp.array(dtype=wp.uint32, ndim=3),  # type: ignore
    output_colors: wp.array(dtype=wp.uint32, ndim=3),  # type: ignore
):
    """Generate random colors given IDs (e.g. semantic IDs).

    Args:
        input_ids: 3D uint32 array for semantic IDs per pixel.
        output_colors: 3D uint32 array for colors per pixel; each word is ``r | (g<<8) | (b<<16) | (a<<24)``.
    """
    i, j, k = wp.tid()
    output_colors[i, j, k] = random_color_from_id_wp(input_ids[i, j, k])


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
    ovrtx_transforms[i] = wp.transpose(wp.mat44d(wp.transform_to_matrix(transform)))
