# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels and device constant for OVRTX renderer."""

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


@wp.func
def color_hash(seed: wp.uint32) -> wp.uint32:
    """Simple hash function for better distribution. Used for colorization."""
    # uint32 integers are promoted to uint64 to avoid overflow during multiplication.
    h = wp.uint64(seed)
    h = h ^ (h >> wp.uint64(16))
    h = h * wp.uint64(wp.uint32(0x85EBCA6B))
    h = h ^ (h >> wp.uint64(13))
    h = h * wp.uint64(wp.uint32(0xC2B2AE35))
    h = h ^ (h >> wp.uint64(16))
    return wp.uint32(h)


@wp.func
def random_color_from_id(input_id: wp.uint32) -> wp.uint32:
    """Generate random color from a single ID.

    Generate visually distinct colours by linearly spacing the hue channel in HSV space and then convert to RGB space.

    Args:
        input_id: uint32 semantic ID

    Returns:
        uint32 color: ``r | (g<<8) | (b<<16) | (a<<24)``
    """
    hash_val = color_hash(input_id)

    # Golden ratio inverse = 1.0 / 1.618033988749895 (Replicator constant)
    GOLDEN_RATIO_INV = wp.float64(1.0) / wp.float64(1.618033988749895)

    # Use golden ratio spacing for maximum hue spread
    hue_tmp = wp.float64(input_id) * GOLDEN_RATIO_INV
    hue = hue_tmp - wp.floor(hue_tmp)

    # Add hash-based perturbation for better distribution
    hue_perturbation = wp.float64(hash_val & wp.uint32(0xFFFF)) / wp.float64(65536.0)
    hue_tmp = hue + hue_perturbation * wp.float64(0.1)
    hue = hue_tmp - wp.floor(hue_tmp)

    # Use hash to determine saturation and value for maximum spread
    sat_part = wp.uint32((hash_val >> wp.uint32(16)) & wp.uint32(0xFF))
    val_part = wp.uint32((hash_val >> wp.uint32(8)) & wp.uint32(0xFF))

    # Saturation: 0.7 to 1.0 for vibrant colors
    saturation = wp.float64(0.7) + wp.float64(0.3) * (wp.float64(sat_part) / wp.float64(255.0))

    # Value: 0.8 to 1.0 for bright colors
    value = wp.float64(0.8) + wp.float64(0.2) * (wp.float64(val_part) / wp.float64(255.0))

    # HSV to RGB conversion (match Replicator: ``f`` uses pre-modulo ``int(hue*6)``, then ``i %= 6``).
    hue_i = wp.int32(hue * wp.float64(6.0))
    hue_f = hue * wp.float64(6.0) - wp.float64(hue_i)
    p = value * (wp.float64(1.0) - saturation)
    q = value * (wp.float64(1.0) - saturation * hue_f)
    t = value * (wp.float64(1.0) - saturation * (wp.float64(1.0) - hue_f))

    r = wp.float64(0.0)
    g = wp.float64(0.0)
    b = wp.float64(0.0)

    hue_i = hue_i % 6

    if hue_i == 0:
        r = value
        g = t
        b = p
    elif hue_i == 1:
        r = q
        g = value
        b = p
    elif hue_i == 2:
        r = p
        g = value
        b = t
    elif hue_i == 3:
        r = p
        g = q
        b = value
    elif hue_i == 4:
        r = t
        g = p
        b = value
    else:
        r = value
        g = p
        b = q

    ri = wp.min(255, wp.max(0, wp.int32(r * wp.float64(255.0))))
    gi = wp.min(255, wp.max(0, wp.int32(g * wp.float64(255.0))))
    bi = wp.min(255, wp.max(0, wp.int32(b * wp.float64(255.0))))
    ai = wp.int32(255)

    color = (
        wp.uint32(ri)
        | (wp.uint32(gi) << wp.uint32(8))
        | (wp.uint32(bi) << wp.uint32(16))
        | (wp.uint32(ai) << wp.uint32(24))
    )
    return color


@wp.kernel
def generate_random_colors_from_ids_kernel(
    input_ids: wp.array(dtype=wp.uint32, ndim=2),  # type: ignore
    output_colors: wp.array(dtype=wp.uint32, ndim=2),  # type: ignore
):
    """Generate random colors given IDs (e.g. semantic IDs).

    Args:
        input_ids: 2D uint32 array of semantic IDs per pixel
        output_data: 2D uint32 array; each word is `r | (g<<8) | (b<<16) | (a<<24)`
    """
    i, j = wp.tid()

    input_id = input_ids[i, j]

    if input_id == wp.uint32(0):
        # BACKGROUND special case
        output_color = wp.uint32(0)
    elif input_id == wp.uint32(1):
        # UNLABELLED special case
        output_color = wp.uint32(0xFF000000)
    else:
        output_color = random_color_from_id(input_id)

    output_colors[i, j] = output_color


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
