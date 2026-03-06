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


@wp.kernel
def compute_crc32_hash_kernel(
    input_data: wp.array(dtype=wp.uint32, ndim=2),  # type: ignore
    output_hash: wp.array(dtype=wp.uint32, ndim=2),  # type: ignore
):
    """Compute a deterministic CRC32 hash for each uint32 value in the input 2D array.

    Uses same algorithm used by Python's binascii.crc32() to produce deterministic hashes for visual distinguishability.

    Args:
        input_data: 2D uint32 array of values to hash
        output_hash: 2D uint32 array of hashes, same shape as input_data
    """
    CRC32_INIT = wp.uint32(0xFFFFFFFF)
    CRC32_POLY = wp.uint32(0xEDB88320)
    CRC32_FINAL_XOR = wp.uint32(0xFFFFFFFF)

    i, j = wp.tid()
    value_uint32 = input_data[i, j]

    crc = CRC32_INIT

    # Extract 4 bytes (little-endian: LSB first, same order as standard CRC32)
    for _ in range(4):
        current_byte = value_uint32 & wp.uint32(0xFF)

        crc = crc ^ current_byte

        for _ in range(8):
            if (crc & wp.uint32(1)) != wp.uint32(0):
                crc = (crc >> wp.uint32(1)) ^ CRC32_POLY
            else:
                crc = crc >> wp.uint32(1)

        value_uint32 = value_uint32 >> wp.uint32(8)

    crc = crc ^ CRC32_FINAL_XOR

    output_hash[i, j] = crc


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
