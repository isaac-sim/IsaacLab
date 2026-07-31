# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-side geometry point scatter helpers for SceneData."""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def scatter_geometry_points_kernel(
    src: wp.array(dtype=wp.vec3f),
    dst: wp.array(dtype=wp.vec3f),
    src_offsets: wp.array(dtype=wp.int32),
    dest_offsets: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
):
    """Copy contiguous geometry slices from ``src`` into ``dst`` for each entity."""
    entity_id = wp.tid()
    dest_offset = dest_offsets[entity_id]
    if dest_offset < 0:
        return
    src_offset = src_offsets[entity_id]
    count = counts[entity_id]
    for i in range(count):
        dst[dest_offset + i] = src[src_offset + i]


@wp.kernel
def pack_body_slices_kernel(
    src: wp.array2d(dtype=wp.vec3f),
    dst: wp.array(dtype=wp.vec3f),
    body_counts: wp.array(dtype=wp.int32),
    write_offsets: wp.array(dtype=wp.int32),
    dest_base_offset: int,
):
    """Pack per-body nodal slices from a ``(body_count, max_nodes)`` view into a flat buffer."""
    body_idx = wp.tid()
    count = body_counts[body_idx]
    write_offset = dest_base_offset + write_offsets[body_idx]
    for i in range(count):
        dst[write_offset + i] = src[body_idx, i]


def scatter_geometry_points(
    src: wp.array(dtype=wp.vec3f),
    dst: wp.array(dtype=wp.vec3f),
    entity_counts: list[int],
    mapping: wp.array(dtype=wp.int32) | None,
    *,
    device: str,
) -> None:
    """Scatter backend geometry entities into a consumer points buffer on device.

    Args:
        src: Flattened source geometry points [m], shape ``[src_count]``, ``wp.vec3f``.
        dst: Pre-allocated destination buffer [m], shape ``[dst_count]``, ``wp.vec3f``.
        entity_counts: Unpadded point count for each backend geometry entity.
        mapping: Optional destination particle offsets per entity (``-1`` skips copy).
        device: Warp device for launch metadata.
    """
    if not entity_counts:
        wp.copy(dst, src)
        return

    num_entities = len(entity_counts)
    src_offsets = np.zeros(num_entities, dtype=np.int32)
    flat = 0
    for index, count in enumerate(entity_counts):
        src_offsets[index] = flat
        flat += int(count)

    if mapping is None:
        dest_offsets = src_offsets.copy()
    else:
        dest_offsets = mapping.numpy().astype(np.int32, copy=False)

    counts = np.asarray(entity_counts, dtype=np.int32)
    wp.launch(
        scatter_geometry_points_kernel,
        dim=num_entities,
        inputs=[
            src,
            dst,
            wp.array(src_offsets, dtype=wp.int32, device=src.device),
            wp.array(dest_offsets, dtype=wp.int32, device=src.device),
            wp.array(counts, dtype=wp.int32, device=src.device),
        ],
        device=device,
    )


def pack_body_nodal_slices(
    nodal: wp.array2d,
    dst: wp.array(dtype=wp.vec3f),
    body_counts: list[int],
    *,
    device: str,
    dest_base_offset: int = 0,
) -> None:
    """Pack per-body nodal positions from a 2D view into ``dst``."""
    if not body_counts:
        return

    write_offsets = np.zeros(len(body_counts), dtype=np.int32)
    flat = 0
    for index, count in enumerate(body_counts):
        write_offsets[index] = flat
        flat += int(count)

    wp.launch(
        pack_body_slices_kernel,
        dim=len(body_counts),
        inputs=[
            nodal,
            dst,
            wp.array(np.asarray(body_counts, dtype=np.int32), dtype=wp.int32, device=nodal.device),
            wp.array(write_offsets, dtype=wp.int32, device=nodal.device),
            dest_base_offset,
        ],
        device=device,
    )
