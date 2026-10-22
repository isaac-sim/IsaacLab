# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Barycentric sim-to-visual remapping for volume deformable shadow rendering."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import warp as wp

logger = logging.getLogger(__name__)

_SPATIAL_ACCEL_MIN_OPS = 10_000


@dataclass
class VolumeVisRemap:
    """Device-resident barycentric remap from sim tet nodes to visual mesh vertices.

    Attributes:
        tet_vertex_indices: Per visual vertex, four sim-slice corner indices,
            shape ``[vis_count, 4]``, int32.
        bary_weights: Per visual vertex, four barycentric weights,
            shape ``[vis_count, 4]``, float32.
    """

    tet_vertex_indices: wp.array2d(dtype=wp.int32)
    bary_weights: wp.array2d(dtype=wp.float32)


@wp.func
def _det3(
    m00: float,
    m01: float,
    m02: float,
    m10: float,
    m11: float,
    m12: float,
    m20: float,
    m21: float,
    m22: float,
) -> float:
    """Return the determinant of a 3×3 matrix."""
    return m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20)


@wp.func
def _tet_aabb_contains_point(
    a: wp.vec3f, b: wp.vec3f, c: wp.vec3f, d: wp.vec3f, point: wp.vec3f, margin: float
) -> bool:
    """Return ``True`` when ``point`` lies inside the tet axis-aligned bounding box expanded by ``margin``."""
    min_x = wp.min(wp.min(a[0], b[0]), wp.min(c[0], d[0])) - margin
    min_y = wp.min(wp.min(a[1], b[1]), wp.min(c[1], d[1])) - margin
    min_z = wp.min(wp.min(a[2], b[2]), wp.min(c[2], d[2])) - margin
    max_x = wp.max(wp.max(a[0], b[0]), wp.max(c[0], d[0])) + margin
    max_y = wp.max(wp.max(a[1], b[1]), wp.max(c[1], d[1])) + margin
    max_z = wp.max(wp.max(a[2], b[2]), wp.max(c[2], d[2])) + margin
    return (
        point[0] >= min_x
        and point[0] <= max_x
        and point[1] >= min_y
        and point[1] <= max_y
        and point[2] >= min_z
        and point[2] <= max_z
    )


@wp.kernel
def _build_volume_vis_barycentric_remap_kernel(
    sim_vertices: wp.array(dtype=wp.vec3f),
    tet_indices: wp.array(dtype=wp.int32),
    vis_vertices: wp.array(dtype=wp.vec3f),
    tet_count: int,
    use_spatial_filter: int,
    tet_vertex_indices: wp.array2d(dtype=wp.int32),
    bary_weights: wp.array2d(dtype=wp.float32),
    assigned: wp.array(dtype=wp.int32),
    clamped: wp.array(dtype=wp.int32),
):
    """Embed each visual vertex in the tet with the largest minimum barycentric weight."""
    vis_idx = wp.tid()
    point = vis_vertices[vis_idx]
    best_neg = float(-1.0e30)
    best_tet = int(-1)
    best_w0 = float(0.0)
    best_w1 = float(0.0)
    best_w2 = float(0.0)
    best_w3 = float(0.0)
    best_i0 = int(0)
    best_i1 = int(0)
    best_i2 = int(0)
    best_i3 = int(0)

    for tet_idx in range(tet_count):
        base = tet_idx * 4
        i0 = tet_indices[base + 0]
        i1 = tet_indices[base + 1]
        i2 = tet_indices[base + 2]
        i3 = tet_indices[base + 3]
        a = sim_vertices[i0]
        b = sim_vertices[i1]
        c = sim_vertices[i2]
        d = sim_vertices[i3]

        inside_aabb = True
        if use_spatial_filter != 0:
            inside_aabb = _tet_aabb_contains_point(a, b, c, d, point, 0.5)

        if inside_aabb:
            ba = b - a
            ca = c - a
            da = d - a
            rhs = point - a
            det = _det3(ba[0], ca[0], da[0], ba[1], ca[1], da[1], ba[2], ca[2], da[2])
            if wp.abs(det) >= 1.0e-12:
                w1 = _det3(rhs[0], ca[0], da[0], rhs[1], ca[1], da[1], rhs[2], ca[2], da[2]) / det
                w2 = _det3(ba[0], rhs[0], da[0], ba[1], rhs[1], da[1], ba[2], rhs[2], da[2]) / det
                w3 = _det3(ba[0], ca[0], rhs[0], ba[1], ca[1], rhs[1], ba[2], ca[2], rhs[2]) / det
                w0 = 1.0 - w1 - w2 - w3
                neg = wp.min(wp.min(w0, w1), wp.min(w2, w3))
                if neg > best_neg:
                    best_neg = neg
                    best_tet = tet_idx
                    best_w0 = w0
                    best_w1 = w1
                    best_w2 = w2
                    best_w3 = w3
                    best_i0 = i0
                    best_i1 = i1
                    best_i2 = i2
                    best_i3 = i3

    if best_tet < 0:
        assigned[vis_idx] = 0
        return

    assigned[vis_idx] = 1
    if best_neg < -1.0e-5:
        clamped[vis_idx] = 1

    tet_vertex_indices[vis_idx, 0] = best_i0
    tet_vertex_indices[vis_idx, 1] = best_i1
    tet_vertex_indices[vis_idx, 2] = best_i2
    tet_vertex_indices[vis_idx, 3] = best_i3
    bary_weights[vis_idx, 0] = best_w0
    bary_weights[vis_idx, 1] = best_w1
    bary_weights[vis_idx, 2] = best_w2
    bary_weights[vis_idx, 3] = best_w3


def build_volume_vis_barycentric_remap(
    sim_vertices: np.ndarray,
    tet_indices: np.ndarray,
    vis_vertices: np.ndarray,
    *,
    device: str = "cpu",
) -> VolumeVisRemap | None:
    """Embed each visual vertex in the closest sim tet and upload remap tables to *device*.

    Visual vertices slightly outside the tet hull are projected onto the nearest tet
    (barycentric extrapolation) instead of failing the whole remap.

    Args:
        sim_vertices: Sim tet node positions [m], shape ``[sim_count, 3]``, float32.
        tet_indices: Flattened tet vertex indices, shape ``[4 * tet_count]``, int32.
        vis_vertices: Visual mesh vertex positions [m], shape ``[vis_count, 3]``, float32.
        device: Warp device for the returned remap arrays.

    Returns:
        Device-resident remap tables, or ``None`` when no tet can be assigned.
    """
    if vis_vertices.size == 0 or sim_vertices.size == 0 or tet_indices.size == 0:
        return None

    tet_count = tet_indices.shape[0] // 4
    vis_count = vis_vertices.shape[0]
    use_spatial_filter = int(vis_count * tet_count >= _SPATIAL_ACCEL_MIN_OPS)

    sim_wp = wp.array(sim_vertices.astype(np.float32, copy=False), dtype=wp.vec3f, device=device)
    tet_wp = wp.array(tet_indices.astype(np.int32, copy=False), dtype=wp.int32, device=device)
    vis_wp = wp.array(vis_vertices.astype(np.float32, copy=False), dtype=wp.vec3f, device=device)

    tet_vertex_indices = wp.zeros((vis_count, 4), dtype=wp.int32, device=device)
    bary_weights = wp.zeros((vis_count, 4), dtype=wp.float32, device=device)
    assigned = wp.zeros(vis_count, dtype=wp.int32, device=device)
    clamped = wp.zeros(vis_count, dtype=wp.int32, device=device)

    wp.launch(
        _build_volume_vis_barycentric_remap_kernel,
        dim=vis_count,
        inputs=[
            sim_wp,
            tet_wp,
            vis_wp,
            tet_count,
            use_spatial_filter,
            tet_vertex_indices,
            bary_weights,
            assigned,
            clamped,
        ],
        device=device,
    )

    assigned_host = assigned.numpy()
    if not bool(np.all(assigned_host)):
        return None

    clamped_count = int(np.sum(clamped.numpy()))
    if clamped_count > 0:
        logger.warning(
            "Volume vis remap clamped %d/%d visual vertices to the nearest sim tet (outside hull).",
            clamped_count,
            vis_count,
        )

    return VolumeVisRemap(tet_vertex_indices=tet_vertex_indices, bary_weights=bary_weights)
