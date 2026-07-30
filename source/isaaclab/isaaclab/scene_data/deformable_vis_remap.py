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

_BARY_EPS = 1e-5


@dataclass
class VolumeVisRemap:
    """Device-resident barycentric remap from sim tet nodes to visual mesh vertices."""

    tet_vertex_indices: wp.array
    bary_weights: wp.array


@wp.kernel
def _remap_volume_vis_positions_kernel(
    sim_particle_q: wp.array(dtype=wp.vec3f),
    render_particle_q: wp.array(dtype=wp.vec3f),
    sim_offset: int,
    render_offset: int,
    tet_vertex_indices: wp.array2d(dtype=wp.int32),
    bary_weights: wp.array2d(dtype=wp.float32),
):
    """Interpolate sim tet nodal positions into visual mesh vertex positions."""
    vis_idx = wp.tid()
    w0 = bary_weights[vis_idx, 0]
    w1 = bary_weights[vis_idx, 1]
    w2 = bary_weights[vis_idx, 2]
    w3 = bary_weights[vis_idx, 3]
    i0 = sim_offset + tet_vertex_indices[vis_idx, 0]
    i1 = sim_offset + tet_vertex_indices[vis_idx, 1]
    i2 = sim_offset + tet_vertex_indices[vis_idx, 2]
    i3 = sim_offset + tet_vertex_indices[vis_idx, 3]
    p = w0 * sim_particle_q[i0] + w1 * sim_particle_q[i1] + w2 * sim_particle_q[i2] + w3 * sim_particle_q[i3]
    render_particle_q[render_offset + vis_idx] = p


def launch_volume_vis_remap(
    sim_particle_q: wp.array,
    render_particle_q: wp.array,
    sim_offset: int,
    render_offset: int,
    remap: VolumeVisRemap,
) -> None:
    """Barycentrically interpolate sim tet nodes into visual render slots.

    Args:
        sim_particle_q: Live sim particle positions [m], shape ``[sim_count, 3]``, float.
        render_particle_q: Shadow render buffer [m], shape ``[render_count, 3]``, float.
        sim_offset: Starting index of this body's sim particles in ``sim_particle_q``.
        render_offset: Starting index of this body's visual particles in ``render_particle_q``.
        remap: Pre-built device-resident barycentric tables for this body.
    """
    vis_count = remap.tet_vertex_indices.shape[0]
    wp.launch(
        _remap_volume_vis_positions_kernel,
        dim=vis_count,
        inputs=[
            sim_particle_q,
            render_particle_q,
            sim_offset,
            render_offset,
            remap.tet_vertex_indices,
            remap.bary_weights,
        ],
        device=sim_particle_q.device,
    )


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
    tet_vertex_indices = np.empty((vis_count, 4), dtype=np.int32)
    bary_weights = np.empty((vis_count, 4), dtype=np.float32)
    clamped_count = 0
    unassigned_count = 0

    for vis_idx in range(vis_count):
        point = vis_vertices[vis_idx]
        best_neg = -np.inf
        best_tet = -1
        best_weights = np.zeros(4, dtype=np.float32)

        for tet_idx in range(tet_count):
            v0, v1, v2, v3 = (int(tet_indices[tet_idx * 4 + j]) for j in range(4))
            a = sim_vertices[v0]
            b = sim_vertices[v1]
            c = sim_vertices[v2]
            d = sim_vertices[v3]

            mat = np.stack((b - a, c - a, d - a), axis=1)
            rhs = point - a
            try:
                sol = np.linalg.solve(mat, rhs)
            except np.linalg.LinAlgError:
                continue

            w1, w2, w3 = sol
            w0 = 1.0 - w1 - w2 - w3
            weights = np.array([w0, w1, w2, w3], dtype=np.float32)
            neg = float(min(weights))
            if neg > best_neg:
                best_neg = neg
                best_tet = tet_idx
                best_weights = weights

        if best_tet < 0:
            unassigned_count += 1
            continue

        if best_neg < -_BARY_EPS:
            clamped_count += 1

        base = best_tet * 4
        tet_vertex_indices[vis_idx] = tet_indices[base : base + 4]
        bary_weights[vis_idx] = best_weights

    if unassigned_count > 0:
        return None

    if clamped_count > 0:
        logger.warning(
            "Volume vis remap clamped %d/%d visual vertices to the nearest sim tet (outside hull).",
            clamped_count,
            vis_count,
        )

    return VolumeVisRemap(
        tet_vertex_indices=wp.array(tet_vertex_indices, dtype=wp.int32, device=device),
        bary_weights=wp.array(bary_weights, dtype=wp.float32, device=device),
    )
