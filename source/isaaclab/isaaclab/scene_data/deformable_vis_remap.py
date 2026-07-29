# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Volume deformable sim-to-visual remapping for shadow rendering."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import warp as wp

logger = logging.getLogger(__name__)

_BARY_EPS = 1e-5


@dataclass(frozen=True)
class VolumeVisRemap:
    """Barycentric embed of visual mesh vertices into a tet sim mesh."""

    tet_vertex_indices: np.ndarray
    """Visual-vertex tet corner indices into the sim body slice, shape ``[vis_count, 4]``, int32."""

    bary_weights: np.ndarray
    """Barycentric weights per visual vertex, shape ``[vis_count, 4]``, float32."""


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


@wp.kernel
def _copy_particle_slice_kernel(
    src: wp.array(dtype=wp.vec3f),
    dst: wp.array(dtype=wp.vec3f),
    src_offset: int,
    dst_offset: int,
):
    """Copy one contiguous particle slice between buffers."""
    idx = wp.tid()
    dst[dst_offset + idx] = src[src_offset + idx]


def launch_volume_vis_remap(
    sim_particle_q: wp.array,
    render_particle_q: wp.array,
    sim_offset: int,
    render_offset: int,
    remap: VolumeVisRemap,
    *,
    device: str,
) -> None:
    """Write one body's remapped visual positions into ``render_particle_q``."""
    vis_count = int(remap.tet_vertex_indices.shape[0])
    tet_indices_wp = wp.array(remap.tet_vertex_indices, dtype=wp.int32, device=device)
    bary_wp = wp.array(remap.bary_weights, dtype=wp.float32, device=device)
    wp.launch(
        _remap_volume_vis_positions_kernel,
        dim=vis_count,
        inputs=[
            sim_particle_q,
            render_particle_q,
            sim_offset,
            render_offset,
            tet_indices_wp,
            bary_wp,
        ],
        device=device,
    )


def launch_copy_particle_slice(
    src: wp.array,
    dst: wp.array,
    src_offset: int,
    dst_offset: int,
    count: int,
    *,
    device: str,
) -> None:
    """Copy ``count`` particles from ``src`` to ``dst`` at the given offsets."""
    if count <= 0:
        return
    wp.launch(
        _copy_particle_slice_kernel,
        dim=count,
        inputs=[src, dst, src_offset, dst_offset],
        device=device,
    )


def _barycentric_coords_tet(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    point: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Return barycentric weights ``(w0,w1,w2,w3)`` for ``point`` in tet ``(v0..v3)``."""
    mat = np.stack((v1 - v0, v2 - v0, v3 - v0), axis=1).astype(np.float64)
    rhs = (point - v0).astype(np.float64)
    try:
        w123 = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return np.zeros(4, dtype=np.float32), False
    w0 = 1.0 - float(np.sum(w123))
    weights = np.array([w0, w123[0], w123[1], w123[2]], dtype=np.float32)
    inside = bool(np.all(weights >= -_BARY_EPS))
    return weights, inside


def build_volume_vis_barycentric_remap(
    sim_vertices: np.ndarray,
    tet_indices: np.ndarray,
    vis_vertices: np.ndarray,
) -> VolumeVisRemap | None:
    """Build a barycentric sim→vis remap for one volume deformable body.

    Args:
        sim_vertices: Sim tet mesh vertex positions in the deformable parent frame [m],
            shape ``[sim_count, 3]``.
        tet_indices: Tetrahedron indices flattened as ``4 * num_tets`` int32.
        vis_vertices: Visual mesh vertex positions in the same frame [m], shape ``[vis_count, 3]``.

    Returns:
        A :class:`VolumeVisRemap` when every visual vertex embeds in a tet, otherwise ``None``.
    """
    sim_vertices = np.asarray(sim_vertices, dtype=np.float32).reshape(-1, 3)
    vis_vertices = np.asarray(vis_vertices, dtype=np.float32).reshape(-1, 3)
    tet_indices = np.asarray(tet_indices, dtype=np.int32).reshape(-1, 4)
    if sim_vertices.shape[0] == 0 or vis_vertices.shape[0] == 0 or tet_indices.shape[0] == 0:
        return None

    tet_vertex_indices = np.zeros((vis_vertices.shape[0], 4), dtype=np.int32)
    bary_weights = np.zeros((vis_vertices.shape[0], 4), dtype=np.float32)

    for vis_idx, point in enumerate(vis_vertices):
        best_weights: np.ndarray | None = None
        best_tet: np.ndarray | None = None
        best_neg = -np.inf
        found_inside = False

        for tet in tet_indices:
            v0, v1, v2, v3 = sim_vertices[tet[0]], sim_vertices[tet[1]], sim_vertices[tet[2]], sim_vertices[tet[3]]
            weights, inside = _barycentric_coords_tet(v0, v1, v2, v3, point)
            if inside:
                best_weights = weights
                best_tet = tet
                found_inside = True
                break
            min_weight = float(np.min(weights))
            if min_weight > best_neg:
                best_neg = min_weight
                best_weights = weights
                best_tet = tet

        if best_tet is None or best_weights is None:
            logger.warning("Failed to embed visual vertex %d into any simulation tet.", vis_idx)
            return None

        if not found_inside and best_neg < -_BARY_EPS:
            logger.warning(
                "Visual vertex %d lies outside the simulation tet hull (min bary weight %.4f).",
                vis_idx,
                best_neg,
            )
            return None

        tet_vertex_indices[vis_idx] = best_tet
        bary_weights[vis_idx] = best_weights

    return VolumeVisRemap(tet_vertex_indices=tet_vertex_indices, bary_weights=bary_weights)
