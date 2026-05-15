# Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Ported from Newton Physics Engine — xcath/xcath_solver.py
#   branch: xpbd_rods_solver_integr
#   author: Przemyslaw Korzeniowski <korzenio@gmail.com>
#
# Modifications for Isaac Lab integration:
#   Copyright (c) 2022-2025, The Isaac Lab Project Developers
#   SPDX-License-Identifier: BSD-3-Clause

"""XPBD rod solver with catheter-in-vessel mesh collision.

This module provides :class:`XCathRodSolver`, a subclass of
:class:`~isaaclab_newton.solvers.XPBDRodSolver` that adds three capabilities
needed for realistic catheter-in-vessel simulation:

**Vessel containment (two collision paths)**

*SDF path (default, cheapest):*
Each rod particle is queried against a :class:`warp.Mesh` BVH via
``wp.mesh_query_point_sign_normal``.  Particles that penetrate the vessel
surface are projected back to a configurable clearance offset
``target_phi`` (typically ``-particle_radius``).

*AABB / mesh-edge path (more robust for tight curvature):*
An AABB broadphase finds all nearby triangles; vertex-vs-triangle and
rod-segment-vs-mesh-edge contacts are resolved into a shared correction
buffer and flushed to ``predicted_positions`` after each Gauss-Seidel
iteration.

**Track-guided insertion**
Non-tip rod particles are gently projected onto a fixed linear insertion
axis (``track_start``, ``track_dir``, ``track_length``), mimicking the
rigid catheter shaft sliding through a guide sheath while only the
steerable distal tip deforms freely.

**Configurable collision ordering**
Vessel containment can run *before* the XPBD constraint solve
(``collision_pre_constraints_enabled``), *after* (default), or both.
``collision_iterations`` controls how many Gauss-Seidel passes are done
per substep.

Usage::

    import numpy as np
    import warp as wp
    from isaaclab_newton.solvers import RodConfig
    from isaaclab_newton.solvers.xcath_rod_solver import XCathRodSolver

    # Build a wp.Mesh from your vessel geometry
    verts = np.load("aorta_verts.npy")   # (N, 3) float32
    faces = np.load("aorta_faces.npy")   # (M, 3) int32
    vessel_mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device="cuda"),
        indices=wp.array(faces.flatten(), dtype=int, device="cuda"),
    )

    cfg = RodConfig()
    solver = XCathRodSolver(
        cfg,
        collision_mesh=vessel_mesh,
        track_start=np.array([0.0, 0.0, 0.0]),
        track_dir=np.array([1.0, 0.0, 0.0]),
        track_length=0.5,
        tip_num_edges=10,
        particle_radius=0.002,
        segment_length=cfg.rod.rest_length / (cfg.rod.num_links - 1),
    )
    for _ in range(300):
        solver.step(cfg.solver.dt)
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .xpbd_rod_solver import XPBDRodSolver, _BatchedWorkspace, _Workspace

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

COLLISION_PROJECTION_STAGE_PRE = "pre"
COLLISION_PROJECTION_STAGE_POST = "post"
_COLLISION_PROJECTION_STAGES = (
    COLLISION_PROJECTION_STAGE_PRE,
    COLLISION_PROJECTION_STAGE_POST,
)
DEFAULT_MESH_EDGE_MAX_TRIANGLES = 64


# ──────────────────────────────────────────────────────────────────────────────
# NumPy utility
# ──────────────────────────────────────────────────────────────────────────────


def compute_smooth_vertex_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals from triangle topology.

    Args:
        vertices: ``(N, 3)`` float32 array of vertex positions.
        indices:  ``(M, 3)`` int32 array of triangle indices.

    Returns:
        ``(N, 3)`` float32 array of unit vertex normals.
    """
    vertices = np.asarray(vertices, dtype=np.float32).reshape((-1, 3))
    triangles = np.asarray(indices, dtype=np.int32).reshape((-1, 3))

    normals = np.zeros_like(vertices, dtype=np.float32)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)

    np.add.at(normals, triangles[:, 0], face_normals)
    np.add.at(normals, triangles[:, 1], face_normals)
    np.add.at(normals, triangles[:, 2], face_normals)

    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


# ──────────────────────────────────────────────────────────────────────────────
# Warp helper functions
# ──────────────────────────────────────────────────────────────────────────────


@wp.func
def _mesh_face_normal(mesh_id: wp.uint64, face_index: int) -> wp.vec3:
    """Return the outward face normal for a triangle in *mesh_id*."""
    mesh = wp.mesh_get(mesh_id)
    i0 = mesh.indices[face_index * 3 + 0]
    i1 = mesh.indices[face_index * 3 + 1]
    i2 = mesh.indices[face_index * 3 + 2]
    v0 = mesh.points[i0]
    v1 = mesh.points[i1]
    v2 = mesh.points[i2]
    return wp.normalize(wp.cross(v1 - v0, v2 - v0))


@wp.func
def _safe_normalize(v: wp.vec3, fallback: wp.vec3) -> wp.vec3:
    length = wp.length(v)
    if length > 1.0e-8:
        return v / length
    return fallback


@wp.func
def _closest_point_triangle(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    cp = point - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + ab * v

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + ac * w

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + (c - b) * w

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


@wp.func
def _closest_point_triangle_barycentric(
    point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3
) -> wp.vec3:
    """Return barycentric coords ``(wa, wb, wc)`` of the closest point."""
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    bp = point - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)

    cp = point - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return wp.vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.vec3(0.0, 1.0 - w, w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def _mesh_barycentric_normal(
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    face_index: int,
    weights: wp.vec3,
    fallback: wp.vec3,
) -> wp.vec3:
    mesh = wp.mesh_get(mesh_id)
    i0 = mesh.indices[face_index * 3 + 0]
    i1 = mesh.indices[face_index * 3 + 1]
    i2 = mesh.indices[face_index * 3 + 2]
    n = smooth_normals[i0] * weights.x + smooth_normals[i1] * weights.y + smooth_normals[i2] * weights.z
    length = wp.length(n)
    if length > 1.0e-8:
        return n / length
    return fallback


@wp.func
def _segment_segment_barycentric(
    a0: wp.vec3, a1: wp.vec3, b0: wp.vec3, b1: wp.vec3
) -> wp.vec4:
    """Return ``(sa, 1-sa, sb, 1-sb)`` barycentric parameters for the closest
    points between segment ``[a0,a1]`` and segment ``[b0,b1]``."""
    da = a1 - a0
    db = b1 - b0
    r = a0 - b0
    a = wp.dot(da, da)
    e = wp.dot(db, db)
    f = wp.dot(db, r)

    sa = float(0.0)
    sb = float(0.0)

    if a <= 1.0e-10 and e <= 1.0e-10:
        return wp.vec4(1.0, 0.0, 1.0, 0.0)

    if a <= 1.0e-10:
        sa = 0.0
        sb = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(da, r)
        if e <= 1.0e-10:
            sb = 0.0
            sa = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b_coeff = wp.dot(da, db)
            denom = a * e - b_coeff * b_coeff
            if wp.abs(denom) > 1.0e-10:
                sa = wp.clamp((b_coeff * f - c * e) / denom, 0.0, 1.0)
            else:
                sa = 0.0
            sb = (b_coeff * sa + f) / e
            if sb < 0.0:
                sb = 0.0
                sa = wp.clamp(-c / a, 0.0, 1.0)
            elif sb > 1.0:
                sb = 1.0
                sa = wp.clamp((b_coeff - c) / a, 0.0, 1.0)

    return wp.vec4(sa, 1.0 - sa, sb, 1.0 - sb)


# ──────────────────────────────────────────────────────────────────────────────
# Warp kernels
# ──────────────────────────────────────────────────────────────────────────────


@wp.kernel
def _track_sliding_kernel(
    predicted_positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    track_start: wp.vec3,
    track_dir: wp.vec3,
    track_length: float,
    track_stiffness: float,
    end_idx: int,
):
    """Project non-tip rod particles toward the insertion axis (track)."""
    i = wp.tid()
    if i >= end_idx:
        return
    if inv_masses[i] <= 0.0:
        return

    p = predicted_positions[i]
    t = wp.clamp(wp.dot(p - track_start, track_dir), 0.0, track_length)
    proj = track_start + track_dir * t
    predicted_positions[i] = p + (proj - p) * track_stiffness


@wp.kernel
def _clear_vec3_kernel(values: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    values[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _project_vessel_containment_kernel(
    current_positions: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    sign_scale: float,
    use_smooth_normals: bool,
    target_phi: float,
    max_dist: float,
):
    """Project predicted positions to the vessel interior signed offset.

    Uses ``wp.mesh_query_point_sign_normal`` (BVH SDF query) to determine
    whether each particle is inside the vessel and how far from the wall it is.
    Particles that violate the ``target_phi`` clearance are projected back.
    The velocity clamp at the end ensures the derived velocity (computed as
    ``(predicted - current) / dt`` during integration) does not point outward.
    """
    i = wp.tid()
    if inv_masses[i] <= 0.0:
        return

    pos = predicted_positions[i]

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    if not wp.mesh_query_point_sign_normal(mesh_id, pos, max_dist, sign, face_index, face_u, face_v):
        return

    closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
    diff = pos - closest
    dist = wp.length(diff)

    side = sign * sign_scale
    if wp.abs(side) < 1.0e-6:
        side = sign_scale

    grad = wp.vec3(0.0, 0.0, 0.0)
    if use_smooth_normals:
        weights = wp.vec3(1.0 - face_u - face_v, face_u, face_v)
        face_normal = _mesh_face_normal(mesh_id, face_index)
        grad = _mesh_barycentric_normal(mesh_id, smooth_normals, face_index, weights, face_normal) * sign_scale
        if dist > 1.0e-8 and wp.dot(grad, diff) * side < 0.0:
            grad = -grad
    elif dist > 1.0e-8:
        grad = (diff / dist) * side
    else:
        grad = _mesh_face_normal(mesh_id, face_index) * side

    phi = side * dist
    if phi <= target_phi:
        return

    projected = pos - grad * (phi - target_phi)
    disp = projected - current_positions[i]
    outward_disp = wp.dot(disp, grad)
    if outward_disp > 0.0:
        projected = projected - grad * outward_disp
    predicted_positions[i] = projected


@wp.kernel
def _project_mesh_vertex_collision_kernel(
    predicted_positions: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    radius: float,
    query_radius: float,
    sign_scale: float,
    use_smooth_normals: bool,
    max_triangles: int,
    corrections: wp.array(dtype=wp.vec3),
):
    """Per-particle vertex-vs-triangle containment (max-penetration variant).

    Picks the single nearby triangle with maximum penetration depth and applies
    that correction — summing all overlapping triangles over-corrects on curved
    surfaces where many nearly-tangent triangles report similar penetrations.
    """
    i = wp.tid()
    if inv_masses[i] <= 0.0:
        return

    p = predicted_positions[i]
    lower = wp.vec3(p.x - query_radius, p.y - query_radius, p.z - query_radius)
    upper = wp.vec3(p.x + query_radius, p.y + query_radius, p.z + query_radius)

    mesh = wp.mesh_get(mesh_id)
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_index = wp.int32(0)
    tri_count = int(0)
    best_pen = float(0.0)
    best_n = wp.vec3(0.0, 0.0, 0.0)

    while wp.mesh_query_aabb_next(query, face_index):
        if tri_count >= max_triangles:
            break
        tri_count = tri_count + 1

        i0 = mesh.indices[face_index * 3 + 0]
        i1 = mesh.indices[face_index * 3 + 1]
        i2 = mesh.indices[face_index * 3 + 2]
        v0 = mesh.points[i0]
        v1 = mesh.points[i1]
        v2 = mesh.points[i2]

        face_normal = _safe_normalize(wp.cross(v1 - v0, v2 - v0), wp.vec3(0.0, 0.0, 1.0))
        bar = _closest_point_triangle_barycentric(p, v0, v1, v2)
        n_outward = face_normal * sign_scale
        if use_smooth_normals:
            n_outward = _mesh_barycentric_normal(
                mesh_id, smooth_normals, face_index, bar, face_normal
            ) * sign_scale

        cp = v0 * bar.x + v1 * bar.y + v2 * bar.z
        signed = wp.dot(p - cp, n_outward)
        penetration = signed + radius
        if penetration > best_pen:
            best_pen = penetration
            best_n = n_outward

    if best_pen > 0.0:
        wp.atomic_add(corrections, i, -best_n * best_pen)


@wp.kernel
def _project_mesh_vertex_collision_kernel_averaged(
    predicted_positions: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    radius: float,
    query_radius: float,
    sign_scale: float,
    use_smooth_normals: bool,
    max_triangles: int,
    corrections: wp.array(dtype=wp.vec3),
):
    """Per-particle vertex-vs-triangle containment (averaged variant).

    Averages corrections from all penetrating nearby triangles.  Less
    aggressive than the max-penetration variant; useful when the SDF path
    causes ringing on highly curved surfaces.
    """
    i = wp.tid()
    if inv_masses[i] <= 0.0:
        return

    p = predicted_positions[i]
    lower = wp.vec3(p.x - query_radius, p.y - query_radius, p.z - query_radius)
    upper = wp.vec3(p.x + query_radius, p.y + query_radius, p.z + query_radius)

    mesh = wp.mesh_get(mesh_id)
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_index = wp.int32(0)
    tri_count = int(0)
    col_count = int(0)
    delta = wp.vec3(0.0, 0.0, 0.0)

    while wp.mesh_query_aabb_next(query, face_index):
        if tri_count >= max_triangles:
            break
        tri_count = tri_count + 1

        i0 = mesh.indices[face_index * 3 + 0]
        i1 = mesh.indices[face_index * 3 + 1]
        i2 = mesh.indices[face_index * 3 + 2]
        v0 = mesh.points[i0]
        v1 = mesh.points[i1]
        v2 = mesh.points[i2]

        face_normal = _safe_normalize(wp.cross(v1 - v0, v2 - v0), wp.vec3(0.0, 0.0, 1.0))
        bar = _closest_point_triangle_barycentric(p, v0, v1, v2)
        n_outward = face_normal * sign_scale
        if use_smooth_normals:
            n_outward = _mesh_barycentric_normal(
                mesh_id, smooth_normals, face_index, bar, face_normal
            ) * sign_scale

        cp = v0 * bar.x + v1 * bar.y + v2 * bar.z
        signed = wp.dot(p - cp, n_outward)
        penetration = signed + radius
        if penetration > 0.0:
            delta = delta - n_outward * penetration
            col_count += 1

    if col_count > 0:
        wp.atomic_add(corrections, i, delta / float(col_count))


@wp.kernel
def _project_mesh_edge_collision_kernel(
    predicted_positions: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    radius: float,
    query_radius: float,
    sign_scale: float,
    use_smooth_normals: bool,
    max_triangles: int,
    corrections: wp.array(dtype=wp.vec3),
):
    """Per-rod-edge: rod-segment vs. mesh-edge collision (max-penetration variant).

    Iterates over the three edges of each nearby mesh triangle (deduped by
    ``ia < ib`` so each undirected edge is visited once), finds the single
    mesh edge with maximum rod-segment penetration, and accumulates the
    correction into the shared buffer.
    """
    edge = wp.tid()

    p0 = predicted_positions[edge]
    p1 = predicted_positions[edge + 1]
    inv0 = inv_masses[edge]
    inv1 = inv_masses[edge + 1]
    if inv0 <= 0.0 and inv1 <= 0.0:
        return

    lower = wp.vec3(
        wp.min(p0.x, p1.x) - query_radius,
        wp.min(p0.y, p1.y) - query_radius,
        wp.min(p0.z, p1.z) - query_radius,
    )
    upper = wp.vec3(
        wp.max(p0.x, p1.x) + query_radius,
        wp.max(p0.y, p1.y) + query_radius,
        wp.max(p0.z, p1.z) + query_radius,
    )

    mesh = wp.mesh_get(mesh_id)
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_index = wp.int32(0)
    tri_count = int(0)
    best_pen = float(0.0)
    best_n = wp.vec3(0.0, 0.0, 0.0)
    best_bar = wp.vec4(0.0, 0.0, 0.0, 0.0)

    while wp.mesh_query_aabb_next(query, face_index):
        if tri_count >= max_triangles:
            break
        tri_count = tri_count + 1

        i0 = mesh.indices[face_index * 3 + 0]
        i1 = mesh.indices[face_index * 3 + 1]
        i2 = mesh.indices[face_index * 3 + 2]
        v0 = mesh.points[i0]
        v1 = mesh.points[i1]
        v2 = mesh.points[i2]
        face_normal = _safe_normalize(wp.cross(v1 - v0, v2 - v0), wp.vec3(0.0, 0.0, 1.0))

        for mesh_edge in range(3):
            ia = i0
            ib = i1
            ea = v0
            eb = v1
            mesh_normal_a = wp.vec3(1.0, 0.0, 0.0)
            mesh_normal_b = wp.vec3(0.0, 1.0, 0.0)
            if mesh_edge == 1:
                ia = i1
                ib = i2
                ea = v1
                eb = v2
                mesh_normal_a = wp.vec3(0.0, 1.0, 0.0)
                mesh_normal_b = wp.vec3(0.0, 0.0, 1.0)
            elif mesh_edge == 2:
                ia = i2
                ib = i0
                ea = v2
                eb = v0
                mesh_normal_a = wp.vec3(0.0, 0.0, 1.0)
                mesh_normal_b = wp.vec3(1.0, 0.0, 0.0)
            if ia >= ib:
                continue

            bar = _segment_segment_barycentric(p0, p1, ea, eb)
            rod_point = p0 * bar.x + p1 * bar.y
            mesh_point = ea * bar.z + eb * bar.w
            n_outward = face_normal * sign_scale
            if use_smooth_normals:
                weights = mesh_normal_a * bar.z + mesh_normal_b * bar.w
                n_outward = _mesh_barycentric_normal(
                    mesh_id, smooth_normals, face_index, weights, face_normal
                ) * sign_scale
            signed = wp.dot(rod_point - mesh_point, n_outward)
            penetration = signed + radius
            if penetration > best_pen:
                best_pen = penetration
                best_n = n_outward
                best_bar = bar

    if best_pen > 0.0:
        denom = inv0 * best_bar.x * best_bar.x + inv1 * best_bar.y * best_bar.y
        if denom > 1.0e-8:
            correction = -best_n * (best_pen / denom)
            if inv0 > 0.0:
                wp.atomic_add(corrections, edge, correction * best_bar.x * inv0)
            if inv1 > 0.0:
                wp.atomic_add(corrections, edge + 1, correction * best_bar.y * inv1)


@wp.kernel
def _project_mesh_edge_collision_kernel_averaged(
    predicted_positions: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    radius: float,
    query_radius: float,
    sign_scale: float,
    use_smooth_normals: bool,
    max_triangles: int,
    corrections: wp.array(dtype=wp.vec3),
):
    """Per-rod-edge: rod-segment vs. mesh-edge collision (averaged variant)."""
    edge = wp.tid()

    p0 = predicted_positions[edge]
    p1 = predicted_positions[edge + 1]
    inv0 = inv_masses[edge]
    inv1 = inv_masses[edge + 1]
    if inv0 <= 0.0 and inv1 <= 0.0:
        return

    lower = wp.vec3(
        wp.min(p0.x, p1.x) - query_radius,
        wp.min(p0.y, p1.y) - query_radius,
        wp.min(p0.z, p1.z) - query_radius,
    )
    upper = wp.vec3(
        wp.max(p0.x, p1.x) + query_radius,
        wp.max(p0.y, p1.y) + query_radius,
        wp.max(p0.z, p1.z) + query_radius,
    )

    mesh = wp.mesh_get(mesh_id)
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_index = wp.int32(0)
    tri_count = int(0)
    col_count = int(0)
    delta0 = wp.vec3(0.0, 0.0, 0.0)
    delta1 = wp.vec3(0.0, 0.0, 0.0)

    while wp.mesh_query_aabb_next(query, face_index):
        if tri_count >= max_triangles:
            break
        tri_count = tri_count + 1

        i0 = mesh.indices[face_index * 3 + 0]
        i1 = mesh.indices[face_index * 3 + 1]
        i2 = mesh.indices[face_index * 3 + 2]
        v0 = mesh.points[i0]
        v1 = mesh.points[i1]
        v2 = mesh.points[i2]
        face_normal = _safe_normalize(wp.cross(v1 - v0, v2 - v0), wp.vec3(0.0, 0.0, 1.0))

        for mesh_edge in range(3):
            ia = i0
            ib = i1
            ea = v0
            eb = v1
            mesh_normal_a = wp.vec3(1.0, 0.0, 0.0)
            mesh_normal_b = wp.vec3(0.0, 1.0, 0.0)
            if mesh_edge == 1:
                ia = i1
                ib = i2
                ea = v1
                eb = v2
                mesh_normal_a = wp.vec3(0.0, 1.0, 0.0)
                mesh_normal_b = wp.vec3(0.0, 0.0, 1.0)
            elif mesh_edge == 2:
                ia = i2
                ib = i0
                ea = v2
                eb = v0
                mesh_normal_a = wp.vec3(0.0, 0.0, 1.0)
                mesh_normal_b = wp.vec3(1.0, 0.0, 0.0)
            if ia >= ib:
                continue

            bar = _segment_segment_barycentric(p0, p1, ea, eb)
            rod_point = p0 * bar.x + p1 * bar.y
            mesh_point = ea * bar.z + eb * bar.w
            n_outward = face_normal * sign_scale
            if use_smooth_normals:
                weights = mesh_normal_a * bar.z + mesh_normal_b * bar.w
                n_outward = _mesh_barycentric_normal(
                    mesh_id, smooth_normals, face_index, weights, face_normal
                ) * sign_scale
            signed = wp.dot(rod_point - mesh_point, n_outward)
            penetration = signed + radius
            if penetration > 0.0:
                denom = inv0 * bar.x * bar.x + inv1 * bar.y * bar.y
                if denom > 1.0e-8:
                    correction = -n_outward * (penetration / denom)
                    if inv0 > 0.0:
                        delta0 = delta0 + correction * bar.x * inv0
                    if inv1 > 0.0:
                        delta1 = delta1 + correction * bar.y * inv1
                    col_count += 1

    if col_count > 0:
        inv_count = 1.0 / float(col_count)
        if inv0 > 0.0:
            wp.atomic_add(corrections, edge, delta0 * inv_count)
        if inv1 > 0.0:
            wp.atomic_add(corrections, edge + 1, delta1 * inv_count)


@wp.kernel
def _apply_mesh_edge_collision_corrections_kernel(
    predicted_positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    corrections: wp.array(dtype=wp.vec3),
):
    """Flush the accumulated edge-collision correction buffer to predicted_positions."""
    i = wp.tid()
    if inv_masses[i] > 0.0:
        predicted_positions[i] = predicted_positions[i] + corrections[i]


# ──────────────────────────────────────────────────────────────────────────────
# Signed-distance sampling (diagnostic / offline use)
# ──────────────────────────────────────────────────────────────────────────────


@wp.kernel
def _sample_signed_distance_kernel(
    points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    max_dist: float,
    sign_scale: float,
    distances: wp.array(dtype=wp.float32),
):
    """Sample the signed distance from each point to the mesh surface."""
    i = wp.tid()
    pos = points[i]
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    if wp.mesh_query_point_sign_normal(mesh_id, pos, max_dist, sign, face_index, face_u, face_v):
        closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
        dist = wp.length(pos - closest)
        distances[i] = sign * sign_scale * dist
    else:
        distances[i] = max_dist


def compute_signed_distances(
    points: np.ndarray,
    mesh: "wp.Mesh",
    max_dist: float = 1.0,
    sign_scale: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """Return the signed distance from each point to *mesh*.

    Args:
        points:     ``(N, 3)`` float32 positions.
        mesh:       A :class:`warp.Mesh` whose BVH has been built.
        max_dist:   Search radius; points farther than this receive
                    ``max_dist`` as their distance.
        sign_scale: Flip sign convention (``+1`` for interior-positive,
                    ``-1`` for interior-negative).
        device:     Warp device string.

    Returns:
        ``(N,)`` float32 signed distances.
    """
    pts = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    n = pts.shape[0]
    pts_wp = wp.array(pts, dtype=wp.vec3, device=device)
    out_wp = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(
        _sample_signed_distance_kernel,
        dim=n,
        inputs=[pts_wp, mesh.id, float(max_dist), float(sign_scale), out_wp],
        device=device,
    )
    return out_wp.numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Stage normalisation helper
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_collision_projection_stage(stage: str) -> str:
    s = stage.strip().lower()
    if s not in _COLLISION_PROJECTION_STAGES:
        raise ValueError(
            f"Invalid collision_projection_stage {stage!r}. "
            f"Choose one of {_COLLISION_PROJECTION_STAGES}."
        )
    return s


# ──────────────────────────────────────────────────────────────────────────────
# XCathRodSolver
# ──────────────────────────────────────────────────────────────────────────────


class XCathRodSolver(XPBDRodSolver):
    """XPBD rod solver with track-guided insertion and vessel mesh containment.

    Extends :class:`~isaaclab_newton.solvers.XPBDRodSolver` with three
    sample-local projections:

    * **Track guidance** — non-tip particles slide along a linear axis.
    * **Vessel containment (SDF)** — BVH signed-distance query per particle.
    * **Vessel containment (AABB/edge)** — vertex + edge contacts via AABB
      broadphase; more robust in tight curvature regions.

    Args:
        *args: Positional arguments forwarded to :class:`XPBDRodSolver`.
        collision_mesh: Closed triangle mesh defining the vessel interior.
            Pass ``None`` to disable collision (track guidance still runs).
        track_start: World-space origin of the insertion axis, shape ``(3,)``.
        track_dir: Unit direction of the insertion axis, shape ``(3,)``.
        track_length: Maximum insertion distance along the axis (metres).
        tip_num_edges: Number of distal rod edges treated as the free tip;
            particles beyond index ``num_points - tip_num_edges`` are *not*
            snapped to the track.
        particle_radius: Effective rod radius for collision detection (metres).
        segment_length: Rest length of a single rod segment (metres).  Used
            to set the default ``max_dist`` BVH search radius.
        track_stiffness: Blend factor for track snapping (``0`` = free,
            ``1`` = perfectly locked to axis).
        track_enabled: Toggle track guidance on/off.
        collision_enabled: Toggle vessel containment on/off.
        collision_mesh_normals: Optional pre-computed smooth vertex normals
            on the *same* device as the solver.  Pass the output of
            :func:`compute_smooth_vertex_normals` wrapped in a
            :class:`warp.array`.
        smooth_collision_normals_enabled: Use area-weighted vertex normals
            instead of flat face normals (reduces facet-aligned ringing).
        collision_iterations: Gauss-Seidel iterations per substep.
        collision_pre_constraints_enabled: Run vessel containment *before*
            the XPBD constraint solve.
        collision_post_constraints_enabled: Run vessel containment *after*
            the XPBD constraint solve (default ``True``, recommended).
        mesh_edge_collision_enabled: Use the AABB / edge collision path
            instead of the SDF path.
        mesh_edge_collision_radius: Effective rod radius for the edge path
            (defaults to ``particle_radius``).
        mesh_edge_collision_query_radius: AABB query radius for the edge path
            (defaults to ``mesh_edge_collision_radius``).
        mesh_edge_collision_max_triangles: Triangle budget per AABB query
            (default ``64``).
        collision_projection_stage: Shorthand for enabling exactly one stage
            (``"pre"`` or ``"post"``); overrides the two boolean flags when set.
        sign_scale: Sign convention flip for the mesh (``+1`` if mesh normals
            point inward, ``-1`` if they point outward).
        target_phi: Target signed-distance clearance from the vessel wall.
            Defaults to ``-particle_radius`` (keep the rod surface just inside).
        max_dist: BVH search radius.  Defaults to
            ``2 * particle_radius + segment_length``.
        **kwargs: Keyword arguments forwarded to :class:`XPBDRodSolver`.
    """

    def __init__(
        self,
        *args,
        collision_mesh: "wp.Mesh | None",
        track_start: np.ndarray,
        track_dir: np.ndarray,
        track_length: float,
        tip_num_edges: int,
        particle_radius: float,
        segment_length: float,
        track_stiffness: float = 1.0,
        track_enabled: bool = True,
        collision_enabled: bool = True,
        collision_mesh_normals: "wp.array | None" = None,
        smooth_collision_normals_enabled: bool = False,
        collision_iterations: int = 2,
        collision_pre_constraints_enabled: bool = False,
        collision_post_constraints_enabled: bool = True,
        mesh_edge_collision_enabled: bool = False,
        mesh_edge_collision_radius: float | None = None,
        mesh_edge_collision_query_radius: float | None = None,
        mesh_edge_collision_max_triangles: int = DEFAULT_MESH_EDGE_MAX_TRIANGLES,
        collision_projection_stage: str | None = None,
        sign_scale: float = 1.0,
        target_phi: float | None = None,
        max_dist: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        track_start = np.asarray(track_start, dtype=np.float32)
        track_dir = np.asarray(track_dir, dtype=np.float32)
        norm = np.linalg.norm(track_dir)
        if norm <= 0.0:
            raise ValueError("track_dir must be non-zero")
        track_dir = track_dir / norm

        self.collision_mesh = collision_mesh
        self.collision_mesh_normals = collision_mesh_normals
        self.smooth_collision_normals_enabled = bool(smooth_collision_normals_enabled)
        self.track_start = wp.vec3(float(track_start[0]), float(track_start[1]), float(track_start[2]))
        self.track_dir = wp.vec3(float(track_dir[0]), float(track_dir[1]), float(track_dir[2]))
        self.track_length = float(track_length)
        self.tip_num_edges = int(tip_num_edges)
        self.track_stiffness = float(track_stiffness)
        self.track_enabled = bool(track_enabled)
        self.collision_enabled = bool(collision_enabled)
        self.collision_iterations = max(1, int(collision_iterations))
        self.collision_pre_constraints_enabled = bool(collision_pre_constraints_enabled)
        self.collision_post_constraints_enabled = bool(collision_post_constraints_enabled)
        self.mesh_edge_collision_enabled = bool(mesh_edge_collision_enabled)
        self.mesh_edge_collision_radius = (
            float(mesh_edge_collision_radius)
            if mesh_edge_collision_radius is not None
            else float(particle_radius)
        )
        self.mesh_edge_collision_query_radius = (
            float(mesh_edge_collision_query_radius)
            if mesh_edge_collision_query_radius is not None
            else self.mesh_edge_collision_radius
        )
        self.mesh_edge_collision_max_triangles = max(1, int(mesh_edge_collision_max_triangles))
        if collision_projection_stage is not None:
            self.set_collision_projection_stage(collision_projection_stage)
        self.sign_scale = float(sign_scale)
        self.target_phi = (
            float(target_phi) if target_phi is not None else -float(particle_radius)
        )
        self.max_dist = (
            float(max_dist)
            if max_dist is not None
            else 2.0 * float(particle_radius) + float(segment_length)
        )

        # Dedicated per-rod correction buffer for the mesh-edge collision path.
        # Decoupled from ws.pos_corrections (which the constraint solver owns).
        self._mesh_edge_corrections: "wp.array | None" = None

        # Per-device dummy normals array for when smooth normals are disabled.
        self._dummy_collision_mesh_normals: dict[str, "wp.array"] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def set_collision_mesh(
        self,
        collision_mesh: "wp.Mesh | None",
        collision_mesh_normals: "wp.array | None" = None,
    ) -> None:
        """Swap the vessel mesh at runtime (also rebuilds the BVH on next step).

        Args:
            collision_mesh: New vessel mesh, or ``None`` to disable collision.
            collision_mesh_normals: Optional pre-computed smooth vertex normals
                on the solver device.
        """
        self.collision_mesh = collision_mesh
        self.collision_mesh_normals = collision_mesh_normals

    def set_collision_projection_stage(self, stage: str) -> None:
        """Configure whether containment runs before or after rod constraints.

        Args:
            stage: ``"pre"`` or ``"post"``.
        """
        normalized = _normalize_collision_projection_stage(stage)
        self.collision_pre_constraints_enabled = normalized == COLLISION_PROJECTION_STAGE_PRE
        self.collision_post_constraints_enabled = normalized == COLLISION_PROJECTION_STAGE_POST

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _collision_normals_for_device(
        self, dev: str
    ) -> "tuple[wp.array, bool]":
        """Return ``(normals_array, use_smooth)`` for the given device."""
        if self.smooth_collision_normals_enabled and self.collision_mesh_normals is not None:
            return self.collision_mesh_normals, True
        if dev not in self._dummy_collision_mesh_normals:
            self._dummy_collision_mesh_normals[dev] = wp.zeros(1, dtype=wp.vec3, device=dev)
        return self._dummy_collision_mesh_normals[dev], False

    def _ensure_edge_corrections(self, ws: _Workspace, dev: str) -> "wp.array":
        """Lazily allocate (or re-use) the per-rod edge-collision buffer."""
        if (
            self._mesh_edge_corrections is None
            or self._mesh_edge_corrections.shape[0] < ws.num_points
        ):
            self._mesh_edge_corrections = wp.zeros(
                max(1, ws.num_points), dtype=wp.vec3, device=dev
            )
        return self._mesh_edge_corrections

    def _project_track_guidance(self, ws: _Workspace, dev: str) -> None:
        """Project non-tip particles onto the insertion axis."""
        if not self.track_enabled:
            return
        end_idx = max(1, ws.num_points - self.tip_num_edges)
        wp.launch(
            _track_sliding_kernel,
            dim=ws.num_points,
            inputs=[
                ws.predicted_positions,
                ws.inv_masses,
                self.track_start,
                self.track_dir,
                float(self.track_length),
                float(self.track_stiffness),
                int(end_idx),
            ],
            device=dev,
        )

    def _project_vessel_containment(self, ws: _Workspace, dev: str) -> None:
        """Run vessel containment for the configured number of iterations."""
        if not self.collision_enabled or self.collision_mesh is None:
            return

        smooth_normals, use_smooth = self._collision_normals_for_device(dev)

        if self.mesh_edge_collision_enabled:
            corrections = self._ensure_edge_corrections(ws, dev)
            for _ in range(self.collision_iterations):
                wp.launch(_clear_vec3_kernel, dim=ws.num_points, inputs=[corrections], device=dev)
                wp.launch(
                    _project_mesh_vertex_collision_kernel_averaged,
                    dim=ws.num_points,
                    inputs=[
                        ws.predicted_positions,
                        self.collision_mesh.id,
                        smooth_normals,
                        ws.inv_masses,
                        float(self.mesh_edge_collision_radius),
                        float(self.mesh_edge_collision_query_radius),
                        float(self.sign_scale),
                        bool(use_smooth),
                        int(self.mesh_edge_collision_max_triangles),
                        corrections,
                    ],
                    device=dev,
                )
                wp.launch(
                    _project_mesh_edge_collision_kernel_averaged,
                    dim=ws.num_edges,
                    inputs=[
                        ws.predicted_positions,
                        self.collision_mesh.id,
                        smooth_normals,
                        ws.inv_masses,
                        float(self.mesh_edge_collision_radius),
                        float(self.mesh_edge_collision_query_radius),
                        float(self.sign_scale),
                        bool(use_smooth),
                        int(self.mesh_edge_collision_max_triangles),
                        corrections,
                    ],
                    device=dev,
                )
                wp.launch(
                    _apply_mesh_edge_collision_corrections_kernel,
                    dim=ws.num_points,
                    inputs=[ws.predicted_positions, ws.inv_masses, corrections],
                    device=dev,
                )
        else:
            for _ in range(self.collision_iterations):
                wp.launch(
                    _project_vessel_containment_kernel,
                    dim=ws.num_points,
                    inputs=[
                        ws.positions,
                        ws.predicted_positions,
                        self.collision_mesh.id,
                        smooth_normals,
                        ws.inv_masses,
                        float(self.sign_scale),
                        bool(use_smooth),
                        float(self.target_phi),
                        float(self.max_dist),
                    ],
                    device=dev,
                )

    # ──────────────────────────────────────────────────────────────────────
    # Hooks wired into XPBDRodSolver._substep
    # ──────────────────────────────────────────────────────────────────────

    def _pre_constraints_hook(self, ws: _Workspace, dt: float, dev: str) -> None:
        """Apply vessel containment *before* XPBD constraint solve when configured."""
        del dt
        if self.collision_pre_constraints_enabled:
            self._project_vessel_containment(ws, dev)

    def _post_constraints_hook(self, ws: _Workspace, dt: float, dev: str) -> None:
        """Apply track guidance and vessel containment *after* XPBD constraint solve."""
        del dt
        self._project_track_guidance(ws, dev)
        if self.collision_post_constraints_enabled:
            self._project_vessel_containment(ws, dev)


__all__ = [
    "XCathRodSolver",
    "COLLISION_PROJECTION_STAGE_POST",
    "COLLISION_PROJECTION_STAGE_PRE",
    "compute_smooth_vertex_normals",
    "compute_signed_distances",
    "_apply_mesh_edge_collision_corrections_kernel",
    "_project_mesh_edge_collision_kernel",
    "_project_mesh_edge_collision_kernel_averaged",
    "_project_mesh_vertex_collision_kernel",
    "_project_mesh_vertex_collision_kernel_averaged",
    "_project_vessel_containment_kernel",
    "_sample_signed_distance_kernel",
]
