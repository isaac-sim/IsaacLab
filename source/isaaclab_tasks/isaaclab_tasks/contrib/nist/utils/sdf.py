# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SDF query mega-kernel and packed collider transforms.

Pure Warp math. Three layers:

- :class:`ColliderTransform` — packed (pos, mat, mat_inv) struct holding
  one collider's rigid pose. Static scale/shear is baked into mesh vertices.
  Articulated collider poses are refreshed from live body poses.
- ``@wp.func`` building blocks (:func:`scale_mat33_columns`,
  :func:`compute_world_point`, :func:`compose_root_mat`,
  :func:`query_collider_sdf`) — composable inline-by-the-JIT helpers.
- :func:`get_signed_distance_mega` — single-launch kernel that iterates
  every (env, body, point, obstacle, collider) tuple to find the minimum
  world-space signed distance.

Consumed by :class:`~.collision_analyzer.CollisionAnalyzer`.
"""

from __future__ import annotations

import warp as wp

# ---------------------------------------------------------------------------
# @wp.struct — packed data structures for the SDF kernel
# ---------------------------------------------------------------------------


@wp.struct
class ColliderTransform:
    """Per-collider rigid transform (root-relative, or world-space for articulated bodies).

    Bounds enclose the statically transformed mesh in its local frame [m]. Equal lower/upper
    bounds disable pruning, preserving callers that only initialize the transform fields.
    """

    pos: wp.vec3
    mat: wp.mat33
    mat_inv: wp.mat33
    bounds_lower: wp.vec3
    bounds_upper: wp.vec3


# ---------------------------------------------------------------------------
# @wp.kernel — pack flat arrays into ColliderTransform struct array (one-time)
# ---------------------------------------------------------------------------


@wp.kernel
def pack_collider_transforms(
    pos: wp.array(dtype=wp.vec3),
    mat: wp.array(dtype=wp.mat33),
    mat_inv: wp.array(dtype=wp.mat33),
    out: wp.array(dtype=ColliderTransform),
):
    """Pack three flat arrays into a contiguous ColliderTransform struct array.

    Launched once at init to avoid Python-side per-element struct construction.
    """
    i = wp.tid()
    ct = ColliderTransform()
    ct.pos = pos[i]
    ct.mat = mat[i]
    ct.mat_inv = mat_inv[i]
    out[i] = ct


@wp.kernel
def _pack_collider_bounds(
    bounds: wp.array(dtype=wp.vec3, ndim=2),
    indices: wp.array(dtype=wp.int32),
    out: wp.array(dtype=ColliderTransform),
):
    """Scatter deduplicated mesh bounds into the existing collider slots once at initialization."""
    i = wp.tid()
    ct = out[i]
    ct.bounds_lower = bounds[indices[i], 0]
    ct.bounds_upper = bounds[indices[i], 1]
    out[i] = ct


# ---------------------------------------------------------------------------
# @wp.func — composable building blocks (inlined by compiler)
# ---------------------------------------------------------------------------


@wp.func
def scale_mat33_columns(m: wp.mat33, s: wp.vec3) -> wp.mat33:
    """Column-scale a 3x3 matrix: result = M @ diag(s)."""
    return wp.mat33(
        m[0, 0] * s[0],
        m[0, 1] * s[1],
        m[0, 2] * s[2],
        m[1, 0] * s[0],
        m[1, 1] * s[1],
        m[1, 2] * s[2],
        m[2, 0] * s[0],
        m[2, 1] * s[1],
        m[2, 2] * s[2],
    )


@wp.func
def compute_world_point(body_pos: wp.vec3, body_quat: wp.quat, local_pt: wp.vec3) -> wp.vec3:
    """Transform a local surface point to world space using body pose."""
    return wp.quat_rotate(body_quat, local_pt) + body_pos


@wp.func
def compose_root_mat(root_quat: wp.quat, root_scale: wp.vec3) -> wp.mat33:
    """Build root_mat = R(quat) @ diag(scale) from runtime pose + static scale."""
    rot = wp.quat_to_matrix(root_quat)
    return scale_mat33_columns(rot, root_scale)


@wp.func
def query_collider_sdf(
    query_world: wp.vec3,
    root_pos: wp.vec3,
    root_mat: wp.mat33,
    root_mat_inv: wp.mat33,
    coll: ColliderTransform,
    mesh_id: wp.uint64,
    max_dist: float,
    check_dist: bool,
    query_origin: wp.vec3 = wp.vec3(0.0),
) -> float:
    """Query signed distance [m] with rigid runtime transforms and statically transformed mesh vertices.

    Nonuniform scale/shear must be baked into the mesh before construction: transforming the
    nearest displacement afterward cannot correct nearest-face selection in a different metric.
    ``query_world`` may be relative to ``query_origin`` [m] to avoid rounding small surface offsets
    when adding them to large world positions. The default retains absolute-world queries.
    Nonfinite transformed queries return negative infinity so invalid geometry fails closed.
    """
    to_world = root_mat * coll.mat
    to_local = coll.mat_inv * root_mat_inv
    q_local = to_local * ((query_origin - root_pos) - root_mat * coll.pos + query_world)
    if not wp.isfinite(q_local):
        return -wp.inf
    # The inverse-transform Frobenius norm bounds the world query radius in mesh coordinates.
    local_max_dist = max_dist * wp.sqrt(wp.ddot(to_local, to_local))
    # Axis-aligned ray votes can misclassify exterior points coplanar with mesh faces.
    mp = wp.mesh_query_point_sign_winding_number(mesh_id, q_local, local_max_dist)
    if mp.result:
        if check_dist:
            closest = wp.mesh_eval_position(mesh_id, mp.face, mp.u, mp.v)
            dist_world = to_world * (q_local - closest)
            signed_dist = wp.length(dist_world) * mp.sign
        else:
            signed_dist = mp.sign
        if not wp.isfinite(signed_dist):
            return -wp.inf
        return signed_dist
    return max_dist


# ---------------------------------------------------------------------------
# @wp.kernel — SDF mega kernel (single GPU dispatch)
# ---------------------------------------------------------------------------


@wp.kernel
def get_signed_distance_mega(
    body_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    body_quat_w: wp.array(dtype=wp.quat, ndim=2),
    body_ids: wp.array(dtype=wp.int32),
    asset_rows: wp.array(dtype=wp.int32),
    local_pts: wp.array(dtype=wp.vec3, ndim=3),
    env_ids: wp.array(dtype=wp.int32),
    source_ids: wp.array(dtype=wp.int32),
    obs_root_pos: wp.array(dtype=wp.vec3, ndim=2),
    obs_root_quat: wp.array(dtype=wp.quat, ndim=2),
    obs_root_scale: wp.array(dtype=wp.vec3),
    colliders: wp.array(dtype=ColliderTransform),
    mesh_handles: wp.array(dtype=wp.uint64),
    prim_counts: wp.array(dtype=wp.int32),
    obstacle_row_offsets: wp.array(dtype=wp.int32),
    collider_rows: wp.array(dtype=wp.int32, ndim=2),
    min_dist_offsets: wp.array(dtype=float),
    max_dist: float,
    min_dist: float,
    check_dist: bool,
    num_bodies: int,
    num_points: int,
    num_obstacles: int,
    max_prims: int,
    signs: wp.array(dtype=float),
):
    """SDF kernel with inline cloud construction and transform composition.

    Each thread handles one (env, body, point) tuple, scanning obstacles and colliders until a
    threshold violation. Rejected outputs are violation witnesses, not necessarily deepest distances.

    Source row indices and obstacle environment indices are independent, allowing hypothetical source poses.
    Returns signed distances adjusted onto the common ``min_dist`` threshold. When every obstacle uses that
    threshold, the output is a world-space signed distance [m] (positive = outside, negative = inside).
    """
    tid = wp.tid()
    env_local = tid // (num_bodies * num_points)
    rem = tid - env_local * num_bodies * num_points
    body_local = rem // num_points
    point_local = rem - body_local * num_points

    env_id = env_ids[env_local]
    source_id = source_ids[env_local]
    body_id = body_ids[body_local]
    asset_row = asset_rows[source_id]

    query_origin = body_pos_w[source_id, body_id]
    query = wp.quat_rotate(
        body_quat_w[source_id, body_id],
        local_pts[asset_row, body_local, point_local],
    )
    if not wp.isfinite(query_origin) or not wp.isfinite(query):
        signs[tid] = -wp.inf
        return

    best = max_dist
    for obs_idx in range(num_obstacles):
        collider_row = collider_rows[obs_idx, env_id]
        packed_row = obstacle_row_offsets[obs_idx] + collider_row
        root_mat = compose_root_mat(
            obs_root_quat[obs_idx, env_id],
            obs_root_scale[packed_row],
        )
        root_mat_inv = wp.inverse(root_mat)
        root_pos = obs_root_pos[obs_idx, env_id]

        prim_base = packed_row * max_prims

        for p in range(prim_counts[packed_row]):
            index = prim_base + p
            mid = mesh_handles[index]
            if mid != wp.uint64(0):
                coll = colliders[index]
                if check_dist and wp.length_sq(coll.bounds_upper - coll.bounds_lower) > 0.0:
                    to_local = coll.mat_inv * root_mat_inv
                    q_local = to_local * ((query_origin - root_pos) - root_mat * coll.pos + query)
                    if not wp.isfinite(q_local):
                        best = -wp.inf
                        break
                    delta = wp.max(wp.max(coll.bounds_lower - q_local, q_local - coll.bounds_upper), wp.vec3(0.0))
                    lower_bound = wp.length(delta) / (wp.sqrt(wp.ddot(to_local, to_local)) * 1.000001)
                    # Never prune interior queries. Preserve the max_dist no-hit sentinel and
                    # per-obstacle offsets, not only the eventual acceptance predicate.
                    if lower_bound > 0.0 and wp.min(lower_bound, max_dist) + min_dist_offsets[obs_idx] >= best:
                        continue
                sd = query_collider_sdf(
                    query,
                    root_pos,
                    root_mat,
                    root_mat_inv,
                    coll,
                    mid,
                    max_dist,
                    check_dist,
                    query_origin,
                )
                adjusted_sd = sd + min_dist_offsets[obs_idx]
                if adjusted_sd < best:
                    best = adjusted_sd
                    if best < min_dist:
                        break

        if best < min_dist:
            break

    signs[tid] = best


# ---------------------------------------------------------------------------
# @wp.kernel — copy obstacle root poses into stacked buffer
# ---------------------------------------------------------------------------


@wp.kernel
def update_root_poses(
    dst_pos: wp.array(dtype=wp.vec3, ndim=2),
    dst_quat: wp.array(dtype=wp.quat, ndim=2),
    src_pos: wp.array(dtype=wp.vec3),
    src_quat: wp.array(dtype=wp.quat),
    obs_idx: int,
):
    """Copy one obstacle's root pos/quat into the stacked buffer at row obs_idx."""
    eid = wp.tid()
    dst_pos[obs_idx, eid] = src_pos[eid]
    dst_quat[obs_idx, eid] = src_quat[eid]


# ---------------------------------------------------------------------------
# @wp.kernel — refresh articulated obstacles' collider transforms per call
# ---------------------------------------------------------------------------


@wp.kernel
def update_articulated_collider_transforms(
    body_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    body_quat_w: wp.array(dtype=wp.quat, ndim=2),
    coll_env_ids: wp.array(dtype=wp.int32),
    coll_body_ids: wp.array(dtype=wp.int32),
    coll_slots: wp.array(dtype=wp.int32),
    rel_pos: wp.array(dtype=wp.vec3),
    rel_mat: wp.array(dtype=wp.mat33),
    out: wp.array(dtype=ColliderTransform),
):
    """Recompose an articulated obstacle's collider transforms from live body poses.

    A rigid obstacle's colliders are static relative to its root, so the
    kernel-side ``root_pose x rel`` composition stays correct as the root
    moves. An articulated obstacle's colliders ride on its BODY LINKS, whose
    poses change with the joint configuration — caching them root-relative
    freezes the obstacle in its spawn articulation. This kernel writes each
    collider's WORLD transform (body pose x static body-relative offset)
    directly into the packed collider slot; the obstacle's stacked root pose
    is kept at identity so the query kernel needs no changes.
    """
    i = wp.tid()
    e = coll_env_ids[i]
    b = coll_body_ids[i]
    rot = wp.quat_to_matrix(body_quat_w[e, b])
    ct = out[coll_slots[i]]
    ct.mat = rot * rel_mat[i]
    ct.mat_inv = wp.inverse(ct.mat)
    ct.pos = body_pos_w[e, b] + rot * rel_pos[i]
    out[coll_slots[i]] = ct
