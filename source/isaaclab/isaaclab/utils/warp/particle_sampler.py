# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sample particle lattices inside solid mesh volumes and hollow mesh cavities."""

from __future__ import annotations

import operator

import numpy as np
import warp as wp


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize a triangle mesh for the sampling queries."""
    vertices_array = np.asarray(vertices, dtype=np.float32)
    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise ValueError(f"`vertices` must have shape (N, 3), got {vertices_array.shape}.")
    if not np.all(np.isfinite(vertices_array)):
        raise ValueError("`vertices` must contain only finite values.")

    faces_array = np.asarray(faces)
    if faces_array.ndim == 1:
        if faces_array.size % 3 != 0:
            raise ValueError(f"Flattened `faces` must contain a multiple of 3 indices, got {faces_array.size}.")
        faces_array = faces_array.reshape(-1, 3)
    elif faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise ValueError(f"`faces` must have shape (N, 3) or (3 * N,), got {faces_array.shape}.")

    if vertices_array.shape[0] < 4 or faces_array.shape[0] < 4:
        raise ValueError(
            f"A closed mesh needs >= 4 vertices and >= 4 faces, "
            f"got {vertices_array.shape[0]} and {faces_array.shape[0]}."
        )
    if (
        not np.issubdtype(faces_array.dtype, np.number)
        or np.issubdtype(faces_array.dtype, np.bool_)
        or np.iscomplexobj(faces_array)
        or not np.all(np.isfinite(faces_array))
        or not np.all(faces_array == np.trunc(faces_array))
    ):
        raise ValueError("`faces` must contain integer indices.")
    if np.any(faces_array < 0) or np.any(faces_array >= vertices_array.shape[0]):
        raise ValueError(f"`faces` indices must be in [0, {vertices_array.shape[0] - 1}].")
    faces_array = faces_array.astype(np.int32, copy=False)

    return np.ascontiguousarray(vertices_array), np.ascontiguousarray(faces_array)


def _validate_lattice_settings(
    spacing: float,
    jitter: float,
    inset: float,
    surface_margin: float,
    max_query_dist: float,
) -> None:
    """Validate settings shared by the solid and cavity samplers."""
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"`spacing` must be positive and finite, got {spacing}.")
    if not np.isfinite(jitter) or not 0.0 <= jitter <= 0.5:
        raise ValueError(f"`jitter` must be in [0, 0.5], got {jitter}.")
    if not np.isfinite(inset) or inset < 0.0:
        raise ValueError(f"`inset` must be non-negative and finite, got {inset}.")
    if not np.isfinite(surface_margin) or surface_margin < 0.0:
        raise ValueError(f"`surface_margin` must be non-negative and finite, got {surface_margin}.")
    if not np.isfinite(max_query_dist) or max_query_dist <= 0.0:
        raise ValueError(f"`max_query_dist` must be positive and finite, got {max_query_dist}.")


def _candidate_lattice(vertices: np.ndarray, spacing: float, inset: float, jitter: float, seed: int) -> np.ndarray:
    """Build a regular, optionally jittered lattice clipped (inset) to the mesh AABB.

    Args:
        vertices: Mesh vertices, shape ``(num_vertices, 3)`` [m].
        spacing: Target distance between neighboring lattice points [m].
        inset: Distance, in units of :paramref:`spacing`, by which the lattice is inset from the AABB
            faces so points do not start exactly on the surface.
        jitter: Uniform position jitter as a fraction of :paramref:`spacing`, in ``[0, 0.5]``.
        seed: RNG seed for the jitter so sampling is reproducible.

    Returns:
        Candidate positions, shape ``(num_candidates, 3)`` float32. May be empty.
    """
    lower = vertices.min(axis=0) + inset * spacing
    upper = vertices.max(axis=0) - inset * spacing
    if np.any(upper < lower):
        return np.empty((0, 3), dtype=np.float32)
    counts = np.maximum(np.floor((upper - lower) / spacing).astype(np.int64) + 1, 1)
    axes = [lower[dim] + np.arange(counts[dim]) * spacing for dim in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    if grid.shape[0] > 0 and jitter > 0.0:
        rng = np.random.default_rng(seed)
        grid += ((rng.random(grid.shape, dtype=np.float32) - 0.5) * (2.0 * float(jitter) * spacing)).astype(np.float32)
    return grid


@wp.kernel
def mark_points_in_mesh_kernel(
    points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    max_query_dist: wp.float32,
    surface_margin: wp.float32,
    inside: wp.array(dtype=wp.float32),
):
    """Flag, per candidate point, whether it lies inside the mesh via the winding-number sign.

    Args:
        points: Candidate positions to test, shape ``(num_candidates,)``.
        mesh_id: Warp mesh id (built with ``support_winding_number=True``).
        max_query_dist: Maximum distance for the closest-point search [m].
        surface_margin: Minimum distance a point must keep from the surface to be accepted [m]. This
            erodes the volume inward, dropping thin features (tubes/shells) and keeping particles off
            the walls. ``0`` disables the check.
        inside: Output containment flags (``1.0`` inside, ``0.0`` outside), shape ``(num_candidates,)``.
    """
    tid = wp.tid()
    p = points[tid]
    query = wp.mesh_query_point_sign_winding_number(mesh_id, p, max_query_dist)
    flag = wp.float32(0.0)
    # Warp convention: a negative winding-number sign means the point is inside the mesh.
    if query.result and query.sign < 0.0:
        if surface_margin <= 0.0:
            flag = wp.float32(1.0)
        else:
            # Erode by the distance to the nearest surface point (closest point of the hit face).
            closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
            if wp.length(p - closest) >= surface_margin:
                flag = wp.float32(1.0)
    inside[tid] = flag


def sample_particles_in_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    spacing: float,
    *,
    device: str,
    jitter: float = 0.0,
    seed: int = 0,
    inset: float = 0.5,
    surface_margin: float = 0.0,
    max_query_dist: float = 1.0e6,
) -> np.ndarray:
    """Sample a regular, optionally jittered particle lattice clipped to a mesh interior.

    A lattice with the requested :paramref:`spacing` is generated over the mesh axis-aligned bounding
    box and every lattice point is tested for containment on the selected Warp device. Only interior
    points are returned, in the same frame as :paramref:`vertices`.

    Args:
        vertices: Mesh vertices, shape ``(num_vertices, 3)`` [m].
        faces: Triangle indices, shape ``(num_faces, 3)`` or flattened ``(3 * num_faces,)``.
        spacing: Target distance between neighboring particles [m]. Must be positive.
        device: Warp/torch device string the sampler runs on (e.g. ``"cuda:0"``).
        jitter: Uniform position jitter as a fraction of :paramref:`spacing`, in ``[0, 0.5]``.
            ``0`` keeps the lattice perfectly regular. Defaults to ``0``.
        seed: RNG seed for the jitter so sampling is reproducible. Defaults to ``0``.
        inset: Distance, in units of :paramref:`spacing`, by which the lattice is inset from the AABB
            faces so particles do not start exactly on the surface. Defaults to ``0.5``.
        surface_margin: Minimum distance kept from the mesh surface [m]. Points closer than this to
            any wall are rejected, which erodes the volume inward: thin features (e.g. a teapot's
            spout/handle tubes or shell) drop out while the bulk keeps a solid core, and particles
            stay clear of thin walls. ``0`` keeps the full winding-number interior. Defaults to ``0``.
        max_query_dist: Maximum distance for the closest-point search [m]. Defaults to a large value
            so the winding-number sign and surface distance are always resolved.

    Returns:
        Interior particle positions, shape ``(num_inside, 3)`` float32, in the input frame. May be
        empty if the lattice resolved no interior points (reduce :paramref:`spacing` or
        :paramref:`surface_margin`).
    """
    vertices, faces = _validate_mesh(vertices, faces)
    _validate_lattice_settings(spacing, jitter, inset, surface_margin, max_query_dist)

    # Candidate lattice over the AABB, inset from the faces by a fraction of the spacing.
    grid = _candidate_lattice(vertices, spacing, inset, jitter, seed)
    if grid.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces.reshape(-1), dtype=wp.int32, device=device),
        support_winding_number=True,
    )
    candidates = wp.array(grid, dtype=wp.vec3, device=device)
    inside = wp.empty(grid.shape[0], dtype=wp.float32, device=device)
    wp.launch(
        mark_points_in_mesh_kernel,
        dim=grid.shape[0],
        inputs=[candidates, mesh.id, float(max_query_dist), float(surface_margin)],
        outputs=[inside],
        device=device,
    )
    mask = inside.numpy() > 0.5
    return np.ascontiguousarray(grid[mask], dtype=np.float32)


@wp.kernel
def mark_cavity_points_kernel(
    points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    max_query_dist: wp.float32,
    surface_margin: wp.float32,
    max_ray_dist: wp.float32,
    min_ray_hits: wp.int32,
    water_level: wp.float32,
    inside: wp.array(dtype=wp.float32),
):
    """Flag whether each candidate passes the hollow-shell cavity heuristic.

    A point qualifies when it is *not* inside the shell material (winding-number sign ``>= 0``) and
    enough of the six axis-aligned rays hit the surrounding surface. This is designed for closed,
    consistently oriented, double-walled assets; it is not a general topological enclosure test.

    Args:
        points: Candidate positions to test, shape ``(num_candidates,)``.
        mesh_id: Warp mesh id (built with ``support_winding_number=True``).
        max_query_dist: Maximum distance for the winding-number closest-point search [m].
        surface_margin: Minimum distance a point must keep from the shell surface [m].
        max_ray_dist: Maximum length of the enclosure rays [m].
        min_ray_hits: Minimum number of the six axis rays that must hit for the point to be enclosed.
        water_level: Points with ``z`` above this value are rejected, so the cavity is filled only up
            to a water line. Use a large value to fill the whole cavity.
        inside: Output flags (``1.0`` cavity, ``0.0`` otherwise), shape ``(num_candidates,)``.
    """
    tid = wp.tid()
    p = points[tid]
    flag = wp.float32(0.0)
    if p[2] <= water_level:
        # Reject points inside the solid shell material; a negative sign means "inside material".
        query = wp.mesh_query_point_sign_winding_number(mesh_id, p, max_query_dist)
        if query.result and query.sign >= 0.0:
            closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
            clear_of_surface = surface_margin <= 0.0 or wp.length(p - closest) >= surface_margin
            hits = wp.int32(0)
            if clear_of_surface:
                if wp.mesh_query_ray(mesh_id, p, wp.vec3(1.0, 0.0, 0.0), max_ray_dist).result:
                    hits += 1
                if wp.mesh_query_ray(mesh_id, p, wp.vec3(-1.0, 0.0, 0.0), max_ray_dist).result:
                    hits += 1
                if wp.mesh_query_ray(mesh_id, p, wp.vec3(0.0, 1.0, 0.0), max_ray_dist).result:
                    hits += 1
                if wp.mesh_query_ray(mesh_id, p, wp.vec3(0.0, -1.0, 0.0), max_ray_dist).result:
                    hits += 1
                if wp.mesh_query_ray(mesh_id, p, wp.vec3(0.0, 0.0, 1.0), max_ray_dist).result:
                    hits += 1
                if wp.mesh_query_ray(mesh_id, p, wp.vec3(0.0, 0.0, -1.0), max_ray_dist).result:
                    hits += 1
                if hits >= min_ray_hits:
                    flag = wp.float32(1.0)
    inside[tid] = flag


def sample_particles_in_cavity(
    vertices: np.ndarray,
    faces: np.ndarray,
    spacing: float,
    *,
    device: str,
    jitter: float = 0.0,
    seed: int = 0,
    inset: float = 0.5,
    surface_margin: float = 0.0,
    min_ray_hits: int = 5,
    water_level: float | None = None,
    max_ray_dist: float | None = None,
    max_query_dist: float = 1.0e6,
) -> np.ndarray:
    """Heuristically sample the cavity of a closed, double-walled shell mesh.

    Unlike :func:`sample_particles_in_mesh` (which fills solid material), this fills the *empty cavity*
    of a consistently oriented asset such as the closed, thin, double-walled Utah teapot. This is
    necessary because the winding-number "interior" is the wall material, not the cavity. Each lattice
    point is kept when it is outside the shell material and most of its six axis rays hit the surface.
    This criterion does not prove enclosure for open vessels or arbitrary concave meshes. Optionally,
    only fill up to a :paramref:`water_level`.

    Args:
        vertices: Mesh vertices, shape ``(num_vertices, 3)`` [m].
        faces: Triangle indices, shape ``(num_faces, 3)`` or flattened ``(3 * num_faces,)``.
        spacing: Target distance between neighboring particles [m]. Must be positive.
        device: Warp/torch device string the sampler runs on (e.g. ``"cuda:0"``).
        jitter: Uniform position jitter as a fraction of :paramref:`spacing`, in ``[0, 0.5]``. Defaults
            to ``0``.
        seed: RNG seed for the jitter so sampling is reproducible. Defaults to ``0``.
        inset: Distance, in units of :paramref:`spacing`, by which the lattice is inset from the AABB
            faces. Defaults to ``0.5``.
        surface_margin: Minimum distance kept from the shell surface [m]. Points closer than this are
            rejected so particles do not start inside a collider's effective thickness. Defaults to ``0``.
        min_ray_hits: How many of the six axis rays must hit the surface for a point to count as
            enclosed, in ``[1, 6]``. Higher values keep only deep interior points (dropping thin
            features like spouts/handles); lower values fill closer to openings. Defaults to ``5``.
        water_level: Absolute ``z`` in the mesh frame above which points are dropped, to fill only up
            to a water line. ``None`` fills the whole cavity. Defaults to ``None``.
        max_ray_dist: Maximum length of the enclosure rays [m]. ``None`` uses the mesh AABB diagonal,
            which always spans the cavity. Defaults to ``None``.
        max_query_dist: Maximum distance for the winding-number closest-point search [m]. Defaults to a
            large value so the sign is always resolved.

    Returns:
        Cavity particle positions, shape ``(num_inside, 3)`` float32, in the input frame. May be empty
        if no enclosed points were found (reduce :paramref:`spacing` or :paramref:`min_ray_hits`).
    """
    vertices, faces = _validate_mesh(vertices, faces)
    _validate_lattice_settings(spacing, jitter, inset, surface_margin, max_query_dist)
    try:
        min_ray_hits = operator.index(min_ray_hits)
    except TypeError:
        raise ValueError(f"`min_ray_hits` must be an integer, got {min_ray_hits}.") from None
    if isinstance(min_ray_hits, bool):
        raise ValueError(f"`min_ray_hits` must be an integer, got {min_ray_hits}.")
    if not 1 <= min_ray_hits <= 6:
        raise ValueError(f"`min_ray_hits` must be in [1, 6], got {min_ray_hits}.")
    if water_level is not None and not np.isfinite(water_level):
        raise ValueError(f"`water_level` must be finite or None, got {water_level}.")
    if max_ray_dist is not None and (not np.isfinite(max_ray_dist) or max_ray_dist <= 0.0):
        raise ValueError(f"`max_ray_dist` must be positive and finite, got {max_ray_dist}.")

    grid = _candidate_lattice(vertices, spacing, inset, jitter, seed)
    if grid.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    if max_ray_dist is None:
        max_ray_dist = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        if max_ray_dist <= 0.0:
            raise ValueError("The mesh AABB diagonal must be positive when `max_ray_dist` is None.")
    level = float(water_level) if water_level is not None else 1.0e30

    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces.reshape(-1), dtype=wp.int32, device=device),
        support_winding_number=True,
    )
    candidates = wp.array(grid, dtype=wp.vec3, device=device)
    inside = wp.empty(grid.shape[0], dtype=wp.float32, device=device)
    wp.launch(
        mark_cavity_points_kernel,
        dim=grid.shape[0],
        inputs=[
            candidates,
            mesh.id,
            float(max_query_dist),
            float(surface_margin),
            float(max_ray_dist),
            int(min_ray_hits),
            level,
        ],
        outputs=[inside],
        device=device,
    )
    mask = inside.numpy() > 0.5
    return np.ascontiguousarray(grid[mask], dtype=np.float32)
