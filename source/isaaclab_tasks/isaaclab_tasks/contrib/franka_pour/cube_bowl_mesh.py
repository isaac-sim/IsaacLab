# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watertight hollow-cube bowl collision mesh for the Franka pour task.

The bowl is an open-top rectangular box (a "hollow cube") generated as a single closed,
consistently outward-wound triangle mesh, so an MPM collider built from it has an unambiguous
inside/outside. The walls and floor must be **at least ~1.5 grid voxels thick**:
MPM resolves mesh colliders at grid-node resolution, so a sub-voxel wall has
no solid interior on the grid and particles tunnel through it.

Local frame: ``z=0`` is the outer base (table-facing); the cavity floor is at ``z=bottom_thickness``;
the rim is at ``z = bottom_thickness + cavity_depth``. The bowl is centred on the z axis in x/y.
"""

from __future__ import annotations

import numpy as np

# Corner ordering for an axis-aligned ring, CCW viewed from +z: (-,-), (+,-), (+,+), (-,+).
_RING_SIGNS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
# Outward (away-from-solid) horizontal normal of outer wall ``k`` (the edge from ring corner k to k+1).
_OUTER_WALL_NORMALS = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))


def _validate_closed_oriented_mesh(indices: np.ndarray, name: str) -> None:
    """Validate that every triangle edge has exactly one oppositely directed partner."""
    triangles = indices.reshape(-1, 3)
    directed_edges: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        edges = (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        )
        for edge in edges:
            directed_edges[edge] = directed_edges.get(edge, 0) + 1

    duplicate_edges = [edge for edge, count in directed_edges.items() if count != 1]
    if duplicate_edges:
        raise RuntimeError(f"{name} mesh winding is inconsistent: {len(duplicate_edges)} directed edges not unique.")
    boundary_edges = [(start, end) for start, end in directed_edges if (end, start) not in directed_edges]
    if boundary_edges:
        raise RuntimeError(f"{name} mesh is not watertight: {len(boundary_edges)} boundary edges.")


def make_cube_bowl_mesh(
    *,
    inner_width: float,
    inner_depth: float,
    cavity_depth: float,
    wall_thickness: float,
    bottom_thickness: float,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a watertight, outward-wound open-top hollow-cube bowl collision mesh.

    Args:
        inner_width: Inner cavity size along x [m].
        inner_depth: Inner cavity size along y [m].
        cavity_depth: Inner cavity height from the cavity floor to the rim [m].
        wall_thickness: Side-wall thickness [m]; keep >= ~1.5 * MPM voxel_size to avoid tunnelling.
        bottom_thickness: Base thickness below the cavity floor [m]; keep >= ~1.5 * voxel_size.
        validate: If True, assert the mesh is a closed, consistently-oriented manifold.

    Returns:
        ``(vertices, indices)`` with vertices ``(V, 3)`` float32 and triangle indices ``(3F,)`` int32.
    """
    ihx, ihy = 0.5 * float(inner_width), 0.5 * float(inner_depth)
    ohx, ohy = ihx + float(wall_thickness), ihy + float(wall_thickness)
    bt = float(bottom_thickness)
    total_h = bt + float(cavity_depth)

    def ring(hx: float, hy: float, z: float) -> np.ndarray:
        return np.array([[sx * hx, sy * hy, z] for sx, sy in _RING_SIGNS], dtype=np.float64)

    outer_base = ring(ohx, ohy, 0.0)  # [0:4)
    outer_top = ring(ohx, ohy, total_h)  # [4:8)
    inner_top = ring(ihx, ihy, total_h)  # [8:12)
    inner_floor = ring(ihx, ihy, bt)  # [12:16)
    vertices = np.vstack([outer_base, outer_top, inner_top, inner_floor]).astype(np.float32)
    outer_base_start, outer_top_start, inner_top_start, inner_floor_start = 0, 4, 8, 12

    faces: list[int] = []

    def quad(i0: int, i1: int, i2: int, i3: int, normal: tuple[float, float, float]) -> None:
        """Emit two triangles for a cyclic quad, wound CCW about ``normal`` (outward from the solid)."""
        p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]
        geo_n = np.cross(p1 - p0, p2 - p0)
        if float(np.dot(geo_n, np.asarray(normal, dtype=np.float64))) < 0.0:
            i1, i3 = i3, i1  # reverse winding
        faces.extend([i0, i1, i2, i0, i2, i3])

    # Outer base (z=0), normal -z.
    quad(
        outer_base_start + 0,
        outer_base_start + 1,
        outer_base_start + 2,
        outer_base_start + 3,
        (0.0, 0.0, -1.0),
    )
    for k in range(4):
        j = (k + 1) % 4
        # Outer side wall (z=0..total_h), outward horizontal normal.
        quad(
            outer_base_start + k,
            outer_base_start + j,
            outer_top_start + j,
            outer_top_start + k,
            _OUTER_WALL_NORMALS[k],
        )
        # Top rim trapezoid (z=total_h), normal +z; the outer-to-inner diagonal tiles the frame.
        quad(
            outer_top_start + k,
            outer_top_start + j,
            inner_top_start + j,
            inner_top_start + k,
            (0.0, 0.0, 1.0),
        )
        # Inner cavity wall (z=bottom_thickness..total_h), normal points into the cavity.
        inner_n = tuple(-c for c in _OUTER_WALL_NORMALS[k])
        quad(
            inner_floor_start + k,
            inner_floor_start + j,
            inner_top_start + j,
            inner_top_start + k,
            inner_n,
        )
    # Cavity floor (z=bottom_thickness), normal +z (into the cavity).
    quad(
        inner_floor_start + 0,
        inner_floor_start + 1,
        inner_floor_start + 2,
        inner_floor_start + 3,
        (0.0, 0.0, 1.0),
    )

    indices = np.asarray(faces, dtype=np.int32)

    # Auto-orient to outward (positive signed volume), matching make_hemisphere_scoop_mesh.
    tris = vertices[indices.reshape(-1, 3)]
    signed_vol = float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum())
    if signed_vol < 0.0:
        indices = indices.reshape(-1, 3)[:, ::-1].reshape(-1).astype(np.int32)

    if validate:
        _validate_closed_oriented_mesh(indices, "Cube bowl")

    return vertices, indices


def cube_bowl_inner_bounds(
    inner_width: float,
    inner_depth: float,
    cavity_depth: float,
    bottom_thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the open inner-cavity axis-aligned bounds in the bowl local frame.

    Args:
        inner_width: Inner cavity size along x [m].
        inner_depth: Inner cavity size along y [m].
        cavity_depth: Inner cavity height [m].
        bottom_thickness: Base thickness below the cavity floor [m].

    Returns:
        ``(lo, hi)`` each ``(3,)`` float32: the cavity floor corner and the rim corner.
    """
    ihx, ihy = 0.5 * float(inner_width), 0.5 * float(inner_depth)
    bt = float(bottom_thickness)
    lo = np.array([-ihx, -ihy, bt], dtype=np.float32)
    hi = np.array([ihx, ihy, bt + float(cavity_depth)], dtype=np.float32)
    return lo, hi
