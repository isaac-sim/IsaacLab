# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Literal

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree


def fix_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Repair a mesh by addressing common issues (holes, duplicated vertices).

    Args:
        mesh: The mesh to repair.

    Returns:
        The repaired mesh.
    """
    mesh.merge_vertices()
    trimesh.repair.fill_holes(mesh)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.unique_faces())
    return mesh


def simplify_mesh(
    mesh: trimesh.Trimesh,
    method: Literal[
        "vertex_clustering",
        "relative_quadric_decimation",
        "absolute_quadric_decimation",
    ],
    parameter: float | int,
) -> trimesh.Trimesh:
    """Simplify a mesh by reducing vertices while preserving shape and vertex colors where possible.

    Args:
        mesh: The input mesh to simplify.
        method: Simplification method to use. Supported values:
            - ``vertex_clustering``: uses a voxel grid; parameter is ``voxel_size``.
            - ``relative_quadric_decimation``: parameter is relative target
              fraction of triangles (0..1).
            - ``absolute_quadric_decimation``: parameter is target number of
              triangles.
        parameter: Method-specific parameter (see description above).

    Returns:
        The simplified mesh as a :class:`trimesh.Trimesh`.
    """
    # Convert to Open3D mesh
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )

    # Simplify
    if method == "vertex_clustering":
        mesh_simplified = mesh_o3d.simplify_vertex_clustering(
            voxel_size=parameter,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
    elif method == "relative_quadric_decimation":
        mesh_simplified = mesh_o3d.simplify_quadric_decimation(int(len(mesh.faces) * parameter))
    elif method == "absolute_quadric_decimation":
        mesh_simplified = mesh_o3d.simplify_quadric_decimation(parameter)
    else:
        raise RuntimeError("Unknown simplification method.")

    # Interpolate colors from the original mesh to the simplified vertices
    # using nearest-neighbor in vertex space.
    vertices_simplified = np.asarray(mesh_simplified.vertices)
    tree = cKDTree(np.asarray(mesh.vertices))
    _, indices = tree.query(vertices_simplified)
    colors_simplified = mesh.visual.vertex_colors[indices, :3]

    # Convert back to Trimesh
    return trimesh.Trimesh(
        vertices=vertices_simplified,
        faces=np.asarray(mesh_simplified.triangles),
        vertex_colors=colors_simplified,
        process=True,
    )


def cut_above(mesh: trimesh.Trimesh, height: float) -> trimesh.Trimesh:
    """Remove vertices above a specified height and cap the mesh, preserving colors.

    Args:
        mesh: Input mesh to slice.
        height: Cut plane height (units match mesh coordinates). Vertices with
            z > height are removed/capped.

    Returns:
        The sliced mesh with vertex colors recovered from the original mesh.
    """
    # cut
    mesh_sliced = mesh.slice_plane(
        plane_origin=np.array([0.0, 0.0, height]),
        plane_normal=np.array([0.0, 0.0, -1.0]),
        cap=True,
    )

    # recover colors
    vertices_sliced = np.asarray(mesh_sliced.vertices)
    tree = cKDTree(np.asarray(mesh.vertices))
    _, indices = tree.query(vertices_sliced)
    mesh_sliced.visual.vertex_colors = mesh.visual.vertex_colors[indices]

    return mesh_sliced
