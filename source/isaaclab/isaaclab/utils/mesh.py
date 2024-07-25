# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for working with meshes."""

import numpy as np
import trimesh

from pxr import Usd, UsdGeom

PRIMITIVE_MESH_TYPES = ["Cube", "Plane"]
"""List of supported primitive mesh types that can be converted to a trimesh."""


def create_trimesh_from_geom_mesh(mesh_prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    """Reads the vertices and faces of a mesh prim.

    The function reads the vertices and faces of a mesh prim and returns it. If the underlying mesh is a quad mesh,
    it converts it to a triangle mesh.

    Args:
        mesh_prim: The mesh prim to read the vertices and faces from.

    Returns:
        A tuple containing the vertices and faces of the mesh.
        Shape of vertices is (n_vertices, 3).
        Shape of faces is (n_faces, 3).
    """
    if mesh_prim.GetTypeName() != "Mesh":
        raise ValueError(f"Prim at path '{mesh_prim.GetPath()}' is not a mesh.")
    # cast into UsdGeomMesh
    mesh = UsdGeom.Mesh(mesh_prim)

    # read the vertices and faces
    points = np.asarray(mesh.GetPointsAttr().Get()).copy()

    # Load faces and convert to triangle if needed. (Default is quads)
    num_vertex_per_face = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    return points, convert_faces_to_triangles(indices, num_vertex_per_face)


def create_mesh_from_geom_shape(prim: Usd.Prim) -> trimesh.Trimesh:
    """Converts a primitive object to a trimesh.

    Args:
        prim: The prim that should be converted to a trimesh.

    Returns:
        A trimesh object representing the primitive.

    Raises:
        ValueError: If the prim is not a supported primitive. Check PRIMITIVE_MESH_TYPES for supported primitives.
    """

    if prim.GetTypeName() not in PRIMITIVE_MESH_TYPES:
        raise ValueError(f"Prim at path '{prim.GetPath()}' is not a primitive mesh. Cannot convert to trimesh.")

    # Create primitive mesh for the provided shapes
    if prim.GetTypeName() == "Plane":
        size = (2e6, 2e6)
        vertices = np.array([[size[0], size[1], 0], [size[0], 0.0, 0], [0.0, size[1], 0], [0.0, 0.0, 0]]) - np.array(
            [size[0] / 2.0, size[1] / 2.0, 0.0]
        )
        faces = np.array([[1, 0, 2], [2, 3, 1]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    elif prim.GetTypeName() == "Cube":
        size = prim.GetAttribute("size").Get()
        extends = [size, size, size]
        mesh = trimesh.creation.box(extends)
    else:
        raise ValueError(f"Prim at path '{prim.GetPath()}' is not a primitive mesh. Cannot convert to trimesh.")

    return mesh


def convert_faces_to_triangles(faces: np.ndarray, point_counts: np.ndarray) -> np.ndarray:
    """Converts quad mesh face indices into triangle face indices.

    This function expects an array of faces (indices) and the number of points per face. It then converts potential
    quads into triangles and returns the new triangle face indices as a numpy array of shape (n_faces_new, 3).

    Args:
        faces: The faces of the quad mesh as a one-dimensional array. Shape is (N,).
        point_counts: The number of points per face. Shape is (N, 3) or (N, 4).

    Returns:
        The new face ids with triangles. Shape is (n_faces_new, 3).
    """
    # check if the mesh is already triangulated
    if (point_counts == 3).all():
        return faces.reshape(-1, 3)  # already triangulated
    all_faces = []

    vertex_counter = 0
    # Iterates over all triangles of the mesh.
    # could be very slow for large meshes
    for num_points in point_counts:
        if num_points == 3:
            # triangle
            all_faces.append(faces[vertex_counter : vertex_counter + 3])
        elif num_points == 4:
            # quads. Subdivide into two triangles
            f = faces[vertex_counter : vertex_counter + 4]
            first_triangle = f[:3]
            second_triangle = np.array([f[0], f[2], f[3]])
            all_faces.append(first_triangle)
            all_faces.append(second_triangle)
        else:
            raise RuntimeError(f"Invalid number of points per face: {num_points}")

        vertex_counter += num_points
    return np.asarray(all_faces)
