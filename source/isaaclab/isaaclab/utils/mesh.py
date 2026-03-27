# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Utility functions for working with meshes."""

from collections.abc import Callable

import numpy as np
import trimesh

from pxr import Usd, UsdGeom

__all__ = [
    "create_trimesh_from_geom_mesh",
    "create_trimesh_from_geom_shape",
    "convert_faces_to_triangles",
    "PRIMITIVE_MESH_TYPES",
]


def create_trimesh_from_geom_mesh(mesh_prim: Usd.Prim) -> trimesh.Trimesh:
    """Reads the vertices and faces of a mesh prim.

    The function reads the vertices and faces of a mesh prim and returns it. If the underlying mesh is a quad mesh,
    it converts it to a triangle mesh.

    Args:
        mesh_prim: The mesh prim to read the vertices and faces from.

    Returns:
        A trimesh.Trimesh object containing the mesh geometry.
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
    return trimesh.Trimesh(points, convert_faces_to_triangles(indices, num_vertex_per_face))


def create_trimesh_from_geom_shape(prim: Usd.Prim) -> trimesh.Trimesh:
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

    return _MESH_CONVERTERS_CALLBACKS[prim.GetTypeName()](prim)


def convert_faces_to_triangles(faces: np.ndarray, point_counts: np.ndarray) -> np.ndarray:
    """Converts quad mesh face indices into triangle face indices.

    This function expects an array of faces (indices) and the number of points per face. It then converts potential
    quads into triangles and returns the new triangle face indices as a numpy array of shape (n_faces_new, 3).

    Args:
        faces: The faces of the quad mesh as a one-dimensional array. Shape is (N,).
        point_counts: The number of points per face. Shape is (N,).

    Returns:
        The new face ids with triangles. Shape is (n_faces_new, 3).
    """
    # check if the mesh is already triangulated
    if (point_counts == 3).all():
        return faces.reshape(-1, 3)  # already triangulated
    all_faces = []

    vertex_counter = 0
    # Iterates over all faces of the mesh to triangulate them.
    # could be very slow for large meshes
    for num_points in point_counts:
        # Triangulate n-gons (n>4) using fan triangulation
        for i in range(num_points - 2):
            triangle = np.array([faces[vertex_counter], faces[vertex_counter + 1 + i], faces[vertex_counter + 2 + i]])
            all_faces.append(triangle)

        vertex_counter += num_points
    return np.asarray(all_faces)


"""
Internal USD Shape Handlers.
"""


def _create_plane_trimesh(prim: Usd.Prim) -> trimesh.Trimesh:
    """Creates a trimesh for a plane primitive."""
    size = (2e6, 2e6)
    vertices = np.array([[size[0], size[1], 0], [size[0], 0.0, 0], [0.0, size[1], 0], [0.0, 0.0, 0]]) - np.array(
        [size[0] / 2.0, size[1] / 2.0, 0.0]
    )
    faces = np.array([[1, 0, 2], [2, 3, 1]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def _create_cube_trimesh(prim: Usd.Prim) -> trimesh.Trimesh:
    """Creates a trimesh for a cube primitive."""
    size = prim.GetAttribute("size").Get()
    extends = [size, size, size]
    return trimesh.creation.box(extends)


def _create_sphere_trimesh(prim: Usd.Prim, subdivisions: int = 2) -> trimesh.Trimesh:
    """Creates a trimesh for a sphere primitive."""
    radius = prim.GetAttribute("radius").Get()
    mesh = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
    return mesh


def _create_cylinder_trimesh(prim: Usd.Prim) -> trimesh.Trimesh:
    """Creates a trimesh for a cylinder primitive."""
    radius = prim.GetAttribute("radius").Get()
    height = prim.GetAttribute("height").Get()
    mesh = trimesh.creation.cylinder(radius=radius, height=height)
    axis = prim.GetAttribute("axis").Get()
    if axis == "X":
        # rotate −90° about Y to point the length along +X
        R = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
        mesh.apply_transform(R)
    elif axis == "Y":
        # rotate +90° about X to point the length along +Y
        R = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
        mesh.apply_transform(R)
    return mesh


def _create_capsule_trimesh(prim: Usd.Prim) -> trimesh.Trimesh:
    """Creates a trimesh for a capsule primitive."""
    radius = prim.GetAttribute("radius").Get()
    height = prim.GetAttribute("height").Get()
    mesh = trimesh.creation.capsule(radius=radius, height=height)
    axis = prim.GetAttribute("axis").Get()
    if axis == "X":
        # rotate −90° about Y to point the length along +X
        R = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
        mesh.apply_transform(R)
    elif axis == "Y":
        # rotate +90° about X to point the length along +Y
        R = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
        mesh.apply_transform(R)
    return mesh


def _create_cone_trimesh(prim: Usd.Prim) -> trimesh.Trimesh:
    """Creates a trimesh for a cone primitive."""
    radius = prim.GetAttribute("radius").Get()
    height = prim.GetAttribute("height").Get()
    mesh = trimesh.creation.cone(radius=radius, height=height)
    # shift all vertices down by height/2 for usd / trimesh cone primitive definition discrepancy
    mesh.apply_translation((0.0, 0.0, -height / 2.0))
    return mesh


_MESH_CONVERTERS_CALLBACKS: dict[str, Callable[[Usd.Prim], trimesh.Trimesh]] = {
    "Plane": _create_plane_trimesh,
    "Cube": _create_cube_trimesh,
    "Sphere": _create_sphere_trimesh,
    "Cylinder": _create_cylinder_trimesh,
    "Capsule": _create_capsule_trimesh,
    "Cone": _create_cone_trimesh,
}

PRIMITIVE_MESH_TYPES = list(_MESH_CONVERTERS_CALLBACKS.keys())
"""List of supported primitive mesh types that can be converted to a trimesh."""
