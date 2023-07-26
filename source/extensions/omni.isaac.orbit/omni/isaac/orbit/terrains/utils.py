# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import trimesh
from typing import List


def color_meshes_by_height(meshes: List[trimesh.Trimesh], **kwargs) -> trimesh.Trimesh:
    """
    Color the vertices of a trimesh object based on the z-coordinate (height) of each vertex,
    using the Turbo colormap. If the z-coordinates are all the same, the vertices will be colored
    with a single color.

    Args:
        meshes (List[trimesh.Trimesh]): A list of trimesh objects.

    Keyword Args:
        color (List[int]): A list of 3 integers in the range [0,255] representing the RGB
            color of the mesh. Used when the z-coordinates of all vertices are the same.

    Returns:
        trimesh.Trimesh: A trimesh object with the vertices colored based on the z-coordinate (height) of each vertex.
    """
    # Combine all meshes into a single mesh
    mesh = trimesh.util.concatenate(meshes)
    # Get the z-coordinates of each vertex
    heights = mesh.vertices[:, 2]
    # Check if the z-coordinates are all the same
    if np.max(heights) == np.min(heights):
        # Obtain a single color: light blue
        color = kwargs.pop("color", [172, 216, 230, 255])
        color = np.asarray(color, dtype=np.uint8)
        # Set the color for all vertices
        mesh.visual.vertex_colors = color
    else:
        # Normalize the heights to [0,1]
        heights_normalized = (heights - np.min(heights)) / (np.max(heights) - np.min(heights))
        # Get the color for each vertex based on the height
        colors = trimesh.visual.color.interpolate(heights_normalized, color_map="turbo")
        # Set the vertex colors
        mesh.visual.vertex_colors = colors
    # Return the mesh
    return mesh


def create_prim_from_mesh(prim_path: str, vertices: np.ndarray, triangles: np.ndarray, **kwargs):
    """Create a USD prim with mesh defined from vertices and triangles.

    The function creates a USD prim with a mesh defined from vertices and triangles. It performs the
    following steps:

    - Create a USD Xform prim at the path :obj:`prim_path`.
    - Create a USD prim with a mesh defined from the input vertices and triangles at the path :obj:`{prim_path}/mesh`.
    - Assign a physics material to the mesh at the path :obj:`{prim_path}/physicsMaterial`.
    - Assign a visual material to the mesh at the path :obj:`{prim_path}/visualMaterial`.

    Args:
        prim_path (str): The path to the primitive to be created.
        vertices (np.ndarray): The vertices of the mesh. Shape is :math:`(N, 3)`, where :math:`N`
            is the number of vertices.
        triangles (np.ndarray): The triangles of the mesh as references to vertices for each triangle.
            Shape is :math:`(M, 3)`, where :math:`M` is the number of triangles / faces.

    Keyword Args:
        translation (Optional[Sequence[float]]): The translation of the terrain. Defaults to None.
        orientation (Optional[Sequence[float]]): The orientation of the terrain. Defaults to None.
        scale (Optional[Sequence[float]]): The scale of the terrain. Defaults to None.
        color (Optional[tuple]): The color of the terrain. Defaults to (0.065, 0.0725, 0.080).
        static_friction (float): The static friction of the terrain. Defaults to 1.0.
        dynamic_friction (float): The dynamic friction of the terrain. Defaults to 1.0.
        restitution (float): The restitution of the terrain. Defaults to 0.0.
        improve_patch_friction (bool): Whether to enable patch friction. Defaults to False.
        combine_mode (str): Determines the way physics materials will be combined during collisions.
            Available options are `average`, `min`, `multiply`, `multiply`, and `max`. Defaults to `average`.
    """
    # need to import these here to prevent isaacsim launching when importing this module
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.materials import PhysicsMaterial, PreviewSurface
    from omni.isaac.core.prims import GeometryPrim, XFormPrim
    from pxr import PhysxSchema

    # create parent prim
    prim_utils.create_prim(prim_path, "Xform")
    # create mesh prim
    prim_utils.create_prim(
        f"{prim_path}/mesh",
        "Mesh",
        translation=kwargs.get("translation"),
        orientation=kwargs.get("orientation"),
        scale=kwargs.get("scale"),
        attributes={
            "points": vertices,
            "faceVertexIndices": triangles.flatten(),
            "faceVertexCounts": np.asarray([3] * len(triangles)),
            "subdivisionScheme": "bilinear",
        },
    )

    # create visual material
    color = kwargs.get("color", (0.065, 0.0725, 0.080))
    if color is not None:
        material = PreviewSurface(f"{prim_path}/visualMaterial", color=np.asarray(color))
        XFormPrim(f"{prim_path}/mesh").apply_visual_material(material)

    # create physics material
    material = PhysicsMaterial(
        f"{prim_path}/physicsMaterial",
        static_friction=kwargs.get("static_friction", 1.0),
        dynamic_friction=kwargs.get("dynamic_friction", 1.0),
        restitution=kwargs.get("restitution", 0.0),
    )
    # apply PhysX Rigid Material schema
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
    # set patch friction property
    improve_patch_friction = kwargs.get("improve_patch_friction", False)
    physx_material_api.CreateImprovePatchFrictionAttr().Set(improve_patch_friction)
    # set combination mode for coefficients
    combine_mode = kwargs.get("combine_mode", "multiply")
    physx_material_api.CreateFrictionCombineModeAttr().Set(combine_mode)
    physx_material_api.CreateRestitutionCombineModeAttr().Set(combine_mode)
    # apply physics material to ground plane
    GeometryPrim(f"{prim_path}/mesh", collision=True).apply_physics_material(material)
