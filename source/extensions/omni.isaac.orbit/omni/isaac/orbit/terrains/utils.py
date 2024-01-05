# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import trimesh


def color_meshes_by_height(meshes: list[trimesh.Trimesh], **kwargs) -> trimesh.Trimesh:
    """
    Color the vertices of a trimesh object based on the z-coordinate (height) of each vertex,
    using the Turbo colormap. If the z-coordinates are all the same, the vertices will be colored
    with a single color.

    Args:
        meshes: A list of trimesh objects.

    Keyword Args:
        color: A list of 3 integers in the range [0,255] representing the RGB
            color of the mesh. Used when the z-coordinates of all vertices are the same.
        color_map: The name of the color map to be used. Defaults to "turbo".

    Returns:
        A trimesh object with the vertices colored based on the z-coordinate (height) of each vertex.
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
        # clip lower and upper bounds to have better color mapping
        heights_normalized = np.clip(heights_normalized, 0.1, 0.9)
        # Get the color for each vertex based on the height
        color_map = kwargs.pop("color_map", "turbo")
        colors = trimesh.visual.color.interpolate(heights_normalized, color_map=color_map)
        # Set the vertex colors
        mesh.visual.vertex_colors = colors
    # Return the mesh
    return mesh


def create_prim_from_mesh(prim_path: str, mesh: trimesh.Trimesh, **kwargs):
    """Create a USD prim with mesh defined from vertices and triangles.

    The function creates a USD prim with a mesh defined from vertices and triangles. It performs the
    following steps:

    - Create a USD Xform prim at the path :obj:`prim_path`.
    - Create a USD prim with a mesh defined from the input vertices and triangles at the path :obj:`{prim_path}/mesh`.
    - Assign a physics material to the mesh at the path :obj:`{prim_path}/physicsMaterial`.
    - Assign a visual material to the mesh at the path :obj:`{prim_path}/visualMaterial`.

    Args:
        prim_path: The path to the primitive to be created.
        mesh: The mesh to be used for the primitive.

    Keyword Args:
        translation: The translation of the terrain. Defaults to None.
        orientation: The orientation of the terrain. Defaults to None.
        visual_material: The visual material to apply. Defaults to None.
        physics_material: The physics material to apply. Defaults to None.
    """
    # need to import these here to prevent isaacsim launching when importing this module
    import omni.isaac.core.utils.prims as prim_utils
    from pxr import UsdGeom

    import omni.isaac.orbit.sim as sim_utils

    # create parent prim
    prim_utils.create_prim(prim_path, "Xform")
    # create mesh prim
    prim = prim_utils.create_prim(
        f"{prim_path}/mesh",
        "Mesh",
        translation=kwargs.get("translation"),
        orientation=kwargs.get("orientation"),
        attributes={
            "points": mesh.vertices,
            "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(mesh.faces)),
            "subdivisionScheme": "bilinear",
        },
    )
    # apply collider properties
    collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    sim_utils.define_collision_properties(prim.GetPrimPath(), collider_cfg)
    # add rgba color to the mesh primvars
    if mesh.visual.vertex_colors is not None:
        # obtain color from the mesh
        rgba_colors = np.asarray(mesh.visual.vertex_colors).astype(np.float32) / 255.0
        # displayColor is a primvar attribute that is used to color the mesh
        color_prim_attr = prim.GetAttribute("primvars:displayColor")
        color_prim_var = UsdGeom.Primvar(color_prim_attr)
        color_prim_var.SetInterpolation(UsdGeom.Tokens.vertex)
        color_prim_attr.Set(rgba_colors[:, :3])
        # displayOpacity is a primvar attribute that is used to set the opacity of the mesh
        display_prim_attr = prim.GetAttribute("primvars:displayOpacity")
        display_prim_var = UsdGeom.Primvar(display_prim_attr)
        display_prim_var.SetInterpolation(UsdGeom.Tokens.vertex)
        display_prim_var.Set(rgba_colors[:, 3])

    # create visual material
    if kwargs.get("visual_material") is not None:
        visual_material_cfg: sim_utils.VisualMaterialCfg = kwargs.get("visual_material")
        # spawn the material
        visual_material_cfg.func(f"{prim_path}/visualMaterial", visual_material_cfg)
        sim_utils.bind_visual_material(prim.GetPrimPath(), f"{prim_path}/visualMaterial")
    # create physics material
    if kwargs.get("physics_material") is not None:
        physics_material_cfg: sim_utils.RigidBodyMaterialCfg = kwargs.get("physics_material")
        # spawn the material
        physics_material_cfg.func(f"{prim_path}/physicsMaterial", physics_material_cfg)
        sim_utils.bind_physics_material(prim.GetPrimPath(), f"{prim_path}/physicsMaterial")
