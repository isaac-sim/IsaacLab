# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import tempfile
import torch

import PIL.Image
from occupancy_map import OccupancyMap
from PIL import ImageDraw
from pxr import Kind, Sdf, Usd, UsdGeom, UsdShade


def occupancy_map_add_to_stage(
    occupancy_map: OccupancyMap,
    stage: Usd.Stage,
    path: str,
    z_offset: float = 0.0,
    draw_path: np.ndarray | torch.Tensor | None = None,
    draw_path_line_width_meter: float = 0.25,
) -> Usd.Prim:

    image_path = os.path.join(tempfile.mkdtemp(), "texture.png")
    image = occupancy_map.ros_image()

    if draw_path is not None:
        if isinstance(draw_path, torch.Tensor):
            draw_path = draw_path.detach().cpu().numpy()
        image = image.copy().convert("RGBA")
        draw = ImageDraw.Draw(image)
        line_coordinates = []
        path_pixels = occupancy_map.world_to_pixel_numpy(draw_path)
        for i in range(len(path_pixels)):
            line_coordinates.append(int(path_pixels[i, 0]))
            line_coordinates.append(int(path_pixels[i, 1]))
        width_pixels = draw_path_line_width_meter / occupancy_map.resolution
        draw.line(line_coordinates, fill="green", width=int(width_pixels / 2), joint="curve")

    # need to flip, ros uses inverted coordinates on y axis
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    image.save(image_path)

    x0, y0 = occupancy_map.top_left_pixel_world_coords()
    x1, y1 = occupancy_map.bottom_right_pixel_world_coords()

    # Add model
    modelRoot = UsdGeom.Xform.Define(stage, path)
    Usd.ModelAPI(modelRoot).SetKind(Kind.Tokens.component)

    # Add mesh
    mesh = UsdGeom.Mesh.Define(stage, os.path.join(path, "mesh"))
    mesh.CreatePointsAttr([(x0, y0, z_offset), (x1, y0, z_offset), (x1, y1, z_offset), (x0, y1, z_offset)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateExtentAttr([(x0, y0, z_offset), (x1, y1, z_offset)])

    texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
    )

    texCoords.Set([(0, 0), (1, 0), (1, 1), (0, 1)])

    # Add material
    material_path = os.path.join(path, "material")
    material = UsdShade.Material.Define(stage, material_path)
    pbrShader = UsdShade.Shader.Define(stage, os.path.join(material_path, "shader"))
    pbrShader.CreateIdAttr("UsdPreviewSurface")
    pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

    # Add texture to material
    stReader = UsdShade.Shader.Define(stage, os.path.join(material_path, "st_reader"))
    stReader.CreateIdAttr("UsdPrimvarReader_float2")
    diffuseTextureSampler = UsdShade.Shader.Define(stage, os.path.join(material_path, "diffuse_texture"))
    diffuseTextureSampler.CreateIdAttr("UsdUVTexture")
    diffuseTextureSampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(image_path)
    diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        stReader.ConnectableAPI(), "result"
    )
    diffuseTextureSampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        diffuseTextureSampler.ConnectableAPI(), "rgb"
    )

    stInput = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
    stInput.Set("st")
    stReader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(stInput)
    mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(mesh).Bind(material)

    return modelRoot
