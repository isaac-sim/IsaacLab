# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Restore the workcell's OmniPBR surface channels in the standalone RTX viewer.

Newton's mesh export does not carry the asset's normal and scalar texture maps into
the viewer. Translate the task assets' direct OmniPBR inputs to UsdPreviewSurface,
using their original UVs and cached textures. This only changes rendered materials.
"""

from collections.abc import Iterable

from pxr import Gf, Sdf, Usd, UsdShade


def restore_workcell_materials(stage: Usd.Stage, source_stage: Usd.Stage, instances: Iterable[tuple[str, str]]) -> None:
    """Bind authored surface channels to rendered robot and table instances.

    Args:
        stage: Viewer stage, before RTX initialization.
        source_stage: Simulation stage with resolved asset references.
        instances: Pairs of rendered instance paths and source shape paths.
    """
    materials = {}
    for instance_path, shape_path in instances:
        if not shape_path.startswith(("/World/envs/env_0/Robot/", "/World/envs/env_0/Table/")):
            continue
        source = source_stage.GetPrimAtPath(shape_path)
        if not source and shape_path.endswith("_visual"):
            source = source_stage.GetPrimAtPath(shape_path.removesuffix("_visual"))
        if not source:
            continue
        material, _ = UsdShade.MaterialBindingAPI(source).ComputeBoundMaterial()
        if not material:
            continue
        shader, _, _ = material.ComputeSurfaceSource("mdl")
        if not shader or shader.GetSourceAssetSubIdentifier("mdl") != "OmniPBR":
            continue
        key = material.GetPath()
        if key not in materials:
            materials[key] = _preview_material(stage, shader, f"/World/WorkcellMaterials/Material_{len(materials)}")
        instance = stage.GetPrimAtPath(instance_path)
        if instance:
            UsdShade.MaterialBindingAPI.Apply(instance).Bind(materials[key])


def _preview_material(stage: Usd.Stage, source: UsdShade.Shader, path: str) -> UsdShade.Material:
    """Translate the direct surface inputs used by the Franka and Seattle table assets."""
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    st = UsdShade.Shader.Define(stage, f"{path}/UV")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for channel, constant, texture, value_type, default in (
        ("diffuseColor", "diffuse_color_constant", "diffuse_texture", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.2)),
        ("roughness", "reflection_roughness_constant", "reflectionroughness_texture", Sdf.ValueTypeNames.Float, 0.5),
        ("metallic", "metallic_constant", "metallic_texture", Sdf.ValueTypeNames.Float, 0.0),
        ("normal", None, "normalmap_texture", Sdf.ValueTypeNames.Normal3f, Gf.Vec3f(0, 0, 1)),
    ):
        target = surface.CreateInput(channel, value_type)
        value = source.GetInput(constant).Get() if constant else None
        target.Set(default if value is None else value)
        asset = source.GetInput(texture).Get()
        if not asset or not asset.path:
            continue
        sampler = UsdShade.Shader.Define(stage, f"{path}/{channel}Texture")
        sampler.CreateIdAttr("UsdUVTexture")
        sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(asset.resolvedPath or asset.path))
        sampler.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
            "sRGB" if channel == "diffuseColor" else "raw"
        )
        for axis in ("S", "T"):
            sampler.CreateInput(f"wrap{axis}", Sdf.ValueTypeNames.Token).Set("repeat")
        sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        if channel == "normal":
            # Tangent-space normal maps encode [-1, 1] as [0, 1].
            sampler.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            sampler.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        elif channel == "diffuseColor":
            bias = source.GetInput("albedo_add").Get()
            if bias is not None:
                sampler.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(bias, bias, bias, 0))
        output = "r" if value_type == Sdf.ValueTypeNames.Float else "rgb"
        sampler.CreateOutput(output, Sdf.ValueTypeNames.Float if output == "r" else Sdf.ValueTypeNames.Float3)
        target.ConnectToSource(sampler.ConnectableAPI(), output)
    return material
