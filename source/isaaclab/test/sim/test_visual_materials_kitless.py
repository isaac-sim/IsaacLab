# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect

from pxr import Sdf, Usd, UsdGeom, UsdShade

from isaaclab.sim.spawners.materials import visual_materials
from isaaclab.sim.spawners.materials.visual_materials_cfg import MdlFileCfg, PreviewSurfaceCfg
from isaaclab.sim.spawners.meshes import meshes
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshRectangleCfg
from isaaclab.sim.utils import prims as prim_utils
from isaaclab.sim.utils.prims import bind_visual_material
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR


def test_spawn_preview_surface_without_kit(monkeypatch):
    """Preview surfaces should use renderer-agnostic USD shader inputs and outputs."""
    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(visual_materials, "get_current_stage", lambda: stage)
    monkeypatch.setattr(prim_utils, "get_current_stage", lambda: stage)
    cfg = PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))

    prim = visual_materials.spawn_preview_surface("/Looks/PreviewSurface", cfg)

    shader = UsdShade.Shader(prim)
    material = UsdShade.Material(stage.GetPrimAtPath("/Looks/PreviewSurface"))
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert shader.GetInput("diffuseColor").GetTypeName() == Sdf.ValueTypeNames.Color3f
    assert shader.GetInput("emissiveColor").GetTypeName() == Sdf.ValueTypeNames.Color3f
    assert shader.GetInput("roughness").GetTypeName() == Sdf.ValueTypeNames.Float
    assert shader.GetOutput("surface").GetTypeName() == Sdf.ValueTypeNames.Token
    assert shader.GetOutput("displacement").GetTypeName() == Sdf.ValueTypeNames.Token
    assert material.GetSurfaceOutput().HasConnectedSource()
    assert material.GetDisplacementOutput().HasConnectedSource()


def test_spawn_and_bind_mdl_material_without_kit(monkeypatch):
    """MDL spawning should create a bindable material without requiring Kit commands."""
    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(visual_materials, "get_current_stage", lambda: stage)
    monkeypatch.setattr(prim_utils, "get_current_stage", lambda: stage)
    cfg = MdlFileCfg(
        mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl",
        project_uvw=True,
        albedo_brightness=0.5,
    )

    prim = visual_materials.spawn_from_mdl_file("/Looks/MdlMaterial", cfg)
    UsdGeom.Cube.Define(stage, "/World/Geometry")
    bind_visual_material("/World/Geometry", "/Looks/MdlMaterial", stage=stage)

    shader = UsdShade.Shader(prim)
    material = UsdShade.Material(stage.GetPrimAtPath("/Looks/MdlMaterial"))
    source_asset = shader.GetSourceAsset("mdl")
    assert source_asset.path == (f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl")
    assert shader.GetSourceAssetSubIdentifier("mdl") == "Aluminum_Anodized"
    assert shader.GetOutput("out").GetRenderType() == "material"
    assert material.GetSurfaceOutput("mdl").HasConnectedSource()
    assert material.GetDisplacementOutput("mdl").HasConnectedSource()
    assert material.GetVolumeOutput("mdl").HasConnectedSource()
    assert prim.GetAttribute("inputs:project_uvw").Get() is True
    assert prim.GetAttribute("inputs:albedo_brightness").Get() == 0.5
    binding = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/Geometry")).GetDirectBinding()
    assert binding.GetMaterialPath() == "/Looks/MdlMaterial"


def test_spawn_mesh_with_visual_material_without_kit(monkeypatch):
    """Mesh spawners should author and bind their inline material without Kit."""
    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(meshes, "get_current_stage", lambda: stage)
    monkeypatch.setattr(visual_materials, "get_current_stage", lambda: stage)
    monkeypatch.setattr(prim_utils, "get_current_stage", lambda: stage)
    cfg = MeshRectangleCfg(size=(0.2, 0.2), visual_material=PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)))

    cfg.func("/World/Cloth", cfg)

    material_path = Sdf.Path("/World/Cloth/geometry/material")
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    binding = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/Cloth/geometry/mesh")).GetDirectBinding()
    assert material
    assert binding.GetMaterialPath() == material_path
    assert UsdShade.Shader(stage.GetPrimAtPath(material_path.AppendChild("Shader"))).GetInput("diffuseColor").Get() == (
        0.95,
        0.85,
        0.1,
    )


def test_visual_material_spawners_do_not_depend_on_kit_commands():
    """Keep visual material ownership in the OpenUSD layer."""
    source = inspect.getsource(visual_materials)
    assert "has_kit" not in source
    assert "omni.usd.commands" not in source
