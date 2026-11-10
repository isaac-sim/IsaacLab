# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect

import pytest

from pxr import Sdf, Usd, UsdGeom, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim import UsdFileCfg, use_stage
from isaaclab.sim.spawners.materials import visual_materials
from isaaclab.sim.spawners.materials.visual_materials_cfg import MdlFileCfg, PbrMdlCfg, PreviewSurfaceCfg
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshRectangleCfg
from isaaclab.sim.utils.prims import bind_visual_material
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR


@pytest.fixture
def stage():
    stage = Usd.Stage.CreateInMemory()
    with use_stage(stage):
        yield stage


def test_spawn_preview_surface_without_kit(stage):
    """Preview surfaces should use renderer-agnostic USD shader inputs and outputs."""
    cfg = PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))

    prim = visual_materials.spawn_preview_surface("/Looks/PreviewSurface", cfg)

    shader = UsdShade.Shader(prim)
    material = UsdShade.Material(stage.GetPrimAtPath("/Looks/PreviewSurface"))
    assert prim.GetPrimTypeInfo().GetTypeName() == "Shader"
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert shader.GetInput("diffuseColor").Get() == cfg.diffuse_color
    assert shader.GetInput("diffuseColor").GetTypeName() == Sdf.ValueTypeNames.Color3f
    assert shader.GetInput("emissiveColor").GetTypeName() == Sdf.ValueTypeNames.Color3f
    assert shader.GetInput("roughness").GetTypeName() == Sdf.ValueTypeNames.Float
    assert shader.GetOutput("surface").GetTypeName() == Sdf.ValueTypeNames.Token
    assert shader.GetOutput("displacement").GetTypeName() == Sdf.ValueTypeNames.Token
    assert material.GetSurfaceOutput().HasConnectedSource()
    assert material.GetDisplacementOutput().HasConnectedSource()


def test_spawn_and_bind_mdl_material_without_kit(stage):
    """MDL spawning should create a bindable material without requiring Kit commands."""
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


def test_authored_material_overrides_preserve_textures_and_clone_bindings(stage, tmp_path):
    """Author surface inputs before cloning without changing textures or other render contexts."""
    path = str(tmp_path / "prop.usda")
    source = Usd.Stage.CreateNew(path)
    source.SetDefaultPrim(UsdGeom.Xform.Define(source, "/Prop").GetPrim())
    source.GetDefaultPrim().SetInstanceable(True)
    for name in ("body", "label"):
        material = UsdShade.Material.Define(source, f"/Prop/Looks/{name}")
        shader = UsdShade.Shader.Define(source, f"{material.GetPath()}/Shader")
        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.CreateOutput("out", Sdf.ValueTypeNames.Token))
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"{name}.png"))
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set((0.2, 0.3, 0.4))
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.1)
        texture = UsdShade.Shader.Define(source, f"{material.GetPath()}/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.25)
        shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).ConnectToSource(
            texture.CreateOutput("texture", Sdf.ValueTypeNames.Asset)
        )
        preview = UsdShade.Shader.Define(source, f"{material.GetPath()}/Preview")
        preview.CreateIdAttr("UsdPreviewSurface")
        preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
        material.CreateSurfaceOutput().ConnectToSource(preview.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        UsdShade.MaterialBindingAPI.Apply(UsdGeom.Cube.Define(source, f"/Prop/{name}").GetPrim()).Bind(material)
    source.GetRootLayer().Save()
    cfg = UsdFileCfg(
        usd_path=path,
        make_uninstanceable=True,
        visual_material_path=None,
        visual_material=PbrMdlCfg(
            diffuse_color_constant=None,
            reflection_roughness_constant=0.7,
            metallic_constant=0.3,
            albedo_brightness=2.0,
            metallic_texture_influence=0.0,
            reflection_roughness_texture_influence=0.0,
        ),
    )
    for index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/env_{index}")
    cfg.func("/World/env_.*/Prop", cfg)
    for index in range(2):
        for name in ("body", "label"):
            root = f"/World/env_{index}/Prop"
            material_path = f"{root}/Looks/{name}"
            binding = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(f"{root}/{name}"))
            assert binding.ComputeBoundMaterial()[0].GetPath() == material_path
            shader = UsdShade.Shader(stage.GetPrimAtPath(f"{material_path}/Shader"))
            assert shader.GetSourceAsset("mdl").path == "OmniPBR.mdl"
            assert shader.GetInput("diffuse_texture").Get().path == f"{name}.png"
            assert shader.GetInput("diffuse_color_constant").Get() == pytest.approx((0.2, 0.3, 0.4))
            assert shader.GetInput("normalmap_texture").GetConnectedSource()[0].GetPath() == f"{material_path}/Texture"
            texture = UsdShade.Shader(stage.GetPrimAtPath(f"{material_path}/Texture"))
            assert [value.GetBaseName() for value in texture.GetInputs()] == ["reflection_roughness_constant"]
            assert texture.GetInput("reflection_roughness_constant").Get() == 0.25
            preview = UsdShade.Shader(stage.GetPrimAtPath(f"{material_path}/Preview"))
            assert [value.GetBaseName() for value in preview.GetInputs()] == ["roughness"]
            assert preview.GetInput("roughness").Get() == pytest.approx(0.2)
            for channel in (
                "reflection_roughness_constant",
                "metallic_constant",
                "albedo_brightness",
                "metallic_texture_influence",
                "reflection_roughness_texture_influence",
            ):
                assert shader.GetInput(channel).Get() == pytest.approx(getattr(cfg.visual_material, channel))


def test_spawn_mesh_with_visual_material_without_kit(stage):
    """Mesh spawners should author and bind their inline material without Kit."""
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
    """Keep authoring private and visual material ownership in the OpenUSD layer."""
    source = inspect.getsource(visual_materials)
    assert "has_kit" not in source
    assert "omni.usd.commands" not in source
    assert "Usd.PrimRange" not in source
    for module in (sim_utils, sim_utils.spawners, sim_utils.spawners.materials, visual_materials):
        assert not hasattr(module, "modify_visual_material")
        assert "_author_material_inputs" not in getattr(module, "__all__", ())
