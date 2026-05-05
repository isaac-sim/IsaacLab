# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PPISP USD/SPG helpers."""

import pytest

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from isaaclab.renderers.ppisp import (
    PPISP_INPUT_RENDER_VAR,
    PPISP_LDR_RENDER_VAR,
    PPISP_NO_ISP_EXPOSURE,
    PPISP_NO_ISP_EXPOSURE_FSTOP,
    PPISP_NO_ISP_EXPOSURE_ISO,
    PPISP_NO_ISP_EXPOSURE_RESPONSIVITY,
    PPISP_NO_ISP_EXPOSURE_TIME,
    PPISP_OUTPUT_RENDER_VAR,
    PPISPCfg,
    author_ppisp_render_product,
    copy_ppisp_spg_files,
    get_ppisp_spg_file_paths,
    normalize_ppisp_cfg,
    parse_render_product,
    ppisp_cfg_from_usd_shader,
)


def test_ppisp_shader_import_uses_first_time_sample():
    stage = Usd.Stage.CreateInMemory()
    shader = UsdShade.Shader.Define(stage, "/Render/RenderProduct/PPISP")

    exposure = shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).GetAttr()
    exposure.Set(1.0)
    exposure.Set(2.0, 10.0)
    exposure.Set(3.0, 20.0)

    color = shader.CreateInput("colorLatentBlue", Sdf.ValueTypeNames.Float2).GetAttr()
    color.Set(Gf.Vec2f(0.0, 0.0))
    color.Set(Gf.Vec2f(0.1, 0.2), 5.0)

    cfg = ppisp_cfg_from_usd_shader(shader)

    assert cfg.inputs["exposureOffset"] == 2.0
    assert cfg.inputs["colorLatentBlue"] == pytest.approx((0.1, 0.2))


def test_author_ppisp_render_product_creates_no_isp_camera_and_graph():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Camera.Define(stage, "/World/Camera")
    stage.DefinePrim("/Render", "Scope")
    render_product = stage.DefinePrim("/Render/RenderProduct", "RenderProduct")
    render_product.CreateRelationship("camera").SetTargets([Sdf.Path("/World/Camera")])

    shader_prim = author_ppisp_render_product(stage, "/Render/RenderProduct", PPISPCfg())

    camera_targets = render_product.GetRelationship("camera").GetTargets()
    assert len(camera_targets) == 1
    no_isp_camera = stage.GetPrimAtPath(camera_targets[0])
    assert no_isp_camera.IsValid()
    assert no_isp_camera.IsHidden()
    assert no_isp_camera.GetInherits().GetAllDirectInherits() == [Sdf.Path("/World/Camera")]
    assert no_isp_camera.GetAttribute("exposure").Get() == PPISP_NO_ISP_EXPOSURE
    assert no_isp_camera.GetAttribute("exposure:fStop").Get() == PPISP_NO_ISP_EXPOSURE_FSTOP
    assert no_isp_camera.GetAttribute("exposure:iso").Get() == PPISP_NO_ISP_EXPOSURE_ISO
    assert no_isp_camera.GetAttribute("exposure:responsivity").Get() == PPISP_NO_ISP_EXPOSURE_RESPONSIVITY
    assert no_isp_camera.GetAttribute("exposure:time").Get() == PPISP_NO_ISP_EXPOSURE_TIME

    ordered_vars = render_product.GetRelationship("orderedVars").GetTargets()
    assert render_product.GetPath().AppendChild(PPISP_INPUT_RENDER_VAR) in ordered_vars
    assert render_product.GetPath().AppendChild(PPISP_LDR_RENDER_VAR) in ordered_vars

    shader = UsdShade.Shader(shader_prim)
    assert shader.GetInput("HdrColor").GetAttr().GetConnections() == [
        render_product.GetPath().AppendChild(PPISP_INPUT_RENDER_VAR).AppendProperty("omni:rtx:aov")
    ]
    assert shader.GetOutput(PPISP_OUTPUT_RENDER_VAR)
    assert stage.GetPrimAtPath(f"/Render/RenderProduct/{PPISP_LDR_RENDER_VAR}").IsValid()


def test_parse_render_product_collects_ppisp_and_camera_xform_samples():
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    translate = camera.AddTranslateOp()
    translate.Set(Gf.Vec3f(0.0, 0.0, 1.0), 1.0)
    translate.Set(Gf.Vec3f(0.0, 0.0, 2.0), 2.0)
    stage.DefinePrim("/Render", "Scope")
    render_product = stage.DefinePrim("/Render/RenderProduct", "RenderProduct")
    render_product.CreateRelationship("camera").SetTargets([Sdf.Path("/World/Camera")])
    render_product.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(32, 16))

    author_ppisp_render_product(stage, "/Render/RenderProduct", PPISPCfg(inputs={"exposureOffset": 1.25}))

    info = parse_render_product(stage, "/Render/RenderProduct")

    assert info.render_product_path == "/Render/RenderProduct"
    assert info.resolution == (32, 16)
    assert len(info.camera_paths) == 1
    assert info.ppisp is not None
    assert info.ppisp.inputs["exposureOffset"] == 1.25
    assert info.camera_xform_time_samples == [1.0, 2.0]


def test_ppisp_spg_assets_are_bundled_and_used_by_default(tmp_path):
    file_paths = get_ppisp_spg_file_paths()
    for file_path in file_paths.values():
        assert file_path.exists()

    copied_paths = copy_ppisp_spg_files(tmp_path)
    for filename in file_paths:
        assert copied_paths[filename].exists()

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Camera.Define(stage, "/World/Camera")
    stage.DefinePrim("/Render", "Scope")
    render_product = stage.DefinePrim("/Render/RenderProduct", "RenderProduct")
    render_product.CreateRelationship("camera").SetTargets([Sdf.Path("/World/Camera")])

    shader_prim = author_ppisp_render_product(stage, "/Render/RenderProduct", PPISPCfg())
    reference_assets = [str(reference.assetPath) for reference in shader_prim.GetMetadata("references").prependedItems]

    assert any(reference_asset.endswith("ppisp_usd_spg.slang.usda") for reference_asset in reference_assets)


def test_normalize_ppisp_cfg_imports_shader_prim_path_from_stage():
    stage = Usd.Stage.CreateInMemory()
    shader = UsdShade.Shader.Define(stage, "/Render/RenderProduct/PPISP")
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(1.5)
    shader.CreateInput("colorLatentRed", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.25, -0.5))

    cfg = normalize_ppisp_cfg({"shader_prim_path": "/Render/RenderProduct/PPISP"}, stage=stage)

    assert cfg.shader_prim_path == "/Render/RenderProduct/PPISP"
    assert cfg.inputs["exposureOffset"] == 1.5
    assert cfg.inputs["colorLatentRed"] == pytest.approx((0.25, -0.5))


def test_normalize_ppisp_cfg_applies_explicit_overrides_after_shader_import():
    stage = Usd.Stage.CreateInMemory()
    shader = UsdShade.Shader.Define(stage, "/Render/RenderProduct/PPISP")
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(1.5)
    shader.CreateInput("colorLatentRed", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.25, -0.5))

    cfg = normalize_ppisp_cfg(
        {
            "shader_prim_path": "/Render/RenderProduct/PPISP",
            "inputs": {"exposureOffset": 2.0},
        },
        stage=stage,
    )

    assert cfg.inputs["exposureOffset"] == 2.0
    assert cfg.inputs["colorLatentRed"] == pytest.approx((0.25, -0.5))
