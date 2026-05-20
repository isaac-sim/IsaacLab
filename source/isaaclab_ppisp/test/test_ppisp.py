# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PPISP USD parsing helpers."""

import pytest
from isaaclab_ppisp import PpispCfg, normalize_ppisp_cfg, ppisp_cfg_from_usd_shader

from pxr import Gf, Sdf, Usd, UsdShade


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


def test_normalize_ppisp_cfg_imports_shader_prim_path_from_stage():
    stage = Usd.Stage.CreateInMemory()
    shader = UsdShade.Shader.Define(stage, "/Render/RenderProduct/PPISP")
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(1.5)
    shader.CreateInput("colorLatentRed", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.25, -0.5))

    cfg = normalize_ppisp_cfg(PpispCfg(shader_prim_path="/Render/RenderProduct/PPISP"), stage=stage)

    assert cfg.shader_prim_path == "/Render/RenderProduct/PPISP"
    assert cfg.inputs["exposureOffset"] == 1.5
    assert cfg.inputs["colorLatentRed"] == pytest.approx((0.25, -0.5))


def test_normalize_ppisp_cfg_applies_explicit_overrides_after_shader_import():
    stage = Usd.Stage.CreateInMemory()
    shader = UsdShade.Shader.Define(stage, "/Render/RenderProduct/PPISP")
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(1.5)
    shader.CreateInput("colorLatentRed", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.25, -0.5))

    cfg = normalize_ppisp_cfg(
        PpispCfg(
            shader_prim_path="/Render/RenderProduct/PPISP",
            inputs={"exposureOffset": 2.0},
        ),
        stage=stage,
    )

    assert cfg.inputs["exposureOffset"] == 2.0
    assert cfg.inputs["colorLatentRed"] == pytest.approx((0.25, -0.5))
