# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`isaaclab.sim.utils.newton_model_utils` (no Kit required)."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from isaaclab.sim.utils.newton_model_utils import (
    UNBOUND_SHAPE_LINEAR_GRAY,
    _linear_to_srgb,
    _linear_to_srgb_float,
    _omnipbr_linear_diffuse_from_material,
    replace_default_shape_colors,
)


def test_linear_to_srgb_float_endpoints():
    assert _linear_to_srgb_float(0.0) == 0.0
    assert _linear_to_srgb_float(1.0) == 1.0


def test_linear_to_srgb_float_mid():
    # ~0.214 linear -> 0.5 sRGB
    assert _linear_to_srgb_float(0.21404114048) == pytest.approx(0.5, rel=1e-5)


def test_linear_to_srgb_triple():
    s = _linear_to_srgb((1.0, 0.21404114048, 0.0))
    assert s[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(0.5, rel=1e-5)
    assert s[2] == pytest.approx(0.0)


def test_omnipbr_linear_diffuse_from_material_applies_srgb_oetf():
    """OmniPBR path encodes diffuse × tint with the sRGB OETF (display-referred)."""
    stage = Usd.Stage.CreateInMemory()
    mat = UsdShade.Material.Define(stage, "/World/Mat")
    shader = UsdShade.Shader.Define(stage, "/World/Mat/OmniPBRShader")
    assert mat is not None and shader is not None
    sp = shader.GetPrim()
    sp.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.2, 0.2))
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))

    out = _omnipbr_linear_diffuse_from_material(sp)
    assert out == pytest.approx(_linear_to_srgb((0.2, 0.2, 0.2)), rel=1e-5)


def test_replace_default_shape_colors_unbound_display_and_gray():
    """No material: copy ``displayColor`` when authored; otherwise use linear unbound gray (no OETF)."""
    stage = Usd.Stage.CreateInMemory()
    mesh_a = UsdGeom.Mesh.Define(stage, "/World/A")
    assert mesh_a is not None
    pv = UsdGeom.PrimvarsAPI(mesh_a).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
    )
    assert pv is not None
    pv.Set([Gf.Vec3f(0.2, 0.4, 0.6)])
    UsdGeom.Mesh.Define(stage, "/World/B")

    n = 2
    shape_color = wp.zeros(n, dtype=wp.vec3, device="cpu")
    model = SimpleNamespace(shape_label=["/World/A", "/World/B"], shape_color=shape_color)

    count = replace_default_shape_colors(model, stage)
    assert count == 2

    after = wp.to_torch(shape_color)
    assert after[0, 0].item() == pytest.approx(0.2)
    assert after[0, 1].item() == pytest.approx(0.4)
    assert after[0, 2].item() == pytest.approx(0.6)
    assert after[1, 0].item() == pytest.approx(UNBOUND_SHAPE_LINEAR_GRAY[0])
    assert after[1, 1].item() == pytest.approx(UNBOUND_SHAPE_LINEAR_GRAY[1])
    assert after[1, 2].item() == pytest.approx(UNBOUND_SHAPE_LINEAR_GRAY[2])


def test_replace_default_shape_colors_skips_non_omnipbr_material():
    """Bound material that is not OmniPBR leaves the row unchanged."""
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    assert mesh is not None
    mesh_prim = mesh.GetPrim()
    mat = UsdShade.Material.Define(stage, "/World/Mat")
    shader = UsdShade.Shader.Define(stage, "/World/Mat/Shader")
    assert mat is not None and shader is not None
    shader.GetPrim().CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("SomeOther.mdl")
    )
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
    UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(UsdShade.Material(mat.GetPrim()))

    shape_color = wp.zeros(1, dtype=wp.vec3, device="cpu")
    before = wp.to_torch(shape_color).clone()
    model = SimpleNamespace(shape_label=["/World/Mesh"], shape_color=shape_color)

    assert replace_default_shape_colors(model, stage) == 0
    assert torch.allclose(wp.to_torch(shape_color), before)


def test_replace_default_shape_colors_omnipbr_binding():
    """Bound OmniPBR: diffuse × tint then sRGB OETF."""
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    assert mesh is not None
    mesh_prim = mesh.GetPrim()
    mat = UsdShade.Material.Define(stage, "/World/Mat")
    shader = UsdShade.Shader.Define(stage, "/World/Mat/OmniPBRShader")
    assert mat is not None and shader is not None
    sp = shader.GetPrim()
    sp.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
    UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(UsdShade.Material(mat.GetPrim()))

    shape_color = wp.zeros(1, dtype=wp.vec3, device="cpu")
    model = SimpleNamespace(shape_label=["/World/Mesh"], shape_color=shape_color)

    assert replace_default_shape_colors(model, stage) == 1
    exp = _linear_to_srgb((1.0, 0.0, 0.0))
    after = wp.to_torch(shape_color)[0]
    assert after[0].item() == pytest.approx(exp[0])
    assert after[1].item() == pytest.approx(exp[1])
    assert after[2].item() == pytest.approx(exp[2])
