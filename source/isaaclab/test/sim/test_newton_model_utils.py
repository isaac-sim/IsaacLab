# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`isaaclab.sim.utils.newton_model_utils` (no Kit required)."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from isaaclab.sim.utils.newton_model_utils import (
    _UNBOUND_DEFAULT_FALLBACK_GRAY,
    _get_omnipbr_albedo,
    replace_newton_shape_colors,
)

_ISAACLAB_REPLACE_NEWTON_SHAPE_COLORS_ENV = "ISAACLAB_REPLACE_NEWTON_SHAPE_COLORS"


def _reference_linear_to_srgb_float(c: float) -> float:
    """Host reference for the sRGB OETF (must match ``_linear_channel_to_srgb_warp`` in the scatter kernel)."""
    if c <= 0.0:
        return 0.0
    if c >= 1.0:
        return 1.0
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _reference_linear_to_srgb(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        _reference_linear_to_srgb_float(rgb[0]),
        _reference_linear_to_srgb_float(rgb[1]),
        _reference_linear_to_srgb_float(rgb[2]),
    )


def test_linear_to_srgb_float_endpoints():
    assert _reference_linear_to_srgb_float(0.0) == 0.0
    assert _reference_linear_to_srgb_float(1.0) == 1.0


def test_linear_to_srgb_float_mid():
    # ~0.214 linear -> 0.5 sRGB
    assert _reference_linear_to_srgb_float(0.21404114048) == pytest.approx(0.5, rel=1e-5)


def test_linear_to_srgb_triple():
    s = _reference_linear_to_srgb((1.0, 0.21404114048, 0.0))
    assert s[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(0.5, rel=1e-5)
    assert s[2] == pytest.approx(0.0)


def test_get_omnipbr_albedo_returns_linear_diffuse_times_tint():
    """OmniPBR helper returns linear RGB (diffuse × tint); scatter kernel applies sRGB OETF."""
    stage = Usd.Stage.CreateInMemory()
    mat = UsdShade.Material.Define(stage, "/World/Mat")
    shader = UsdShade.Shader.Define(stage, "/World/Mat/OmniPBRShader")
    assert mat is not None and shader is not None
    sp = shader.GetPrim()
    sp.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.2, 0.2))
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))

    out = _get_omnipbr_albedo(sp)
    assert out == pytest.approx((0.2, 0.2, 0.2), rel=1e-5)


def test_replace_newton_shape_colors_unbound_display_and_gray():
    """No material: ``displayColor`` or unbound gray as linear RGB, then sRGB OETF into ``shape_color``."""
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

    count = replace_newton_shape_colors(model, stage)
    assert count == 2

    after = wp.to_torch(shape_color)
    exp_a = _reference_linear_to_srgb((0.2, 0.4, 0.6))
    exp_b = _reference_linear_to_srgb(_UNBOUND_DEFAULT_FALLBACK_GRAY)
    assert after[0, 0].item() == pytest.approx(exp_a[0])
    assert after[0, 1].item() == pytest.approx(exp_a[1])
    assert after[0, 2].item() == pytest.approx(exp_a[2])
    assert after[1, 0].item() == pytest.approx(exp_b[0])
    assert after[1, 1].item() == pytest.approx(exp_b[1])
    assert after[1, 2].item() == pytest.approx(exp_b[2])


def test_replace_newton_shape_colors_skips_non_omnipbr_material():
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

    assert replace_newton_shape_colors(model, stage) == 0
    assert torch.allclose(wp.to_torch(shape_color), before)


def test_replace_newton_shape_colors_omnipbr_binding():
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

    assert replace_newton_shape_colors(model, stage) == 1
    exp = _reference_linear_to_srgb((1.0, 0.0, 0.0))
    after = wp.to_torch(shape_color)[0]
    assert after[0].item() == pytest.approx(exp[0])
    assert after[1].item() == pytest.approx(exp[1])
    assert after[2].item() == pytest.approx(exp[2])


def test_replace_newton_shape_colors_respects_stronger_than_descendants_binding():
    """Parent stronger-than-descendants binding overrides direct child binding."""
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/World/Parent")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Parent/Mesh")
    assert parent is not None and mesh is not None

    child_mat = UsdShade.Material.Define(stage, "/World/ChildMat")
    child_shader = UsdShade.Shader.Define(stage, "/World/ChildMat/OmniPBRShader")
    assert child_mat is not None and child_shader is not None
    child_shader.GetPrim().CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("OmniPBR.mdl")
    )
    child_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
    child_shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))

    parent_mat = UsdShade.Material.Define(stage, "/World/ParentMat")
    parent_shader = UsdShade.Shader.Define(stage, "/World/ParentMat/OmniPBRShader")
    assert parent_mat is not None and parent_shader is not None
    parent_shader.GetPrim().CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("OmniPBR.mdl")
    )
    parent_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.0))
    parent_shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))

    mesh_prim = mesh.GetPrim()
    parent_prim = parent.GetPrim()
    UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    UsdShade.MaterialBindingAPI.Apply(parent_prim)
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(UsdShade.Material(child_mat.GetPrim()))
    UsdShade.MaterialBindingAPI(parent_prim).Bind(
        UsdShade.Material(parent_mat.GetPrim()),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )

    shape_color = wp.zeros(1, dtype=wp.vec3, device="cpu")
    model = SimpleNamespace(shape_label=["/World/Parent/Mesh"], shape_color=shape_color)

    assert replace_newton_shape_colors(model, stage) == 1
    exp = _reference_linear_to_srgb((0.0, 1.0, 0.0))
    after = wp.to_torch(shape_color)[0]
    assert after[0].item() == pytest.approx(exp[0])
    assert after[1].item() == pytest.approx(exp[1])
    assert after[2].item() == pytest.approx(exp[2])


def test_replace_newton_shape_colors_isaac_lab_env_labels_deduplicated():
    """Multiple Newton rows pointing at the same USD path share one color resolution (per-key cache)."""
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/obj")
    assert mesh is not None
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
    )
    assert pv is not None
    pv.Set([Gf.Vec3f(0.1, 0.2, 0.3)])

    n = 2
    shape_color = wp.zeros(n, dtype=wp.vec3, device="cpu")
    model = SimpleNamespace(
        shape_label=["/World/envs/env_0/obj", "/World/envs/env_0/obj"],
        shape_color=shape_color,
    )

    assert replace_newton_shape_colors(model, stage) == 2
    after = wp.to_torch(shape_color)
    exp = _reference_linear_to_srgb((0.1, 0.2, 0.3))
    for row in range(2):
        assert after[row, 0].item() == pytest.approx(exp[0])
        assert after[row, 1].item() == pytest.approx(exp[1])
        assert after[row, 2].item() == pytest.approx(exp[2])


def test_replace_newton_shape_colors_skips_guide_shapes():
    """Guide-purpose shapes keep Newton's default color."""
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/GuideMesh")
    assert mesh is not None
    UsdGeom.Imageable(mesh).GetPurposeAttr().Set(UsdGeom.Tokens.guide)

    shape_color = wp.zeros(1, dtype=wp.vec3, device="cpu")
    before = wp.to_torch(shape_color).clone()
    model = SimpleNamespace(shape_label=["/World/GuideMesh"], shape_color=shape_color)

    assert replace_newton_shape_colors(model, stage) == 0
    assert torch.allclose(wp.to_torch(shape_color), before)


def test_replace_newton_shape_colors_emits_future_warning_when_active(monkeypatch: pytest.MonkeyPatch):
    """Enabling the workaround issues a single :exc:`FutureWarning` before validation short-circuits."""
    monkeypatch.delenv(_ISAACLAB_REPLACE_NEWTON_SHAPE_COLORS_ENV, raising=False)

    model = SimpleNamespace(shape_label=None, shape_color=None)
    with pytest.warns(FutureWarning, match="Newton shape color replacement is enabled"):
        replace_newton_shape_colors(model, stage=None)


def test_replace_newton_shape_colors_can_be_disabled_by_env_var(monkeypatch: pytest.MonkeyPatch):
    """Disabled workaround leaves shape colors unchanged and reports no updates."""
    monkeypatch.setenv(_ISAACLAB_REPLACE_NEWTON_SHAPE_COLORS_ENV, "0")

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/A")
    assert mesh is not None
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
    )
    assert pv is not None
    pv.Set([Gf.Vec3f(0.2, 0.4, 0.6)])

    shape_color = wp.zeros(1, dtype=wp.vec3, device="cpu")
    before = wp.to_torch(shape_color).clone()
    model = SimpleNamespace(shape_label=["/World/A"], shape_color=shape_color)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert replace_newton_shape_colors(model, stage) == 0
    assert not any(issubclass(w.category, FutureWarning) for w in recorded)
    assert torch.allclose(wp.to_torch(shape_color), before)
