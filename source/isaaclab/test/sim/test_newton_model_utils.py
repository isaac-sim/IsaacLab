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
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from isaaclab.sim.utils.newton_model_utils import (
    _OMNIPBR_DEFAULTS,
    _UNBOUND_DEFAULT_FALLBACK_GRAY,
    _get_omnipbr_albedo,
    _resolve_shape_color,
    replace_newton_builder_shape_colors,
)

pytestmark = pytest.mark.integration

_WARNING_MESSAGE = "Newton shape color replacement is enabled; this workaround will be deprecated in a future release."

_OMNIPBR_ALBEDO_INPUT_CASES = [
    pytest.param((0.2, 0.4, 0.6), (0.5, 0.25, 2.0), id="both_authored"),
    pytest.param((0.25, 0.5, 0.75), None, id="diffuse_only"),
    pytest.param(None, (0.4, 0.5, 2.0), id="tint_only"),
    pytest.param(None, None, id="defaults"),
]

Color = tuple[float, float, float]


def _replace_colors(builder: object, stage: Usd.Stage) -> int:
    """Call :func:`replace_newton_builder_shape_colors` with its :class:`FutureWarning` suppressed."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_WARNING_MESSAGE, category=FutureWarning)
        return replace_newton_builder_shape_colors(builder, stage)


def _builder(labels: list[str], color: Color = (0.0, 0.0, 0.0)) -> SimpleNamespace:
    """Minimal Newton builder stand-in with one ``shape_color`` row per label."""
    return SimpleNamespace(shape_label=list(labels), shape_color=[wp.vec3(*color) for _ in labels])


def _colors(builder: SimpleNamespace) -> list[Color]:
    return [tuple(color) for color in builder.shape_color]


def _expected_omnipbr_linear_albedo(diffuse_color_constant: Color | None, diffuse_tint: Color | None) -> Color:
    cd = diffuse_color_constant or _OMNIPBR_DEFAULTS["diffuse_color_constant"]
    td = diffuse_tint or _OMNIPBR_DEFAULTS["diffuse_tint"]
    return (cd[0] * td[0], cd[1] * td[1], cd[2] * td[2])


def _linear_to_srgb(rgb: Color) -> Color:
    """Reference sRGB OETF for linear ``rgb``; channels may lie outside ``[0, 1]``."""

    def encode(c: float) -> float:
        if c <= 0.0:
            return 0.0
        if c >= 1.0:
            return 1.0
        if c <= 0.0031308:
            return 12.92 * c
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055

    return tuple(encode(c) for c in rgb)


def _define_omnipbr_material(
    stage: Usd.Stage,
    material_path: str,
    diffuse_color_constant: Color | None = None,
    diffuse_tint: Color | None = None,
) -> UsdShade.Material:
    """Define a ``UsdShade.Material`` with a minimal OmniPBR shader; ``None`` inputs stay unauthored."""
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/OmniPBRShader")
    shader.GetPrim().CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniPBR.mdl"))
    if diffuse_color_constant is not None:
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_color_constant))
    if diffuse_tint is not None:
        shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_tint))
    return material


def _define_preview_surface_material(stage: Usd.Stage, material_path: str) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _define_bound_mesh(stage: Usd.Stage, mesh_path: str, material: UsdShade.Material, **binding_kwargs) -> Usd.Prim:
    prim = UsdGeom.Mesh.Define(stage, mesh_path).GetPrim()
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(material, **binding_kwargs)
    return prim


def _set_display_color(mesh: UsdGeom.Mesh, color: Color) -> None:
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
    )
    primvar.Set([Gf.Vec3f(*color)])


@pytest.mark.parametrize(("diffuse_color_constant", "diffuse_tint"), _OMNIPBR_ALBEDO_INPUT_CASES)
def test_get_omnipbr_albedo(diffuse_color_constant, diffuse_tint):
    """``_get_omnipbr_albedo`` is diffuse × tint per channel; unauthored inputs use ``_OMNIPBR_DEFAULTS``."""
    stage = Usd.Stage.CreateInMemory()
    _define_omnipbr_material(stage, "/World/Mat", diffuse_color_constant, diffuse_tint)

    shader_prim = stage.GetPrimAtPath("/World/Mat/OmniPBRShader")
    expected = _expected_omnipbr_linear_albedo(diffuse_color_constant, diffuse_tint)
    assert _get_omnipbr_albedo(shader_prim) == pytest.approx(expected, rel=1e-5)


def test_replace_newton_builder_shape_colors_warning():
    """A :exc:`FutureWarning` is expected by default."""
    builder = SimpleNamespace(shape_label=None, shape_color=None)

    with pytest.warns(FutureWarning, match=_WARNING_MESSAGE):
        replace_newton_builder_shape_colors(builder, stage=Usd.Stage.CreateInMemory())


def _guide_mesh(stage: Usd.Stage) -> str:
    mesh = UsdGeom.Mesh.Define(stage, "/World/GuideMesh")
    UsdGeom.Imageable(mesh).GetPurposeAttr().Set(UsdGeom.Tokens.guide)
    return "/World/GuideMesh"


def _preview_surface_mesh(stage: Usd.Stage) -> str:
    _define_bound_mesh(stage, "/World/Mesh", _define_preview_surface_material(stage, "/World/Mat"))
    return "/World/Mesh"


@pytest.mark.parametrize(
    "make_label",
    [
        pytest.param(lambda stage: "/World/Missing", id="missing_prim"),
        pytest.param(_guide_mesh, id="guide_purpose"),
        pytest.param(_preview_surface_mesh, id="non_omnipbr_material"),
    ],
)
def test_unresolvable_labels_keep_newton_colors(make_label):
    """Missing prims, guide geometry, and non-OmniPBR materials resolve to ``None`` and leave the row unchanged."""
    stage = Usd.Stage.CreateInMemory()
    label = make_label(stage)
    initial = (0.1, 0.2, 0.3)
    builder = _builder([label], initial)

    assert _resolve_shape_color(stage, label, {}) is None
    assert _replace_colors(builder, stage) == 0
    assert _colors(builder) == [pytest.approx(initial)]


def test_unbound_meshes_use_display_color_or_fallback_gray():
    """No material: ``displayColor`` or the unbound gray is resolved as linear RGB and stored sRGB-encoded."""
    stage = Usd.Stage.CreateInMemory()
    display_color = (0.11, 0.55, 0.9)
    _set_display_color(UsdGeom.Mesh.Define(stage, "/World/A"), display_color)
    UsdGeom.Mesh.Define(stage, "/World/B")

    assert _resolve_shape_color(stage, "/World/A", {}) == pytest.approx(display_color, rel=1e-5)
    assert _resolve_shape_color(stage, "/World/B", {}) == pytest.approx(_UNBOUND_DEFAULT_FALLBACK_GRAY, rel=1e-5)

    builder = _builder(["/World/A", "/World/B"])
    assert _replace_colors(builder, stage) == 2
    assert _colors(builder) == [
        pytest.approx(_linear_to_srgb(display_color), rel=1e-5),
        pytest.approx(_linear_to_srgb(_UNBOUND_DEFAULT_FALLBACK_GRAY), rel=1e-5),
    ]


@pytest.mark.parametrize(("diffuse_color_constant", "diffuse_tint"), _OMNIPBR_ALBEDO_INPUT_CASES)
def test_omnipbr_bound_meshes_use_albedo(diffuse_color_constant, diffuse_tint):
    """Bound OmniPBR: diffuse × tint is resolved as linear RGB and stored sRGB-encoded."""
    stage = Usd.Stage.CreateInMemory()
    material = _define_omnipbr_material(stage, "/World/Mat", diffuse_color_constant, diffuse_tint)
    _define_bound_mesh(stage, "/World/Mesh", material)
    albedo = _expected_omnipbr_linear_albedo(diffuse_color_constant, diffuse_tint)

    assert _resolve_shape_color(stage, "/World/Mesh", {}) == pytest.approx(albedo, rel=1e-5)

    builder = _builder(["/World/Mesh"])
    assert _replace_colors(builder, stage) == 1
    assert _colors(builder) == [pytest.approx(_linear_to_srgb(albedo), rel=1e-5)]


def test_replace_newton_builder_shape_colors_respects_binding_strength():
    """A parent binding that is stronger than descendants overrides the mesh's own binding."""
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/World/Parent").GetPrim()
    green = _define_omnipbr_material(stage, "/World/GreenMat", diffuse_color_constant=(0.0, 1.0, 0.0))
    red = _define_omnipbr_material(stage, "/World/RedMat", diffuse_color_constant=(1.0, 0.0, 0.0))
    UsdShade.MaterialBindingAPI.Apply(parent)
    UsdShade.MaterialBindingAPI(parent).Bind(green, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    _define_bound_mesh(stage, "/World/Parent/Mesh", red)

    builder = _builder(["/World/Parent/Mesh"])
    assert _replace_colors(builder, stage) == 1
    assert _colors(builder) == [pytest.approx((0.0, 1.0, 0.0))]


def test_replace_newton_builder_shape_colors_instanced():
    """Instance-proxy shape labels resolve colors from their prototype via displayColor."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Prototype")
    _set_display_color(UsdGeom.Mesh.Define(stage, "/World/Prototype/Mesh"), (0.1, 0.2, 0.3))
    for env in ("env_0", "env_1"):
        instance = stage.DefinePrim(f"/World/envs/{env}", "Xform")
        instance.GetReferences().AddInternalReference("/World/Prototype")
        instance.SetInstanceable(True)

    proxy_paths = ["/World/envs/env_0/Mesh", "/World/envs/env_1/Mesh"]
    assert all(stage.GetPrimAtPath(path).IsInstanceProxy() for path in proxy_paths)

    builder = _builder(proxy_paths)
    assert _replace_colors(builder, stage) == 2
    assert _colors(builder) == [pytest.approx(_linear_to_srgb((0.1, 0.2, 0.3)), rel=1e-5)] * 2


def test_replace_newton_builder_shape_colors_updates_only_labels_with_prims():
    """Only labels backed by a prim are colorized; clone labels without prims keep their row for replication."""
    stage = Usd.Stage.CreateInMemory()
    color = (0.2, 0.4, 0.6)
    _set_display_color(UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Robot/Mesh"), color)

    builder = _builder(["/World/envs/env_0/Robot/Mesh", "/World/envs/env_1/Robot/Mesh"])
    assert _replace_colors(builder, stage) == 1
    assert _colors(builder) == [pytest.approx(_linear_to_srgb(color)), pytest.approx((0.0, 0.0, 0.0))]
