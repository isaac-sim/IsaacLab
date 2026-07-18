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

_WARNING_MESSAGE = "Newton shape color replacement is enabled; this workaround will be deprecated in a future release."


def _replace_newton_builder_shape_colors_wrapper(builder: object, stage: Usd.Stage) -> int:
    """Call :func:`replace_newton_builder_shape_colors` with :class:`FutureWarning` suppressed in test reports."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_WARNING_MESSAGE, category=FutureWarning)
        return replace_newton_builder_shape_colors(builder, stage)


_OMNIPBR_ALBEDO_INPUT_CASES = [
    pytest.param((0.2, 0.4, 0.6), (0.5, 0.25, 2.0), id="both_authored"),
    pytest.param((0.25, 0.5, 0.75), None, id="diffuse_only"),
    pytest.param(None, (0.4, 0.5, 2.0), id="tint_only"),
    pytest.param(None, None, id="defaults"),
]


def _expected_omnipbr_linear_albedo(
    diffuse_color_constant: tuple[float, float, float] | None,
    diffuse_tint: tuple[float, float, float] | None,
) -> tuple[float, float, float]:
    cd = diffuse_color_constant or _OMNIPBR_DEFAULTS["diffuse_color_constant"]
    td = diffuse_tint or _OMNIPBR_DEFAULTS["diffuse_tint"]
    return (cd[0] * td[0], cd[1] * td[1], cd[2] * td[2])


def _make_omnipbr_test_shader(
    stage: Usd.Stage,
    material_prim_path: str,
) -> UsdShade.Shader:
    """Define a ``UsdShade.Material`` and minimal OmniPBR ``UsdShade.Shader`` (MDL asset only).

    Diffuse and tint inputs are unauthored until the caller sets them (MDL defaults apply in readers).

    Args:
        stage: Stage to author on.
        material_prim_path: Absolute prim path for ``UsdShade.Material.Define``.

    Returns:
        The defined shader API.
    """
    UsdShade.Material.Define(stage, material_prim_path)
    shader = UsdShade.Shader.Define(stage, f"{material_prim_path}/OmniPBRShader")
    assert shader.GetPrim().IsValid()
    shader_prim = shader.GetPrim()
    mdl_asset_attr = shader_prim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset)
    assert mdl_asset_attr.IsValid()
    mdl_asset_attr.Set(Sdf.AssetPath("OmniPBR.mdl"))
    return shader


def _define_mesh_and_bind_material(stage: Usd.Stage, mesh_path: str, material: UsdShade.Material) -> UsdGeom.Mesh:
    """Define a mesh at ``mesh_path`` and bind ``material`` via :class:`UsdShade.MaterialBindingAPI`."""
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    assert mesh.GetPrim().IsValid()
    mesh_prim = mesh.GetPrim()
    UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)
    return mesh


def _make_preview_surface_bound_mesh_stage() -> tuple[Usd.Stage, str]:
    """In-memory stage with ``/World/Mesh`` bound to ``UsdPreviewSurface``."""
    stage = Usd.Stage.CreateInMemory()

    mat = UsdShade.Material.Define(stage, "/World/Mat")
    shader = UsdShade.Shader.Define(stage, "/World/Mat/PreviewSurface")
    assert mat.GetPrim().IsValid() and shader.GetPrim().IsValid()
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _define_mesh_and_bind_material(stage, "/World/Mesh", mat)
    return stage, "/World/Mesh"


def _make_mesh_bound_to_omnipbr_test_material(
    diffuse_color_constant: tuple[float, float, float] | None,
    diffuse_tint: tuple[float, float, float] | None,
) -> tuple[Usd.Stage, UsdShade.Shader, str]:
    """Reuse :func:`_make_omnipbr_test_shader`, author inputs, add ``/World/Mesh`` bound to ``/World/Mat``."""
    stage = Usd.Stage.CreateInMemory()

    shader = _make_omnipbr_test_shader(stage, "/World/Mat")
    if diffuse_color_constant is not None:
        diffuse_inp = shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
        diffuse_inp.Set(Gf.Vec3f(*diffuse_color_constant))
    if diffuse_tint is not None:
        tint_inp = shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f)
        tint_inp.Set(Gf.Vec3f(*diffuse_tint))

    mat_prim = stage.GetPrimAtPath("/World/Mat")
    assert mat_prim.IsValid()
    mat = UsdShade.Material(mat_prim)
    _define_mesh_and_bind_material(stage, "/World/Mesh", mat)
    return stage, shader, "/World/Mesh"


def _reference_linear_to_srgb(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Host sRGB OETF for linear ``rgb``.

    Args:
        rgb: Linear RGB triple; channels may lie outside ``[0, 1]``.

    Returns:
        Encoded RGB in ``[0, 1]`` as three floats.
    """

    def linear_to_srgb(c: float) -> float:
        if c <= 0.0:
            return 0.0
        if c >= 1.0:
            return 1.0
        if c <= 0.0031308:
            return 12.92 * c
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055

    r, g, b = rgb
    return (linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b))


@pytest.mark.parametrize(("diffuse_color_constant", "diffuse_tint"), _OMNIPBR_ALBEDO_INPUT_CASES)
def test_get_omnipbr_albedo(
    diffuse_color_constant: tuple[float, float, float] | None,
    diffuse_tint: tuple[float, float, float] | None,
):
    """``_get_omnipbr_albedo`` is diffuse × tint per channel; ``None`` means that shader input is not authored.

    Unauthored inputs use ``_OMNIPBR_DEFAULTS`` in ``newton_model_utils``.
    """
    stage = Usd.Stage.CreateInMemory()

    shader = _make_omnipbr_test_shader(stage, "/World/Mat")
    if diffuse_color_constant is not None:
        diffuse_inp = shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
        diffuse_inp.Set(Gf.Vec3f(*diffuse_color_constant))
    if diffuse_tint is not None:
        tint_inp = shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f)
        tint_inp.Set(Gf.Vec3f(*diffuse_tint))

    expected_albedo = _expected_omnipbr_linear_albedo(diffuse_color_constant, diffuse_tint)

    shader_prim = shader.GetPrim()
    assert shader_prim.IsValid()
    assert _get_omnipbr_albedo(shader_prim) == pytest.approx(expected_albedo, rel=1e-5)


def test_resolve_shape_color_invalid_prim():
    """Invalid prim path yields ``None`` (no replacement)."""
    stage = Usd.Stage.CreateInMemory()
    assert _resolve_shape_color(stage, "/World/Missing", {}) is None


def test_resolve_shape_color_guide_purpose():
    """Guide-purpose geometry is left on Newton's palette (no resolved replacement)."""
    stage = Usd.Stage.CreateInMemory()

    mesh = UsdGeom.Mesh.Define(stage, "/World/GuideMesh")
    assert mesh.GetPrim().IsValid()
    purpose_attr = UsdGeom.Imageable(mesh).GetPurposeAttr()
    assert purpose_attr.IsValid()
    purpose_attr.Set(UsdGeom.Tokens.guide)

    assert _resolve_shape_color(stage, "/World/GuideMesh", {}) is None


def test_resolve_shape_color_no_material_binding():
    """Unbound mesh without ``displayColor``: neutral linear gray fallback."""
    stage = Usd.Stage.CreateInMemory()

    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    assert mesh.GetPrim().IsValid()

    # Default fallback gray should be returned when there is no material binding and no display color.
    material_color_cache: dict[str, tuple[float, float, float] | None] = {}
    out = _resolve_shape_color(stage, "/World/Mesh", material_color_cache)
    assert out == pytest.approx(_UNBOUND_DEFAULT_FALLBACK_GRAY, rel=1e-5)

    # Add display color primvar
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
    )

    # Set an arbitrary color.
    display_color = (0.11, 0.55, 0.9)
    pv.Set([Gf.Vec3f(*display_color)])

    # The display color should be returned instead of the fallback gray.
    out = _resolve_shape_color(stage, "/World/Mesh", material_color_cache)
    assert out == pytest.approx(display_color, rel=1e-5)


@pytest.mark.parametrize(("diffuse_color_constant", "diffuse_tint"), _OMNIPBR_ALBEDO_INPUT_CASES)
def test_resolve_shape_color_omnipbr_binding(
    diffuse_color_constant: tuple[float, float, float] | None,
    diffuse_tint: tuple[float, float, float] | None,
):
    """Bound OmniPBR mesh: :func:`_resolve_shape_color` matches diffuse × tint."""
    stage, _shader, mesh_path = _make_mesh_bound_to_omnipbr_test_material(diffuse_color_constant, diffuse_tint)
    expected_albedo = _expected_omnipbr_linear_albedo(diffuse_color_constant, diffuse_tint)

    out = _resolve_shape_color(stage, mesh_path, {})
    assert out == pytest.approx(expected_albedo, rel=1e-5)


def test_resolve_shape_color_neutral_material_binding():
    """Bound ``UsdPreviewSurface`` material: not OmniPBR, so resolution is ``None`` (Newton row unchanged)."""
    stage, mesh_path = _make_preview_surface_bound_mesh_stage()
    assert _resolve_shape_color(stage, mesh_path, {}) is None


def test_replace_newton_builder_shape_colors_warning():
    """A :exc:`FutureWarning` is expected by default."""
    builder = SimpleNamespace(shape_label=None, shape_color=None)

    with pytest.warns(FutureWarning, match=_WARNING_MESSAGE):
        replace_newton_builder_shape_colors(builder, stage=Usd.Stage.CreateInMemory())


def test_replace_newton_builder_shape_colors_invalid_prim():
    """Invalid prim path leaves ``shape_color`` entry unchanged."""
    stage = Usd.Stage.CreateInMemory()

    initial = (0.1, 0.2, 0.3)
    builder = SimpleNamespace(shape_label=["/World/Missing"], shape_color=[wp.vec3(*initial)])

    assert _replace_newton_builder_shape_colors_wrapper(builder, stage) == 0
    assert tuple(builder.shape_color[0]) == pytest.approx(initial)


def test_replace_newton_builder_shape_colors_guide_purpose():
    """Guide-purpose mesh leaves ``shape_color`` entry unchanged."""
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/GuideMesh")
    assert mesh.GetPrim().IsValid()
    purpose_attr = UsdGeom.Imageable(mesh).GetPurposeAttr()
    assert purpose_attr.IsValid()
    purpose_attr.Set(UsdGeom.Tokens.guide)

    initial = (0.1, 0.2, 0.3)
    builder = SimpleNamespace(shape_label=["/World/GuideMesh"], shape_color=[wp.vec3(*initial)])

    assert _replace_newton_builder_shape_colors_wrapper(builder, stage) == 0
    assert tuple(builder.shape_color[0]) == pytest.approx(initial)


def test_replace_newton_builder_shape_colors_no_material_binding():
    """No material: ``displayColor`` or unbound gray as linear RGB, then sRGB OETF into ``shape_color``."""
    stage = Usd.Stage.CreateInMemory()

    # Mesh A has a display color primvar.
    mesh_a = UsdGeom.Mesh.Define(stage, "/World/A")
    pv = UsdGeom.PrimvarsAPI(mesh_a).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
    )
    color_a = (0.1, 0.2, 0.3)
    pv.Set([Gf.Vec3f(*color_a)])

    # Mesh B has no material binding and no display color primvar.
    UsdGeom.Mesh.Define(stage, "/World/B")

    builder = SimpleNamespace(
        shape_label=["/World/A", "/World/B"],
        shape_color=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
    )

    assert _replace_newton_builder_shape_colors_wrapper(builder, stage) == 2
    assert tuple(builder.shape_color[0]) == pytest.approx(_reference_linear_to_srgb(color_a), rel=1e-5)
    assert tuple(builder.shape_color[1]) == pytest.approx(
        _reference_linear_to_srgb(_UNBOUND_DEFAULT_FALLBACK_GRAY), rel=1e-5
    )


@pytest.mark.parametrize(("diffuse_color_constant", "diffuse_tint"), _OMNIPBR_ALBEDO_INPUT_CASES)
def test_replace_newton_builder_shape_colors_omnipbr_binding(
    diffuse_color_constant: tuple[float, float, float] | None,
    diffuse_tint: tuple[float, float, float] | None,
):
    """Bound OmniPBR: diffuse × tint then sRGB OETF."""
    stage, _shader, mesh_path = _make_mesh_bound_to_omnipbr_test_material(diffuse_color_constant, diffuse_tint)

    builder = SimpleNamespace(shape_label=[mesh_path], shape_color=[wp.vec3(0.0, 0.0, 0.0)])

    assert _replace_newton_builder_shape_colors_wrapper(builder, stage) == 1
    exp = _reference_linear_to_srgb(_expected_omnipbr_linear_albedo(diffuse_color_constant, diffuse_tint))
    assert tuple(builder.shape_color[0]) == pytest.approx(exp, rel=1e-5)


def test_replace_newton_builder_shape_colors_neutral_material():
    """Bound ``UsdPreviewSurface`` material leaves ``shape_color`` entry unchanged."""
    stage, mesh_path = _make_preview_surface_bound_mesh_stage()

    initial = (0.1, 0.2, 0.3)
    builder = SimpleNamespace(shape_label=[mesh_path], shape_color=[wp.vec3(*initial)])

    assert _replace_newton_builder_shape_colors_wrapper(builder, stage) == 0
    assert tuple(builder.shape_color[0]) == pytest.approx(initial)


def test_replace_newton_builder_shape_colors_respects_binding_strength():
    """Parent stronger-than-descendants binding overrides direct child binding."""
    # Scene graph (``ComputeBoundMaterial`` on the mesh yields ParentMat / green, not ChildMat / red):
    #
    #   /World
    #   +-- Parent ........................ [bind: GreenMat, strongerThanDescendants]
    #   |     \-- Mesh .................... [bind: RedMat]
    #   +-- GreenMat ...................... OmniPBRShader .. diffuse (0, 1, 0)
    #   +-- RedMat ........................ OmniPBRShader .. diffuse (1, 0, 0)
    #
    stage = Usd.Stage.CreateInMemory()

    parent = UsdGeom.Xform.Define(stage, "/World/Parent")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Parent/Mesh")
    assert parent.GetPrim().IsValid() and mesh.GetPrim().IsValid()

    # Parent material is green
    green_shader = _make_omnipbr_test_shader(stage, "/World/GreenMat")
    green_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.0))

    # Child material is red
    red_shader = _make_omnipbr_test_shader(stage, "/World/RedMat")
    red_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))

    # Bind GreenMat to Parent
    parent_prim = parent.GetPrim()
    parent_mat = UsdShade.Material(stage.GetPrimAtPath("/World/GreenMat"))
    UsdShade.MaterialBindingAPI.Apply(parent_prim)
    UsdShade.MaterialBindingAPI(parent_prim).Bind(parent_mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    # Bind RedMat to Mesh
    mesh_prim = mesh.GetPrim()
    child_mat = UsdShade.Material(stage.GetPrimAtPath("/World/RedMat"))
    UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(child_mat)

    builder = SimpleNamespace(shape_label=["/World/Parent/Mesh"], shape_color=[wp.vec3(0.0, 0.0, 0.0)])
    assert _replace_newton_builder_shape_colors_wrapper(builder, stage) == 1

    # Expected color is green which is inherited from the parent xform prim
    assert tuple(builder.shape_color[0]) == pytest.approx((0.0, 1.0, 0.0))


def test_replace_newton_builder_shape_colors_instanced():
    """Instance-proxy shape labels resolve colors from their prototype via displayColor."""
    stage = Usd.Stage.CreateInMemory()

    prototype = UsdGeom.Xform.Define(stage, "/World/Prototype")
    assert prototype.GetPrim().IsValid()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Prototype/Mesh")
    assert mesh.GetPrim().IsValid()
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
    ).Set([Gf.Vec3f(0.1, 0.2, 0.3)])

    # Create an instance from the prototype
    env_0 = stage.DefinePrim("/World/envs/env_0", "Xform")
    env_0.GetReferences().AddInternalReference("/World/Prototype")
    env_0.SetInstanceable(True)

    # Create another instance from the prototype
    env_1 = stage.DefinePrim("/World/envs/env_1", "Xform")
    env_1.GetReferences().AddInternalReference("/World/Prototype")
    env_1.SetInstanceable(True)

    proxy_paths = [
        "/World/envs/env_0/Mesh",
        "/World/envs/env_1/Mesh",
    ]
    for proxy_path in proxy_paths:
        inst_proxy = stage.GetPrimAtPath(proxy_path)
        assert inst_proxy.IsValid(), f"Proxy path {proxy_path} is not valid"
        assert inst_proxy.IsInstanceProxy(), f"Proxy path {proxy_path} is not an instance proxy"

    builder = SimpleNamespace(
        shape_label=proxy_paths,
        shape_color=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
    )
    assert _replace_newton_builder_shape_colors_wrapper(builder, stage) == 2

    exp = _reference_linear_to_srgb((0.1, 0.2, 0.3))
    assert tuple(builder.shape_color[0]) == pytest.approx(exp, rel=1e-5)
    assert tuple(builder.shape_color[1]) == pytest.approx(exp, rel=1e-5)


def test_replace_newton_builder_shape_colors_updates_source_builder():
    """Source builders are colorized once before clone replication copies colors forward."""
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Robot/Mesh")
    color = (0.2, 0.4, 0.6)
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
    )
    primvar.Set([Gf.Vec3f(*color)])

    builder = SimpleNamespace(
        shape_label=["/World/envs/env_0/Robot/Mesh", "/World/envs/env_1/Robot/Mesh"],
        shape_color=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
    )

    assert _replace_newton_builder_shape_colors_wrapper(builder, stage) == 1
    assert tuple(builder.shape_color[0]) == pytest.approx(_reference_linear_to_srgb(color))
    assert tuple(builder.shape_color[1]) == pytest.approx((0.0, 0.0, 0.0))


def test_replace_newton_builder_shape_colors_skips_missing_prim_labels():
    """Labels with no matching USD prim leave the corresponding ``shape_color`` entry unchanged."""
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Robot/Mesh")
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
    )
    primvar.Set([Gf.Vec3f(0.2, 0.4, 0.6)])

    initial = (0.1, 0.2, 0.3)
    builder = SimpleNamespace(
        shape_label=["/World/envs/env_1/Robot/Mesh"],
        shape_color=[wp.vec3(*initial)],
    )

    assert _replace_newton_builder_shape_colors_wrapper(builder, stage) == 0
    assert tuple(builder.shape_color[0]) == pytest.approx(initial)
