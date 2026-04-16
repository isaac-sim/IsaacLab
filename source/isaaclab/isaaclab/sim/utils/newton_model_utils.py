# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DO NOT USE ANY FUNCTION IN THIS FILE."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

logger = logging.getLogger(__name__)

# Set to a non-empty value (e.g. ``1``) to log preprocess / USD loop / ``wp.copy`` timings for
# :func:`replace_default_shape_colors` at INFO level.
_PROFILE_REPLACE_DEFAULT_SHAPE_COLORS_ENV = "ISAACLAB_PROFILE_REPLACE_DEFAULT_SHAPE_COLORS"


@contextmanager
def _profile_span(profile: bool, timings: dict[str, float], key: str) -> Iterator[None]:
    """Record wall time [s] for the ``with`` body in ``timings[key]`` when ``profile`` is True."""
    if not profile:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = time.perf_counter() - t0


def _canonical_prim_lookup_key(prim: Usd.Prim) -> str | None:
    """Pick a single USD path for lookup, to maxmize cache hits."""
    if not prim.IsValid():
        return None

    if prim.IsInstanceProxy():
        proto = prim.GetPrimInPrototype()
        if proto.IsValid():
            return proto.GetPath().pathString

    return prim.GetPath().pathString


@wp.kernel
def _scatter_shape_color_rows_kernel(
    shape_colors: wp.array(dtype=wp.vec3),  # type: ignore
    row_indices: wp.array(dtype=wp.int32),  # type: ignore
    row_colors: wp.array(dtype=wp.vec3),  # type: ignore
):
    """Write per-row RGB updates into ``shape_colors``."""
    tid = wp.tid()
    row = row_indices[tid]
    shape_colors[row] = row_colors[tid]


# MDL OmniPBR default when ``diffuse_color_constant`` is not authored (typical MDL default).
_OMNIPBR_DEFAULT_DIFFUSE_COLOR_CONSTANT = (0.2, 0.2, 0.2)

# MDL OmniPBR default when ``diffuse_tint`` is not authored (typical MDL default).
_OMNIPBR_DEFAULT_DIFFUSE_TINT = (1.0, 1.0, 1.0)

# Neutral linear RGB when a shape has no material binding and no ``displayColor`` override.
_DEFAULT_FALLBACK_GRAY = (0.18, 0.18, 0.18)


def _linear_to_srgb_float(c: float) -> float:
    """Apply sRGB OETF: linear channel ``[0, 1]`` to sRGB-encoded ``[0, 1]``."""
    if c <= 0.0:
        return 0.0
    if c >= 1.0:
        return 1.0
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _linear_to_srgb(rgb: tuple[float, float, float] | Gf.Vec3f | Gf.Vec3d) -> tuple[float, float, float]:
    """Convert linear RGB ``[0, 1]`` to sRGB triple ``[0, 1]`` (per-channel OETF)."""
    return (
        _linear_to_srgb_float(float(rgb[0])),
        _linear_to_srgb_float(float(rgb[1])),
        _linear_to_srgb_float(float(rgb[2])),
    )


def _asset_path_to_str(ap: Any) -> str:
    if ap is None:
        return ""
    if hasattr(ap, "path"):
        return str(ap.path)
    return str(ap)


def _shader_is_omnipbr(shader_prim: Usd.Prim) -> bool:
    """Return True if the shader prim references the OmniPBR MDL module."""
    if shader_prim.IsValid():
        for attr_name in ("info:mdl:sourceAsset", "inputs:mdl:sourceAsset"):
            attr = shader_prim.GetAttribute(attr_name)
            if attr and attr.HasAuthoredValue():
                s = _asset_path_to_str(attr.Get())
                if "OmniPBR" in s:
                    return True
        for attr_name in ("info:id", "info:sourceAsset"):
            attr = shader_prim.GetAttribute(attr_name)
            if attr and attr.HasAuthoredValue():
                s = str(attr.Get())
                if "OmniPBR" in s:
                    return True
    return False


def _resolve_visual_material_path(stage: Usd.Stage, shape_prim: Usd.Prim) -> Sdf.Path | None:
    """Resolve the bound *visual* material path by walking up from the geometry prim."""
    prim = shape_prim

    while prim.IsValid():
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            api = UsdShade.MaterialBindingAPI(prim)
            for purpose in (None, "render", "preview"):
                try:
                    db = api.GetDirectBinding() if purpose is None else api.GetDirectBinding(purpose)
                except Exception:
                    continue

                mat_path = db.GetMaterialPath() if db else Sdf.Path()
                if mat_path and not mat_path.isEmpty:
                    mat_prim = stage.GetPrimAtPath(mat_path)
                    if mat_prim.IsValid():
                        return mat_path
        prim = prim.GetParent()

    return None


def _get_input_value(shader: UsdShade.Shader, name: str) -> tuple[float, float, float] | None:
    """Fetch the effective input value from a shader, following connections."""
    inp = shader.GetInput(name)
    if inp is not None:
        attrs = UsdShade.Utils.GetValueProducingAttributes(inp)
        if attrs and len(attrs) > 0:
            value = attrs[0].Get()
            if value is not None:
                return _coerce_color(value)

    return None


def _get_surface_shader(material_prim: Usd.Prim) -> Usd.Prim | None:
    """Get the surface shader from a material."""
    material = UsdShade.Material(material_prim)
    surface_output = material.GetSurfaceOutput()
    if not surface_output:
        surface_output = material.GetOutput("surface")
    if not surface_output:
        surface_output = material.GetOutput("mdl:surface")

    shader_prim = None

    if surface_output:
        connected_source = surface_output.GetConnectedSource()
        if connected_source:
            shader_prim = connected_source[0].GetPrim()

    if shader_prim is None:
        for child in material_prim.GetChildren():
            if child.IsA(UsdShade.Shader):
                shader_prim = child
                break

    return shader_prim


def _omnipbr_linear_diffuse_from_material(shader_prim: Usd.Prim) -> tuple[float, float, float]:
    """Return linear RGB from OmniPBR diffuse × tint."""
    surface_shader = UsdShade.Shader(shader_prim)

    constant = _get_input_value(surface_shader, "diffuse_color_constant")
    if constant is None:
        constant = _OMNIPBR_DEFAULT_DIFFUSE_COLOR_CONSTANT

    tint = _get_input_value(surface_shader, "diffuse_tint")
    if tint is None:
        tint = _OMNIPBR_DEFAULT_DIFFUSE_TINT

    return (constant[0] * tint[0], constant[1] * tint[1], constant[2] * tint[2])


def _coerce_color(value: Any) -> tuple[float, float, float] | None:
    """Coerce a value to an RGB color tuple, or None if not possible."""
    if value is None:
        return None
    color_np = np.array(value, dtype=np.float32).reshape(-1)
    if color_np.size >= 3:
        return (float(color_np[0]), float(color_np[1]), float(color_np[2]))
    return None


def _get_primvar_display_color(shape_prim: Usd.Prim) -> tuple[float, float, float] | None:
    """Get authored ``primvars:displayColor`` from a shape prim as linear RGB."""
    primvars_api = UsdGeom.PrimvarsAPI(shape_prim)
    if not primvars_api.HasPrimvar("displayColor"):
        return None

    primvar = primvars_api.GetPrimvar("displayColor")
    if primvar is None:
        return None

    return _coerce_color(primvar.Get())


def _resolve_shape_color(
    stage: Usd.Stage,
    prim_path: str,
    material_color_cache: dict[str, tuple[float, float, float] | None],
) -> tuple[float, float, float] | None:
    """Resolve replacement sRGB for one prim path.

    Returns:
        Color to write into ``shape_color``, or ``None`` to leave the row unchanged.
    """
    shape_prim = stage.GetPrimAtPath(prim_path)
    if not shape_prim.IsValid():
        return None

    # Newton's random color palette is designed for guide shapes so we keep them unchanged.
    imageable = UsdGeom.Imageable(shape_prim)
    if bool(imageable) and imageable.ComputePurpose() == UsdGeom.Tokens.guide:
        return None

    material_path = _resolve_visual_material_path(stage, shape_prim)
    material_prim = stage.GetPrimAtPath(material_path) if material_path else None

    if material_prim is None or not material_prim.IsValid():
        display_color = _get_primvar_display_color(shape_prim)
        return _linear_to_srgb(display_color or _DEFAULT_FALLBACK_GRAY)

    material_key = _canonical_prim_lookup_key(material_prim)
    if material_key in material_color_cache:
        return material_color_cache[material_key]

    shader_prim = _get_surface_shader(material_prim)
    if _shader_is_omnipbr(shader_prim):
        linear_color = _omnipbr_linear_diffuse_from_material(shader_prim)
        color = _linear_to_srgb(linear_color)
    else:
        color = None

    material_color_cache[material_key] = color
    return color


def replace_default_shape_colors(model: Any, stage: Usd.Stage | None = None) -> int:
    """Replace default shape colors in the Newton model.

    The Newton model builder assigns default colors from a predefined color palette to shapes that have no material
    binding or use a supported material (e.g. OmniPBR), leading to noticeable color difference between the Newton model
    and the USD stage. This function replaces the default colors for shapes:

    - Shapes with no material binding will use primvars:displayColor if it is authored on the prim, otherwise 18% gray.
    - Shapes with the OmniPBR material will use the albedo color defined in the OmniPBR material.

    Colors of all the other shapes will remain unchanged.

    Args:
        model: Finalized Newton model with ``shape_label`` and ``shape_color``.
        stage: USD stage. If ``None``, uses :func:`~isaaclab.sim.utils.stage.get_current_stage`.

    Returns:
        Number of shapes whose colors were updated.

    Note:
        Set environment variable ``ISAACLAB_PROFILE_REPLACE_DEFAULT_SHAPE_COLORS`` (e.g. to ``1``) to log preprocess,
        USD traversal loop, and ``wp.copy`` timings at INFO.
    """
    profile = True  # _profile_replace_default_shape_colors_enabled()
    timings: dict[str, float] = {}

    with _profile_span(profile, timings, "preprocess"):
        if stage is None:
            from .stage import get_current_stage

            stage = get_current_stage()

        # Use duck typing to avoid introducing hard dependencies on newton.
        shape_labels = getattr(model, "shape_label", None)
        shape_colors = getattr(model, "shape_color", None)

        if shape_labels is None or shape_colors is None:
            logger.debug("missing shape_label or shape_color")
            return 0

        if len(shape_labels) == 0 or len(shape_colors) == 0:
            logger.debug("shape_label or shape_color is empty")
            return 0

        if len(shape_labels) != len(shape_colors):
            logger.debug(
                "mismatching number of elements in shape_label and shape_color: %d != %d",
                len(shape_labels),
                len(shape_colors),
            )
            return 0

        n = len(shape_labels)

    resolved_color_cache: dict[str, tuple[float, float, float] | None] = {}
    material_color_cache: dict[str, tuple[float, float, float] | None] = {}

    with _profile_span(profile, timings, "resolve_colors"):
        shape_keys: list[str] = []
        for label in shape_labels:
            prim = stage.GetPrimAtPath(label)
            key = _canonical_prim_lookup_key(prim)
            shape_keys.append(key or label)

        unique_keys = dict.fromkeys(shape_keys)
        for key in unique_keys:
            resolved_color_cache[key] = _resolve_shape_color(stage, key, material_color_cache)

    updated = 0

    with _profile_span(profile, timings, "scatter_rows"):
        shape_indices_np = np.empty(n, dtype=np.int32)
        shape_colors_np = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            rgb = resolved_color_cache[shape_keys[i]]
            if rgb is not None:
                shape_indices_np[updated] = i
                shape_colors_np[updated] = rgb
                updated += 1

        if updated != 0:
            shape_indices_wp = wp.from_numpy(shape_indices_np[:updated], dtype=wp.int32, device=shape_colors.device)
            shape_colors_wp = wp.from_numpy(shape_colors_np[:updated], dtype=wp.vec3, device=shape_colors.device)
            wp.launch(
                kernel=_scatter_shape_color_rows_kernel,
                dim=updated,
                inputs=[shape_colors, shape_indices_wp, shape_colors_wp],
                device=shape_colors.device,
            )

    if profile:
        logger.debug(
            "replace_default_shape_colors updated=%d/%d preprocess=%.2fms resolve_colors=%.2fms scatter_rows=%.2fms "
            "copy=%.2fms unique_keys=%d unique_material_keys=%d",
            updated,
            n,
            timings.get("preprocess", 0.0) * 1000.0,
            timings.get("resolve_colors", 0.0) * 1000.0,
            timings.get("scatter_rows", 0.0) * 1000.0,
            timings.get("copy", 0.0) * 1000.0,
            len(unique_keys),
            len(material_color_cache),
        )

    logger.debug("updated %d / %d shapes", updated, n)
    return updated
