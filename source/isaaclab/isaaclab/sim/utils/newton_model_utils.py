# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DO NOT USE ANY FUNCTION IN THIS FILE.

This module exists only while Isaac Lab and Isaac Sim content still relies on NVIDIA-specific MDL and OmniPBR
materials; after migration to neutral USD materials that Newton can consume directly, this module is expected
to be deprecated and removed.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import warp as wp

from pxr import Usd, UsdGeom, UsdShade

__all__ = ["replace_newton_builder_shape_colors"]

logger = logging.getLogger(__name__)

# MDL OmniPBR defaults when inputs are not authored (typical MDL defaults). Keys match shader input names.
_OMNIPBR_DEFAULTS: dict[str, tuple[float, float, float]] = {
    "diffuse_color_constant": (0.2, 0.2, 0.2),
    "diffuse_tint": (1.0, 1.0, 1.0),
}

# Neutral linear RGB when a shape has no material binding and no ``displayColor`` override.
_UNBOUND_DEFAULT_FALLBACK_GRAY = (0.18, 0.18, 0.18)


def _linear_channel_to_srgb(c: float) -> float:
    """Per-channel sRGB OETF on host: linear ``[0, 1]`` to sRGB-encoded ``[0, 1]``."""
    if c <= 0.0:
        return 0.0
    if c <= 0.0031308:
        return 12.92 * c
    if c >= 1.0:
        return 1.0
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _canonical_prim_lookup_key(prim: Usd.Prim) -> str:
    """Pick a single USD path for lookup, to maximize cache hits."""
    assert prim.IsValid()

    if prim.IsInstanceProxy():
        proto = prim.GetPrimInPrototype()
        if proto.IsValid():
            return proto.GetPath().pathString

    return prim.GetPath().pathString


def _asset_path_to_str(asset_path: Any) -> str:
    """Stringify an asset path."""
    if asset_path is None:
        return ""
    return str(asset_path.path) if hasattr(asset_path, "path") else str(asset_path)


def _is_omnipbr_shader(shader_prim: Usd.Prim) -> bool:
    """Return True if the shader prim references the OmniPBR MDL module (MDL-in-USD metadata)."""
    if shader_prim.IsValid():
        attr = shader_prim.GetAttribute("info:mdl:sourceAsset")
        if attr and attr.HasAuthoredValue() and _asset_path_to_str(attr.Get()).endswith("OmniPBR.mdl"):
            return True

        attr = shader_prim.GetAttribute("info:mdl:sourceAsset:subIdentifier")
        if attr and attr.HasAuthoredValue() and str(attr.Get()) == "OmniPBR":
            return True

    return False


def _get_bound_material_prim(shape_prim: Usd.Prim) -> Usd.Prim:
    """Resolve the effective bound *visual* material path for a geometry prim.

    This uses :meth:`UsdShade.MaterialBindingAPI.ComputeBoundMaterial` so inherited bindings and
    binding-strength semantics (e.g. ``strongerThanDescendants``) are handled correctly.
    """
    if shape_prim.IsValid():
        material, _ = UsdShade.MaterialBindingAPI(shape_prim).ComputeBoundMaterial()
        if material:
            material_prim = material.GetPrim()
            if material_prim.IsValid():
                return material_prim

    return Usd.Prim()


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


def _get_surface_shader(material_prim: Usd.Prim) -> Usd.Prim:
    """Get the surface shader from a material."""
    material = UsdShade.Material(material_prim)
    surface_output = material.GetSurfaceOutput()
    if not surface_output:
        surface_output = material.GetOutput("surface")
    if not surface_output:
        surface_output = material.GetOutput("mdl:surface")

    shader_prim = Usd.Prim()

    if surface_output:
        connected_source = surface_output.GetConnectedSource()
        if connected_source:
            shader_prim = connected_source[0].GetPrim()

    if not shader_prim.IsValid():
        for child in material_prim.GetChildren():
            if child.IsA(UsdShade.Shader):
                shader_prim = child
                break

    return shader_prim


def _get_omnipbr_input(shader: UsdShade.Shader, input_name: str) -> tuple[float, float, float]:
    """Return authored linear RGB for ``input_name`` if it exists, else the MDL OmniPBR default."""
    value = _get_input_value(shader, input_name)
    return value or _OMNIPBR_DEFAULTS[input_name]


def _get_omnipbr_albedo(shader_prim: Usd.Prim) -> tuple[float, float, float]:
    """Return diffuse albedo as linear RGB (``diffuse_color_constant`` × ``diffuse_tint``)."""
    surface_shader = UsdShade.Shader(shader_prim)
    c0, c1, c2 = _get_omnipbr_input(surface_shader, "diffuse_color_constant")
    t0, t1, t2 = _get_omnipbr_input(surface_shader, "diffuse_tint")
    return (c0 * t0, c1 * t1, c2 * t2)


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
    """Resolve replacement linear RGB for one prim path (sRGB encoding is applied in the scatter kernel).

    Returns:
        Linear RGB to pass, or ``None`` to leave the row unchanged.
    """
    shape_prim = stage.GetPrimAtPath(prim_path)
    if not shape_prim.IsValid():
        return None

    # Newton's random color palette is designed for guide shapes so we keep them unchanged.
    imageable = UsdGeom.Imageable(shape_prim)
    if bool(imageable) and imageable.ComputePurpose() == UsdGeom.Tokens.guide:
        return None

    material_prim = _get_bound_material_prim(shape_prim)
    if not material_prim.IsValid():
        display_color = _get_primvar_display_color(shape_prim)
        return display_color or _UNBOUND_DEFAULT_FALLBACK_GRAY

    material_key = _canonical_prim_lookup_key(material_prim)
    if material_key in material_color_cache:
        return material_color_cache[material_key]

    # We only overwrite color if the material is OmniPBR. Otherwise, we leave the existing color unchanged.
    shader_prim = _get_surface_shader(material_prim)
    material_color = _get_omnipbr_albedo(shader_prim) if _is_omnipbr_shader(shader_prim) else None

    material_color_cache[material_key] = material_color
    return material_color


def replace_newton_builder_shape_colors(builder: Any, stage: Usd.Stage) -> int:
    """Align a Newton ``ModelBuilder``'s shape colors with the USD stage before clone replication.

    Overwrites entries in ``builder.shape_color`` so that colors match the authored USD data:

    - **No bound material**: use authored ``primvars:displayColor`` (treated as linear RGB), or a
      neutral 18% linear gray if ``displayColor`` is not authored.
    - **OmniPBR**: use ``diffuse_color_constant`` × ``diffuse_tint`` (linear RGB, with MDL defaults
      when inputs are not authored).
    - **Other materials**: leave the existing Newton color for that shape unchanged.
    - **Guide purpose** prims (``UsdGeom.Tokens.guide``): leave unchanged so guide visualization
      stays on the Newton palette.

    Linear RGB values are encoded to sRGB before being written into ``builder.shape_color``.

    Args:
        builder: Object with ``shape_label`` (``list`` of USD prim paths) and ``shape_color``
            (``list`` of ``wp.vec3``), typically a Newton ``ModelBuilder`` before finalization.
        stage: USD stage to read material and primvar data from.

    Returns:
        Number of shapes that had their colors replaced.
    """
    warnings.warn(
        "Newton shape color replacement is enabled; this workaround will be deprecated in a future release.",
        FutureWarning,
        stacklevel=2,
    )

    # Use duck typing to avoid introducing hard dependencies on newton.
    shape_labels = getattr(builder, "shape_label", None)
    shape_colors = getattr(builder, "shape_color", None)

    if not isinstance(shape_labels, list):
        logger.debug("shape_label must be a list, got %s", type(shape_labels))
        return 0

    if not isinstance(shape_colors, list):
        logger.debug("shape_color must be a list, got %s", type(shape_colors))
        return 0

    if len(shape_labels) != len(shape_colors):
        raise ValueError(
            f"Mismatching length of shape_label and shape_color: {len(shape_labels)} != {len(shape_colors)}"
        )

    from isaaclab.utils.timer import Timer

    with Timer(
        f"[INFO]: Time taken for replace_newton_builder_shape_colors for {len(shape_labels)} shapes", enable=False
    ):
        num_color_updates = 0
        material_color_cache: dict[str, tuple[float, float, float] | None] = {}
        for i, label in enumerate(shape_labels):
            rgb = _resolve_shape_color(stage, label, material_color_cache)
            if rgb is not None:
                shape_colors[i] = wp.vec3(
                    _linear_channel_to_srgb(rgb[0]),
                    _linear_channel_to_srgb(rgb[1]),
                    _linear_channel_to_srgb(rgb[2]),
                )
                num_color_updates += 1

        logger.debug("Replaced builder colors for %d / %d shapes", num_color_updates, len(shape_labels))
        return num_color_updates
