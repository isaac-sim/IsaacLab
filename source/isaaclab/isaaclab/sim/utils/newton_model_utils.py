# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DO NOT USE ANY FUNCTION IN THIS FILE."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import warp as wp

from pxr import Usd, UsdGeom, UsdShade

logger = logging.getLogger(__name__)

# MDL OmniPBR default when ``diffuse_color_constant`` is not authored (typical MDL default).
_OMNIPBR_DEFAULT_DIFFUSE_COLOR_CONSTANT = (0.2, 0.2, 0.2)

# MDL OmniPBR default when ``diffuse_tint`` is not authored (typical MDL default).
_OMNIPBR_DEFAULT_DIFFUSE_TINT = (1.0, 1.0, 1.0)

# Neutral linear RGB when a shape has no material binding and no ``displayColor`` override.
_UNBOUND_DEFAULT_FALLBACK_GRAY = (0.18, 0.18, 0.18)


@wp.func
def _linear_channel_to_srgb_warp(c: float) -> float:
    """Per-channel sRGB OETF on device: linear ``[0, 1]`` to sRGB-encoded ``[0, 1]``."""
    if c <= 0.0:
        return 0.0
    if c <= 0.0031308:
        return 12.92 * c
    if c >= 1.0:
        return 1.0
    return 1.055 * wp.pow(c, 1.0 / 2.4) - 0.055


@wp.func
def _linear_rgb_to_srgb_warp(linear_rgb: wp.vec3) -> wp.vec3:
    """Apply sRGB OETF per channel: linear RGB ``[0, 1]`` to sRGB-encoded ``[0, 1]``."""
    return wp.vec3(
        _linear_channel_to_srgb_warp(linear_rgb[0]),
        _linear_channel_to_srgb_warp(linear_rgb[1]),
        _linear_channel_to_srgb_warp(linear_rgb[2]),
    )


@wp.kernel
def _scatter_shape_color_rows_kernel(
    shape_colors: wp.array(dtype=wp.vec3),  # type: ignore
    row_indices: wp.array(dtype=wp.int32),  # type: ignore
    row_colors: wp.array(dtype=wp.vec3),  # type: ignore
):
    """Write per-row sRGB colors into ``shape_colors``."""
    tid = wp.tid()
    index = row_indices[tid]
    color = row_colors[tid]
    shape_colors[index] = _linear_rgb_to_srgb_warp(color)


def _canonical_prim_lookup_key(prim: Usd.Prim) -> str:
    """Pick a single USD path for lookup, to maxmize cache hits."""
    assert prim.IsValid()

    if prim.IsInstanceProxy():
        proto = prim.GetPrimInPrototype()
        if proto.IsValid():
            return proto.GetPath().pathString

    return prim.GetPath().pathString


def _asset_path_to_str(ap: Any) -> str:
    if ap is None:
        return ""
    if hasattr(ap, "path"):
        return str(ap.path)
    return str(ap)


def _is_omnipbr(shader_prim: Usd.Prim) -> bool:
    """Return True if the shader prim references the OmniPBR MDL module."""
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


def _get_omnipbr_albedo(shader_prim: Usd.Prim) -> tuple[float, float, float]:
    """Return albedo (linear RGB) from OmniPBR."""
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
    """Resolve replacement linear RGB for one prim path (sRGB encoding is applied in the scatter kernel).

    Returns:
        Linear RGB to pass to :func:`_scatter_shape_color_rows_kernel`, or ``None`` to leave the row unchanged.
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

    shader_prim = _get_surface_shader(material_prim)
    color = _get_omnipbr_albedo(shader_prim) if shader_prim.IsValid() and _is_omnipbr(shader_prim) else None

    material_color_cache[material_key] = color

    return color


def replace_newton_shape_colors(model: Any, stage: Usd.Stage | None = None):
    """Align Newton visualization colors with the USD stage.

    Newton assigns a per-shape palette to ``shape_color``. This overwrites those rows so rendering matches authored USD
    data where supported:

    - **No bound material**: use authored ``primvars:displayColor`` (treated as linear RGB), or a neutral 18% linear
      gray if ``displayColor`` is not authored; linear values are encoded to sRGB in the scatter kernel.
    - **OmniPBR**: use ``diffuse_color_constant`` times ``diffuse_tint`` (linear RGB, with MDL defaults when inputs are
      not authored), encoded to sRGB in the scatter kernel.
    - **Other materials**: leave the existing Newton color for that shape.
    - **Guide purpose** prims (``UsdGeom.Tokens.guide``): leave unchanged so guide visualization stays on the palette.

    Args:
        model: Object with ``shape_label`` (``list`` of USD prim paths) and ``shape_color`` (``wp.array`` of
            ``wp.vec3``), typically a finalized Newton model.
        stage: USD stage to read from. If ``None``, uses :func:`~isaaclab.sim.utils.stage.get_current_stage`.

    Note:
        Set ``ISAACLAB_REPLACE_NEWTON_SHAPE_COLORS`` to ``0``, ``false``, ``off``, or ``no`` to skip this pass
        entirely (returns ``0``).

        Wall time for USD resolution and the GPU scatter is measured with :class:`~isaaclab.utils.timer.Timer`, which
        may print a timing summary when the timer is enabled.
    """
    env_val = os.getenv("ISAACLAB_REPLACE_NEWTON_SHAPE_COLORS")
    if env_val is not None and env_val.strip().lower() in ["false", "0", "off", "no"]:
        logger.debug("Newton shape color replacement is disabled")
        return

    # Use duck typing to avoid introducing hard dependencies on newton.
    shape_labels = getattr(model, "shape_label", None)
    shape_colors = getattr(model, "shape_color", None)

    if not isinstance(shape_labels, list):
        logger.debug("shape_label must be a list, got %s", type(shape_labels))
        return

    if not isinstance(shape_colors, wp.array):
        logger.debug("shape_color must be a Warp array, got %s", type(shape_colors))
        return

    num_shapes = len(shape_labels)
    if num_shapes == 0:
        logger.debug("Found empty list of shape labels")
        return

    if num_shapes != len(shape_colors):
        logger.debug("Mismatching length of shape_labels and shape_colors: %d != %d", num_shapes, len(shape_colors))
        return

    from isaaclab.utils.timer import Timer

    with Timer(f"[INFO]: Time taken for replace_newton_shape_colors for {num_shapes} shapes", enable=False):
        if stage is None:
            from .stage import get_current_stage

            stage = get_current_stage()

        shape_keys: list[str] = []

        for label in shape_labels:
            prim = stage.GetPrimAtPath(label)
            shape_keys.append(_canonical_prim_lookup_key(prim) if prim.IsValid() else label)

        # shape_keys must stay the same length as shape labels, to guarantee the correctness of
        # shape indices that will be used in the scatter kernel.
        assert num_shapes == len(shape_keys)

        resolved_color_cache: dict[str, tuple[float, float, float] | None] = {}
        material_color_cache: dict[str, tuple[float, float, float] | None] = {}

        unique_keys = dict.fromkeys(shape_keys)
        for key in unique_keys:
            color = _resolve_shape_color(stage, key, material_color_cache)
            resolved_color_cache[key] = color

        # Prepare the indices and colors for the scatter kernel:
        # - Indices point to the slots in the shape_colors array that should be updated
        # - Colors are the new values to write into the slots
        indices_np = np.empty(num_shapes, dtype=np.int32)
        colors_np = np.empty((num_shapes, 3), dtype=np.float32)
        num_color_updates = 0

        for i in range(num_shapes):
            if rgb := resolved_color_cache.get(shape_keys[i]):
                indices_np[num_color_updates] = i
                colors_np[num_color_updates] = rgb
                num_color_updates += 1

        # If there are any color updates, launch the scatter kernel to update the shape_colors array.
        if num_color_updates != 0:
            indices_wp = wp.from_numpy(indices_np[:num_color_updates], dtype=wp.int32, device=shape_colors.device)
            colors_wp = wp.from_numpy(colors_np[:num_color_updates], dtype=wp.vec3, device=shape_colors.device)

            wp.launch(
                kernel=_scatter_shape_color_rows_kernel,
                dim=num_color_updates,
                inputs=[shape_colors, indices_wp, colors_wp],
                device=shape_colors.device,
            )

        logger.debug("Replaced colors for %d / %d shapes", num_color_updates, num_shapes)
