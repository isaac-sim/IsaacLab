# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DO NOT USE ANY FUNCTION IN THIS FILE."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# MDL OmniPBR default when ``diffuse_color_constant`` is not authored (typical MDL default).
_OMNIPBR_DEFAULT_DIFFUSE_COLOR_CONSTANT = (0.2, 0.2, 0.2)

# MDL OmniPBR default when ``diffuse_tint`` is not authored (typical MDL default).
_OMNIPBR_DEFAULT_DIFFUSE_TINT = (1.0, 1.0, 1.0)

# Neutral linear RGB when a shape has no material binding and no ``displayColor`` override.
UNBOUND_SHAPE_LINEAR_GRAY = (0.18, 0.18, 0.18)


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


def _resolve_visual_material_path(stage: Usd.Stage, prim_path_str: str) -> Sdf.Path | None:
    """Resolve the bound *visual* material path by walking up from the geometry prim."""
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path_str))
    while prim and prim.IsValid():
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            api = UsdShade.MaterialBindingAPI(prim)
            for purpose in (None, "render", "preview"):
                try:
                    db = api.GetDirectBinding() if purpose is None else api.GetDirectBinding(purpose)
                except Exception:
                    continue
                mat_path = db.GetMaterialPath() if db else Sdf.Path()
                if mat_path and not mat_path.isEmpty:
                    bound = stage.GetPrimAtPath(mat_path)
                    if bound is not None and bound.IsValid():
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


# HDC_TODO: Profiling and optimization for the function
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
    """
    if stage is None:
        from .stage import get_current_stage

        stage = get_current_stage()

    # Use duck typing to avoid introducing hard dependencies on newton.
    shape_label = getattr(model, "shape_label", None)
    shape_color = getattr(model, "shape_color", None)

    if shape_label is None or shape_color is None:
        logger.debug("missing shape_label or shape_color")
        return 0

    if len(shape_label) == 0 or len(shape_color) == 0:
        logger.debug("shape_label or shape_color is empty")
        return 0

    if len(shape_label) != len(shape_color):
        logger.debug(
            "mismatching number of elements in shape_label and shape_color: %d != %d",
            len(shape_label),
            len(shape_color),
        )
        return 0

    try:
        colors_t = wp.to_torch(shape_color)
    except Exception as exc:
        logger.warning("could not read shape_color: %s", exc)
        return 0

    # Staging: clone on the same device as ``shape_color``; row updates then wp.copy back.
    n = len(shape_label)
    if colors_t.numel() == n * 3:
        colors_work = colors_t.reshape(n, 3).clone()
    elif colors_t.ndim == 2 and colors_t.shape[0] == n and colors_t.shape[1] == 3:
        colors_work = colors_t.clone()
    else:
        logger.warning(
            "unexpected shape_color layout labels=%d tensor_shape=%s",
            n,
            tuple(colors_t.shape),
        )
        return 0

    updated = 0

    for label, color in zip(shape_label, colors_work, strict=True):
        shape_prim = stage.GetPrimAtPath(label)
        if not shape_prim:
            logger.debug("skipped %s: prim not found in the USD stage", label)
            continue

        material_path = _resolve_visual_material_path(stage, label)
        material_prim = stage.GetPrimAtPath(material_path) if material_path else None

        # If the prim has no material binding, use the display color if it is authored on the prim, otherwise 18% gray.
        if not material_prim:
            linear_color = None

            primvars_api = UsdGeom.PrimvarsAPI(shape_prim)
            if primvars_api.HasPrimvar("displayColor"):
                primvar = primvars_api.GetPrimvar("displayColor")
                if primvar is not None:
                    linear_color = _coerce_color(primvar.Get())

            if not linear_color:
                linear_color = UNBOUND_SHAPE_LINEAR_GRAY

            srgb_color = _linear_to_srgb(linear_color)
            color[0] = srgb_color[0]
            color[1] = srgb_color[1]
            color[2] = srgb_color[2]

            updated += 1
            continue

        shader_prim = _get_surface_shader(material_prim)
        if not shader_prim or not _shader_is_omnipbr(shader_prim):
            continue

        # If the prim uses an OmniPBR shader, use the albedo color defined in the OmniPBR shader.
        linear_color = _omnipbr_linear_diffuse_from_material(shader_prim)
        srgb_color = _linear_to_srgb(linear_color)
        color[0] = srgb_color[0]
        color[1] = srgb_color[1]
        color[2] = srgb_color[2]

        updated += 1

    if updated == 0:
        return 0

    try:
        src_wp = wp.from_torch(colors_work.contiguous(), dtype=wp.vec3)
        wp.copy(shape_color, src_wp)
    except Exception as exc:
        logger.warning("wp.copy failed: %s", exc)
        return 0

    logger.debug("updated %d / %d shapes", updated, n)
    return updated
