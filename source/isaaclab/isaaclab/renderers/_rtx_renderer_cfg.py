# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared configuration for Isaac RTX and OVRTX renderers.

This private module is a temporary shared location. It will be removed when Isaac RTX is
deprecated and the configuration moves into OVRTX. Import public aliases from the backend
configuration modules instead of depending on this module.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..sensors.camera.camera_cfg import CameraCfg


class _RtxRenderMode(StrEnum):
    """Per-render-product RTX execution modes shared by Isaac RTX and OVRTX."""

    REAL_TIME_PATH_TRACING = "RealTimePathTracing"
    """Render with real-time path tracing."""

    MINIMAL = "Minimal"
    """Render with simplified material evaluation and color rendering enabled."""


class _RtxMinimalShadingMode(IntEnum):
    """Material evaluation modes for the RTX Minimal renderer.

    Values match the native RTX settings. Use enum members in Python configuration;
    serialized configurations retain their numeric values.
    """

    CONSTANT_DIFFUSE = 1
    """Shade every surface with one constant diffuse color."""

    DIFFUSE_MDL = 2
    """Evaluate textured diffuse MDL materials."""

    FULL_MDL = 3
    """Evaluate diffuse, glossy, and emissive MDL materials."""


_RTX_RENDER_MODE_ATTR = "omni:rtx:rendermode"
_RTX_MINIMAL_MODE_ATTR = "omni:rtx:minimal:mode"


def _apply_render_mode(camera_cfg: CameraCfg, mode: _RtxRenderMode) -> None:
    """Set a camera's RTX execution mode without changing its shared renderer.

    Args:
        camera_cfg: Camera whose render-product settings are updated in place.
        mode: Execution mode. Simple-shading outputs override this selection.

    Raises:
        ValueError: If mode is not a RtxRenderMode member.
    """
    if not isinstance(mode, _RtxRenderMode):
        raise ValueError("mode must be a RtxRenderMode member.")
    camera_cfg.render_settings = {**camera_cfg.render_settings, _RTX_RENDER_MODE_ATTR: ("Token", mode.value)}


def _apply_minimal_shading_mode(camera_cfg: CameraCfg, mode: _RtxMinimalShadingMode) -> None:
    """Set a camera's Minimal shading level, leaving its execution mode unchanged.

    Args:
        camera_cfg: Camera whose render-product settings are updated in place.
        mode: Shading level. Simple-shading outputs override this selection.

    Raises:
        ValueError: If mode is not a RtxMinimalShadingMode member.
    """
    if not isinstance(mode, _RtxMinimalShadingMode):
        raise ValueError("mode must be a RtxMinimalShadingMode member.")
    camera_cfg.render_settings = {**camera_cfg.render_settings, _RTX_MINIMAL_MODE_ATTR: ("Int", int(mode))}


_SIMPLE_SHADING_MODES = {
    "simple_shading_constant_diffuse": _RtxMinimalShadingMode.CONSTANT_DIFFUSE,
    "simple_shading_diffuse_mdl": _RtxMinimalShadingMode.DIFFUSE_MDL,
    "simple_shading_full_mdl": _RtxMinimalShadingMode.FULL_MDL,
}
"""Public simple-shading output names and their RTX Minimal shading modes."""


def _resolve_rtx_render_mode(
    render_settings: dict[str, tuple[str, Any]],
    data_types: list[str],
) -> tuple[str, int]:
    """Resolve per-product RTX settings, including simple-shading output shortcuts.

    A ``simple_shading_*`` output overrides both configured settings with Minimal rendering
    and its matching shading mode. Other outputs leave the configured settings unchanged.

    Args:
        render_settings: Per-camera render-product attributes, including serialized mode values.
        data_types: Requested camera output names, including implicit ISP inputs.

    Returns:
        Effective render mode and native integer shading level to author on the render product.

    Raises:
        ValueError: If a setting is invalid, different shading outputs are requested, or the
            effective mode cannot produce a requested output.
    """
    mode_type, mode_value = render_settings.get(
        _RTX_RENDER_MODE_ATTR, ("Token", _RtxRenderMode.REAL_TIME_PATH_TRACING)
    )
    shading_type, shading_value = render_settings.get(
        _RTX_MINIMAL_MODE_ATTR, ("Int", _RtxMinimalShadingMode.FULL_MDL)
    )
    if mode_type != "Token":
        raise ValueError(f"{_RTX_RENDER_MODE_ATTR} requires USD type 'Token'.")
    if shading_type != "Int":
        raise ValueError(f"{_RTX_MINIMAL_MODE_ATTR} requires USD type 'Int'.")
    try:
        render_mode = _RtxRenderMode(mode_value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Unsupported render mode {mode_value!r}.") from exc
    if not isinstance(shading_value, int) or isinstance(shading_value, bool):
        raise ValueError("Minimal shading mode must be an integer RtxMinimalShadingMode value.")
    try:
        minimal_shading_mode = _RtxMinimalShadingMode(shading_value)
    except ValueError as exc:
        raise ValueError(f"Unsupported Minimal shading mode {shading_value!r}.") from exc
    requested = {name for name in data_types if name in _SIMPLE_SHADING_MODES}
    if len(requested) > 1:
        raise ValueError("A render product supports at most one simple shading level; use separate cameras.")
    if requested:
        render_mode = _RtxRenderMode.MINIMAL
        minimal_shading_mode = _SIMPLE_SHADING_MODES[next(iter(requested))]
    if render_mode == _RtxRenderMode.MINIMAL and "rgb_hdr" in data_types:
        raise ValueError(
            "RTX Minimal does not support 'rgb_hdr' or ISP, which requires HDR. "
            "Remove the HDR output and isp_cfg, or use real-time path tracing without simple_shading_* outputs."
        )
    return render_mode.value, int(minimal_shading_mode)
