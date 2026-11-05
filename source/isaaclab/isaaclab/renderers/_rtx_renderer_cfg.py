# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared configuration for Isaac RTX and OVRTX renderers.

This private module is a temporary shared location. It will be removed when Isaac RTX is
deprecated and the configuration moves into OVRTX. Import public aliases from the backend
configuration modules instead of depending on this module.
"""

from enum import IntEnum, StrEnum

from ..utils import configclass
from .renderer_cfg import RendererCfg


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


@configclass
class _RtxRendererCfg(RendererCfg):
    """Shared per-product RTX execution settings for Isaac RTX and OVRTX."""

    render_mode: _RtxRenderMode = _RtxRenderMode.REAL_TIME_PATH_TRACING
    """Per-product execution mode. Defaults to real-time path tracing.

    A ``simple_shading_*`` camera output is a shortcut that overrides this setting with
    ``RtxRenderMode.MINIMAL`` and selects its matching shading mode. Other outputs respect
    this setting. String values such as ``"Minimal"`` are also accepted in configuration overrides.
    """

    minimal_shading_mode: _RtxMinimalShadingMode = _RtxMinimalShadingMode.FULL_MDL
    """Minimal material evaluation mode. Defaults to ``RtxMinimalShadingMode.FULL_MDL``.

    A ``simple_shading_*`` camera output overrides this setting with its matching mode.
    """


_SIMPLE_SHADING_MODES = {
    "simple_shading_constant_diffuse": _RtxMinimalShadingMode.CONSTANT_DIFFUSE,
    "simple_shading_diffuse_mdl": _RtxMinimalShadingMode.DIFFUSE_MDL,
    "simple_shading_full_mdl": _RtxMinimalShadingMode.FULL_MDL,
}
"""Public simple-shading output names and their RTX Minimal shading modes."""


def _resolve_rtx_render_mode(
    render_mode: _RtxRenderMode,
    minimal_shading_mode: _RtxMinimalShadingMode,
    data_types: list[str],
) -> tuple[str, int]:
    """Resolve per-product RTX settings, including simple-shading output shortcuts.

    A ``simple_shading_*`` output overrides both configured settings with Minimal rendering
    and its matching shading mode. Other outputs leave the configured settings unchanged.

    Args:
        render_mode: Configured RTX execution mode.
        minimal_shading_mode: Configured Minimal material evaluation mode.
        data_types: Requested camera output names, including implicit ISP inputs.

    Returns:
        Effective render mode and native integer shading level to author on the render product.

    Raises:
        ValueError: If a setting is invalid, different shading outputs are requested, or the
            effective mode cannot produce a requested output.
    """
    if not isinstance(render_mode, _RtxRenderMode):
        raise ValueError(
            f"Unsupported render_mode {render_mode!r}; use RtxRenderMode.MINIMAL "
            "or RtxRenderMode.REAL_TIME_PATH_TRACING."
        )
    if not isinstance(minimal_shading_mode, _RtxMinimalShadingMode):
        supported = ", ".join(f"RtxMinimalShadingMode.{mode.name}" for mode in _RtxMinimalShadingMode)
        raise ValueError(f"minimal_shading_mode must be one of: {supported}.")
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
