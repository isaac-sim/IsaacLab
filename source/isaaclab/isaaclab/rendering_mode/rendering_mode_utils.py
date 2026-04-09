# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility helpers for applying rendering mode profiles."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from .rendering_mode_cfg import RenderingModeCfg
from .rendering_mode_presets import get_kit_rendering_preset

_logger = logging.getLogger(__name__)

# Log at most once if carb cannot read the CLI profile leaf.
_cli_rendering_mode_resolution_warned = False

# Leaf path for the CLI rendering-mode profile name (``performance`` / ``balanced`` / ``quality``, or empty).
# AppLauncher writes here with ``set_string`` only—never the parent ``.../rendering_mode`` path—so
# ``get_string`` / ``get_setting`` return a string instead of a dict subtree.
CLI_RENDERING_MODE_PROFILE_PATH = "/isaaclab/rendering/rendering_mode/profile"

_KIT_FIELD_TO_CARB: dict[str, str] = {
    "kit_enable_translucency": "/rtx/translucency/enabled",
    "kit_enable_reflections": "/rtx/reflections/enabled",
    "kit_enable_global_illumination": "/rtx/indirectDiffuse/enabled",
    "kit_enable_dlssg": "/rtx-transient/dlssg/enabled",
    "kit_enable_dl_denoiser": "/rtx-transient/dldenoiser/enabled",
    "kit_dlss_mode": "/rtx/post/dlss/execMode",
    "kit_enable_direct_lighting": "/rtx/directLighting/enabled",
    "kit_samples_per_pixel": "/rtx/directLighting/sampledLighting/samplesPerPixel",
    "kit_enable_shadows": "/rtx/shadows/enabled",
    "kit_enable_ambient_occlusion": "/rtx/ambientOcclusion/enabled",
    "kit_dome_light_upper_lower_strategy": "/rtx/domeLight/upperLowerStrategy",
}


def _read_cli_rendering_mode_profile_name(get_setting: Any) -> str | None:
    """Read CLI rendering mode profile name (``performance`` / ``balanced`` / ``quality``).

    Reads :data:`CLI_RENDERING_MODE_PROFILE_PATH` (a string leaf written by :class:`~isaaclab.app.AppLauncher`).
    """
    global _cli_rendering_mode_resolution_warned

    with contextlib.suppress(Exception):
        import carb

        gs = carb.settings.get_settings()
        if gs is not None and hasattr(gs, "get_string"):
            with contextlib.suppress(Exception):
                s = gs.get_string(CLI_RENDERING_MODE_PROFILE_PATH)
                if s is not None:
                    out = str(s).strip()
                    if out:
                        return out

    raw = get_setting(CLI_RENDERING_MODE_PROFILE_PATH)
    if isinstance(raw, str):
        out = raw.strip()
        return out if out else None

    if raw is not None:
        if not _cli_rendering_mode_resolution_warned:
            _cli_rendering_mode_resolution_warned = True
            _logger.warning(
                "Could not read %s as a string profile name (got %s). CLI rendering mode override may be ignored.",
                CLI_RENDERING_MODE_PROFILE_PATH,
                type(raw).__name__,
            )
    return None


def resolve_effective_rendering_mode_name(get_setting: Any, cfg: Any) -> str | None:
    """CLI explicit flag wins; otherwise use ``cfg.rendering_mode``."""
    if bool(get_setting("/isaaclab/rendering/rendering_mode/explicit")):
        return _read_cli_rendering_mode_profile_name(get_setting)
    return getattr(cfg, "rendering_mode", None)


def apply_kit_rendering_preset(set_setting: Any, preset_name: str) -> None:
    """Apply a named kit preset via provided setting setter."""
    preset = get_kit_rendering_preset(preset_name)
    for key, value in preset.items():
        set_setting(key, value)


def apply_kit_rendering_mode_cfg(set_setting: Any, mode_cfg: RenderingModeCfg) -> None:
    """Apply kit-specific rendering mode fields."""
    if mode_cfg.rendering_mode_preset:
        apply_kit_rendering_preset(set_setting, mode_cfg.rendering_mode_preset)

    # Replicator's set_render_rtx_realtime() can reset other RTX carb flags. Run it before applying
    # explicit kit_* carb paths so user overrides remain authoritative.
    if mode_cfg.kit_antialiasing_mode is not None:
        with contextlib.suppress(Exception):
            import omni.replicator.core as rep

            rep.settings.set_render_rtx_realtime(antialiasing=mode_cfg.kit_antialiasing_mode)

    for field_name, carb_key in _KIT_FIELD_TO_CARB.items():
        value = getattr(mode_cfg, field_name, None)
        if value is not None:
            set_setting(carb_key, value)


def apply_mode_profile_to_renderer_cfg(
    get_setting: Any,
    set_setting: Any,
    renderer_cfg: Any,
    mode_cfgs: dict[str, RenderingModeCfg],
    logger: Any,
) -> None:
    """Resolve and apply a rendering mode profile to a Kit/RTX :class:`~isaaclab.renderers.RendererCfg` (in place)."""
    rtype = getattr(renderer_cfg, "renderer_type", None)
    if rtype not in ("default", "isaac_rtx", "rtx"):
        return
    mode_name = resolve_effective_rendering_mode_name(get_setting, renderer_cfg)
    mode_cfg = resolve_rendering_mode_cfg(mode_name, mode_cfgs, logger)
    if mode_cfg is None:
        return
    apply_kit_rendering_mode_cfg(set_setting, mode_cfg)


def resolve_rendering_mode_cfg(
    mode_name: str | None, mode_cfgs: dict[str, RenderingModeCfg], logger: Any
) -> RenderingModeCfg | None:
    """Fetch rendering mode cfg by name and log if missing."""
    if not mode_name:
        return None
    if not isinstance(mode_name, str):
        logger.warning(
            "[SimulationContext] Rendering mode name must be str, got %s; skipping profile lookup.",
            type(mode_name).__name__,
        )
        return None
    mode_cfg = mode_cfgs.get(mode_name)
    if mode_cfg is None:
        logger.warning(
            "[SimulationContext] Rendering mode '%s' not found in SimulationCfg.rendering_mode_cfgs.",
            mode_name,
        )
        return None
    return mode_cfg


def apply_mode_profile_to_visualizer_cfg(
    get_setting: Any,
    set_setting: Any,
    visualizer_cfg: Any,
    mode_cfgs: dict[str, RenderingModeCfg],
    logger: Any,
    *,
    cache: dict[int, str | None] | None = None,
    cache_key: int | None = None,
) -> None:
    """Resolve and apply rendering mode profile for a Kit visualizer (RTX / carb settings).

    Pass ``cache`` and ``cache_key`` (typically ``id(viz)``) when updating an active visualizer each
    frame so we skip redundant carb work when the effective profile name is unchanged—including
    when the name is missing from ``mode_cfgs`` (avoids repeated warnings).
    """
    if (cache is None) != (cache_key is None):
        raise ValueError("apply_mode_profile_to_visualizer_cfg: pass both `cache` and `cache_key`, or neither.")
    if getattr(visualizer_cfg, "visualizer_type", None) != "kit":
        return
    mode_name = resolve_effective_rendering_mode_name(get_setting, visualizer_cfg)

    if cache is not None and cache_key is not None:
        if cache.get(cache_key) == mode_name:
            return

    mode_cfg = resolve_rendering_mode_cfg(mode_name, mode_cfgs, logger)
    if mode_cfg is None:
        if cache is not None and cache_key is not None:
            cache[cache_key] = mode_name
        return

    apply_kit_rendering_mode_cfg(set_setting, mode_cfg)
    if cache is not None and cache_key is not None:
        cache[cache_key] = mode_name
