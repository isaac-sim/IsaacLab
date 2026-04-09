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

# Log at most once if carb + heuristics cannot resolve CLI mode.
_cli_rendering_mode_resolution_warned = False

_KNOWN_RENDERING_MODE_PRESETS = frozenset({"performance", "balanced", "quality"})

# Try leaf paths first: AppLauncher uses set_string; generic get() may return a dict subtree.
_CLI_MODE_STRING_PATHS = (
    "/isaaclab/rendering/rendering_mode",
    "/isaaclab/rendering/rendering_mode/value",
    "/isaaclab/rendering/rendering_mode/default",
)

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


def _coerce_carb_rendering_mode_value(raw: Any) -> str | None:
    """Best-effort profile name from carb ``get()`` (often a ``dict`` subtree).

    Path is typically ``/isaaclab/rendering/rendering_mode``.
    """

    def collect_str_leaves(obj: Any, out: list[str], depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(obj, str):
            s = obj.strip()
            if s:
                out.append(s)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                collect_str_leaves(v, out, depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                collect_str_leaves(v, out, depth + 1)

    if raw is None:
        return None
    if isinstance(raw, str):
        out = raw.strip()
        return out if out else None
    if isinstance(raw, dict):
        for key in ("value", "default", "profile", "name", "rendering_mode"):
            v = raw.get(key)
            if isinstance(v, str):
                s = v.strip()
                if s:
                    return s
            if isinstance(v, dict):
                nested = _coerce_carb_rendering_mode_value(v)
                if nested:
                    return nested
    strings: list[str] = []
    collect_str_leaves(raw, strings)
    if not strings:
        return None
    for s in strings:
        if s in _KNOWN_RENDERING_MODE_PRESETS:
            return s
    return strings[0]


def _read_cli_rendering_mode_profile_name(get_setting: Any) -> str | None:
    """Read CLI rendering mode profile name (``performance`` / ``balanced`` / ``quality``).

    ``AppLauncher`` stores the profile with ``set_string``; ``carb.settings.get()`` on the same path
    may return a subtree ``dict``. Prefer ``get_string`` on the leaf path first, then
    :func:`_coerce_carb_rendering_mode_value`.
    """
    global _cli_rendering_mode_resolution_warned

    with contextlib.suppress(Exception):
        import carb

        gs = carb.settings.get_settings()
        if gs is not None and hasattr(gs, "get_string"):
            for path in _CLI_MODE_STRING_PATHS:
                with contextlib.suppress(Exception):
                    s = gs.get_string(path)
                    if s is not None:
                        out = str(s).strip()
                        if out:
                            return out

    raw = get_setting("/isaaclab/rendering/rendering_mode")
    coerced = _coerce_carb_rendering_mode_value(raw)
    if coerced:
        return coerced

    if raw is not None and not isinstance(raw, str):
        if not _cli_rendering_mode_resolution_warned:
            _cli_rendering_mode_resolution_warned = True
            _logger.warning(
                "Could not read /isaaclab/rendering/rendering_mode as a profile name (got %s). "
                "CLI rendering mode override may be ignored.",
                type(raw).__name__,
            )
    return None


def _normalize_rendering_mode_profile_name(name: Any) -> str | None:
    """Return a non-empty string profile name, or None if invalid."""
    if name is None:
        return None
    if isinstance(name, str):
        out = name.strip()
        return out if out else None
    _logger.warning(
        "rendering_mode must be a non-empty str, got %s; ignoring.",
        type(name).__name__,
    )
    return None


def _resolve_effective_rendering_mode_name(get_setting: Any, cfg: Any) -> str | None:
    """CLI explicit flag wins; otherwise use ``cfg.rendering_mode``."""
    if bool(get_setting("/isaaclab/rendering/rendering_mode/explicit")):
        return _read_cli_rendering_mode_profile_name(get_setting)
    return _normalize_rendering_mode_profile_name(getattr(cfg, "rendering_mode", None))


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


def resolve_rendering_mode_name_for_renderer_cfg(get_setting: Any, renderer_cfg: Any) -> str | None:
    """Resolve effective rendering mode profile name for a camera/renderer cfg."""
    return _resolve_effective_rendering_mode_name(get_setting, renderer_cfg)


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
    mode_name = resolve_rendering_mode_name_for_renderer_cfg(get_setting, renderer_cfg)
    mode_cfg = resolve_rendering_mode_cfg(mode_name, mode_cfgs, logger)
    if mode_cfg is None:
        return
    apply_kit_rendering_mode_cfg(set_setting, mode_cfg)


def resolve_rendering_mode_name_for_visualizer_cfg(get_setting: Any, visualizer_cfg: Any) -> str | None:
    """Resolve effective rendering mode profile name for a visualizer cfg."""
    return _resolve_effective_rendering_mode_name(get_setting, visualizer_cfg)


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
) -> None:
    """Resolve and apply rendering mode profile to a Kit visualizer config (RTX / carb settings)."""
    if getattr(visualizer_cfg, "visualizer_type", None) != "kit":
        return
    mode_name = resolve_rendering_mode_name_for_visualizer_cfg(get_setting, visualizer_cfg)
    mode_cfg = resolve_rendering_mode_cfg(mode_name, mode_cfgs, logger)
    if mode_cfg is None:
        return
    apply_kit_rendering_mode_cfg(set_setting, mode_cfg)


def apply_runtime_mode_profile_to_visualizer(
    get_setting: Any,
    set_setting: Any,
    viz: Any,
    visualizer_mode_keys: dict[int, str | None],
    mode_cfgs: dict[str, RenderingModeCfg],
    logger: Any,
) -> None:
    """Resolve and apply runtime rendering mode profile to an active Kit visualizer."""
    if getattr(viz.cfg, "visualizer_type", None) != "kit":
        return
    mode_name = resolve_rendering_mode_name_for_visualizer_cfg(get_setting, viz.cfg)
    viz_id = id(viz)
    if visualizer_mode_keys.get(viz_id) == mode_name:
        return

    mode_cfg = resolve_rendering_mode_cfg(mode_name, mode_cfgs, logger)
    if mode_cfg is None:
        visualizer_mode_keys[viz_id] = mode_name
        return

    apply_kit_rendering_mode_cfg(set_setting, mode_cfg)
    visualizer_mode_keys[viz_id] = mode_name
