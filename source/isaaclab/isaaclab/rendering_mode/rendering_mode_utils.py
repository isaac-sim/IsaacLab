# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility helpers for applying rendering mode profiles."""

from __future__ import annotations

import logging
from typing import Any

from .rendering_mode_cfg import RenderingModeCfg
from .rendering_mode_presets import get_kit_rendering_preset

_logger = logging.getLogger(__name__)

# Log at most once if carb + heuristics cannot resolve CLI mode.
_cli_rendering_mode_resolution_warned = False

_KNOWN_RENDERING_MODE_PRESETS = frozenset({"performance", "balanced", "quality"})


def _collect_str_leaves(obj: Any, out: list[str], depth: int = 0) -> None:
    """Collect non-empty strings from nested dict/list structures (carb subtrees)."""
    if depth > 12:
        return
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append(s)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_str_leaves(v, out, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _collect_str_leaves(v, out, depth + 1)


def _coerce_carb_rendering_mode_value(raw: Any) -> str | None:
    """Best-effort string profile name from carb get()/subtree dicts."""
    if raw is None:
        return None
    if isinstance(raw, str):
        out = raw.strip()
        return out if out else None
    strings: list[str] = []
    _collect_str_leaves(raw, strings)
    if not strings:
        return None
    for s in strings:
        if s in _KNOWN_RENDERING_MODE_PRESETS:
            return s
    return strings[0]


def _read_cli_rendering_mode_profile_name(get_setting: Any) -> str | None:
    """Read CLI rendering mode profile name (``performance`` / ``balanced`` / ``quality``).

    :class:`~isaaclab.app.settings_manager.SettingsManager` and :class:`~isaaclab.sim.simulation_context.SettingsHelper` resolve
    ``/isaaclab/rendering/rendering_mode`` via ``get_string`` when using carb so readers see the string
    set by ``set_string``. Generic ``carb.settings.get()`` may still return a subtree ``dict``; we coerce
    that below and try alternate leaf paths.
    """
    global _cli_rendering_mode_resolution_warned

    raw = get_setting("/isaaclab/rendering/rendering_mode")
    coerced = _coerce_carb_rendering_mode_value(raw)
    if coerced:
        return coerced

    # Typed carb reads (leaf path and common alternates some Kit builds use).
    try:
        import carb

        gs = carb.settings.get_settings()
        if gs is not None and hasattr(gs, "get_string"):
            for path in (
                "/isaaclab/rendering/rendering_mode",
                "/isaaclab/rendering/rendering_mode/value",
                "/isaaclab/rendering/rendering_mode/default",
            ):
                try:
                    s = gs.get_string(path)
                    if s is not None:
                        out = str(s).strip()
                        if out:
                            return out
                except Exception:
                    continue
    except Exception:
        pass

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
        try:
            import omni.replicator.core as rep

            rep.settings.set_render_rtx_realtime(antialiasing=mode_cfg.kit_antialiasing_mode)
        except Exception:
            pass

    field_to_carb = {
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
    for field_name, carb_key in field_to_carb.items():
        value = getattr(mode_cfg, field_name, None)
        if value is not None:
            set_setting(carb_key, value)


def resolve_rendering_mode_name_for_renderer_cfg(get_setting: Any, renderer_cfg: Any) -> str | None:
    """Resolve effective rendering mode profile name for a camera/renderer cfg."""
    cli_mode_explicit = bool(get_setting("/isaaclab/rendering/rendering_mode/explicit"))
    if cli_mode_explicit:
        return _read_cli_rendering_mode_profile_name(get_setting)
    return _normalize_rendering_mode_profile_name(getattr(renderer_cfg, "rendering_mode", None))


def apply_newton_warp_mode_cfg_to_renderer_cfg(renderer_cfg: Any, mode_cfg: RenderingModeCfg) -> None:
    """Apply Newton Warp tiled-camera fields from a rendering mode profile onto renderer cfg."""
    override_fields = {
        "newton_warp_enable_textures": "enable_textures",
        "newton_warp_enable_shadows": "enable_shadows",
        "newton_warp_enable_ambient_lighting": "enable_ambient_lighting",
        "newton_warp_enable_backface_culling": "enable_backface_culling",
        "newton_warp_max_distance": "max_distance",
        "newton_warp_create_default_light": "create_default_light",
    }
    for mode_field, ren_field in override_fields.items():
        value = getattr(mode_cfg, mode_field, None)
        if value is not None and hasattr(renderer_cfg, ren_field):
            setattr(renderer_cfg, ren_field, value)


def apply_mode_profile_to_renderer_cfg(
    get_setting: Any,
    set_setting: Any,
    renderer_cfg: Any,
    mode_cfgs: dict[str, RenderingModeCfg],
    logger: Any,
) -> None:
    """Resolve and apply a rendering mode profile to a :class:`~isaaclab.renderers.RendererCfg` (in place)."""
    mode_name = resolve_rendering_mode_name_for_renderer_cfg(get_setting, renderer_cfg)
    mode_cfg = resolve_rendering_mode_cfg(mode_name, mode_cfgs, logger)
    if mode_cfg is None:
        return

    rtype = getattr(renderer_cfg, "renderer_type", None)
    if rtype in ("default", "isaac_rtx", "rtx"):
        apply_kit_rendering_mode_cfg(set_setting, mode_cfg)
    elif rtype == "newton_warp":
        apply_newton_warp_mode_cfg_to_renderer_cfg(renderer_cfg, mode_cfg)


def apply_newton_mode_cfg_to_visualizer_cfg(visualizer_cfg: Any, mode_cfg: RenderingModeCfg) -> None:
    """Apply Newton rendering mode values to a visualizer cfg object."""
    override_fields = {
        "newton_enable_shadows": "enable_shadows",
        "newton_enable_sky": "enable_sky",
        "newton_enable_wireframe": "enable_wireframe",
        "newton_sky_upper_color": "sky_upper_color",
        "newton_sky_lower_color": "sky_lower_color",
        "newton_light_color": "light_color",
    }
    for mode_field, viz_field in override_fields.items():
        value = getattr(mode_cfg, mode_field, None)
        if value is not None and hasattr(visualizer_cfg, viz_field):
            setattr(visualizer_cfg, viz_field, value)


def apply_newton_mode_cfg_to_viewer(viewer: Any, mode_cfg: RenderingModeCfg) -> None:
    """Apply Newton rendering mode values to a live Newton viewer renderer, if available."""
    if viewer is None or not hasattr(viewer, "renderer"):
        return

    if mode_cfg.newton_enable_shadows is not None:
        viewer.renderer.draw_shadows = mode_cfg.newton_enable_shadows
    if mode_cfg.newton_enable_sky is not None:
        viewer.renderer.draw_sky = mode_cfg.newton_enable_sky
    if mode_cfg.newton_enable_wireframe is not None:
        viewer.renderer.draw_wireframe = mode_cfg.newton_enable_wireframe
    if mode_cfg.newton_sky_upper_color is not None:
        viewer.renderer.sky_upper = mode_cfg.newton_sky_upper_color
    if mode_cfg.newton_sky_lower_color is not None:
        viewer.renderer.sky_lower = mode_cfg.newton_sky_lower_color
    if mode_cfg.newton_light_color is not None:
        viewer.renderer._light_color = mode_cfg.newton_light_color


def resolve_rendering_mode_name_for_visualizer_cfg(get_setting: Any, visualizer_cfg: Any) -> str | None:
    """Resolve effective rendering mode profile name for a visualizer cfg."""
    cli_mode_explicit = bool(get_setting("/isaaclab/rendering/rendering_mode/explicit"))
    if cli_mode_explicit:
        return _read_cli_rendering_mode_profile_name(get_setting)
    return _normalize_rendering_mode_profile_name(getattr(visualizer_cfg, "rendering_mode", None))


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
    """Resolve and apply rendering mode profile to a visualizer config."""
    mode_name = resolve_rendering_mode_name_for_visualizer_cfg(get_setting, visualizer_cfg)
    mode_cfg = resolve_rendering_mode_cfg(mode_name, mode_cfgs, logger)
    if mode_cfg is None:
        return

    visualizer_type = getattr(visualizer_cfg, "visualizer_type", None)
    if visualizer_type == "kit":
        apply_kit_rendering_mode_cfg(set_setting, mode_cfg)
    elif visualizer_type == "newton":
        apply_newton_mode_cfg_to_visualizer_cfg(visualizer_cfg, mode_cfg)


def apply_runtime_mode_profile_to_visualizer(
    get_setting: Any,
    set_setting: Any,
    viz: Any,
    visualizer_mode_keys: dict[int, str | None],
    mode_cfgs: dict[str, RenderingModeCfg],
    logger: Any,
    force: bool = False,
) -> None:
    """Resolve and apply runtime rendering mode profile to an active visualizer."""
    mode_name = resolve_rendering_mode_name_for_visualizer_cfg(get_setting, viz.cfg)
    viz_id = id(viz)
    if not force and visualizer_mode_keys.get(viz_id) == mode_name:
        return

    mode_cfg = resolve_rendering_mode_cfg(mode_name, mode_cfgs, logger)
    if mode_cfg is None:
        visualizer_mode_keys[viz_id] = mode_name
        return

    viz_type = getattr(viz.cfg, "visualizer_type", None)
    if viz_type == "kit":
        apply_kit_rendering_mode_cfg(set_setting, mode_cfg)
    elif viz_type == "newton":
        apply_newton_mode_cfg_to_visualizer_cfg(viz.cfg, mode_cfg)
        apply_newton_mode_cfg_to_viewer(getattr(viz, "_viewer", None), mode_cfg)

    visualizer_mode_keys[viz_id] = mode_name
