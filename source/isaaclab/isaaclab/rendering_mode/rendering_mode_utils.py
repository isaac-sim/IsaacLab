# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility helpers for applying rendering mode profiles."""

from __future__ import annotations

from typing import Any

from .rendering_mode_cfg import RenderingModeCfg
from .rendering_mode_presets import get_kit_rendering_preset


def apply_kit_rendering_preset(set_setting: Any, preset_name: str) -> None:
    """Apply a named kit preset via provided setting setter."""
    preset = get_kit_rendering_preset(preset_name)
    for key, value in preset.items():
        set_setting(key, value)


def apply_kit_rendering_mode_cfg(set_setting: Any, mode_cfg: RenderingModeCfg) -> None:
    """Apply kit-specific rendering mode fields."""
    if mode_cfg.rendering_mode_preset:
        apply_kit_rendering_preset(set_setting, mode_cfg.rendering_mode_preset)

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

    if mode_cfg.kit_antialiasing_mode is not None:
        try:
            import omni.replicator.core as rep

            rep.settings.set_render_rtx_realtime(antialiasing=mode_cfg.kit_antialiasing_mode)
        except Exception:
            pass


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
    cli_mode = get_setting("/isaaclab/rendering/rendering_mode")
    if cli_mode_explicit:
        return cli_mode if cli_mode else None
    mode_name = getattr(visualizer_cfg, "rendering_mode", None)
    return mode_name if mode_name else None


def resolve_rendering_mode_cfg(
    mode_name: str | None, mode_cfgs: dict[str, RenderingModeCfg], logger: Any
) -> RenderingModeCfg | None:
    """Fetch rendering mode cfg by name and log if missing."""
    if not mode_name:
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
