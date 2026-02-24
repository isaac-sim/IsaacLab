# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility helpers for applying rendering quality profiles."""

from __future__ import annotations

from typing import Any

from .rendering_quality_cfg import RenderingQualityCfg
from .rendering_quality_presets import get_kit_rendering_preset


def apply_kit_rendering_preset(set_setting: Any, preset_name: str) -> None:
    """Apply a named kit preset via provided setting setter."""
    preset = get_kit_rendering_preset(preset_name)
    for key, value in preset.items():
        set_setting(key, value)


def apply_kit_rendering_quality_cfg(set_setting: Any, quality_cfg: RenderingQualityCfg) -> None:
    """Apply kit-specific quality fields."""
    if quality_cfg.kit_rendering_preset:
        apply_kit_rendering_preset(set_setting, quality_cfg.kit_rendering_preset)

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
        value = getattr(quality_cfg, field_name, None)
        if value is not None:
            set_setting(carb_key, value)

    if quality_cfg.kit_antialiasing_mode is not None:
        try:
            import omni.replicator.core as rep

            rep.settings.set_render_rtx_realtime(antialiasing=quality_cfg.kit_antialiasing_mode)
        except Exception:
            pass


def apply_newton_quality_cfg_to_visualizer_cfg(visualizer_cfg: Any, quality_cfg: RenderingQualityCfg) -> None:
    """Apply Newton quality values to a visualizer cfg object."""
    override_fields = {
        "newton_enable_shadows": "enable_shadows",
        "newton_enable_sky": "enable_sky",
        "newton_enable_wireframe": "enable_wireframe",
        "newton_sky_upper_color": "sky_upper_color",
        "newton_sky_lower_color": "sky_lower_color",
        "newton_light_color": "light_color",
    }
    for quality_field, viz_field in override_fields.items():
        value = getattr(quality_cfg, quality_field, None)
        if value is not None and hasattr(visualizer_cfg, viz_field):
            setattr(visualizer_cfg, viz_field, value)


def resolve_rendering_quality_name_for_visualizer_cfg(get_setting: Any, visualizer_cfg: Any) -> str | None:
    """Resolve effective quality profile name for a visualizer cfg."""
    cli_quality_explicit = bool(get_setting("/isaaclab/rendering/rendering_quality/explicit"))
    cli_quality = get_setting("/isaaclab/rendering/rendering_quality")
    if cli_quality_explicit:
        return cli_quality if cli_quality else None
    quality_name = getattr(visualizer_cfg, "rendering_quality", None)
    return quality_name if quality_name else None


def resolve_rendering_quality_cfg(
    quality_name: str | None, quality_cfgs: dict[str, RenderingQualityCfg], logger: Any
) -> RenderingQualityCfg | None:
    """Fetch quality cfg by name and log if missing."""
    if not quality_name:
        return None
    quality_cfg = quality_cfgs.get(quality_name)
    if quality_cfg is None:
        logger.warning(
            "[SimulationContext] Rendering quality '%s' not found in SimulationCfg.rendering_quality_cfgs.",
            quality_name,
        )
        return None
    return quality_cfg
