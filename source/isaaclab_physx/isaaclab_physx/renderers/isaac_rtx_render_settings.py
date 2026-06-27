# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Translation of :class:`~isaaclab.sim.RenderCfg` overrides into RTX carb settings.

This module owns the RTX-specific mapping from each :class:`~isaaclab.sim.RenderCfg` field to its
``/rtx/...`` carb setting. The mapping is applied on top of the RTX defaults from the rendering
experience files, so any value set on :class:`~isaaclab.sim.RenderCfg` overrides them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.sim import RenderCfg

# RenderCfg fields mapped to their RTX carb setting paths.
_FIELD_TO_SETTING: dict[str, str] = {
    "enable_translucency": "/rtx/translucency/enabled",
    "enable_reflections": "/rtx/reflections/enabled",
    "enable_global_illumination": "/rtx/indirectDiffuse/enabled",
    "enable_dlssg": "/rtx-transient/dlssg/enabled",
    "enable_dl_denoiser": "/rtx-transient/dldenoiser/enabled",
    "dlss_mode": "/rtx/post/dlss/execMode",
    "enable_direct_lighting": "/rtx/directLighting/enabled",
    "samples_per_pixel": "/rtx/directLighting/sampledLighting/samplesPerPixel",
    "enable_shadows": "/rtx/shadows/enabled",
    "enable_ambient_occlusion": "/rtx/ambientOcclusion/enabled",
    "dome_light_upper_lower_strategy": "/rtx/domeLight/upperLowerStrategy",
    "ambient_light_intensity": "/rtx/sceneDb/ambientLightIntensity",
    "ambient_occlusion_denoiser_mode": "/rtx/ambientOcclusion/denoiserMode",
    "subpixel_mode": "/rtx/raytracing/subpixel/mode",
    "enable_cached_raytracing": "/rtx/raytracing/cached/enabled",
    "max_samples_per_launch": "/rtx/pathtracing/maxSamplesPerLaunch",
    "view_tile_limit": "/rtx/viewTile/limit",
    # RT2 path tracing settings
    "max_bounces": "/rtx/rtpt/maxBounces",
    "split_glass": "/rtx/rtpt/splitGlass",
    "split_clearcoat": "/rtx/rtpt/splitClearcoat",
    "split_rough_reflection": "/rtx/rtpt/splitRoughReflection",
}


def apply_rtx_render_settings(render_cfg: RenderCfg, set_setting: Callable[[str, Any], None]) -> None:
    """Apply :class:`~isaaclab.sim.RenderCfg` overrides on top of the experience-file RTX defaults.

    Each non-``None`` :class:`~isaaclab.sim.RenderCfg` field is written to its mapped ``/rtx/...`` carb
    path, entries in :attr:`~isaaclab.sim.RenderCfg.carb_settings` are normalized to carb paths and
    written verbatim, and :attr:`~isaaclab.sim.RenderCfg.antialiasing_mode` is applied through Replicator
    when available. Call once before :meth:`~isaaclab.sim.SimulationContext.reset`.

    Args:
        render_cfg: The render configuration whose set fields override the experience-file defaults.
        set_setting: Callable that writes a carb setting, e.g. :meth:`isaaclab.sim.SimulationContext.set_setting`.
    """
    # Mapped RenderCfg fields (carb_settings and antialiasing_mode are handled separately below).
    for key, value in vars(render_cfg).items():
        if value is None or key in {"carb_settings", "antialiasing_mode"}:
            continue
        setting_path = _FIELD_TO_SETTING.get(key)
        if setting_path is not None:
            set_setting(setting_path, value)

    # Raw overrides supplied with native carb / .kit / python key formats.
    extra_settings = getattr(render_cfg, "carb_settings", None)
    if extra_settings:
        for key, value in extra_settings.items():
            if "_" in key:
                path = "/" + key.replace("_", "/")
            elif "." in key:
                path = "/" + key.replace(".", "/")
            else:
                path = key
            set_setting(path, value)

    # Optional anti-aliasing mode via Replicator (best-effort, may use Omniverse APIs).
    antialiasing_mode = getattr(render_cfg, "antialiasing_mode", None)
    if antialiasing_mode is not None:
        try:
            import omni.replicator.core as rep

            rep.settings.set_render_rtx_realtime(antialiasing=antialiasing_mode)
        except Exception:
            pass
