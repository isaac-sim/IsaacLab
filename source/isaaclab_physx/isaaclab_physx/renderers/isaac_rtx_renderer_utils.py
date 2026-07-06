# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Isaac RTX renderer integration."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import tomllib

import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import SettingsManager, get_settings_manager

from .isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg

logger = logging.getLogger(__name__)

_RTX_FIELD_TO_SETTING = {
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

# Module-level dedup stamp: tracks the last (sim instance, physics step, render generation) at
# which Kit's ``app.update()`` was pumped.  Keyed on ``id(sim)`` so that a
# new ``SimulationContext`` (e.g. in a new test) automatically invalidates
# any stale stamp from a previous instance.
_last_render_update_key: tuple[int, int, int] = (0, -1, -1)

_STREAMING_WAIT_TIMEOUT_S: float = 30.0


def _setting_path_from_key(key: str) -> str:
    """Convert a user-friendly carb setting key to a carb path."""
    if key.startswith("/"):
        return key
    if "_" in key:
        return "/" + key.replace("_", "/")
    if "." in key:
        return "/" + key.replace(".", "/")
    return key


def _apply_nested_preset(settings: SettingsManager, data: dict[str, Any], path: str = "") -> None:
    """Apply nested preset dictionaries loaded from a .kit file."""
    for key, value in data.items():
        key_path = f"{path}/{key}" if path else f"/{key}"
        if isinstance(value, dict):
            _apply_nested_preset(settings, value, key_path)
        else:
            settings.set(key_path.replace(".", "/"), value)


def _apply_rendering_mode_preset(settings: SettingsManager, rendering_mode: str) -> None:
    """Apply an Isaac Lab rendering-mode preset."""
    supported_rendering_modes = {"performance", "balanced", "quality"}
    if rendering_mode not in supported_rendering_modes:
        raise ValueError(
            f"IsaacRtxRendererCfg rendering mode '{rendering_mode}' not in "
            "supported modes "
            f"{sorted(supported_rendering_modes)}."
        )

    isaaclab_app_exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "apps")
    from isaaclab.utils.version import get_isaac_sim_version

    if get_isaac_sim_version().major < 6:
        isaaclab_app_exp_path = os.path.join(isaaclab_app_exp_path, "isaacsim_5")

    preset_filename = os.path.join(isaaclab_app_exp_path, f"rendering_modes/{rendering_mode}.kit")
    if os.path.exists(preset_filename):
        with open(preset_filename, "rb") as file:
            _apply_nested_preset(settings, tomllib.load(file))
    else:
        logger.warning("[isaac_rtx] Render preset file not found: %s", preset_filename)


def apply_isaac_rtx_global_settings(
    global_settings: IsaacRtxRendererGlobalSettingsCfg,
    settings: SettingsManager | None = None,
) -> None:
    """Apply global Isaac RTX settings before renderer initialization.

    Args:
        global_settings: Global Isaac RTX settings to apply.
        settings: Settings manager to apply settings through. If None, the global settings manager is used.
    """
    if settings is None:
        settings = get_settings_manager()
    _apply_isaac_rtx_global_settings(global_settings, settings)


def _apply_isaac_rtx_global_settings(
    global_settings: IsaacRtxRendererGlobalSettingsCfg,
    settings: SettingsManager,
) -> None:
    """Apply global Isaac RTX settings to the provided settings manager."""

    rendering_mode = getattr(global_settings, "rendering_mode", None)
    if rendering_mode:
        _apply_rendering_mode_preset(settings, rendering_mode)

    for field_name, setting_path in _RTX_FIELD_TO_SETTING.items():
        value = getattr(global_settings, field_name, None)
        if value is not None:
            settings.set(setting_path, value)

    extra_settings = getattr(global_settings, "carb_settings", None)
    if extra_settings:
        for key, value in extra_settings.items():
            settings.set(_setting_path_from_key(key), value)

    antialiasing_mode = getattr(global_settings, "antialiasing_mode", None)
    if antialiasing_mode is not None:
        try:
            import omni.replicator.core as rep

            rep.settings.set_render_rtx_realtime(antialiasing=antialiasing_mode)
        except Exception:
            pass


def _get_stage_streaming_busy() -> bool:
    """Synchronously query whether RTX stage streaming is still in progress."""
    import omni.usd

    usd_context = omni.usd.get_context()
    if usd_context is None:
        return False
    return usd_context.get_stage_streaming_status()


def _wait_for_streaming_complete() -> None:
    """Pump ``app.update()`` until RTX streaming reports idle or timeout.

    After streaming finishes a final ``app.update()`` is issued so that the
    frame captured by downstream annotators reflects the newly loaded textures.
    """
    import omni.kit.app

    start = time.monotonic()
    while _get_stage_streaming_busy() and (time.monotonic() - start) < _STREAMING_WAIT_TIMEOUT_S:
        omni.kit.app.get_app().update()

    elapsed = time.monotonic() - start
    if _get_stage_streaming_busy():
        logger.warning(
            "RTX streaming did not complete within %.1f s – proceeding anyway.",
            _STREAMING_WAIT_TIMEOUT_S,
        )
    elif elapsed > 0.01:
        logger.info("RTX streaming completed in %.2f s.", elapsed)

    omni.kit.app.get_app().update()


def ensure_rtx_hydra_engine_attached() -> None:
    """Attach the RTX Hydra engine to the USD context if not already attached.

    Headless app files such as ``isaaclab.python.headless.rendering.kit`` intentionally
    omit ``omni.kit.viewport.window`` to avoid pulling in the ``omni.ui``-based viewport
    stack. However, ``ViewportWindow`` is normally responsible for calling
    :func:`omni.usd.create_hydra_engine` at startup; without it the RTX Hydra engine is
    never bound to the :class:`omni.usd.UsdContext`, and the first Replicator tiled
    render product runs against a cold pipeline. On some GPUs this manifests as
    ``cudaErrorIllegalAddress`` inside ``omni.rtx`` (CUDA ``freeAsync``) and/or all
    tiles rendering as black.

    This helper replicates only the activation step ``ViewportWindow`` performs,
    without creating a UI or a window. It is idempotent: when the engine is already
    attached (e.g. GUI runs that do load ``omni.kit.viewport.window``, or a previous
    call already attached it) the function is a no-op. Failures are logged as errors
    and do not propagate, so non-RTX contexts (e.g. unit tests importing this module
    without a running Kit app) continue to work.
    """
    try:
        ctx = omni.usd.get_context()
        if ctx is None:
            return
        if "rtx" in ctx.get_attached_hydra_engine_names():
            return
        omni.usd.create_hydra_engine("rtx", ctx)
    except Exception as e:  # noqa: BLE001
        logger.error("RTX Hydra engine attach failed: %s", e)


def ensure_isaac_rtx_render_update() -> None:
    """Ensure the Isaac RTX renderer has been pumped for the current sim step.

    This keeps the Kit-specific ``app.update()`` logic inside the renderers
    package rather than in the backend-agnostic ``SimulationContext``.

    Safe to call from multiple ``Camera`` instances per step —
    only the first call triggers ``app.update()``.  Subsequent calls are no-ops
    because the module-level ``_last_render_update_key`` already matches the
    current ``(id(sim), step_count, render_generation)`` tuple.

    The key is a ``(sim_instance_id, step_count, render_generation)`` tuple so that:
    - creating a new ``SimulationContext`` invalidates stale stamps, and
    - render/reset transitions that do not advance physics step count still force a fresh update.

    After the initial ``app.update()`` the streaming subsystem is queried
    synchronously via ``UsdContext.get_stage_streaming_status()``.  If textures
    are still loading, additional ``app.update()`` calls are pumped until the
    subsystem reports idle (or a timeout is reached).

    No-op conditions:
        * Already called this step (dedup across camera instances).
        * A visualizer already pumps ``app.update()`` (e.g. KitVisualizer).
        * Rendering is not active.
    """
    global _last_render_update_key

    sim = sim_utils.SimulationContext.instance()
    if sim is None:
        return

    render_generation = getattr(sim, "render_generation", getattr(sim, "_render_generation", 0))
    key = (id(sim), sim._physics_step_count, render_generation)
    if _last_render_update_key == key:
        return  # Already pumped this step (by another camera or a visualizer)

    # If a visualizer already pumps the Kit app loop, mark as done and skip.
    # However, on the very first call for a new SimulationContext, the visualizer
    # has not had a chance to pump yet (sim.render() was never called), so we
    # must perform the initial app.update() ourselves to populate annotator buffers.
    first_call_for_sim = _last_render_update_key[0] != id(sim)
    if not first_call_for_sim and any(viz.pumps_app_update() for viz in sim.visualizers):
        _last_render_update_key = key
        return

    if not sim.is_rendering:
        return

    # Sync physics results → Fabric so RTX sees updated positions.
    # physics_manager.step() only runs simulate()/fetch_results() and does NOT
    # call _update_fabric(), so without this the render would lag one frame behind.
    sim.physics_manager.forward()

    import omni.kit.app

    sim.set_setting("/app/player/playSimulations", False)
    omni.kit.app.get_app().update()

    if _get_stage_streaming_busy():
        _wait_for_streaming_complete()

    sim.set_setting("/app/player/playSimulations", True)

    _last_render_update_key = key


def pump_kit_app_for_headless_video_render_if_needed(sim: Any) -> None:
    """Pump Kit app-loop for headless rgb-array rendering when needed.

    Isaac Sim / RTX specific; kept out of backend-agnostic :class:`~isaaclab.sim.SimulationContext`.
    """
    if not bool(sim.get_setting("/isaaclab/video/enabled")):
        return

    from isaaclab.utils.version import has_kit

    if not has_kit():
        return
    if any(viz.pumps_app_update() for viz in sim.visualizers):
        return
    try:
        ensure_isaac_rtx_render_update()
    except (ImportError, AttributeError, ModuleNotFoundError) as exc:
        logger.debug("[isaac_rtx] Skipping Kit app-loop pump in render() (non-Kit env): %s", exc)
    except Exception as exc:
        logger.warning(
            "[isaac_rtx] Kit app-loop pump failed in render() — video frames may be stale or black: %s",
            exc,
        )
