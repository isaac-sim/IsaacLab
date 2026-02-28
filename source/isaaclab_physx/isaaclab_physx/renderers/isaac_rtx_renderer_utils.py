# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Isaac RTX renderer integration."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.utils.version import get_isaac_sim_version

RTX_DISABLE_COLOR_RENDER_SETTING = "/rtx/sdg/force/disableColorRender"
RTX_SENSORS_SETTING = "/isaaclab/render/rtx_sensors"
SIMPLE_SHADING_AOV = "SimpleShadingSD"
SIMPLE_SHADING_MODE_SETTING = "/rtx/sdg/simpleShading/mode"
SIMPLE_SHADING_MODES = {
    "simple_shading_constant_diffuse": 0,
    "simple_shading_diffuse_mdl": 1,
    "simple_shading_full_mdl": 2,
}
_SUPPORTED_FAST_TYPES = frozenset({
    "distance_to_camera",
    "distance_to_image_plane",
    "depth",
    "albedo",
})


def apply_rtx_sensors_setup(data_types: list[str]) -> None:
    """Set RTX sensors flag and apply version-specific setup.

    Sets /isaaclab/render/rtx_sensors to True so SimulationContext enables rendering.
    Logs warnings for Isaac Sim < 6.0 when albedo or simple shading types are requested.
    """
    import logging

    logger = logging.getLogger(__name__)
    settings = get_settings_manager()
    settings.set_bool(RTX_SENSORS_SETTING, True)

    if get_isaac_sim_version().major < 6:
        if "albedo" in data_types:
            logger.warning(
                "Albedo annotator is only supported in Isaac Sim 6.0+. The albedo data type will be ignored."
            )
        if any(dt in SIMPLE_SHADING_MODES for dt in data_types):
            logger.warning(
                "Simple shading annotators are only supported in Isaac Sim 6.0+. The simple shading data types"
                " will be ignored."
            )


def apply_simple_shading_mode(data_types: list[str]) -> None:
    """Set RTX simple shading mode if requested in data types."""
    requested = [dt for dt in data_types if dt in SIMPLE_SHADING_MODES]
    if not requested:
        return
    if len(requested) > 1:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            "Multiple simple shading modes requested (%s). Using '%s' only.",
            requested,
            requested[0],
        )
    settings = get_settings_manager()
    settings.set_int(SIMPLE_SHADING_MODE_SETTING, SIMPLE_SHADING_MODES[requested[0]])


def apply_rtx_disable_color_render(data_types: list[str]) -> None:
    """Set RTX disableColorRender for fast path when only depth/albedo requested.

    Isaac Sim 6.0+ only. Sets True when all data types are depth/albedo; otherwise False.
    If GUI is enabled, always False so viewport is not black.
    """
    if get_isaac_sim_version().major < 6:
        return
    settings = get_settings_manager()
    if settings.get("/isaaclab/has_gui"):
        settings.set_bool(RTX_DISABLE_COLOR_RENDER_SETTING, False)
    elif all(dt in _SUPPORTED_FAST_TYPES for dt in data_types):
        settings.set_bool(RTX_DISABLE_COLOR_RENDER_SETTING, True)
    else:
        settings.set_bool(RTX_DISABLE_COLOR_RENDER_SETTING, False)


# Module-level dedup stamp: tracks the last (sim instance, physics step) at
# which Kit's ``app.update()`` was pumped.  Keyed on ``id(sim)`` so that a
# new ``SimulationContext`` (e.g. in a new test) automatically invalidates
# any stale stamp from a previous instance.
_last_render_update_key: tuple[int, int] = (0, -1)


def ensure_isaac_rtx_render_update() -> None:
    """Ensure the Isaac RTX renderer has been pumped for the current physics step.

    This keeps the Kit-specific ``app.update()`` logic inside the renderers
    package rather than in the backend-agnostic ``SimulationContext``.

    Safe to call from multiple ``Camera`` / ``TiledCamera`` instances per step —
    only the first call triggers ``app.update()``.  Subsequent calls are no-ops
    because the module-level ``_last_render_update_key`` already matches the
    current ``(id(sim), step_count)`` pair.

    The key is a ``(sim_instance_id, step_count)`` tuple so that creating a new
    ``SimulationContext`` (e.g. in a subsequent test) automatically invalidates
    any stale stamp left over from a previous instance.

    No-op conditions:
        * Already called this step (dedup across camera instances).
        * A visualizer already pumps ``app.update()`` (e.g. KitVisualizer).
        * Rendering is not active.
    """
    global _last_render_update_key

    sim = sim_utils.SimulationContext.instance()
    if sim is None:
        return

    key = (id(sim), sim._physics_step_count)
    if _last_render_update_key == key:
        return  # Already pumped this step (by another camera or a visualizer)

    # If a visualizer already pumps the Kit app loop, mark as done and skip.
    if any(viz.pumps_app_update() for viz in sim.visualizers):
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
    sim.set_setting("/app/player/playSimulations", True)

    _last_render_update_key = key

