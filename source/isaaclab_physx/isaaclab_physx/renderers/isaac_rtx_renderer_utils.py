# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Isaac RTX renderer integration."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.utils import has_kit
from isaaclab.utils.version import get_isaac_sim_version

logger = logging.getLogger(__name__)

SIMPLE_SHADING_MODES: dict[str, int] = {
    "simple_shading_constant_diffuse": 0,
    "simple_shading_diffuse_mdl": 1,
    "simple_shading_full_mdl": 2,
}
SIMPLE_SHADING_MODE_SETTING: str = "/rtx/sdg/simpleShading/mode"

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


def configure_isaac_rtx_settings(data_types: Sequence[str]) -> None:
    """Configure global Isaac RTX settings based on requested data types.

    On Isaac Sim 6.0+ this enables the fast SDG path when no RGB/RGBA
    annotators are requested (unless a GUI viewport is active).

    On older Isaac Sim versions it logs warnings for data types that are
    not yet supported (``albedo``, simple-shading modes).

    Args:
        data_types: The sensor data types requested by the camera.
    """
    if not has_kit():
        return

    settings = get_settings_manager()
    isaac_sim_version = get_isaac_sim_version()

    if isaac_sim_version.major >= 6:
        needs_color_render = "rgb" in data_types or "rgba" in data_types
        if not needs_color_render:
            settings.set_bool("/rtx/sdg/force/disableColorRender", True)
        if settings.get("/isaaclab/has_gui"):
            settings.set_bool("/rtx/sdg/force/disableColorRender", False)
    else:
        if "albedo" in data_types:
            logger.warning(
                "Albedo annotator is only supported in Isaac Sim 6.0+. The albedo data type will be ignored."
            )
        if any(dt in SIMPLE_SHADING_MODES for dt in data_types):
            logger.warning(
                "Simple shading annotators are only supported in Isaac Sim 6.0+. The simple shading data types"
                " will be ignored."
            )


def resolve_simple_shading_mode(data_types: Sequence[str]) -> int | None:
    """Resolve the requested simple shading mode from data types.

    Args:
        data_types: The sensor data types requested by the camera.

    Returns:
        The integer mode value for the first matching simple-shading data type,
        or ``None`` if none were requested.
    """
    requested = [dt for dt in data_types if dt in SIMPLE_SHADING_MODES]
    if not requested:
        return None
    if len(requested) > 1:
        logger.warning(
            "Multiple simple shading modes requested (%s). Using '%s' only.",
            requested,
            requested[0],
        )
    return SIMPLE_SHADING_MODES[requested[0]]
