# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Isaac RTX renderer integration."""

from __future__ import annotations

import logging
import time

import isaaclab.sim as sim_utils

logger = logging.getLogger(__name__)

# Module-level dedup stamp: tracks the last (sim instance, physics step) at
# which Kit's ``app.update()`` was pumped.  Keyed on ``id(sim)`` so that a
# new ``SimulationContext`` (e.g. in a new test) automatically invalidates
# any stale stamp from a previous instance.
_last_render_update_key: tuple[int, int] = (0, -1)

_STREAMING_WAIT_TIMEOUT_S: float = 30.0


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

    if _get_stage_streaming_busy():
        _wait_for_streaming_complete()

    sim.set_setting("/app/player/playSimulations", True)

    _last_render_update_key = key
