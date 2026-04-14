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


def _is_stage_loading_or_streaming() -> bool:
    """Return whether the USD stage is still loading or streaming.

    Queries ``omni.usd`` directly rather than relying on asynchronous event
    callbacks, so the result is always consistent with the most recent
    ``app.update()`` tick.
    """
    import omni.usd

    usd_context = omni.usd.get_context()
    _, files_loaded, total_files = usd_context.get_stage_loading_status()
    if files_loaded or total_files:
        return True
    return bool(usd_context.get_stage_streaming_status())


def _wait_for_streaming_complete() -> None:
    """Pump ``app.update()`` until RTX streaming reports idle or timeout.

    The caller is expected to have already pumped ``app.update()`` so that the
    render pipeline has had a chance to report its loading/streaming status.

    After streaming finishes a final ``app.update()`` is issued so that the
    frame captured by downstream annotators reflects the newly loaded textures.
    """
    if not _is_stage_loading_or_streaming():
        return

    import omni.kit.app

    app = omni.kit.app.get_app()

    start = time.monotonic()
    while _is_stage_loading_or_streaming() and (time.monotonic() - start) < _STREAMING_WAIT_TIMEOUT_S:
        app.update()

    elapsed = time.monotonic() - start
    if _is_stage_loading_or_streaming():
        logger.warning(
            "RTX streaming did not complete within %.1f s -- proceeding anyway.",
            _STREAMING_WAIT_TIMEOUT_S,
        )
    elif elapsed > 0.01:
        logger.info("RTX streaming completed in %.2f s.", elapsed)

    app.update()


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

    If RTX texture/geometry streaming is in progress, additional
    ``app.update()`` calls are pumped until the streaming subsystem reports
    idle (or a timeout is reached).  Streaming status is polled directly via
    :meth:`omni.usd.UsdContext.get_stage_streaming_status` rather than through
    asynchronous event callbacks, avoiding any race between event delivery and
    the busy-flag check.

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
    _wait_for_streaming_complete()
    sim.set_setting("/app/player/playSimulations", True)

    _last_render_update_key = key
