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

# ---------------------------------------------------------------------------
# RTX streaming status tracking
# ---------------------------------------------------------------------------
_RTX_STREAMING_STATUS_EVENT: str = "omni.streamingstatus:streaming_status"

_streaming_is_busy: bool = False
_streaming_subscription = None
_streaming_subscribed: bool = False

_STREAMING_WAIT_TIMEOUT_S: float = 30.0


def _on_streaming_status_event(event) -> None:
    """Callback fired by the RTX renderer whenever streaming status changes."""
    global _streaming_is_busy
    try:
        is_busy = event["isBusy"]
        if is_busy is not None:
            _streaming_is_busy = bool(is_busy)
    except (KeyError, TypeError):
        pass


def _ensure_streaming_subscription() -> None:
    """Subscribe to RTX streaming status events (idempotent)."""
    global _streaming_subscription, _streaming_subscribed
    if _streaming_subscribed:
        return

    # Mark initialization as attempted even if dispatcher lookup fails.
    # This flag enforces no-retry behavior after the first attempt.
    _streaming_subscribed = True

    from carb.eventdispatcher import get_eventdispatcher

    dispatcher = get_eventdispatcher()
    if dispatcher is None:
        logger.warning("carb event dispatcher unavailable – RTX streaming wait will be inactive.")
    else:
        _streaming_subscription = dispatcher.observe_event(
            observer_name="isaaclab_rtx_streaming_wait",
            event_name=_RTX_STREAMING_STATUS_EVENT,
            on_event=_on_streaming_status_event,
        )


def _wait_for_streaming_complete() -> None:
    """Pump ``app.update()`` until RTX streaming reports idle or timeout.

    After streaming finishes a final ``app.update()`` is issued so that the
    frame captured by downstream annotators reflects the newly loaded textures.
    """
    import omni.kit.app

    start = time.monotonic()
    while _streaming_is_busy and (time.monotonic() - start) < _STREAMING_WAIT_TIMEOUT_S:
        omni.kit.app.get_app().update()

    elapsed = time.monotonic() - start
    if _streaming_is_busy:
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

    If RTX texture/geometry streaming is in progress, additional
    ``app.update()`` calls are pumped until the streaming subsystem reports
    idle (or a timeout is reached).

    No-op conditions:
        * Already called this step (dedup across camera instances).
        * A visualizer already pumps ``app.update()`` (e.g. KitVisualizer).
        * Rendering is not active.
    """
    global _last_render_update_key, _streaming_is_busy, _streaming_subscribed, _streaming_subscription

    sim = sim_utils.SimulationContext.instance()
    if sim is None:
        return

    key = (id(sim), sim._physics_step_count)
    if _last_render_update_key == key:
        return  # Already pumped this step (by another camera or a visualizer)

    # Reset stale streaming state when a new SimulationContext is detected.
    if key[0] != _last_render_update_key[0]:
        _streaming_is_busy = False
        _streaming_subscribed = False
        _streaming_subscription = None

    # If a visualizer already pumps the Kit app loop, mark as done and skip.
    if any(viz.pumps_app_update() for viz in sim.visualizers):
        _last_render_update_key = key
        return

    if not sim.is_rendering:
        return

    _ensure_streaming_subscription()

    # Sync physics results → Fabric so RTX sees updated positions.
    # physics_manager.step() only runs simulate()/fetch_results() and does NOT
    # call _update_fabric(), so without this the render would lag one frame behind.
    sim.physics_manager.forward()

    import omni.kit.app

    sim.set_setting("/app/player/playSimulations", False)
    omni.kit.app.get_app().update()

    if _streaming_is_busy:
        _wait_for_streaming_complete()

    sim.set_setting("/app/player/playSimulations", True)

    _last_render_update_key = key
