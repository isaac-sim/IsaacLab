# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Isaac RTX renderer integration."""

from __future__ import annotations

import logging
import time
from typing import Any

import omni.usd

import isaaclab.sim as sim_utils

logger = logging.getLogger(__name__)

# Module-level dedup stamp: tracks the last (sim instance, physics step, render generation) at
# which Kit's ``app.update()`` was pumped.  Keyed on ``id(sim)`` so that a
# new ``SimulationContext`` (e.g. in a new test) automatically invalidates
# any stale stamp from a previous instance.
_last_render_update_key: tuple[int, int, int] = (0, -1, -1)

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
