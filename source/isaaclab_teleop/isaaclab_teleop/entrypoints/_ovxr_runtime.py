# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Control-loop pacing, rejected frames, shutdown signals, cleanup and reset policy shared by the teleoperation
entrypoints."""

from __future__ import annotations

import logging
import signal
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._ovxr_presentation import OvxrPresentation

logger = logging.getLogger(__name__)

RENDERER_ATTACH_TIMEOUT_S = 1200.0
"""Allow cold RTX preparation before failing attachment; client connection waits separately.

A first start compiles shaders, and a GPU shared with other rendering processes slows attachment;
either can take several minutes. An expired wait cannot cancel the attachment: the process still
waits for it before it exits.
"""


def requires_hard_sim_reset(physics_backend: str | None) -> bool:
    """Return whether the simulator must be rebuilt before an environment reset."""
    # Only Kit PhysX needs articulation-view rebinding for Pink IK. Both kitless
    # backends retain their constructed solver and any leased presentation stage.
    return physics_backend in (None, "physx")


def pace_ovxr_control_loop(
    using_ovxr: bool,
    action_applied: bool,
    *,
    iteration_started_s: float,
    action_period_s: float,
    step_hz: float | None = None,
    wait_for_shutdown: Callable[[float], bool] | None = None,
    clock: Callable[[], float] | None = None,
) -> None:
    """Limit host control independently of the XR frame clock.

    Credit elapsed input, physics and publication work; an overrun adds no wait
    or catch-up steps. By default the target period is ``action_period_s``, the
    environment step, so simulation time keeps pace with real time when the host
    can. A positive ``step_hz`` replaces it with ``1 / step_hz``. Zero leaves
    active control unpaced. Paused polling targets ``action_period_s`` unless a
    positive ``step_hz`` is given.

    ``clock`` is the clock that measured ``iteration_started_s``; it defaults to
    :func:`time.perf_counter`.
    """
    if not using_ovxr or (action_applied and step_hz == 0.0):
        return
    period_s = 1.0 / step_hz if step_hz else action_period_s
    remaining_s = period_s - ((time.perf_counter if clock is None else clock)() - iteration_started_s)
    if remaining_s > 0.0:
        if wait_for_shutdown is None:
            time.sleep(remaining_s)
        else:
            wait_for_shutdown(remaining_s)


def hold_teleop_frame(env, haptic_driver, *, render: bool = True) -> None:
    """Hold the scene without a physics step while an unsafe teleop frame is rejected.

    ``render`` is false under OVXR, which owns rendering and republishes the unchanged scene.
    """
    if render:
        env.sim.render()
    if haptic_driver is not None:
        haptic_driver.stop()


def warn_rejected_teleop_frame(issue: str, skipped_frames: int) -> None:
    """Log a rejected teleop frame: the first five, then every 200th."""
    if skipped_frames <= 5 or skipped_frames % 200 == 0:
        logger.warning(
            "Rendering without a physics step for unsafe teleop frame (%s); skipped=%d", issue, skipped_frames
        )


class OvxrShutdownSignals:
    """Route SIGINT and SIGTERM to the owner of an OVXR session.

    Before the render worker starts, the first signal unwinds startup with :class:`KeyboardInterrupt`.
    Afterwards, or once cleanup began, a signal requests session shutdown so it cannot interrupt the drain.
    :func:`close_teleop_resources` restores :attr:`previous_handlers`.
    """

    def __init__(self) -> None:
        self.previous_handlers: dict = {}
        """Handlers that preceded the first installation."""
        self.shutting_down = False
        """Whether the owner is unwinding. Set it before cleanup."""

    def install(self, presentation: OvxrPresentation) -> None:
        """Route the signals to ``presentation``, keeping the handlers that preceded the first installation."""

        def request_shutdown(_signum, _frame) -> None:
            if not presentation.started and not self.shutting_down:
                self.shutting_down = True
                raise KeyboardInterrupt
            presentation.session.request_shutdown()

        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous_handlers.setdefault(sig, signal.signal(sig, request_shutdown))


def restore_ovxr_shutdown_handlers(previous_handlers: dict) -> None:
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def close_teleop_resources(
    env,
    camera_feed_session,
    presentation: OvxrPresentation | None,
    previous_shutdown_handlers,
    teleop_interface=None,
    *,
    previous_context=None,
) -> None:
    """Preserve the first failure, drain XR, then close the environment on this thread.

    An undrained XR session stays open. Closing the environment dispatches the simulation's STOP, where the
    session retires or keeps its renderer and Stage alive, so the environment closes either way.
    """
    failure = sys.exception()
    cleanup_failure = None
    environment_closed = False

    def report_cleanup_error(error: Exception) -> None:
        nonlocal cleanup_failure
        logger.exception("Teleoperation cleanup failed")
        if failure is not None:
            failure.add_note(f"Cleanup also failed: {error!r}")
        elif cleanup_failure is None:
            cleanup_failure = error
        else:
            cleanup_failure.add_note(f"Cleanup also failed: {error!r}")

    try:
        input_drain_failed = getattr(teleop_interface, "session_cleanup_failed", False) is True
        if presentation is not None:
            # Releases the session only after its input device drained.
            try:
                presentation.close()
            except Exception as error:
                report_cleanup_error(error)
        elif input_drain_failed:
            report_cleanup_error(
                RuntimeError("Isaac Capture cleanup failed; borrowed XR/environment drain is unproved")
            )
        if not input_drain_failed:
            # A successfully bound feed exits before Capture. This is also the
            # cleanup path for preparation that never reached bind().
            try:
                camera_feed_session.close()
            except Exception as error:
                report_cleanup_error(error)
        if env is not None:
            try:
                env.close()
                environment_closed = True
            except Exception as error:
                report_cleanup_error(error)
        else:
            from isaaclab.sim import SimulationContext

            # Construction may have created a context before returning env; preserve older contexts.
            context = SimulationContext.instance()
            if context is not None and context is not previous_context:
                try:
                    SimulationContext.clear_instance()
                except Exception as error:
                    report_cleanup_error(error)
    finally:
        # Keep repeated SIGINT/SIGTERM on the owner throughout this cleanup attempt.
        try:
            restore_ovxr_shutdown_handlers(previous_shutdown_handlers)
        except Exception as error:
            report_cleanup_error(error)
    if cleanup_failure is not None:
        raise cleanup_failure
    if environment_closed:
        print("Environment closed")
