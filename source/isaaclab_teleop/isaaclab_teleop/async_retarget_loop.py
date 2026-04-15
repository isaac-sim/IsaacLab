# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EMA-paced, consumption-gated background retarget loop for :class:`IsaacTeleopDevice`.

The background thread produces exactly one retarget result per
:meth:`consume` call.  After posting a result it waits for the main
thread to consume it before retargeting again, ensuring the retarget
pipeline is invoked at the same cadence as ``advance()`` (just like the
synchronous path).

Within each cycle the thread uses an Exponential Moving Average (EMA)
to predict *when* the next :meth:`consume` will happen, then delays
reading inputs and retargeting until just before that deadline.  This
maximises input freshness while still overlapping the retarget
computation with ``env.step()``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AsyncRetargetLoop:
    """Background loop that retargets once per :meth:`consume` call.

    The loop runs in a daemon thread and combines two mechanisms:

    1. **Consumption gate** -- after posting a result the thread blocks
       until :meth:`consume` has read it, preventing more than one
       retarget per ``advance()`` call.
    2. **EMA pacer** -- once the gate opens, the thread sleeps until the
       optimal moment to start retargeting so inputs are as fresh as
       possible when read.

    The EMA tracks two quantities:

    * **step period** -- time between consecutive :meth:`consume` calls
      (i.e. the main loop's frame period).
    * **retarget duration** -- wall-clock cost of one ``step_fn`` call.

    With those estimates the pacer computes::

        sleep = predicted_next_consume - now - retarget_duration - margin

    so it wakes up just early enough to finish retargeting right before
    the caller needs the result.

    Timeline sketch (one consume-to-consume period)::

        main: advance() → update_inputs() → consume(block) [waits] →
                                                      gets result → env.step()

        bg:   [gate: wait consumed] → [ema sleep] → read inputs →
                                       retarget → post result → [gate] …

    Thread-safety contract:
        All USD/Kit/Omniverse API calls must stay on the main thread.
        The main thread pushes pre-computed inputs (anchor matrix,
        target-frame transform) via :meth:`update_inputs`, and the
        background thread only runs the retarget computation with those
        inputs.
    """

    _MAX_CONSECUTIVE_FAILURES = 5
    _MIN_LOOP_PERIOD = 0.001  # 1 ms floor to prevent busy-spinning

    def __init__(
        self,
        step_fn: Callable[[np.ndarray, np.ndarray | torch.Tensor | None], torch.Tensor | None],
        *,
        ema_alpha: float = 0.3,
        margin_s: float = 0.005,
    ):
        """Create a new loop (stopped).  Call :meth:`start` to begin.

        Args:
            step_fn: Retarget function called each iteration with
                ``(anchor_matrix, target_T_world)`` and returning an
                action tensor or ``None``.  **Called from a background
                thread**; must not touch USD/Kit/Omniverse APIs directly.
            ema_alpha: EMA smoothing factor in ``(0, 1]``.
            margin_s: Safety margin [s] subtracted from the predicted
                deadline.
        """
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")

        self._step_fn = step_fn

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._gen = 0
        self._consumed_gen = 0
        self._latest: torch.Tensor | None = None
        self._exc: BaseException | None = None

        # Input state: written by main thread, read by background thread.
        self._input_lock = threading.Lock()
        self._anchor_matrix: np.ndarray = np.eye(4, dtype=np.float32)
        self._target_T_world: np.ndarray | torch.Tensor | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._ema_alpha = ema_alpha
        self._margin_s = margin_s
        self._step_period = 0.022
        self._retarget_dur = 0.005
        self._last_consume = 0.0

    def start(self) -> None:
        """Start the retarget loop in a daemon thread."""
        self._stop.clear()
        with self._cond:
            self._gen = 0
            self._consumed_gen = 0
            self._latest = None
            self._exc = None
        self._last_consume = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the loop to exit and wait for the thread to finish."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.error(
                    "Background retarget thread did not exit within 2 s; "
                    "it will continue running as a daemon until interpreter exit."
                )
            self._thread = None

    def update_inputs(
        self,
        anchor_matrix: np.ndarray,
        target_T_world: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        """Push updated inputs from the main thread.

        Called every frame before :meth:`consume` so the background
        thread always has fresh USD-derived data without touching
        USD/Kit APIs itself.

        Args:
            anchor_matrix: The (4, 4) world-to-anchor transform [m].
            target_T_world: Optional (4, 4) rebase transform [m].
        """
        with self._input_lock:
            self._anchor_matrix = np.array(anchor_matrix, dtype=np.float32, copy=True)
            if isinstance(target_T_world, np.ndarray):
                target_T_world = np.array(target_T_world, dtype=np.float32, copy=True)
            elif isinstance(target_T_world, torch.Tensor):
                target_T_world = target_T_world.clone()
            self._target_T_world = target_T_world

    def consume(self, block: bool = True) -> tuple[torch.Tensor | None, bool]:
        """Read the most recent retarget result.

        Updates the EMA estimate of the consumer's call period so the
        pacer can predict the next deadline.

        Args:
            block: When ``True`` (default), wait until the background
                thread has produced a fresh result since the last
                :meth:`consume`.  When ``False``, return immediately
                with whatever result is available (may be stale).

        Returns:
            ``(action, is_fresh)`` where *is_fresh* is ``True`` when the
            result has been updated since the last :meth:`consume`.

        Raises:
            Exception: Re-raises any exception that killed the background
                thread, chained under a :class:`RuntimeError`.
        """
        with self._cond:
            if block:
                while not self._stop.is_set() and self._gen == self._consumed_gen:
                    if self._thread is not None and not self._thread.is_alive():
                        break
                    self._cond.wait(timeout=0.1)

            now = time.monotonic()
            if self._last_consume > 0:
                dt = now - self._last_consume
                if 0.001 < dt < 1.0:
                    self._step_period = (
                        self._ema_alpha * dt + (1 - self._ema_alpha) * self._step_period
                    )
            self._last_consume = now

            exc = self._exc
            self._exc = None
            fresh = self._gen != self._consumed_gen
            self._consumed_gen = self._gen
            result = self._latest
            # Wake the BG thread now that we have consumed the result.
            self._cond.notify_all()

        if exc is not None:
            raise RuntimeError("Background retarget thread failed") from exc

        if block and not fresh and self._thread is not None and not self._thread.is_alive():
            raise RuntimeError("Background retarget thread died unexpectedly")

        return result, fresh

    # ------------------------------------------------------------------

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
            # Gate: wait until the previous result has been consumed so
            # the pipeline is invoked exactly once per advance() call.
            with self._cond:
                while not self._stop.is_set() and self._gen > self._consumed_gen:
                    self._cond.wait(timeout=0.1)
            if self._stop.is_set():
                break

            # EMA pace: delay reading inputs until just before the
            # predicted next consume() so that inputs are as fresh as
            # possible.
            self._pace()
            if self._stop.is_set():
                break

            with self._input_lock:
                anchor_matrix = self._anchor_matrix
                target_T_world = self._target_T_world

            t0 = time.monotonic()
            try:
                action = self._step_fn(anchor_matrix, target_T_world)
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                if consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                    with self._cond:
                        self._exc = exc
                        self._gen += 1
                        self._cond.notify_all()
                    return
                logger.warning(
                    "Retarget failed (%d/%d), retrying: %s",
                    consecutive_failures,
                    self._MAX_CONSECUTIVE_FAILURES,
                    exc,
                )
                self._stop.wait(timeout=min(0.1 * consecutive_failures, 0.5))
                continue
            elapsed = time.monotonic() - t0

            self._retarget_dur = (
                self._ema_alpha * elapsed + (1 - self._ema_alpha) * self._retarget_dur
            )

            if action is not None:
                action = action.clone()
                if action.is_cuda:
                    torch.cuda.current_stream(action.device).synchronize()

            with self._cond:
                self._latest = action
                self._gen += 1
                self._cond.notify_all()

    def _pace(self) -> None:
        """Sleep until the optimal moment to begin the next retarget.

        Goal: read inputs and start retargeting as late as possible
        while still finishing before the predicted next :meth:`consume`.
        If there is no timing data yet (first iteration), return
        immediately so the first result is produced as fast as possible.

        The consumption gate in :meth:`_run` guarantees at most one
        retarget per consume, so this method never needs to worry about
        a second retarget sneaking in.
        """
        if self._last_consume <= 0:
            return

        elapsed_since_consume = time.monotonic() - self._last_consume
        time_until_next = self._step_period - elapsed_since_consume
        ideal_sleep = time_until_next - self._retarget_dur - self._margin_s

        self._stop.wait(timeout=max(ideal_sleep, self._MIN_LOOP_PERIOD))
