# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EMA-paced background retarget loop used by :class:`IsaacTeleopDevice`."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AsyncRetargetLoop:
    """Background loop that retargets with EMA-paced scheduling.

    The loop runs continuously in a background thread and uses an
    Exponential Moving Average (EMA) to predict *when* the next
    :meth:`consume` will happen and *how long* retargeting takes.  The
    EMA is a weighted running average where each new sample contributes
    ``alpha`` and the accumulated history contributes ``1 - alpha``,
    giving a smooth estimate that adapts to changing timing without
    reacting to every jitter spike.

    With those two estimates the loop computes::

        sleep = predicted_next_consume - now - retarget_duration - margin

    so it wakes up just early enough to finish retargeting right before
    the caller needs the result.

    Timeline sketch (one consume-to-consume period)::

        last_consume                        next_consume (predicted)
            |                                       |
            |-------sleep--------->|--retarget--|   |
                                   ^            ^   ^
                                wake here    ready  consume

    When :meth:`consume` is called with ``block=True`` (the default), it
    waits until the background thread has produced a fresh result since
    the previous :meth:`consume`.  Because the EMA pacer aims to finish
    just in time, the wait is almost always instantaneous; the block acts
    as a safety net when retarget occasionally runs longer than predicted.

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
        # alpha=0.3 gives an EMA half-life of ~2 samples (ln2/ln(1/0.7)≈1.9).
        # At a 45 Hz target frame rate that adapts within ~44 ms — fast
        # enough to track frame-rate changes (e.g. VSync toggle, workload
        # shifts) while still filtering single-frame timing jitter.
        self._step_fn = step_fn

        # Result state protected by a Condition rather than a bare Event to
        # avoid a TOCTOU (Time-of-Check-to-Time-of-Use) race: with Event the
        # consumer would check is_set() then separately read _latest, but
        # the producer could overwrite _latest between those two steps.
        # Condition bundles the "is there a new result?" check and the read
        # under a single lock acquisition.
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
        self._retarget_dur = 0.020
        self._last_consume = 0.0

    def start(self) -> None:
        """Start the retarget loop in a daemon thread."""
        self._stop.clear()
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
        if block:
            with self._cond:
                while not self._stop.is_set() and self._gen == self._consumed_gen:
                    self._cond.wait(timeout=0.1)

        now = time.monotonic()
        if self._last_consume > 0:
            dt = now - self._last_consume
            if 0.001 < dt < 1.0:
                # EMA update: blend the new measurement (dt) at weight alpha
                # with the running estimate at weight (1 - alpha).  With the
                # default alpha=0.3 the effective window is ~1/alpha ≈ 3
                # samples, so the estimate tracks frame-rate changes within
                # 2-3 frames while filtering single-frame jitter.
                self._step_period = self._ema_alpha * dt + (1 - self._ema_alpha) * self._step_period
        self._last_consume = now

        with self._cond:
            exc = self._exc
            self._exc = None
            fresh = self._gen != self._consumed_gen
            self._consumed_gen = self._gen
            result = self._latest

        if exc is not None:
            raise RuntimeError("Background retarget thread failed") from exc

        return result, fresh

    # ------------------------------------------------------------------

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
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

            # Same EMA formula as _step_period but applied to retarget
            # wall-clock duration, so the pacer knows how much lead time
            # to reserve before the next predicted consume().
            self._retarget_dur = self._ema_alpha * elapsed + (1 - self._ema_alpha) * self._retarget_dur

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

        Goal: finish retargeting just before the predicted next
        :meth:`consume` call.  If there is no timing data yet (first
        iteration), return immediately so the first result is produced
        as fast as possible.


        ``margin_s`` is subtracted so we wake slightly *early* rather
        than risk finishing late.  ``_MIN_LOOP_PERIOD`` clamps the sleep
        to at least 1 ms to prevent busy-spinning when the deadline has
        already passed.
        """
        if self._last_consume <= 0:
            return

        # How much time has already elapsed since the last consume().
        elapsed_since_consume = time.monotonic() - self._last_consume
        # EMA-predicted time remaining until the next consume() call.
        time_until_next = self._step_period - elapsed_since_consume
        # Subtract the predicted retarget cost and a safety margin so we
        # wake up just early enough to have a fresh result ready.
        ideal_sleep = time_until_next - self._retarget_dur - self._margin_s

        self._stop.wait(timeout=max(ideal_sleep, self._MIN_LOOP_PERIOD))
