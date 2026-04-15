# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Consumption-gated background retarget loop with pluggable timing estimation.

The background thread produces exactly one retarget result per
:meth:`consume` call.  After posting a result it waits for the main
thread to consume it before retargeting again, ensuring the retarget
pipeline is invoked at the same cadence as ``advance()`` (just like the
synchronous path).

Within each cycle a :class:`TimingEstimator` predicts *when* the next
:meth:`consume` will happen and delays reading inputs until just before
that deadline.  The default strategy uses Exponential Moving Average
(EMA) via :class:`EmaTimingEstimator`.  To swap in a different
prediction algorithm, subclass :class:`TimingEstimator` and
:class:`TimingEstimatorCfg`, then pass the custom cfg to
:class:`AsyncRetargetLoop`.
"""

from __future__ import annotations

import abc
import logging
import threading
import time
from collections.abc import Callable

import numpy as np
import torch

from isaaclab.utils import configclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing estimator interface + default EMA implementation
# ---------------------------------------------------------------------------


class TimingEstimator(abc.ABC):
    """Base class for retarget-loop timing estimators.

    Subclasses predict *when* to start each retarget cycle so the result
    is ready just before the consumer calls :meth:`consume`.

    Subclass both this class and :class:`TimingEstimatorCfg` to
    implement a custom estimation strategy.
    """

    def __init__(self, cfg: TimingEstimatorCfg):
        """Initialize the estimator.

        Args:
            cfg: Configuration for this estimator instance.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset to initial state (no timing data)."""

    @abc.abstractmethod
    def record_consume(self, timestamp: float) -> None:
        """Record a consume() call, updating the step-period estimate.

        Args:
            timestamp: :func:`time.monotonic` value at the consume call.
        """

    @abc.abstractmethod
    def record_retarget(self, duration_s: float) -> None:
        """Record a completed retarget, updating the cost estimate.

        Args:
            duration_s: Wall-clock time [s] of the retarget computation.
        """

    @abc.abstractmethod
    def compute_sleep(self, now: float) -> float:
        """Return how long to sleep before starting the next retarget [s].

        Returns ``0.0`` when no timing data is available yet (first
        iteration), signalling that the caller should proceed immediately.

        Args:
            now: Current :func:`time.monotonic` value.
        """


@configclass
class TimingEstimatorCfg:
    """Base configuration for :class:`TimingEstimator` subclasses.

    Subclass to add parameters for a custom estimation strategy and set
    :attr:`class_type` to the corresponding :class:`TimingEstimator`
    subclass.
    """

    class_type: type[TimingEstimator] | None = None
    """The :class:`TimingEstimator` subclass to instantiate.  Must be
    set by concrete cfg subclasses."""


class EmaTimingEstimator(TimingEstimator):
    """Timing estimator using Exponential Moving Average (EMA).

    Tracks two quantities via EMA:

    * **Step period** -- interval between consecutive consumer calls
      (i.e. how often a fresh result is needed).
    * **Retarget duration** -- wall-clock cost of one retarget computation.

    :meth:`compute_sleep` combines both estimates to return how long the
    background thread should sleep before starting the next retarget,
    aiming to finish just before the consumer arrives.
    """

    _MIN_SLEEP = 0.001  # 1 ms floor to prevent busy-spinning

    def __init__(self, cfg: EmaTimingEstimatorCfg):
        """Initialize the EMA estimator.

        Args:
            cfg: Configuration for this estimator instance.
        """
        super().__init__(cfg)
        self._alpha = cfg.ema_alpha
        self._margin_s = cfg.margin_s
        # Seed estimates assume ~45 Hz consume cadence and ~5 ms retarget
        # cost.  The EMA converges within a few frames regardless of the
        # actual rates, so these only affect the first 2-3 pacing decisions.
        self._step_period = 0.022
        self._retarget_dur = 0.005
        self._last_consume = 0.0

    @property
    def step_period(self) -> float:
        """Current estimate of the consumer call interval [s]."""
        return self._step_period

    @property
    def retarget_dur(self) -> float:
        """Current estimate of the retarget computation cost [s]."""
        return self._retarget_dur

    def reset(self) -> None:
        """Reset to initial state (no timing data)."""
        self._last_consume = 0.0

    def record_consume(self, timestamp: float) -> None:
        """Record a consume() call, updating the step-period estimate.

        Args:
            timestamp: :func:`time.monotonic` value at the consume call.
        """
        if self._last_consume > 0:
            dt = timestamp - self._last_consume
            if 0.001 < dt < 1.0:
                self._step_period = self._alpha * dt + (1 - self._alpha) * self._step_period
        self._last_consume = timestamp

    def record_retarget(self, duration_s: float) -> None:
        """Record a completed retarget, updating the cost estimate.

        Args:
            duration_s: Wall-clock time [s] of the retarget computation.
        """
        self._retarget_dur = self._alpha * duration_s + (1 - self._alpha) * self._retarget_dur

    def compute_sleep(self, now: float) -> float:
        """Return how long to sleep before starting the next retarget [s].

        Returns ``0.0`` when no timing data is available yet (first
        iteration), signalling that the caller should proceed immediately.

        Args:
            now: Current :func:`time.monotonic` value.
        """
        if self._last_consume <= 0:
            return 0.0
        elapsed_since_consume = now - self._last_consume
        time_until_next = self._step_period - elapsed_since_consume
        ideal = time_until_next - self._retarget_dur - self._margin_s
        return max(ideal, self._MIN_SLEEP)


@configclass
class EmaTimingEstimatorCfg(TimingEstimatorCfg):
    """Configuration for :class:`EmaTimingEstimator`."""

    class_type: type = EmaTimingEstimator
    """The estimator class to instantiate."""

    ema_alpha: float = 0.3
    """EMA smoothing factor in ``(0, 1]``.  Higher values adapt faster
    but are more sensitive to jitter.  The default 0.3 gives a half-life
    of ~2 samples (``ln2 / ln(1 / 0.7) ≈ 1.9``)."""

    margin_s: float = 0.005
    """Safety margin [s] subtracted from the predicted deadline so the
    retarget finishes slightly early."""

    def __post_init__(self):
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")


# ---------------------------------------------------------------------------
# Async retarget loop
# ---------------------------------------------------------------------------


class AsyncRetargetLoop:
    """Background loop that retargets once per :meth:`consume` call.

    The loop runs in a daemon thread and combines two mechanisms:

    1. **Consumption gate** -- after posting a result the thread blocks
       until :meth:`consume` has read it, preventing more than one
       retarget per ``advance()`` call.
    2. **Timing estimator** -- once the gate opens, the estimator sleeps
       until the optimal moment to start retargeting so inputs are as
       fresh as possible when read.

    Timeline (one cycle, time flows left → right)::

        main  ──consume()──env.step()────────────────────consume()──
                  ↑                                         ↑
                  │                                         │
        bg    ──[gate]──|                              |──[gate]──
                        ├────────sleep──────┤          │
                                            ├─retarget─┤
                                                       │
                                                    result ready

    When :meth:`consume` is called with ``block=True`` (the default), it
    waits until the background thread has produced a fresh result since
    the previous :meth:`consume`.  Because the pacer aims to finish
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

    def __init__(
        self,
        step_fn: Callable[[np.ndarray, np.ndarray | torch.Tensor | None], torch.Tensor | None],
        *,
        timing_cfg: TimingEstimatorCfg | None = None,
    ):
        """Create a new loop (stopped).  Call :meth:`start` to begin.

        Args:
            step_fn: Retarget function called each iteration with
                ``(anchor_matrix, target_T_world)`` and returning an
                action tensor or ``None``.  **Called from a background
                thread**; must not touch USD/Kit/Omniverse APIs directly.
            timing_cfg: Configuration for the timing estimator.  Defaults
                to :class:`EmaTimingEstimatorCfg` when ``None``.
        """
        if timing_cfg is None:
            timing_cfg = EmaTimingEstimatorCfg()
        if timing_cfg.class_type is None:
            raise ValueError("timing_cfg.class_type must be set to a TimingEstimator subclass")

        self._step_fn = step_fn
        self._timing: TimingEstimator = timing_cfg.class_type(timing_cfg)

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

    def start(self) -> None:
        """Start the retarget loop in a daemon thread."""
        self._stop.clear()
        with self._cond:
            self._gen = 0
            self._consumed_gen = 0
            self._latest = None
            self._exc = None
        self._timing.reset()
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

        Updates the timing estimator so the pacer can predict the next
        deadline.

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

            self._timing.record_consume(time.monotonic())

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

            # Timing estimator pace: delay reading inputs until just
            # before the predicted next consume() so that inputs are as
            # fresh as possible.
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

            self._timing.record_retarget(elapsed)

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

        Delegates to :meth:`TimingEstimator.compute_sleep` for the
        actual duration.  If there is no timing data yet (first
        iteration), returns immediately so the first result is produced
        as fast as possible.

        The consumption gate in :meth:`_run` guarantees at most one
        retarget per consume, so this method never needs to worry about
        a second retarget sneaking in.
        """
        sleep = self._timing.compute_sleep(time.monotonic())
        if sleep > 0:
            self._stop.wait(timeout=sleep)
