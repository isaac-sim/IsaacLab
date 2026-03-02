# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for a timer class that can be used for performance measurements.

Note:
    This module has a hard dependency on `warp` because the :class:`Timer` calls
    ``wp.synchronize()`` on stop to flush pending GPU work before sampling the clock.
    Since IsaacLab workloads are predominantly GPU-bound, an unsynchronized timer would
    under-report wall time by returning before device kernels have finished executing.
"""

from __future__ import annotations

import math
import time
from contextlib import ContextDecorator
from typing import Any, ClassVar, Literal

import warp as wp


class TimerError(Exception):
    """A custom exception used to report errors in use of :class:`Timer` class."""

    pass


class Timer(ContextDecorator):
    """A timer for performance measurements.

    A class to keep track of time for performance measurement.
    It allows timing via context managers and decorators as well.

    It uses the `time.perf_counter` function to measure time. This function
    returns the number of seconds since the epoch as a float. It has the
    highest resolution available on the system.

    As a regular object:

    .. code-block:: python

        import time

        from isaaclab.utils.timer import Timer

        timer = Timer()
        timer.start()
        time.sleep(1)
        print(1 <= timer.time_elapsed <= 2)  # Output: True

        time.sleep(1)
        timer.stop()
        print(2 <= timer.total_run_time)  # Output: True

    As a context manager:

    .. code-block:: python

        import time

        from isaaclab.utils.timer import Timer

        with Timer() as timer:
            time.sleep(1)
            print(1 <= timer.time_elapsed <= 2)  # Output: True

    Reference: https://gist.github.com/sumeet/1123871
    """

    timing_info: ClassVar[dict[str, dict[str, float]]] = dict()
    """Dictionary for storing the elapsed time per timer instances globally.

    This dictionary logs the timer information. The keys are the names given to the timer class
    at its initialization. If no :attr:`name` is passed to the constructor, no time
    is recorded in the dictionary.
    """

    _welford_state: ClassVar[dict[str, float]] = dict()
    """Internal accumulator (m2) for Welford's online algorithm, keyed by timer name."""

    enable: ClassVar[bool] = True
    """Whether to enable the timer."""

    enable_display_output: ClassVar[bool] = True
    """Whether to enable the display output."""

    _UNIT_MULTIPLIERS: ClassVar[dict[str, float]] = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    """Mapping from time unit string to multiplier (seconds -> unit)."""

    def __init__(
        self,
        msg: str | None = None,
        name: str | None = None,
        enable: bool = True,
        time_unit: Literal["s", "ms", "us", "ns"] = "s",
    ):
        """Initializes the timer.

        Args:
            msg: The message to display when using the timer
                class in a context manager. Defaults to None.
            name: The name to use for logging times in a global
                dictionary. Defaults to None.
            enable: Whether to enable the timer. Defaults to True.
            time_unit: The unit to use for the elapsed time. Defaults to "s".
        """
        self._msg = msg
        self._name = name
        self._start_time = None
        self._stop_time = None
        self._elapsed_time = None
        self._enable = enable if Timer.enable else False

        if time_unit not in Timer._UNIT_MULTIPLIERS:
            raise ValueError(f"Invalid time_unit, {time_unit} is not in {list(Timer._UNIT_MULTIPLIERS)}")

        self._format = time_unit
        self._multiplier = Timer._UNIT_MULTIPLIERS[time_unit]

    def __str__(self) -> str:
        """A string representation of the class object.

        Returns:
            A string containing the elapsed time.
        """
        return f"{(self.total_run_time * self._multiplier):0.6f} {self._format}"

    """
    Properties
    """

    @property
    def time_elapsed(self) -> float:
        """The number of seconds that have elapsed since this timer started timing.

        Note:
            This always returns seconds regardless of the configured ``time_unit``.
            It is used for checking how much time has elapsed while the timer is still running.
        """
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def total_run_time(self) -> float:
        """The number of seconds that elapsed from when the timer started to when it ended.

        Note:
            This always returns seconds regardless of the configured ``time_unit``.
        """
        if self._elapsed_time is None:
            return 0.0
        return self._elapsed_time

    """
    Operations
    """

    def start(self):
        """Start timing."""
        if not self._enable:
            return

        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop timing."""
        if not self._enable:
            return

        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Synchronize the device to make sure we time the whole operation
        wp.synchronize()

        # Get the elapsed time
        self._stop_time = time.perf_counter()
        self._elapsed_time = self._stop_time - self._start_time
        self._start_time = None

        if self._name is not None:
            self._update_welford(self._elapsed_time)

    def _update_welford(self, value: float):
        """Update the running statistics using Welford's online algorithm."""
        info = Timer.timing_info.get(self._name)
        if info is None:
            Timer.timing_info[self._name] = {"mean": value, "std": 0.0, "n": 1, "last": value}
            Timer._welford_state[self._name] = 0.0
        else:
            m2 = Timer._welford_state[self._name]
            n = info["n"] + 1
            delta = value - info["mean"]
            mean = info["mean"] + delta / n
            delta2 = value - mean
            m2 = m2 + delta * delta2
            std = math.sqrt(m2 / (n - 1)) if n > 1 else 0.0
            Timer.timing_info[self._name] = {"mean": mean, "std": std, "n": n, "last": value}
            Timer._welford_state[self._name] = m2

    """
    Context managers
    """

    def __enter__(self) -> Timer:
        """Start timing and return this `Timer` instance."""
        self.start()
        return self

    def __exit__(self, *exc_info: Any):
        """Stop timing."""
        self.stop()
        # print message
        if self._enable:
            if (self._msg is not None) and (Timer.enable_display_output):
                parts = [f"Last: {(self._elapsed_time * self._multiplier):0.6f} {self._format}"]
                if self._name is not None:
                    info = Timer.timing_info[self._name]
                    parts.append(f"Mean: {(info['mean'] * self._multiplier):0.6f} {self._format}")
                    parts.append(f"Std: {(info['std'] * self._multiplier):0.6f} {self._format}")
                    parts.append(f"N: {info['n']}")
                print(self._msg, ", ".join(parts))

    """
    Static Methods
    """

    @staticmethod
    def reset(name: str | None = None):
        """Reset statistics for a named timer, or all timers if name is None.

        Args:
            name: Name of the timer to reset. If None, resets all timers.
        """
        if name is None:
            Timer.timing_info.clear()
            Timer._welford_state.clear()
        else:
            Timer.timing_info.pop(name, None)
            Timer._welford_state.pop(name, None)

    @staticmethod
    def get_timer_info(name: str) -> float:
        """Retrieves the time logged in the global dictionary
            based on name.

        Args:
            name: Name of the the entry to be retrieved.

        Raises:
            TimerError: If name doesn't exist in the log.

        Returns:
            A float containing the time logged if the name exists.
        """
        if name not in Timer.timing_info:
            raise TimerError(f"Timer {name} does not exist")
        return Timer.timing_info.get(name)["last"]

    @staticmethod
    def get_timer_statistics(name: str) -> dict[str, float]:
        """Retrieves the timer statistics logged in the global dictionary
            based on name.

        Args:
            name: Name of the entry to be retrieved.

        Raises:
            TimerError: If name doesn't exist in the log.

        Returns:
            A dictionary containing the mean, std, and n for the named timer.
        """

        if name not in Timer.timing_info:
            raise TimerError(f"Timer {name} does not exist")

        keys = ["mean", "std", "n", "last"]

        return {k: Timer.timing_info[name][k] for k in keys}
