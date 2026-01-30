# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for a timer class that can be used for performance measurements."""

from __future__ import annotations

import math
import time
from contextlib import ContextDecorator
from typing import Any, ClassVar


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
        print(2 <= stopwatch.total_run_time)  # Output: True

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

    In each of the dictionaries, we store the following information:
    - last: The last elapsed time
    - m2: The sum of squares of differences from the current mean (Intermediate value in Welford's Algorithm)
    - mean: The mean of the elapsed time
    - std: The standard deviation of the elapsed time
    - n: The number of samples
    """

    def __init__(self, msg: str | None = None, name: str | None = None):
        """Initializes the timer.

        Args:
            msg: The message to display when using the timer
                class in a context manager. Defaults to None.
            name: The name to use for logging times in a global
                dictionary. Defaults to None.
        """
        self._msg = msg
        self._name = name
        self._start_time = None
        self._stop_time = None
        self._elapsed_time = None

        # Online welford's algorithm to compute the mean and std of the elapsed time
        self._mean = 0.0
        self._m2 = 0.0
        self._std = 0.0
        self._n = 0

    def __str__(self) -> str:
        """A string representation of the class object.

        Returns:
            A string containing the elapsed time.
        """
        return f"{self.time_elapsed:0.6f} seconds"

    """
    Properties
    """

    @property
    def time_elapsed(self) -> float:
        """The number of seconds that have elapsed since this timer started timing.

        Note:
            This is used for checking how much time has elapsed while the timer is still running.
        """
        return time.perf_counter() - self._start_time

    @property
    def total_run_time(self) -> float:
        """The number of seconds that elapsed from when the timer started to when it ended."""
        return self._elapsed_time

    """
    Operations
    """

    def start(self):
        """Start timing."""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop timing."""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        self._stop_time = time.perf_counter()
        self._elapsed_time = self._stop_time - self._start_time
        self._start_time = None

        if self._name:
            # Update the welford's algorithm
            self._update_welford(self._elapsed_time)
            # Store the information in the global dictionary
            Timer.timing_info[self._name] = {
                "last": self._elapsed_time,
                "m2": self._m2,
                "mean": self._mean,
                "std": self._std,
                "n": self._n,
            }

    """
    Internal helpers.
    """

    def _update_welford(self, value: float):
        """Update the welford's algorithm with a new value.

        This algorithm computes the mean and standard deviation of the elapsed time in a numerically stable way.
        It may become numerically unstable if n becomes very large.

        Note: We use the global dictionary to retrieve the current values. We do this to make the timer
        instances stateful.

        Args:
            value: The new value to add to the statistics.
        """
        try:
            self._n = Timer.timing_info[self._name]["n"] + 1
            delta = value - Timer.timing_info[self._name]["mean"]
            self._mean = Timer.timing_info[self._name]["mean"] + delta / self._n
            delta2 = value - self._mean
            self._m2 = Timer.timing_info[self._name]["m2"] + delta * delta2
        except KeyError:
            self._n = 1
            self._mean = value
            self._m2 = 0.0

        # Update the std (sample standard deviation with Bessel's correction)
        if self._n > 1:
            self._std = math.sqrt(self._m2 / (self._n - 1))
        else:
            self._std = 0.0

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
        if self._msg is not None:
            print(self._msg, f": {self._elapsed_time:0.6f} seconds")

    """
    Static Methods
    """

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
        # Non-breaking change: return the last elapsed time
        return Timer.timing_info.get(name)["last"]

    @staticmethod
    def get_timer_statistics(name: str) -> dict[str, float]:
        """Retrieves the statistics of the time logged in the global dictionary based on name.

        Returns a dictionary containing the mean, standard deviation, and number of samples as
        well as the last measurement. Available keys are:
        - mean: The mean of the elapsed time
        - std: The standard deviation of the elapsed time
        - n: The number of samples
        - last: The last elapsed time

        Args:
            name: Name of the the entry to be retrieved.

        Raises:
            TimerError: If name doesn't exist in the log.

        Returns:
            A dictionary containing the time logged for all timers.
        """

        if name not in Timer.timing_info:
            raise TimerError(f"Timer {name} does not exist")

        keys = ["mean", "std", "n", "last"]

        return {k: Timer.timing_info.get(name)[k] for k in keys}
