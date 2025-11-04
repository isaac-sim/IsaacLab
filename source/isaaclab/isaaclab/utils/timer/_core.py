# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

    .. caution::
        The Timer class is not optimized for performance. For decorators, use the :func:`timer` decorator instead.
        The :func:`timer` decorator turns into a no-op when the timer is disabled, adding no overhead.

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

    As a global object:

    .. code-block:: python

        import time

        @timer("math_op", name="add_op", msg="Math add op took:", enable=True, format="us")
        def math_add_op(a, b):
            return a + b

        # Enable the timer (enabled by default)
        toggle_timer_group("math_op", True)
        # Enable the display output (enabled by default)
        toggle_timer_group_display_output("math_op", True)

        math_add_op(1, 2)
        # Disable the display output
        toggle_timer_group_display_output("math_op", False)
        math_add_op(1, 2)
        # Get the statistics
        Timer.get_timer_statistics("add_op")

        # Disable the timer
        toggle_timer_group("math_op", False)

        math_add_op(1, 2)
        Timer.get_timer_statistics("add_op")


    Initial reference: https://gist.github.com/sumeet/1123871
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

    group_enable: ClassVar[dict[str, bool]] = dict()
    """Whether to enable the timer for a given group.

    This dictionary allows to enable/disable the timer for a given group.
    If the group is not in the dictionary, the timer is enabled by default.
    """

    group_display_output: ClassVar[dict[str, bool]] = dict()
    """Whether to display the output for a given group.

    This dictionary allows to enable/disable the display output for a given group.
    If the group is not in the dictionary, the display output is enabled by default.
    """

    def __init__(
        self,
        msg: str | None = None,
        name: str | None = None,
        enable: bool = True,
        format: Literal["s", "ms", "us", "ns"] = "s",
        group: str | None = None,
    ):
        """Initializes the timer.

        Note: The format flag is only used for display purposes, and does not affect the values stored in the dictionary.
        This is done to avoid any confusion with the unit of the values stored in the dictionary. All the values
        in the dictionary are stored in seconds.

        Args:
            msg: The message to display when using the timer
                class in a context manager. Defaults to None.
            name: The name to use for logging times in a global
                dictionary. Defaults to None.
            enable: Whether to enable the timer. Defaults to True.
            format: The format to use for the elapsed time. Defaults to "s".
            group: The group to use for the timer. Defaults to None.
        """
        self._msg = msg
        self._start_time = None
        self._stop_time = None
        self._elapsed_time = None
        self._enable = enable
        self._format = format
        self._group = group
        self._name = name

        # Check if the format is valid
        if format not in ["s", "ms", "us", "ns"]:
            raise ValueError(f"Invalid format, {format} is not in [s, ms, us, ns]")

        # Convert the format to a multiplier
        self._unit_multiplier = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[format]

        # Online welford's algorithm to compute the mean and std of the elapsed time
        self._mean = 0.0
        self._m2 = 0.0
        self._std = 0.0
        self._n = 0

    @classmethod
    def register_name(cls, name: str | None = None):
        """Register the name of the timer.

        Args:
            name: The name to register.
        """
        # If no name is provided, use a default name
        if name is None:
            name = "unnamed_timer_instance"

        # Check if the name is already registered, if it is, append a number to the name until it is unique
        if name in Timer.timing_info:
            i = 1
            while f"{name}_{i}" in Timer.timing_info:
                i += 1
            name = f"{name}_{i}"
        Timer.timing_info[name] = {
            "last": 0.0,
            "m2": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "n": 0,
        }
        return name

    def __str__(self) -> str:
        """A string representation of the class object.

        Returns:
            A string containing the elapsed time.
        """
        return f"{(self.time_elapsed * self._unit_multiplier):0.6f} {self._format}"

    """
    Properties
    """

    @property
    def time_elapsed(self) -> float:
        """The number of seconds that have elapsed since this timer started timing.

        Note:
            This is used for checking how much time has elapsed while the timer is still running.
        """
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

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
        if (not self._enable) or (self._group is not None and not Timer.group_enable.get(self._group, True)):
            return

        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop timing."""
        if not self._enable or (self._group is not None and not Timer.group_enable.get(self._group, True)):
            return

        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Synchronize the device to make sure we time the whole operation
        try:
            wp.synchronize_device()
        except:
            pass

        # Get the elapsed time
        self._stop_time = time.perf_counter()
        self._elapsed_time = self._stop_time - self._start_time
        self._start_time = None

        if (self._name is not None) and (self._enable):
            # Update the welford's algorithm
            self._update_welford(self._elapsed_time)

            # Update the timing info
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

        # Update the std
        self._std = math.sqrt(self._m2 / self._n)

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
        if (
            self._enable
            and (Timer.group_display_output.get(self._group, True))
            and (Timer.group_enable.get(self._group, True))
        ):
            if self._msg is not None:
                print(
                    self._msg,
                    f"Last: {(self._elapsed_time * self._unit_multiplier):0.6f} {self._format}, "
                    f"Mean: {(self._mean * self._unit_multiplier):0.6f} {self._format}, "
                    f"Std: {(self._std * self._unit_multiplier):0.6f} {self._format}, "
                    f"N: {self._n}",
                )
    
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

        return {k: Timer.timing_info[name][k] for k in keys}

    """
    Class methods
    """

    @classmethod
    def set_group(cls, group: str, enabled: bool) -> None:
        """Set the enable flag for a given group.

        Args:
            group: The group to set the enable flag for.
            enabled: The enable flag to set.
        """
        cls.group_enable[group] = bool(enabled)

    @classmethod
    def set_group_display_output(cls, group: str, enabled: bool) -> None:
        """Set the display output flag for a given group.

        Args:
            group: The group to set the display output flag for.
            enabled: The display output flag to set.
        """
        cls.group_display_output[group] = bool(enabled)

    @classmethod
    def clear_all_timers(cls) -> None:
        """Clear all timers.

        Resets the global dictionary to its initial state.
        """
        cls.timing_info.clear()
        cls.group_enable.clear()
        cls.group_display_output.clear()

    @classmethod
    def clear_timer(cls, name: str) -> None:
        """Clear a timer.

        Note: This does not affect the enable flag for the group.

        Args:
            name: The name of the timer to clear.
        """
        cls.timing_info.pop(name, None)

    @classmethod
    def get_group_names(cls) -> list[str]:
        """Get the names of all groups.

        Returns:
            A list of all group names.
        """
        return list(cls.group_enable.keys())
