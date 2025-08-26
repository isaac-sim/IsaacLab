# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# NOTE: While we don't actually use the simulation app in this test, we still need to launch it
#       because warp is only available in the context of a running simulation
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import random
import time

import pytest

from isaaclab.utils.timer import (
    Instrumented,
    Timer,
    timer,
    timer_dynamic,
    toggle_timer_group,
    toggle_timer_group_display_output,
)

# number of decimal places to check
PRECISION_PLACES = 2
PRECISION_PLACES_MS = 3


@pytest.fixture(autouse=True)
def reset_timers():
    Timer.clear_all_timers()
    for group in Timer.get_group_names():
        toggle_timer_group(group, True)
        toggle_timer_group_display_output(group, True)
    yield


def test_timer_as_object():
    """Test using a `Timer` as a regular object."""

    # Make sure that the timer is enabled by default
    timer = Timer(group="test", name="test_timer_as_object", msg="Test timer as object took:", enable=True, format="us")

    toggle_timer_group_display_output("test", False)

    timer.start()
    assert abs(0 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
    time.sleep(1)
    assert abs(1 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
    timer.stop()
    assert abs(1 - timer.total_run_time) < 10 ** (-PRECISION_PLACES)

    # Make sure that if the timer is not running, the total run time is None
    timer = Timer(group="test", name="test_timer_as_object", msg="Test timer as object took:", enable=True, format="us")
    toggle_timer_group("test", False)
    timer.start()
    time.sleep(1)
    timer.stop()
    assert timer.total_run_time is None


def test_timer_as_context_manager():
    """Test using a `Timer` as a context manager."""

    with Timer() as timer:
        assert abs(0 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
        time.sleep(1)
        assert abs(1 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)


def test_timer_as_decorator():
    """Test using a `Timer` as a decorator."""

    @timer(name="add_op", msg="Math add op took:", enable=True, format="us")
    def math_add_op(a, b):
        return a + b

    for i in range(1000):
        math_add_op(i, i)

    timer_stats = Timer.get_timer_statistics("add_op")
    assert timer_stats["n"] == 1000


def test_timer_mean_and_std():
    """Test the mean and std of the timer."""

    for i in range(1000):
        with Timer(name="test_timer_mean_and_std", msg="Test timer mean and std took:", enable=True, format="us"):
            time.sleep(random.normalvariate(0.002, 0.0001))

    timer_stats = Timer.get_timer_statistics("test_timer_mean_and_std")
    assert abs(timer_stats["mean"] - 0.002) < 10 ** (-PRECISION_PLACES_MS)
    assert abs(timer_stats["std"] - 0.0001) < 10 ** (-PRECISION_PLACES_MS)
    assert timer_stats["n"] == 1000


def test_timer_global_enable_decorator_free_functions():
    """Test the global enable flag with a decorator."""

    @timer_dynamic(group="math_op", name="add_op", msg="Math add op took:", enable=True, format="us")
    def math_add_op(a, b):
        return a + b

    toggle_timer_group("math_op", True)
    toggle_timer_group_display_output("math_op", False)

    start_time = time.perf_counter()
    for i in range(100000):
        math_add_op(i, i)
    end_time = time.perf_counter()
    instrumented_time = end_time - start_time

    toggle_timer_group("math_op", False)
    start_time = time.perf_counter()
    for i in range(100000):
        math_add_op(i, i)
    end_time = time.perf_counter()
    instrumented_disabled_time = end_time - start_time

    # We cannot compare to the non instrumented time because we can't rebind local functions
    assert instrumented_disabled_time < instrumented_time


def test_timer_global_enable_decorator_class_methods():
    """Test the global enable flag with a decorator."""

    class TestClass(Instrumented):
        @timer(group="math_op", name="add_op", msg="Math add op took:", enable=True, format="us")
        def math_add_op(self, a: float, b: float):
            return a + b

        def math_add_op_non_instrumented(self, a: float, b: float):
            return a + b

    test_class = TestClass()

    toggle_timer_group("math_op", True)
    toggle_timer_group_display_output("math_op", False)

    start_time = time.perf_counter()
    for i in range(100000):
        test_class.math_add_op(i, i)
    end_time = time.perf_counter()
    instrumented_time = end_time - start_time

    toggle_timer_group("math_op", False)
    start_time = time.perf_counter()
    for i in range(100000):
        test_class.math_add_op(i, i)
    end_time = time.perf_counter()
    instrumented_disabled_time = end_time - start_time

    start_time = time.perf_counter()
    for i in range(100000):
        test_class.math_add_op_non_instrumented(i, i)
    end_time = time.perf_counter()
    non_instrumented_time = end_time - start_time

    assert instrumented_disabled_time < instrumented_time
    assert abs(non_instrumented_time - instrumented_disabled_time) < 10 ** (-PRECISION_PLACES)


def test_timer_global_enable_context_manager():
    """Test the global enable flag with a context manager."""

    # Enable the timer group, should record samples
    toggle_timer_group("math_op", True)
    for i in range(1000):
        with Timer(group="sleep_op", name="add_op", msg="Math add op took:", enable=True, format="us"):
            time.sleep(0.0001)

    timer_stats = Timer.get_timer_statistics("add_op")
    assert timer_stats["n"] == 1000

    # Disable the timer group, should not record any new samples
    toggle_timer_group("sleep_op", False)
    for i in range(1000):
        with Timer(group="sleep_op", name="add_op", msg="Math add op took:", enable=True, format="us"):
            time.sleep(0.0001)

    timer_stats = Timer.get_timer_statistics("add_op")
    assert timer_stats["n"] == 1000


def test_timer_global_enable_display_output_decorator(capsys):
    """Test the global enable display output flag."""

    @timer(group="math_op", name="add_op", msg="Math add op took:", enable=True, format="us")
    def math_add_op(a, b):
        return a + b

    toggle_timer_group("math_op", True)
    toggle_timer_group_display_output("math_op", True)

    math_add_op(1, 2)

    captured = capsys.readouterr()
    assert "Math add op took:" in captured.out

    toggle_timer_group_display_output("math_op", False)

    math_add_op(1, 2)

    captured = capsys.readouterr()
    assert "Math add op took:" not in captured.out

    toggle_timer_group_display_output("math_op", True)

    math_add_op(1, 2)

    captured = capsys.readouterr()
    assert "Math add op took:" in captured.out

    toggle_timer_group("math_op", False)

    math_add_op(1, 2)
    captured = capsys.readouterr()
    assert "Math add op took:" not in captured.out


def test_timer_global_enable_display_output_context_manager(capsys):
    """Test the global enable display output flag with a context manager."""

    toggle_timer_group("sleep_op", True)
    toggle_timer_group_display_output("sleep_op", True)

    with Timer(group="sleep_op", name="sleep_op", msg="Sleep op took:", enable=True, format="us"):
        time.sleep(0.0001)

    captured = capsys.readouterr()
    assert "Sleep op took:" in captured.out

    toggle_timer_group_display_output("sleep_op", False)

    with Timer(group="sleep_op", name="sleep_op", msg="Sleep op took:", enable=True, format="us"):
        time.sleep(0.0001)

    captured = capsys.readouterr()
    assert "Sleep op took:" not in captured.out


def test_timer_format():
    """Test the format of the timer."""
    # Check that all formats are supported
    timer_s = Timer(name="test_timer_format_s", msg="Test timer format took:", enable=True, format="s")
    timer_ms = Timer(name="test_timer_format_ms", msg="Test timer format took:", enable=True, format="ms")
    timer_us = Timer(name="test_timer_format_us", msg="Test timer format took:", enable=True, format="us")
    timer_ns = Timer(name="test_timer_format_ns", msg="Test timer format took:", enable=True, format="ns")

    # Check that an invalid format raises an error
    with pytest.raises(ValueError):
        Timer(name="test_timer_format_invalid", msg="Test timer format took:", enable=True, format="invalid")

    timer_s.start()
    time.sleep(0.1)
    timer_s.stop()
    # Check that format does not affect the total run time
    assert abs(timer_s.total_run_time - 0.1) < 10 ** (-PRECISION_PLACES)
    # Check the the multiplier is correct
    assert timer_s._unit_multiplier == 1.0

    timer_ms.start()
    time.sleep(0.1)
    timer_ms.stop()
    # Check that format does not affect the total run time
    assert abs(timer_ms.total_run_time - 0.1) < 10 ** (-PRECISION_PLACES)
    # Check the the multiplier is correct
    assert timer_ms._unit_multiplier == 1e3

    timer_us.start()
    time.sleep(0.1)
    timer_us.stop()
    # Check that format does not affect the total run time
    assert abs(timer_us.total_run_time - 0.1) < 10 ** (-PRECISION_PLACES)
    # Check the the multiplier is correct
    assert timer_us._unit_multiplier == 1e6

    timer_ns.start()
    time.sleep(0.1)
    timer_ns.stop()
    # Check that format does not affect the total run time
    assert abs(timer_ns.total_run_time - 0.1) < 10 ** (-PRECISION_PLACES)
    # Check the the multiplier is correct
    assert timer_ns._unit_multiplier == 1e9
