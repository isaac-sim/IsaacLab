# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import time

import pytest

from isaaclab.utils.timer import Timer, TimerError

# number of decimal places to check
PRECISION_PLACES = 2


def test_timer_as_object():
    """Test using a `Timer` as a regular object."""
    timer = Timer()
    timer.start()
    assert abs(0 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
    time.sleep(1)
    assert abs(1 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
    timer.stop()
    assert abs(1 - timer.total_run_time) < 10 ** (-PRECISION_PLACES)


def test_timer_as_context_manager():
    """Test using a `Timer` as a context manager."""
    with Timer() as timer:
        assert abs(0 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
        time.sleep(1)
        assert abs(1 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)


def test_timer_with_name_logs_to_global_dict():
    """Test that a named timer logs to the global timing_info dict with correct keys."""
    Timer.timing_info.clear()
    timer_name = "test_named_timer"

    with Timer(name=timer_name):
        time.sleep(0.01)

    assert timer_name in Timer.timing_info
    info = Timer.timing_info[timer_name]
    assert "last" in info
    assert "m2" in info
    assert "mean" in info
    assert "std" in info
    assert "n" in info
    assert info["n"] == 1


def test_get_timer_info_returns_last_elapsed():
    """Test that get_timer_info returns the last elapsed time (backward compatibility)."""
    Timer.timing_info.clear()
    timer_name = "test_get_info"

    with Timer(name=timer_name):
        time.sleep(0.02)

    last_time = Timer.get_timer_info(timer_name)
    assert isinstance(last_time, float)
    assert last_time >= 0.02
    assert last_time == Timer.timing_info[timer_name]["last"]


def test_get_timer_info_nonexistent_raises():
    """Test that get_timer_info raises TimerError for non-existent timer."""
    Timer.timing_info.clear()

    with pytest.raises(TimerError):
        Timer.get_timer_info("nonexistent_timer")


def test_get_timer_statistics():
    """Test get_timer_statistics returns correct keys and values for single measurement."""
    Timer.timing_info.clear()
    timer_name = "test_statistics"

    with Timer(name=timer_name):
        time.sleep(0.02)

    stats = Timer.get_timer_statistics(timer_name)
    assert "mean" in stats
    assert "std" in stats
    assert "n" in stats
    assert "last" in stats
    assert stats["n"] == 1
    # For single measurement, mean equals last
    assert stats["mean"] == stats["last"]
    # For single measurement, std should be 0
    assert stats["std"] == 0.0


def test_get_timer_statistics_nonexistent_raises():
    """Test that get_timer_statistics raises TimerError for non-existent timer."""
    Timer.timing_info.clear()

    with pytest.raises(TimerError):
        Timer.get_timer_statistics("nonexistent_timer")


def test_welford_statistics_multiple_iterations():
    """Test that Welford's algorithm correctly computes statistics over multiple iterations."""
    Timer.timing_info.clear()
    timer_name = "test_welford"
    num_iterations = 5
    sleep_duration = 0.02
    measurements = []

    for _ in range(num_iterations):
        with Timer(name=timer_name):
            time.sleep(sleep_duration)
        measurements.append(Timer.timing_info[timer_name]["last"])

    stats = Timer.get_timer_statistics(timer_name)

    # Check n incremented correctly
    assert stats["n"] == num_iterations

    # Check mean is approximately correct
    expected_mean = sum(measurements) / len(measurements)
    assert abs(stats["mean"] - expected_mean) < 1e-9

    # Check std is non-negative and reasonable
    assert stats["std"] >= 0
    # Std should be bounded by the range of measurements
    measurement_range = max(measurements) - min(measurements)
    assert stats["std"] <= measurement_range


def test_multiple_timer_instances_same_name():
    """Test that different timer instances with same name share statistics in global dict."""
    Timer.timing_info.clear()
    timer_name = "shared_timer"

    # First timer instance
    timer1 = Timer(name=timer_name)
    timer1.start()
    time.sleep(0.01)
    timer1.stop()

    assert Timer.timing_info[timer_name]["n"] == 1
    first_mean = Timer.timing_info[timer_name]["mean"]

    # Second timer instance with same name
    timer2 = Timer(name=timer_name)
    timer2.start()
    time.sleep(0.02)
    timer2.stop()

    assert Timer.timing_info[timer_name]["n"] == 2
    # Mean should have changed
    assert Timer.timing_info[timer_name]["mean"] != first_mean
