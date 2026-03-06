# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import time

import pytest
import warp as wp

wp.init()

from isaaclab.utils.timer import Timer, TimerError

# number of decimal places to check
PRECISION_PLACES = 2


def test_timer_as_object():
    """Test using a `Timer` as a regular object."""
    Timer.reset()
    timer = Timer()
    timer.start()
    assert abs(0 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
    time.sleep(1)
    assert abs(1 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
    timer.stop()
    assert abs(1 - timer.total_run_time) < 10 ** (-PRECISION_PLACES)


def test_timer_as_context_manager():
    """Test using a `Timer` as a context manager."""
    Timer.reset()
    with Timer() as timer:
        assert abs(0 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
        time.sleep(1)
        assert abs(1 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)


def test_timer_with_name_logs_to_global_dict():
    """Test that a named timer logs to the global timing_info dict with correct keys."""
    Timer.reset()
    timer_name = "test_named_timer"

    with Timer(name=timer_name):
        time.sleep(0.01)

    assert timer_name in Timer.timing_info
    info = Timer.timing_info[timer_name]
    assert "last" in info
    assert "mean" in info
    assert "std" in info
    assert "n" in info
    assert info["n"] == 1


def test_get_timer_info_returns_last_elapsed():
    """Test that get_timer_info returns the last elapsed time (backward compatibility)."""
    Timer.reset()
    timer_name = "test_get_info"

    with Timer(name=timer_name):
        time.sleep(0.02)

    last_time = Timer.get_timer_info(timer_name)
    assert isinstance(last_time, float)
    assert last_time >= 0.02
    assert last_time == Timer.timing_info[timer_name]["last"]


def test_get_timer_info_nonexistent_raises():
    """Test that get_timer_info raises TimerError for non-existent timer."""
    Timer.reset()

    with pytest.raises(TimerError):
        Timer.get_timer_info("nonexistent_timer")


def test_get_timer_statistics():
    """Test get_timer_statistics returns correct keys and values for single measurement."""
    Timer.reset()
    timer_name = "test_statistics"

    with Timer(name=timer_name):
        time.sleep(0.02)

    stats = Timer.get_timer_statistics(timer_name)
    assert "mean" in stats
    assert "std" in stats
    assert "n" in stats
    assert stats["n"] == 1
    # For single measurement, std should be 0
    assert stats["std"] == 0.0


def test_get_timer_statistics_nonexistent_raises():
    """Test that get_timer_statistics raises TimerError for non-existent timer."""
    Timer.reset()

    with pytest.raises(TimerError):
        Timer.get_timer_statistics("nonexistent_timer")


def test_welford_statistics_multiple_iterations():
    """Test that Welford's algorithm correctly computes statistics over multiple iterations."""
    Timer.reset()
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
    Timer.reset()
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


def test_global_enable_toggle():
    """Test that Timer.enable globally disables all timers."""
    Timer.reset()
    Timer.enable = True

    try:
        # Create timer while globally disabled
        Timer.enable = False
        timer = Timer(name="disabled_timer")
        timer.start()
        time.sleep(0.01)
        timer.stop()

        # Should not have recorded anything
        assert "disabled_timer" not in Timer.timing_info
        assert timer.total_run_time == 0.0
    finally:
        Timer.enable = True


def test_instance_enable_toggle():
    """Test that per-instance enable=False disables a single timer."""
    Timer.reset()

    timer = Timer(name="instance_disabled", enable=False)
    timer.start()
    time.sleep(0.01)
    timer.stop()

    assert "instance_disabled" not in Timer.timing_info
    assert timer.total_run_time == 0.0


def test_enable_display_output(capsys):
    """Test that Timer.enable_display_output controls context manager print output."""
    Timer.reset()
    Timer.enable_display_output = True

    try:
        # With display enabled
        with Timer(msg="visible"):
            time.sleep(0.01)
        captured = capsys.readouterr()
        assert "visible" in captured.out

        # With display disabled
        Timer.enable_display_output = False
        with Timer(msg="hidden"):
            time.sleep(0.01)
        captured = capsys.readouterr()
        assert captured.out == ""
    finally:
        Timer.enable_display_output = True


def test_time_unit_multiplier():
    """Test that time_unit correctly scales the string representation."""
    Timer.reset()

    timer = Timer(time_unit="ms")
    timer.start()
    time.sleep(0.01)
    timer.stop()

    # total_run_time always returns seconds
    assert timer.total_run_time >= 0.01
    # __str__ should show milliseconds
    output = str(timer)
    assert "ms" in output
    # The numeric value should be >= 10 (0.01s = 10ms)
    numeric_part = float(output.split()[0])
    assert numeric_part >= 10.0


def test_time_unit_us_and_ns():
    """Test microsecond and nanosecond time units."""
    timer_us = Timer(time_unit="us")
    timer_us.start()
    time.sleep(0.001)
    timer_us.stop()
    assert "us" in str(timer_us)

    timer_ns = Timer(time_unit="ns")
    timer_ns.start()
    time.sleep(0.001)
    timer_ns.stop()
    assert "ns" in str(timer_ns)


def test_invalid_time_unit_raises():
    """Test that an invalid time_unit raises ValueError."""
    with pytest.raises(ValueError, match="Invalid time_unit"):
        Timer(time_unit="hours")


def test_reset_specific_timer():
    """Test that Timer.reset(name) only resets the specified timer."""
    Timer.reset()

    with Timer(name="keep"):
        time.sleep(0.01)
    with Timer(name="remove"):
        time.sleep(0.01)

    assert "keep" in Timer.timing_info
    assert "remove" in Timer.timing_info

    Timer.reset("remove")

    assert "keep" in Timer.timing_info
    assert "remove" not in Timer.timing_info


def test_get_timer_statistics_includes_last():
    """Test that get_timer_statistics includes the 'last' key."""
    Timer.reset()

    with Timer(name="stats_last"):
        time.sleep(0.01)

    stats = Timer.get_timer_statistics("stats_last")
    assert "last" in stats
    assert stats["last"] >= 0.01
    # For single measurement, mean equals last
    assert stats["mean"] == stats["last"]
