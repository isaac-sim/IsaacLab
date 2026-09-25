# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import statistics
import time

import pytest
import warp as wp

wp.init()

import isaaclab.utils.timer as timer_module
from isaaclab.utils.timer import Timer, TimerError

pytestmark = pytest.mark.unit


class _FakeClock:
    """Manually advanced replacement for :func:`time.perf_counter`."""

    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> _FakeClock:
    """Patch the timer clock so elapsed times are exact and tests do not sleep."""
    fake_clock = _FakeClock()
    monkeypatch.setattr(timer_module.time, "perf_counter", fake_clock)
    return fake_clock


def test_timer_as_object(clock):
    """Test using a `Timer` as a regular object."""
    Timer.reset()
    timer = Timer()
    timer.start()
    assert timer.time_elapsed == 0.0
    clock.advance(1.0)
    assert timer.time_elapsed == 1.0
    timer.stop()
    assert timer.total_run_time == 1.0


def test_timer_as_context_manager(clock):
    """Test using a `Timer` as a context manager."""
    Timer.reset()
    with Timer() as timer:
        assert timer.time_elapsed == 0.0
        clock.advance(1.0)
        assert timer.time_elapsed == 1.0
    assert timer.total_run_time == 1.0


def test_named_timer_statistics(clock):
    """Test that a named timer logs its single measurement to the global dict and the getters."""
    Timer.reset()
    timer_name = "test_named_timer"

    with Timer(name=timer_name):
        clock.advance(0.5)

    info = Timer.timing_info[timer_name]
    assert set(info) == {"last", "mean", "std", "n"}
    assert info["n"] == 1
    assert info["last"] == 0.5
    # For a single measurement, std is 0 and mean equals last
    assert info["std"] == 0.0
    assert info["mean"] == info["last"]
    # get_timer_info returns the last elapsed time (backward compatibility)
    last_time = Timer.get_timer_info(timer_name)
    assert isinstance(last_time, float)
    assert last_time == info["last"]
    assert Timer.get_timer_statistics(timer_name) == info


@pytest.mark.parametrize("getter", [Timer.get_timer_info, Timer.get_timer_statistics])
def test_timer_getters_nonexistent_raise(getter):
    """Test that the timer getters raise TimerError for a non-existent timer."""
    Timer.reset()

    with pytest.raises(TimerError):
        getter("nonexistent_timer")


def test_welford_statistics_multiple_iterations(clock):
    """Test that Welford's algorithm correctly computes statistics over multiple timer instances."""
    Timer.reset()
    timer_name = "test_welford"
    durations = [1.0, 2.0, 3.0, 4.0, 5.0]

    for duration in durations:
        with Timer(name=timer_name):
            clock.advance(duration)

    stats = Timer.get_timer_statistics(timer_name)
    assert stats["n"] == len(durations)
    assert stats["last"] == durations[-1]
    assert stats["mean"] == pytest.approx(statistics.mean(durations))
    assert stats["std"] == pytest.approx(statistics.stdev(durations))


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


@pytest.mark.parametrize("time_unit, multiplier", [("ms", 1e3), ("us", 1e6), ("ns", 1e9)])
def test_time_unit_multiplier(clock, time_unit, multiplier):
    """Test that time_unit correctly scales the string representation."""
    Timer.reset()

    timer = Timer(time_unit=time_unit)
    timer.start()
    clock.advance(0.5)
    timer.stop()

    # total_run_time always returns seconds
    assert timer.total_run_time == 0.5
    # __str__ should show the configured unit
    value, unit = str(timer).split()
    assert unit == time_unit
    assert float(value) == pytest.approx(0.5 * multiplier)


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
