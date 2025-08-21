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

from isaaclab.utils.timer import Timer, TimerError

# number of decimal places to check
PRECISION_PLACES = 2
PRECISION_PLACES_MS = 4


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


def test_timer_mean_and_std():
    """Test the mean and std of the timer."""

    for i in range(1000):
        with Timer(name="test_timer_mean_and_std", msg="Test timer mean and std took:", enable=True, format="us"):
            time.sleep(random.normalvariate(0.002, 0.0001))

    timer_stats = Timer.get_timer_statistics("test_timer_mean_and_std")
    assert abs(timer_stats["mean"] - 0.002) < 10 ** (-PRECISION_PLACES_MS)
    assert abs(timer_stats["std"] - 0.0001) < 10 ** (-PRECISION_PLACES_MS)
    assert timer_stats["n"] == 1000


def test_timer_global_enable():
    """Test the global enable flag."""
    Timer.global_enable = False
    timer = Timer(name="test_timer_global_enable", msg="Test timer global enable took:", enable=True, format="us")
    timer.start()
    time.sleep(1)
    timer.stop()

    with pytest.raises(TimerError):
        timer.time_elapsed


def test_timer_global_enable_display_output(capsys):
    """Test the global enable display output flag."""
    Timer.global_enable = True
    with Timer(
        name="test_timer_global_enable_display_output",
        msg="Test timer global enable display output took:",
        enable=True,
        format="us",
    ):
        time.sleep(1)

    captured = capsys.readouterr()
    assert "Test timer global enable display output took:" in captured.out

    Timer.enable_display_output = False
    with Timer(
        name="test_timer_global_enable_display_output",
        msg="Test timer global enable display output took:",
        enable=True,
        format="us",
    ):
        time.sleep(1)

    captured = capsys.readouterr()
    assert "Test timer global enable display output took:" not in captured.out


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
