# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import statistics
import time

import pytest
import warp as wp

from isaaclab.utils.timer import Timer, TimerError

pytestmark = pytest.mark.unit

wp.init()


@pytest.fixture(autouse=True)
def timer_state(monkeypatch):
    monkeypatch.setattr(Timer, "timing_info", {})
    monkeypatch.setattr(Timer, "_welford_state", {})
    monkeypatch.setattr(Timer, "enable", True)
    monkeypatch.setattr(Timer, "enable_display_output", True)


def test_timer_as_object():
    timer = Timer()
    assert timer.time_elapsed == timer.total_run_time == 0.0
    timer.start()
    elapsed = timer.time_elapsed
    time.sleep(0.001)
    assert timer.time_elapsed >= elapsed + 0.001
    timer.stop()
    assert timer.total_run_time >= elapsed + 0.001


def test_named_timer_statistics():
    measurements = []
    for _ in range(5):
        with Timer(name="shared") as timer:
            assert timer.time_elapsed >= 0.0
        measurements.append(timer.total_run_time)
        stats = Timer.get_timer_statistics("shared")
        assert set(stats) == {"last", "mean", "std", "n"}
        assert stats["last"] == Timer.get_timer_info("shared") == measurements[-1]
        assert stats["n"] == len(measurements)
        assert stats["mean"] == pytest.approx(statistics.mean(measurements))
        expected_std = statistics.stdev(measurements) if len(measurements) > 1 else 0.0
        assert stats["std"] == pytest.approx(expected_std)


@pytest.mark.parametrize("getter", [Timer.get_timer_info, Timer.get_timer_statistics])
def test_missing_timer_raises(getter):
    with pytest.raises(TimerError):
        getter("missing")


@pytest.mark.parametrize("global_disable", [True, False], ids=["global", "instance"])
def test_disabled_timer(monkeypatch, global_disable):
    monkeypatch.setattr(Timer, "enable", not global_disable)
    with Timer(name="disabled", enable=global_disable) as timer:
        assert timer.time_elapsed == 0.0
    assert timer.total_run_time == 0.0
    assert "disabled" not in Timer.timing_info


@pytest.mark.parametrize("display", [True, False])
def test_display_output(monkeypatch, capsys, display):
    monkeypatch.setattr(Timer, "enable_display_output", display)
    with Timer(msg="elapsed"):
        pass
    output = capsys.readouterr().out
    if display:
        assert "elapsed" in output
    else:
        assert output == ""


@pytest.mark.parametrize(("unit", "multiplier"), [("s", 1.0), ("ms", 1e3), ("us", 1e6), ("ns", 1e9)])
def test_time_unit(unit, multiplier):
    with Timer(time_unit=unit) as timer:
        pass
    value, suffix = str(timer).split()
    assert suffix == unit
    assert float(value) == pytest.approx(timer.total_run_time * multiplier, abs=1e-6)


def test_invalid_time_unit_raises():
    with pytest.raises(ValueError, match="Invalid time_unit"):
        Timer(time_unit="hours")


def test_reset():
    for name in ("keep", "remove"):
        with Timer(name=name):
            pass

    Timer.reset("remove")
    assert set(Timer.timing_info) == {"keep"}
    with pytest.raises(TimerError):
        Timer.get_timer_info("remove")

    Timer.reset()
    assert Timer.timing_info == {}
