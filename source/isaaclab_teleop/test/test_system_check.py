# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for the teleop workstation capability check.

These tests exercise pure logic (no Omniverse/Isaac Sim stack required). Every
hardware probe is monkeypatched so the thresholds are evaluated against fixed
synthetic values -- the results must not depend on the machine running the
suite, and the CPU microbenchmark must not actually run.
"""

from __future__ import annotations

import pytest
from isaaclab_teleop import system_check
from isaaclab_teleop.system_check import SystemCheckItem, SystemCheckResult, check_system_requirements


class _FakeFreq:
    def __init__(self, max_mhz: float):
        self.max = max_mhz


class _FakeGpuProps:
    def __init__(self, name: str, total_memory: int, major: int, minor: int):
        self.name = name
        self.total_memory = total_memory
        self.major = major
        self.minor = minor


class _FakeVirtualMemory:
    def __init__(self, total: int):
        self.total = total


def _at_spec(monkeypatch, **overrides):
    """Patch every probe to reference-machine values, then apply *overrides*.

    Returns the item name -> item mapping produced by
    :func:`check_system_requirements` under those conditions.
    """
    values = {
        "score": 1.0,
        "governor": "performance",
        "clock_mhz": 5362.0,
        "cores": 24,
        "cuda_available": True,
        "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "vram_bytes": 102 * 10**9,
        "capability": (12, 0),
        "driver": "580.173.02",
        "ram_bytes": 64 * 2**30,
        "machine": "x86_64",
    }
    values.update(overrides)

    import psutil
    import torch

    monkeypatch.setattr(system_check, "measure_cpu_single_thread_score", lambda: values["score"])
    monkeypatch.setattr(system_check, "_read_cpu_governor", lambda: values["governor"])
    monkeypatch.setattr(system_check, "_read_driver_version", lambda: values["driver"])
    monkeypatch.setattr(
        psutil, "cpu_freq", lambda: None if values["clock_mhz"] is None else _FakeFreq(values["clock_mhz"])
    )
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: values["cores"])
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVirtualMemory(values["ram_bytes"]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: values["cuda_available"])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: _FakeGpuProps(
            values["gpu_name"], values["vram_bytes"], values["capability"][0], values["capability"][1]
        ),
    )
    monkeypatch.setattr(system_check.platform, "machine", lambda: values["machine"])

    result = check_system_requirements()
    return result, {item.name: item for item in result.items}


class TestSystemCheckAtSpec:
    """The reference workstation must pass cleanly."""

    def test_reference_machine_passes_every_item(self, monkeypatch):
        result, items = _at_spec(monkeypatch)
        assert result.passed
        assert result.failures == ()
        assert not any(item.skipped for item in result.items)

    def test_passing_result_produces_no_notice_items(self, monkeypatch):
        result, _ = _at_spec(monkeypatch)
        assert result.to_message()["message"]["items"] == []


class TestSystemCheckThresholds:
    """Each threshold fails independently and identifies the right item."""

    @pytest.mark.parametrize(
        "overrides,failing_item",
        [
            ({"score": 0.5}, "CPU single-thread"),
            ({"governor": "powersave"}, "CPU governor"),
            ({"clock_mhz": 3200.0}, "CPU boost clock"),
            ({"cores": 4}, "CPU physical cores"),
            ({"vram_bytes": 16 * 10**9}, "GPU memory"),
            ({"capability": (8, 6)}, "GPU architecture"),
            ({"driver": "550.100.01"}, "NVIDIA driver"),
            ({"ram_bytes": 32 * 2**30}, "System memory"),
            ({"machine": "aarch64"}, "CPU architecture"),
        ],
    )
    def test_below_threshold_fails_only_that_item(self, monkeypatch, overrides, failing_item):
        result, _ = _at_spec(monkeypatch, **overrides)
        assert not result.passed
        assert [item.name for item in result.failures] == [failing_item]

    def test_score_exactly_at_threshold_passes(self, monkeypatch):
        # Guards against an off-by-one flip from >= to >, which would warn on
        # machines that exactly meet the documented bar.
        result, _ = _at_spec(monkeypatch, score=system_check.CPU_SCORE_MIN)
        assert result.passed

    def test_missing_gpu_fails_with_a_single_item(self, monkeypatch):
        result, _ = _at_spec(monkeypatch, cuda_available=False)
        assert not result.passed
        assert [item.name for item in result.failures] == ["NVIDIA GPU"]

    def test_core_count_floor_is_permissive(self, monkeypatch):
        # An 8-core high-IPC part must pass: Pink IK is single-thread bound, so
        # gating near the reference CPU's 24 cores would fail machines that
        # teleoperate well.
        result, _ = _at_spec(monkeypatch, cores=8)
        assert result.passed


class TestSystemCheckSkips:
    """An unavailable probe must never manufacture a warning."""

    @pytest.mark.parametrize(
        "overrides,skipped_item",
        [
            ({"score": None}, "CPU single-thread"),
            ({"governor": None}, "CPU governor"),
            ({"clock_mhz": None}, "CPU boost clock"),
            ({"driver": None}, "NVIDIA driver"),
        ],
    )
    def test_unavailable_probe_is_skipped_not_failed(self, monkeypatch, overrides, skipped_item):
        result, items = _at_spec(monkeypatch, **overrides)
        assert items[skipped_item].skipped
        assert items[skipped_item].passed
        assert result.passed

    def test_unparsable_driver_version_is_skipped(self, monkeypatch):
        result, items = _at_spec(monkeypatch, driver="unknown")
        assert items["NVIDIA driver"].skipped
        assert result.passed

    def test_probe_raising_does_not_abort_the_check(self, monkeypatch):
        # A broken probe group must drop only itself: the check runs at teleop
        # startup and must never prevent a session from starting.
        def _boom():
            raise RuntimeError("probe exploded")

        _at_spec(monkeypatch)
        monkeypatch.setattr(system_check, "_check_gpu", _boom)
        result = check_system_requirements()

        assert result.items
        assert not any(item.name.startswith("GPU") for item in result.items)


class TestSystemCheckMessage:
    """The wire envelope the XR client consumes."""

    def test_message_carries_only_failing_items(self, monkeypatch):
        result, _ = _at_spec(monkeypatch, governor="powersave", cores=4)
        payload = result.to_message()

        assert payload["type"] == "system_notice"
        assert payload["message"]["level"] == "warning"
        assert {item["name"] for item in payload["message"]["items"]} == {"CPU governor", "CPU physical cores"}

    def test_governor_failure_carries_the_fix_command(self, monkeypatch):
        result, _ = _at_spec(monkeypatch, governor="powersave")
        (item,) = result.to_message()["message"]["items"]
        assert item["detail"] == system_check.CPU_GOVERNOR_FIX
        assert item["actual"] == "powersave"

    def test_message_is_json_serializable(self, monkeypatch):
        import json

        result, _ = _at_spec(monkeypatch, score=0.4)
        assert json.loads(json.dumps(result.to_message()))["type"] == "system_notice"


class TestSystemCheckFormatting:
    """Terminal output."""

    def test_table_marks_each_status(self):
        result = SystemCheckResult(
            items=(
                SystemCheckItem("Passing", True, "10", ">= 5"),
                SystemCheckItem("Failing", False, "1", ">= 5", detail="do the thing"),
                SystemCheckItem("Unknown", True, "n/a", ">= 5", skipped=True),
            )
        )
        table = result.format_table()

        assert "[  ok] Passing" in table
        assert "[WARN] Failing" in table
        assert "[skip] Unknown" in table
        assert "-> do the thing" in table
        assert "1 requirement(s) below" in table

    def test_detail_is_hidden_for_passing_items(self):
        result = SystemCheckResult(items=(SystemCheckItem("Passing", True, "10", ">= 5", detail="noise"),))
        assert "noise" not in result.format_table()
        assert "PASSED" in result.format_table()

    def test_empty_result_does_not_crash(self):
        assert "no requirements" in SystemCheckResult().format_table()


class TestCpuBenchmark:
    """The microbenchmark itself, which the other tests stub out."""

    def test_score_is_none_without_a_calibrated_reference(self, monkeypatch):
        monkeypatch.setattr(system_check, "_CPU_REFERENCE_NS_PER_ITER", 0.0)
        assert system_check.measure_cpu_single_thread_score() is None

    def test_high_variance_reading_is_rejected(self, monkeypatch):
        # A busy machine produces a wide spread; the check must decline to judge
        # rather than emit a false warning.
        timings = iter([100.0, 100.0, 1000.0])
        monkeypatch.setattr(system_check.time, "perf_counter", lambda: next(timings, 1000.0) * 1e-9)
        assert system_check._measure_cpu_ns_per_iter() is None

    def test_benchmark_produces_a_positive_score(self):
        # Real run, no stubs: guards against the kernel or reference constant
        # being changed in a way that makes the score nonsensical.
        score = system_check.measure_cpu_single_thread_score()
        if score is None:
            pytest.skip("CPU benchmark variance too high on this machine")
        assert 0.0 < score < 100.0
