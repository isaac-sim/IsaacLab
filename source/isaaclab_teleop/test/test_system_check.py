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

import json

import pytest
from isaaclab_teleop import system_check
from isaaclab_teleop.system_check import SystemCheckItem, SystemCheckResult, check_system_requirements


class _FakeFreq:
    def __init__(self, max_mhz: float):
        self.max = max_mhz


class _FakeWarpDevice:
    """Stand-in for ``warp.Device`` exposing only what the check reads."""

    def __init__(self, alias: str, name: str, total_memory: int, arch: int, is_cuda: bool = True):
        self.alias = alias
        self.name = name
        self.total_memory = total_memory
        self.arch = arch
        self.is_cuda = is_cuda


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
        "gpus": {0: ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 102 * 10**9, 120)},
        "current_device": 0,
        "device": None,
        "driver": "580.173.02",
        "ram_bytes": 64 * 2**30,
        "machine": "x86_64",
    }
    values.update(overrides)

    import psutil
    import warp as wp

    monkeypatch.setattr(system_check, "measure_cpu_single_thread_score", lambda: values["score"])
    monkeypatch.setattr(system_check, "_read_cpu_governor", lambda: values["governor"])
    monkeypatch.setattr(system_check, "_read_driver_version", lambda: values["driver"])
    monkeypatch.setattr(
        psutil, "cpu_freq", lambda: None if values["clock_mhz"] is None else _FakeFreq(values["clock_mhz"])
    )
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: values["cores"])
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVirtualMemory(values["ram_bytes"]))
    # Per-ordinal devices so tests can prove the *right* adapter was measured.
    devices = {
        f"cuda:{ordinal}": _FakeWarpDevice(f"cuda:{ordinal}", name, vram, arch)
        for ordinal, (name, vram, arch) in values["gpus"].items()
    }
    devices["cpu"] = _FakeWarpDevice("cpu", "CPU", 0, 0, is_cuda=False)

    def _get_device(ident=None):
        if ident is None or ident == "cuda":
            return devices[f"cuda:{values['current_device']}"]
        return devices[str(ident)]  # KeyError mimics warp on an unknown alias

    monkeypatch.setattr(wp, "is_cuda_available", lambda: values["cuda_available"])
    monkeypatch.setattr(wp, "get_device", _get_device)
    monkeypatch.setattr(wp, "get_cuda_devices", lambda: [d for k, d in devices.items() if k.startswith("cuda")])
    monkeypatch.setattr(system_check.platform, "machine", lambda: values["machine"])

    result = check_system_requirements(device=values["device"])
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
            ({"clock_mhz": 3200.0}, "CPU boost clock"),
            ({"cores": 4}, "CPU physical cores"),
            ({"gpus": {0: ("Weak GPU", 16 * 10**9, 120)}}, "GPU memory"),
            ({"gpus": {0: ("Ampere GPU", 102 * 10**9, 86)}}, "GPU architecture"),
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


class TestGovernorGating:
    """The governor is only flagged when the benchmark does not clear the bar.

    It is a proxy for delivered throughput, so warning about it on a machine
    that already measures fast enough is noise -- measurements on the reference
    workstation showed powersave and performance within run-to-run variance.
    """

    def test_passing_score_suppresses_the_governor_warning(self, monkeypatch):
        result, items = _at_spec(monkeypatch, score=1.0, governor="powersave")
        assert result.passed
        assert items["CPU governor"].passed
        assert items["CPU governor"].actual == "powersave"

    def test_score_exactly_at_the_bar_suppresses_it(self, monkeypatch):
        result, _ = _at_spec(monkeypatch, score=system_check.CPU_SCORE_MIN, governor="powersave")
        assert result.passed

    def test_failing_score_still_flags_the_governor(self, monkeypatch):
        # A slow machine on powersave must still be told about the fix -- this
        # is the case the check exists for.
        result, items = _at_spec(monkeypatch, score=0.5, governor="powersave")
        assert not result.passed
        assert {item.name for item in result.failures} == {"CPU single-thread", "CPU governor"}
        assert items["CPU governor"].detail == system_check.CPU_GOVERNOR_FIX

    def test_unmeasurable_score_still_flags_the_governor(self, monkeypatch):
        # Without a score there is no evidence the machine is fast enough, so
        # fall back to gating on the governor.
        result, items = _at_spec(monkeypatch, score=None, governor="powersave")
        assert not result.passed
        assert [item.name for item in result.failures] == ["CPU governor"]

    def test_performance_governor_passes_regardless_of_score(self, monkeypatch):
        _, items = _at_spec(monkeypatch, score=0.5, governor="performance")
        assert items["CPU governor"].passed

    def test_requirement_text_names_the_alternative(self, monkeypatch):
        _, items = _at_spec(monkeypatch, governor="performance")
        assert "single-thread" in items["CPU governor"].required


class TestMultiGpuDeviceSelection:
    """The GPU probe must measure the adapter teleop actually runs on.

    Regression coverage for a fixed ``get_device_properties(0)``, which on a
    multi-GPU workstation reports the wrong adapter -- warning about a capable
    machine, or passing an incapable one.
    """

    # cuda:0 is a weak display adapter, cuda:1 is the capable simulation GPU --
    # the layout that makes an ordinal-0 probe visibly wrong.
    _MIXED = {
        0: ("NVIDIA T400", 4 * 10**9, 75),
        1: ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 102 * 10**9, 120),
    }

    def test_selected_device_is_measured_not_ordinal_zero(self, monkeypatch):
        result, items = _at_spec(monkeypatch, gpus=self._MIXED, current_device=0, device="cuda:1")
        assert result.passed
        assert "RTX PRO 6000" in items["GPU memory"].actual

    def test_weak_selected_device_is_not_masked_by_a_capable_ordinal_zero(self, monkeypatch):
        # The inverse error: passing because ordinal 0 happens to be capable.
        flipped = {0: self._MIXED[1], 1: self._MIXED[0]}
        result, _ = _at_spec(monkeypatch, gpus=flipped, current_device=0, device="cuda:1")
        assert not result.passed
        assert {item.name for item in result.failures} == {"GPU memory", "GPU architecture"}

    def test_reported_ordinal_identifies_the_measured_adapter(self, monkeypatch):
        _, items = _at_spec(monkeypatch, gpus=self._MIXED, current_device=0, device="cuda:1")
        assert "cuda:1" in items["GPU memory"].actual

    def test_unset_device_falls_back_to_the_current_device(self, monkeypatch):
        result, items = _at_spec(monkeypatch, gpus=self._MIXED, current_device=1, device=None)
        assert result.passed
        assert "cuda:1" in items["GPU memory"].actual

    @pytest.mark.parametrize("device", ["cuda", "cpu", None])
    def test_non_ordinal_devices_use_the_current_device(self, monkeypatch, device):
        # A CPU simulation device still warrants a GPU probe: CloudXR encodes on
        # one regardless of where physics runs.
        result, items = _at_spec(monkeypatch, gpus=self._MIXED, current_device=1, device=device)
        assert result.passed
        assert "cuda:1" in items["GPU memory"].actual

    def test_integer_device_is_accepted(self, monkeypatch):
        _, items = _at_spec(monkeypatch, gpus=self._MIXED, current_device=0, device=1)
        assert "cuda:1" in items["GPU memory"].actual

    def test_out_of_range_device_falls_back_without_crashing(self, monkeypatch):
        # An out-of-range ordinal must not take down a teleop session.
        result, items = _at_spec(monkeypatch, gpus=self._MIXED, current_device=0, device="cuda:7")
        assert "cuda:0" in items["GPU memory"].actual
        assert not result.passed  # ordinal 0 is the weak adapter here


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
        result, _ = _at_spec(monkeypatch, cores=4, ram_bytes=32 * 2**30)
        payload = result.to_message()

        assert payload["type"] == "system_notice"
        assert payload["message"]["level"] == "warning"
        assert {item["name"] for item in payload["message"]["items"]} == {
            "CPU physical cores",
            "System memory",
        }

    def test_governor_failure_carries_the_fix_command(self, monkeypatch):
        # A failing score is what makes the governor gate at all now.
        result, _ = _at_spec(monkeypatch, score=0.5, governor="powersave")
        items = {item["name"]: item for item in result.to_message()["message"]["items"]}
        assert items["CPU governor"]["detail"] == system_check.CPU_GOVERNOR_FIX
        assert items["CPU governor"]["actual"] == "powersave"

    def test_message_is_json_serializable(self, monkeypatch):
        import json

        result, _ = _at_spec(monkeypatch, score=0.4)
        assert json.loads(json.dumps(result.to_message()))["type"] == "system_notice"


class TestMessageBudget:
    """A notice must fit the teleop channel, which does not fragment.

    Server-to-client opaque data is tunnelled as a single haptics event, so an
    oversized notice is dropped outright. The machines this check exists for
    fail the most requirements and therefore produce the largest payloads, so
    an unbudgeted notice goes missing exactly when it matters most.
    """

    LIMIT = 900

    def _failing(self, count: int) -> SystemCheckResult:
        return SystemCheckResult(
            items=tuple(
                SystemCheckItem(
                    f"Requirement {i}",
                    False,
                    "powersave",
                    "performance (or single-thread >= 0.80)",
                    detail=system_check.CPU_GOVERNOR_FIX,
                )
                for i in range(count)
            )
        )

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 9, 20])
    def test_notice_always_fits_the_budget(self, count):
        message = self._failing(count).to_message(max_bytes=self.LIMIT)
        assert len(json.dumps(message).encode()) <= self.LIMIT

    def test_every_check_failing_at_once_fits(self):
        # The realistic worst case: one failure per check the module performs.
        message = self._failing(9).to_message(max_bytes=self.LIMIT)
        assert len(json.dumps(message).encode()) <= self.LIMIT
        assert message["message"]["items"]

    def test_small_notice_keeps_its_remediation_hints(self):
        # Budgeting must not strip the actionable part when there is room.
        message = self._failing(1).to_message(max_bytes=self.LIMIT)
        assert message["message"]["items"][0]["detail"] == system_check.CPU_GOVERNOR_FIX

    def test_details_are_dropped_before_requirements(self):
        # Losing a hint is better than losing a whole requirement.
        message = self._failing(5).to_message(max_bytes=self.LIMIT)
        assert len(message["message"]["items"]) == 5
        assert not any(item["detail"] for item in message["message"]["items"])

    def test_truncation_says_how_many_were_omitted(self):
        message = self._failing(9).to_message(max_bytes=self.LIMIT)
        shown = len(message["message"]["items"])
        assert f"{9 - shown} more not shown" in message["message"]["summary"]

    def test_no_budget_returns_the_full_notice(self):
        message = self._failing(9).to_message()
        assert len(message["message"]["items"]) == 9
        assert all(item["detail"] for item in message["message"]["items"])

    def test_budget_matches_the_transport_limit(self):
        # Pins the test budget to what the lifecycle actually enforces, so the
        # two cannot drift apart.
        from isaaclab_teleop import session_lifecycle

        assert session_lifecycle._MAX_CLIENT_MESSAGE_BYTES == self.LIMIT


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
