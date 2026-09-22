# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for method-level micro-benchmark sampling."""

import json
from unittest.mock import patch

import pytest

from isaaclab.benchmark import MethodBenchmarkDefinition, MethodBenchmarkRunner, MethodBenchmarkRunnerConfig

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize(
    ("field", "value"),
    (("num_iterations", 0), ("warmup_steps", -1), ("num_instances", 0), ("num_bodies", 0), ("num_joints", -1)),
)
def test_config_rejects_invalid_workload_sizes(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        MethodBenchmarkRunnerConfig(**{field: value})
    assert MethodBenchmarkRunnerConfig(num_joints=0).num_joints == 0


def _runner(tmp_path, *, num_iterations: int = 3, warmup_steps: int = 0, **config) -> MethodBenchmarkRunner:
    return MethodBenchmarkRunner(
        benchmark_name="asset_benchmark",
        config=MethodBenchmarkRunnerConfig(
            num_iterations=num_iterations, warmup_steps=warmup_steps, device="cpu", **config
        ),
        backend_type="omniperf",
        output_path=str(tmp_path),
        use_recorders=False,
        physics_variant="newton_kamino",
    )


def test_runner_writes_workload_metadata_and_mode_phases(tmp_path) -> None:
    """Exact backend selectors stay distinguishable and results are grouped by input mode."""
    runner = _runner(tmp_path, num_iterations=2, mode="fast")
    calls: list[str] = []
    target = type("Target", (), {"write": lambda self, **inputs: calls.append(inputs["mode"])})()
    definition = MethodBenchmarkDefinition(
        name="write",
        method_name="write",
        input_generators={"fast": lambda config: {"mode": "fast"}, "slow": lambda config: {"mode": "slow"}},
    )

    runner.run_benchmarks([definition], target)
    runner.finalize()

    with open(runner.output_file_path) as f:
        data = json.load(f)
    assert data["benchmark_info"]["physics_variant"] == "newton_kamino"
    assert data["benchmark_info"]["num_iterations"] == 2
    assert data["fast"]["write_n"] == 2
    assert "slow" not in data
    assert calls == ["fast"] * 3  # preflight plus two timed iterations


def test_method_benchmark_propagates_failures(tmp_path) -> None:
    """Failures abort the workload instead of silently reducing the sample count."""
    runner = _runner(tmp_path, num_iterations=1, warmup_steps=1)
    call_count = 0

    def operation() -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match=r"example.*warmup.*iteration 0"):
        runner._benchmark_method(operation, "example", lambda _config: {})

    call_count = 0
    with pytest.raises(RuntimeError, match=r"example.*timed.*iteration 0"):
        _runner(tmp_path, num_iterations=3)._benchmark_method(operation, "example", lambda _config: {})

    class Data:
        @property
        def dependency(self) -> int:
            raise RuntimeError("boom")

        @property
        def value(self) -> int:
            return 1

    with pytest.raises(RuntimeError, match=r"value.*timed preparation.*iteration 0"):
        _runner(tmp_path, num_iterations=1)._benchmark_property(Data(), "value", lambda _config: {}, ["dependency"])


def test_method_benchmark_collects_exact_requested_samples(tmp_path) -> None:
    """Preflight and warm-up calls run untimed, preparation stays outside the measured window."""
    runner = _runner(tmp_path, num_iterations=1, warmup_steps=1)
    events: list[str] = []

    def clock() -> int:
        events.append("clock")
        return events.count("clock") * 1_000

    with patch("isaaclab.benchmark.method_benchmark.time.perf_counter_ns", side_effect=clock):
        result = runner._benchmark_method(
            lambda: events.append("operation"),
            "example",
            lambda _config: events.append("inputs") or {},
            prepare_target=lambda: events.append("target"),
        )

    assert result == {"mean": 1.0, "std": 0.0, "n": 1}
    assert events == ["target", "inputs", "operation"] * 2 + ["target", "inputs", "clock", "operation", "clock"]


def test_method_benchmark_uses_sample_standard_deviation(tmp_path) -> None:
    runner = _runner(tmp_path, num_iterations=2)

    with patch("isaaclab.benchmark.method_benchmark.time.perf_counter_ns", side_effect=(0, 1_000, 0, 3_000)):
        result = runner._benchmark_method(lambda: None, "example", lambda _config: {})

    assert result == {"mean": pytest.approx(2.0), "std": pytest.approx(2**0.5), "n": 2}


def test_property_benchmark_skips_unsupported_and_missing_properties(tmp_path) -> None:
    class Data:
        @property
        def value(self) -> int:
            raise NotImplementedError("unsupported")

    runner = _runner(tmp_path, num_iterations=1)

    assert runner._benchmark_property(Data(), "value", lambda _config: {}, []) == {
        "skipped": True,
        "skip_reason": "NotImplementedError: unsupported",
    }
    assert runner._benchmark_property(Data(), "missing", lambda _config: {}, []) is None
    assert runner._benchmark_method(None, "missing", lambda _config: {}) is None


def test_sync_device_synchronizes_configured_device(tmp_path) -> None:
    runner = _runner(tmp_path)
    runner._config = MethodBenchmarkRunnerConfig(device="cuda:1")

    with patch("warp.synchronize_device") as synchronize_device:
        runner._sync_device()

    synchronize_device.assert_called_once_with("cuda:1")
