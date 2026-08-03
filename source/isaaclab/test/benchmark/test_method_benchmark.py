# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for method-level micro-benchmark sampling."""

import inspect
from dataclasses import fields
from unittest.mock import patch

import pytest

from isaaclab.benchmark import MethodBenchmarkDefinition, MethodBenchmarkRunner, MethodBenchmarkRunnerConfig

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("num_iterations", 0),
        ("warmup_steps", -1),
        ("num_instances", 0),
        ("num_bodies", 0),
        ("num_joints", -1),
    ),
)
def test_config_rejects_invalid_workload_sizes(field: str, value: int) -> None:
    """Invalid workload sizes should fail before benchmark setup."""
    with pytest.raises(ValueError, match=field):
        MethodBenchmarkRunnerConfig(**{field: value})


def test_config_accepts_zero_joints_for_rigid_assets() -> None:
    """Rigid-object benchmarks should represent their zero-joint workload accurately."""
    config = MethodBenchmarkRunnerConfig(num_joints=0)

    assert config.num_joints == 0


def test_runner_records_exact_physics_variant_in_workflow_metadata(tmp_path) -> None:
    """Exact backend selectors should remain distinguishable in output metadata."""
    with patch("isaaclab.benchmark.method_benchmark.BaseIsaacLabBenchmark.__init__") as base_init:
        MethodBenchmarkRunner(
            benchmark_name="asset_benchmark",
            config=MethodBenchmarkRunnerConfig(device="cpu"),
            output_path=str(tmp_path),
            use_recorders=False,
            physics_variant="newton_kamino",
        )

    metadata = base_init.call_args.kwargs["workflow_metadata"]["metadata"]
    assert {"name": "physics_variant", "data": "newton_kamino"} in metadata


def _runner(*, num_iterations: int = 3, warmup_steps: int = 0) -> MethodBenchmarkRunner:
    runner = object.__new__(MethodBenchmarkRunner)
    runner._config = MethodBenchmarkRunnerConfig(
        num_iterations=num_iterations,
        warmup_steps=warmup_steps,
        device="cpu",
    )
    return runner


def test_method_benchmark_propagates_timed_failure() -> None:
    """A failed timed iteration should abort instead of reducing the sample count."""
    runner = _runner()
    call_count = 0

    def operation() -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match=r"example.*timed.*iteration 0"):
        runner._benchmark_method(operation, "example", lambda _config: {})


def test_method_benchmark_propagates_warmup_failure() -> None:
    """A failed warm-up iteration should abort before measurement starts."""
    runner = _runner(num_iterations=1, warmup_steps=1)
    call_count = 0

    def operation() -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match=r"example.*warmup.*iteration 0"):
        runner._benchmark_method(operation, "example", lambda _config: {})


def test_property_benchmark_propagates_dependency_failure() -> None:
    """A dependency failure should not turn a derived property into another workload."""

    class Data:
        @property
        def dependency(self) -> int:
            raise RuntimeError("boom")

        @property
        def value(self) -> int:
            return 1

    runner = _runner(num_iterations=1)

    with pytest.raises(RuntimeError, match=r"value.*timed preparation.*iteration 0"):
        runner._benchmark_property(Data(), "value", lambda _config: {}, dependencies=["dependency"])


def test_method_benchmark_collects_exact_requested_samples() -> None:
    """Preflight and warm-up calls should not change the requested sample count."""
    runner = _runner(num_iterations=3, warmup_steps=2)
    call_count = 0

    def operation() -> None:
        nonlocal call_count
        call_count += 1

    result = runner._benchmark_method(operation, "example", lambda _config: {})

    assert result is not None
    assert result["n"] == 3
    assert call_count == 6


def test_method_benchmark_prepares_target_outside_timed_operation() -> None:
    """Target cache preparation should run for every call without contributing to measured latency."""
    runner = _runner(num_iterations=1, warmup_steps=1)
    events: list[str] = []

    def prepare_target() -> None:
        events.append("target")

    def generator(_config) -> dict[str, object]:
        events.append("inputs")
        return {}

    def operation() -> None:
        events.append("operation")

    def clock() -> int:
        events.append("clock")
        return events.count("clock") * 1_000

    with patch("isaaclab.benchmark.method_benchmark.time.perf_counter_ns", side_effect=clock):
        result = runner._benchmark_method(
            operation,
            "example",
            generator,
            prepare_target=prepare_target,
        )

    assert result is not None
    assert events == [
        "target",
        "inputs",
        "operation",
        "target",
        "inputs",
        "operation",
        "target",
        "inputs",
        "clock",
        "operation",
        "clock",
    ]


def test_method_dependency_surface_is_not_published() -> None:
    """The unused method dependency arguments should not become public API."""
    definition_fields = {field.name for field in fields(MethodBenchmarkDefinition)}

    assert "dependencies" not in definition_fields
    assert "dependencies" not in inspect.signature(MethodBenchmarkRunner.run_benchmarks).parameters


def test_method_benchmark_uses_sample_standard_deviation() -> None:
    """Method statistics should match the regular benchmark convention."""
    runner = _runner(num_iterations=2)

    with patch(
        "isaaclab.benchmark.method_benchmark.time.perf_counter_ns",
        side_effect=(0, 1_000, 0, 3_000),
    ):
        result = runner._benchmark_method(lambda: None, "example", lambda _config: {})

    assert result is not None
    assert result["mean"] == pytest.approx(2.0)
    assert result["std"] == pytest.approx(2**0.5)
    assert result["n"] == 2


def test_property_benchmark_skips_not_implemented_property() -> None:
    """A property that explicitly raises NotImplementedError should be reported as skipped."""

    class Data:
        @property
        def value(self) -> int:
            raise NotImplementedError("unsupported")

    result = _runner(num_iterations=1)._benchmark_property(
        Data(),
        "value",
        lambda _config: {},
        dependencies=[],
    )

    assert result == {
        "skipped": True,
        "skip_reason": "NotImplementedError: unsupported",
    }


def test_sync_device_synchronizes_configured_device_once() -> None:
    """The timing boundary should synchronize only the configured device."""
    runner = _runner()
    runner._config = MethodBenchmarkRunnerConfig(device="cuda:1")

    with patch("warp.synchronize_device") as synchronize_device:
        runner._sync_device()

    synchronize_device.assert_called_once_with("cuda:1")
