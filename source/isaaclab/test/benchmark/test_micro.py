# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared micro-benchmark latency sampling."""

from collections.abc import Callable
from types import SimpleNamespace

import pytest

from isaaclab.benchmark.micro import (
    LatencyBenchmarkRunner,
    add_latency_measurements,
    measure_latency,
    summarize_latency,
)
from isaaclab.benchmark.sensor_suites import (
    add_sensor_latency_measurements,
    collect_sensor_latency_samples,
    run_contact_sensor_workload,
)

pytestmark = pytest.mark.benchmark


class _MeasurementSink:
    """Collect benchmark measurements without constructing formatter services."""

    def __init__(self) -> None:
        self.measurements = []

    def add_measurement(self, phase_name: str, measurement) -> None:
        self.measurements.append((phase_name, measurement))


def _clock(values: list[int]) -> Callable[[], int]:
    """Return a deterministic nanosecond clock."""
    iterator = iter(values)
    return lambda: next(iterator)


def test_measure_latency_synchronizes_at_both_boundaries() -> None:
    """Pending work must be drained before the operation and after its submission."""
    events: list[str] = []

    sample = measure_latency(
        operation=lambda: events.append("operation"),
        synchronize=lambda: events.append("synchronize"),
        clock_ns=_clock([10, 30, 70]),
    )

    assert events == ["synchronize", "operation", "synchronize"]
    assert sample.submission_s == pytest.approx(20e-9)
    assert sample.synchronized_s == pytest.approx(60e-9)


def test_measure_latency_excludes_pending_work_before_operation() -> None:
    """The pre-boundary synchronization must not be charged to the operation."""
    now_ns = 0
    pending_ns = 100

    def synchronize() -> None:
        nonlocal now_ns, pending_ns
        now_ns += pending_ns
        pending_ns = 0

    def operation() -> None:
        nonlocal now_ns, pending_ns
        now_ns += 20
        pending_ns = 30

    sample = measure_latency(operation=operation, synchronize=synchronize, clock_ns=lambda: now_ns)

    assert sample.submission_s == pytest.approx(20e-9)
    assert sample.synchronized_s == pytest.approx(50e-9)


def test_measure_latency_propagates_operation_failure() -> None:
    """A failed workload must abort instead of producing a partial sample."""

    def operation() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        measure_latency(operation=operation, synchronize=lambda: None)


def test_summarize_latency_rejects_empty_samples() -> None:
    """Empty benchmark results have no meaningful latency statistics."""
    with pytest.raises(ValueError, match="at least one"):
        summarize_latency([])


def test_summarize_latency_uses_zero_std_for_one_sample() -> None:
    """A one-sample run should remain reportable without an undefined sample deviation."""
    stats = summarize_latency([0.5])

    assert stats.std_s == 0.0
    assert stats.p50_s == 0.5
    assert stats.p95_s == 0.5


def test_summarize_latency_uses_sample_std_and_interpolated_percentiles() -> None:
    """Latency summaries should match the regular benchmark statistical convention."""
    stats = summarize_latency([1.0, 2.0, 4.0])

    assert stats.mean_s == pytest.approx(7.0 / 3.0)
    assert stats.std_s == pytest.approx(1.5275252316519468)
    assert stats.p50_s == pytest.approx(2.0)
    assert stats.p95_s == pytest.approx(3.8)
    assert stats.n == 3


def test_add_latency_measurements_converts_seconds_and_std_to_milliseconds() -> None:
    """Structured results should publish all latency fields in milliseconds."""
    sink = _MeasurementSink()

    add_latency_measurements(sink, "sensor", "Synchronized Update", [0.001, 0.003, 0.005])

    assert [phase for phase, _ in sink.measurements] == ["sensor", "sensor", "sensor"]
    statistical = sink.measurements[0][1]
    assert statistical.name == "Synchronized Update"
    assert statistical.mean == pytest.approx(3.0)
    assert statistical.std == pytest.approx(2.0)
    assert statistical.n == 3
    assert statistical.unit == "ms"
    assert sink.measurements[1][1].value == pytest.approx(3.0)
    assert sink.measurements[2][1].value == pytest.approx(4.8)


def test_latency_runner_publishes_paired_samples() -> None:
    """The runner should publish host and synchronized series with consistent names."""
    sink = _MeasurementSink()
    runner = object.__new__(LatencyBenchmarkRunner)
    runner.add_measurement = sink.add_measurement
    samples = [
        measure_latency(lambda: None, lambda: None, clock_ns=_clock([0, 1, 3])),
        measure_latency(lambda: None, lambda: None, clock_ns=_clock([0, 2, 6])),
    ]

    stats = runner.add_latency_samples("sensor", samples)

    assert stats.mean_s == pytest.approx(4.5e-9)
    assert sink.measurements[0][1].name == "Synchronized Completion"
    assert sink.measurements[3][1].name == "Host Submission"


def test_latency_runner_samples_recorders_before_finalize(monkeypatch) -> None:
    """One-shot latency benchmarks should collect recorder data before writing results."""
    events: list[str] = []
    runner = object.__new__(LatencyBenchmarkRunner)
    runner._use_recorders = True
    runner.update_manual_recorders = lambda: events.append("recorders")
    monkeypatch.setattr(
        "isaaclab.benchmark.micro.BaseIsaacLabBenchmark.finalize",
        lambda _self: events.append("finalize") or (),
    )

    assert runner.finalize() == ()
    assert events == ["recorders", "finalize"]


def test_collect_sensor_latency_samples_preserves_visible_timing_order() -> None:
    """Physics stepping should remain outside update timing and native reads should remain separate."""
    events: list[str] = []

    samples = collect_sensor_latency_samples(
        num_steps=2,
        step=lambda: events.append("step"),
        update=lambda: events.append("update"),
        synchronize=lambda: events.append("synchronize"),
        native_read=lambda: events.append("native_read"),
    )

    assert events == [
        "step",
        "synchronize",
        "update",
        "synchronize",
        "step",
        "synchronize",
        "update",
        "synchronize",
        "synchronize",
        "synchronize",
        "synchronize",
        "synchronize",
        "synchronize",
        "native_read",
        "synchronize",
        "synchronize",
        "native_read",
        "synchronize",
    ]
    assert len(samples.updates) == 2
    assert len(samples.observer_s) == 2
    assert len(samples.native_reads) == 2


def test_add_sensor_latency_measurements_reports_standard_phases_and_native_remainder() -> None:
    """The shared reporter should retain update, observer, native-read, and validation phases."""

    class Benchmark:
        def __init__(self) -> None:
            self.calls = []

        def add_latency_samples(self, phase_name, samples):
            self.calls.append(("latency", phase_name, samples))
            return SimpleNamespace(mean_s=0.004 if phase_name == "sensor_update" else 0.001)

        def add_synchronized_samples(self, phase_name, name, samples):
            self.calls.append(("synchronized", phase_name, name, samples))

        def add_measurement(self, phase_name, measurement):
            self.calls.append(("measurement", phase_name, measurement))

    benchmark = Benchmark()
    samples = SimpleNamespace(
        updates=[SimpleNamespace(synchronized_s=0.004)],
        observer_s=[0.0001],
        native_reads=[SimpleNamespace(synchronized_s=0.001)],
    )
    validation = [SimpleNamespace(name="Finite Values")]

    add_sensor_latency_measurements(
        benchmark,
        samples=samples,
        validation=validation,
        update_phase="sensor_update",
        observer_phase="observer",
        validation_phase="validation",
        native_read_phase="native_read",
    )

    assert [(call[0], call[1]) for call in benchmark.calls] == [
        ("latency", "sensor_update"),
        ("latency", "native_read"),
        ("measurement", "sensor_update"),
        ("synchronized", "observer"),
        ("measurement", "validation"),
    ]
    estimated = benchmark.calls[2][2]
    assert estimated.name == "Estimated Synchronized Non-read Time"
    assert estimated.value == pytest.approx(3.0)


def test_contact_sensor_workload_runs_shared_cadence_validation_and_reporting(monkeypatch) -> None:
    """The contact workload should keep physics outside each timed sensor operation."""
    events: list[str] = []

    class Benchmark:
        def __init__(self, **kwargs) -> None:
            events.append(f"benchmark:{kwargs['benchmark_name']}")

        def add_latency_samples(self, phase_name, samples):
            events.append(f"latency:{phase_name}:{len(samples)}")
            return SimpleNamespace(mean_s=0.0)

        def add_synchronized_samples(self, phase_name, _name, samples):
            events.append(f"synchronized:{phase_name}:{len(samples)}")

        def add_measurement(self, phase_name, _measurement):
            events.append(f"measurement:{phase_name}")

        def finalize(self) -> None:
            events.append("finalize")

    monkeypatch.setattr("isaaclab.benchmark.sensor_suites.contact.LatencyBenchmarkRunner", Benchmark)

    run_contact_sensor_workload(
        benchmark_name="contact",
        formatter_type="summary",
        output_path=".",
        metadata={},
        num_steps=2,
        warmup_steps=1,
        decimation=2,
        expected_contacts=4,
        step=lambda: events.append("step"),
        update=lambda: events.append("update"),
        read=lambda: events.append("read"),
        count_contacts=lambda: 4,
        synchronize=lambda: events.append("sync"),
    )

    assert events.count("step") == 6
    assert events.count("update") == 6
    assert events.count("read") == 3
    assert "latency:sensor_cadence:2" in events
    assert "synchronized:observer:2" in events
    assert events[-1] == "finalize"
