# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared micro-benchmark latency sampling."""

from collections.abc import Callable

import pytest

from isaaclab.benchmark.micro import (
    LatencyBenchmarkRunner,
    add_latency_measurements,
    measure_latency,
    summarize_latency,
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
