# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared micro-benchmark latency sampling and the sensor suite helpers."""

import json
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from isaaclab.benchmark.measurements import SingleMeasurement
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


def _clock(values: list[int]) -> Callable[[], int]:
    """Return a deterministic nanosecond clock."""
    iterator = iter(values)
    return lambda: next(iterator)


def _runner(tmp_path, **kwargs) -> LatencyBenchmarkRunner:
    return LatencyBenchmarkRunner("sensor", "omniperf", str(tmp_path), use_recorders=False, **kwargs)


def test_measure_latency_synchronizes_at_both_boundaries() -> None:
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
    def failing() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        measure_latency(operation=failing, synchronize=lambda: None)


def test_summarize_latency_uses_sample_std_and_interpolated_percentiles() -> None:
    stats = summarize_latency([1.0, 2.0, 4.0])
    assert stats.mean_s == pytest.approx(7.0 / 3.0)
    assert stats.std_s == pytest.approx(1.5275252316519468)
    assert (stats.p50_s, stats.p95_s, stats.n) == (pytest.approx(2.0), pytest.approx(3.8), 3)

    single = summarize_latency([0.5])
    assert (single.std_s, single.p50_s, single.p95_s) == (0.0, 0.5, 0.5)

    with pytest.raises(ValueError, match="at least one"):
        summarize_latency([])


def test_latency_runner_publishes_paired_series_in_milliseconds(tmp_path) -> None:
    runner = _runner(tmp_path, metadata={"label": "current"})
    samples = [
        measure_latency(lambda: None, lambda: None, clock_ns=_clock([0, 1_000_000, 3_000_000])),
        measure_latency(lambda: None, lambda: None, clock_ns=_clock([0, 2_000_000, 6_000_000])),
    ]

    stats = runner.add_latency_samples("sensor", samples)
    add_latency_measurements(runner, "observer", "Floor", [0.001, 0.003, 0.005])
    runner.finalize()

    assert stats.mean_s == pytest.approx(4.5e-3)
    with open(runner.output_file_path) as f:
        data = json.load(f)
    assert data["benchmark_info"]["label"] == "current"
    sensor = data["sensor"]
    assert sensor["Synchronized Completion_mean"] == pytest.approx(4.5)
    assert sensor["Synchronized Completion_n"] == 2
    assert sensor["Host Submission_mean"] == pytest.approx(1.5)
    assert sensor["Host Submission p95"] == pytest.approx(1.95)
    observer = data["observer"]
    assert (observer["Floor_mean"], observer["Floor_std"], observer["Floor_n"]) == (
        pytest.approx(3.0),
        pytest.approx(2.0),
        3,
    )
    assert (observer["Floor p50"], observer["Floor p95"]) == (pytest.approx(3.0), pytest.approx(4.8))


def test_latency_runner_samples_recorders_before_finalize(tmp_path, monkeypatch) -> None:
    events: list[str] = []
    runner = _runner(tmp_path)
    runner._use_recorders = True
    runner.update_manual_recorders = lambda: events.append("recorders")
    monkeypatch.setattr(
        "isaaclab.benchmark.micro.BaseIsaacLabBenchmark.finalize", lambda _self: events.append("finalize") or ()
    )

    assert runner.finalize() == ()
    assert events == ["recorders", "finalize"]


def test_collect_sensor_latency_samples_keeps_stepping_outside_timed_updates() -> None:
    events: list[str] = []

    samples = collect_sensor_latency_samples(
        num_steps=2,
        step=lambda: events.append("step"),
        update=lambda: events.append("update"),
        synchronize=lambda: events.append("synchronize"),
        native_read=lambda: events.append("native_read"),
    )

    update_phase = ["step", "synchronize", "update", "synchronize"] * 2
    observer_phase = ["synchronize", "synchronize"] * 2
    native_phase = ["synchronize", "native_read", "synchronize"] * 2
    assert events == update_phase + observer_phase + native_phase
    assert (len(samples.updates), len(samples.observer_s), len(samples.native_reads)) == (2, 2, 2)


def test_add_sensor_latency_measurements_reports_standard_phases_and_native_remainder(tmp_path) -> None:
    runner = _runner(tmp_path)
    sample = measure_latency(lambda: None, lambda: None, clock_ns=_clock([0, 1_000_000, 4_000_000]))
    native = measure_latency(lambda: None, lambda: None, clock_ns=_clock([0, 500_000, 1_000_000]))

    add_sensor_latency_measurements(
        runner,
        samples=SimpleNamespace(updates=[sample], observer_s=[0.0001], native_reads=[native]),
        validation=[SingleMeasurement(name="Finite Values", value=1.0, unit="count")],
        update_phase="sensor_update",
        observer_phase="observer",
        validation_phase="validation",
    )

    phases = runner._phases
    assert list(phases) == ["benchmark_info", "sensor_update", "native_read", "observer", "validation"]
    remainder = next(
        m for m in phases["sensor_update"].measurements if m.name == "Estimated Synchronized Non-read Time"
    )
    assert remainder.value == pytest.approx(3.0)
    assert phases["observer"].measurements[0].name == "Synchronized Observer Floor"
    assert phases["validation"].measurements[0].name == "Finite Values"


def test_contact_sensor_workload_runs_shared_cadence_validation_and_reporting(tmp_path) -> None:
    events: list[str] = []

    paths = run_contact_sensor_workload(
        benchmark_name="contact",
        formatter_type="omniperf",
        output_path=str(tmp_path),
        metadata={"decimation": 2},
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

    # One warm-up cadence plus two measured cadences of two physics steps each.
    assert (events.count("step"), events.count("update"), events.count("read")) == (6, 6, 3)
    with open(paths[0]) as f:
        data = json.load(f)
    assert data["benchmark_info"]["decimation"] == 2
    assert data["sensor_cadence"]["Synchronized Completion_n"] == 2
    assert data["observer"]["Synchronized Observer Floor_n"] == 2
    assert data["validation"]["Sensors in Contact"] == 4

    with pytest.raises(RuntimeError, match="Expected 3 contacting sensors"):
        run_contact_sensor_workload(
            benchmark_name="contact",
            formatter_type="omniperf",
            output_path=str(tmp_path),
            metadata={},
            num_steps=1,
            warmup_steps=0,
            decimation=1,
            expected_contacts=3,
            step=lambda: None,
            update=lambda: None,
            read=lambda: None,
            count_contacts=lambda: 4,
            synchronize=lambda: None,
        )
