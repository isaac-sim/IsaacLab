# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared latency sampling and reporting for micro-benchmarks."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .benchmark_core import BaseIsaacLabBenchmark
from .measurements import Measurement, SingleMeasurement, StatisticalMeasurement


class _MeasurementSink(Protocol):
    """Benchmark object that accepts measurements grouped by phase."""

    def add_measurement(self, phase_name: str, measurement: Measurement | Sequence[Measurement]) -> None:
        """Add one or more measurements to a phase."""


@dataclass(frozen=True)
class LatencySample:
    """One host-submission and device-synchronized latency sample.

    Attributes:
        submission_s: Time [s] until the operation returned to the host.
        synchronized_s: Time [s] until all submitted device work completed.
    """

    submission_s: float
    synchronized_s: float


@dataclass(frozen=True)
class LatencyStatistics:
    """Aggregate statistics for a latency sample series.

    Attributes:
        mean_s: Arithmetic mean latency [s].
        std_s: Sample standard deviation [s], or zero for one sample.
        p50_s: Linearly interpolated 50th percentile latency [s].
        p95_s: Linearly interpolated 95th percentile latency [s].
        n: Number of samples.
    """

    mean_s: float
    std_s: float
    p50_s: float
    p95_s: float
    n: int


def measure_latency(
    operation: Callable[[], None],
    synchronize: Callable[[], None],
    *,
    clock_ns: Callable[[], int] | None = None,
) -> LatencySample:
    """Measure host submission and synchronized completion latency.

    The pre-boundary synchronization prevents asynchronous work submitted before
    :paramref:`operation` from being charged to the sample. The post-boundary
    synchronization includes all device work submitted by the operation.

    Args:
        operation: Workload to measure.
        synchronize: Function that blocks until pending device work completes.
        clock_ns: Monotonic nanosecond clock. Defaults to :func:`time.perf_counter_ns`.

    Returns:
        Host-submission and device-synchronized latency [s].
    """
    if clock_ns is None:
        clock_ns = time.perf_counter_ns

    synchronize()
    start_ns = clock_ns()
    operation()
    submitted_ns = clock_ns()
    synchronize()
    finished_ns = clock_ns()
    return LatencySample(
        submission_s=(submitted_ns - start_ns) / 1e9,
        synchronized_s=(finished_ns - start_ns) / 1e9,
    )


def summarize_latency(samples_s: Sequence[float]) -> LatencyStatistics:
    """Summarize a non-empty latency sample series.

    Args:
        samples_s: Latency samples [s].

    Returns:
        Mean, sample standard deviation, and interpolated percentiles [s].

    Raises:
        ValueError: If :paramref:`samples_s` is empty.
    """
    if not samples_s:
        raise ValueError("Latency statistics require at least one sample.")
    samples = np.asarray(samples_s, dtype=np.float64)
    return LatencyStatistics(
        mean_s=float(np.mean(samples)),
        std_s=float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
        p50_s=float(np.percentile(samples, 50, method="linear")),
        p95_s=float(np.percentile(samples, 95, method="linear")),
        n=len(samples),
    )


def add_latency_measurements(
    benchmark: _MeasurementSink,
    phase_name: str,
    name: str,
    samples_s: Sequence[float],
) -> LatencyStatistics:
    """Add latency mean, standard deviation, and percentiles to a benchmark.

    Args:
        benchmark: Benchmark receiving the measurements.
        phase_name: Phase that owns the measurements.
        name: Base measurement name.
        samples_s: Latency samples [s].

    Returns:
        Computed latency statistics [s].
    """
    stats = summarize_latency(samples_s)
    benchmark.add_measurement(
        phase_name,
        measurement=StatisticalMeasurement(
            name=name,
            mean=stats.mean_s * 1000.0,
            std=stats.std_s * 1000.0,
            n=stats.n,
            unit="ms",
        ),
    )
    benchmark.add_measurement(
        phase_name,
        measurement=SingleMeasurement(name=f"{name} p50", value=stats.p50_s * 1000.0, unit="ms"),
    )
    benchmark.add_measurement(
        phase_name,
        measurement=SingleMeasurement(name=f"{name} p95", value=stats.p95_s * 1000.0, unit="ms"),
    )
    return stats


class LatencyBenchmarkRunner(BaseIsaacLabBenchmark):
    """One-shot runner for latency micro-benchmarks.

    Args:
        benchmark_name: Name used in output metadata and filenames.
        formatter_type: Formatter used to report results.
        output_path: Directory for result files.
        metadata: Workload metadata stored with the result.
        use_recorders: Whether to collect hardware and version information.
    """

    def __init__(
        self,
        benchmark_name: str,
        formatter_type: str,
        output_path: str,
        metadata: dict[str, str | int | float | dict] | None = None,
        use_recorders: bool = True,
    ) -> None:
        workflow_metadata = {"metadata": [{"name": name, "data": value} for name, value in (metadata or {}).items()]}
        super().__init__(
            benchmark_name=benchmark_name,
            formatter_type=formatter_type,
            output_path=output_path,
            output_prefix=benchmark_name,
            workflow_metadata=workflow_metadata,
            use_recorders=use_recorders,
        )

    def add_latency_samples(self, phase_name: str, samples: Sequence[LatencySample]) -> LatencyStatistics:
        """Add synchronized completion and host submission latency series.

        Args:
            phase_name: Phase that owns the measurements.
            samples: Paired host and synchronized latency samples [s].

        Returns:
            Synchronized completion latency statistics [s].
        """
        synchronized_stats = add_latency_measurements(
            self, phase_name, "Synchronized Completion", [sample.synchronized_s for sample in samples]
        )
        add_latency_measurements(self, phase_name, "Host Submission", [sample.submission_s for sample in samples])
        return synchronized_stats

    def add_synchronized_samples(self, phase_name: str, name: str, samples_s: Sequence[float]) -> LatencyStatistics:
        """Add a synchronized-only latency series.

        Args:
            phase_name: Phase that owns the measurements.
            name: Measurement name.
            samples_s: Synchronized latency samples [s].

        Returns:
            Synchronized latency statistics [s].
        """
        return add_latency_measurements(self, phase_name, name, samples_s)

    def finalize(self) -> tuple[Path, ...]:
        """Sample recorders and write results.

        Returns:
            Paths written by the selected formatters.
        """
        if self._use_recorders:
            self.update_manual_recorders()
        return super().finalize()
