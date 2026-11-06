# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latency sampling and reporting shared by sensor micro-benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..measurements import Measurement, SingleMeasurement
from ..micro import LatencyBenchmarkRunner, LatencySample, measure_latency


@dataclass(frozen=True)
class SensorLatencySamples:
    """Latency samples collected during a sensor benchmark.

    Attributes:
        updates: Sensor update latency samples [s].
        observer_s: Synchronized no-op observer latency samples [s].
        native_reads: Native sensor read latency samples [s].
    """

    updates: list[LatencySample]
    observer_s: list[float]
    native_reads: list[LatencySample]


def collect_sensor_latency_samples(
    *,
    num_steps: int,
    step: Callable[[], None],
    update: Callable[[], None],
    synchronize: Callable[[], None],
    native_read: Callable[[], None] | None = None,
) -> SensorLatencySamples:
    """Collect the standard sensor update, observer, and native-read samples.

    Physics stepping remains outside the measured sensor update boundary. A
    synchronized no-op sample captures observer overhead, and an optional
    native read remains separately visible.

    Args:
        num_steps: Number of measured steps.
        step: Physics stepping operation.
        update: Sensor update operation.
        synchronize: Function that blocks until pending device work completes.
        native_read: Optional native sensor data read operation.

    Returns:
        Collected sensor latency samples [s].
    """
    updates: list[LatencySample] = []
    observer_s: list[float] = []
    native_reads: list[LatencySample] = []
    for _ in range(num_steps):
        step()
        updates.append(measure_latency(update, synchronize))
    for _ in range(num_steps):
        observer_s.append(measure_latency(lambda: None, synchronize).synchronized_s)
    if native_read is not None:
        for _ in range(num_steps):
            native_reads.append(measure_latency(native_read, synchronize))
    return SensorLatencySamples(updates=updates, observer_s=observer_s, native_reads=native_reads)


def add_sensor_latency_measurements(
    benchmark: LatencyBenchmarkRunner,
    *,
    samples: SensorLatencySamples,
    validation: Sequence[Measurement],
    update_phase: str,
    observer_phase: str,
    validation_phase: str,
    native_read_phase: str = "native_read",
) -> None:
    """Add the standard sensor latency and validation measurements.

    Args:
        benchmark: Benchmark receiving the measurements.
        samples: Sensor latency samples to report [s].
        validation: Sensor-specific correctness measurements.
        update_phase: Phase containing sensor update measurements.
        observer_phase: Phase containing observer overhead measurements.
        validation_phase: Phase containing validation measurements.
        native_read_phase: Phase containing optional native-read measurements.
    """
    update_stats = benchmark.add_latency_samples(update_phase, samples.updates)
    if samples.native_reads:
        native_read_stats = benchmark.add_latency_samples(native_read_phase, samples.native_reads)
        benchmark.add_measurement(
            update_phase,
            SingleMeasurement(
                name="Estimated Synchronized Non-read Time",
                value=(update_stats.mean_s - native_read_stats.mean_s) * 1000.0,
                unit="ms",
            ),
        )
    benchmark.add_synchronized_samples(
        observer_phase,
        "Synchronized Observer Floor",
        samples.observer_s,
    )
    benchmark.add_measurement(validation_phase, validation)
