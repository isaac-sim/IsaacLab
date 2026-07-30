# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared raw articulation index-kernel benchmark."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from isaaclab.benchmark.asset_suites import index_kernel
from isaaclab.benchmark.method_benchmark import MethodBenchmarkRunnerConfig

pytestmark = pytest.mark.benchmark


class _MeasurementRunner:
    def __init__(self) -> None:
        self.measurements: list[tuple[str, object]] = []

    def add_measurement(self, phase: str, measurement: object) -> None:
        self.measurements.append((phase, measurement))


def test_index_kernel_int32_and_int64_specializations_preserve_outputs_on_cpu() -> None:
    """Changing only selector width should preserve every production-kernel output."""
    device = wp.get_device("cpu")
    env_values = np.array([2, 0], dtype=np.int64)
    joint_values = np.array([2, 0], dtype=np.int64)
    int32_launch = index_kernel._prepare_launch(env_values, joint_values, wp.int32, device, 4, 3)
    int64_launch = index_kernel._prepare_launch(env_values, joint_values, wp.int64, device, 4, 3)

    index_kernel._launch_once(int32_launch, device)
    index_kernel._launch_once(int64_launch, device)
    wp.synchronize_device(device)

    for int32_output, int64_output in zip(int32_launch.outputs, int64_launch.outputs, strict=True):
        np.testing.assert_array_equal(int32_output.numpy(), int64_output.numpy())


def test_index_kernel_registers_paired_dtype_measurements_for_each_fill(monkeypatch) -> None:
    """Every fill phase should record both widths and their hand-derived latency deltas."""
    runner = _MeasurementRunner()
    fake_device = SimpleNamespace(is_cuda=True)
    monkeypatch.setattr(index_kernel, "_resolve_cuda_device", lambda _device: fake_device)
    monkeypatch.setattr(index_kernel, "_prepare_launch", lambda *args: args[2])
    monkeypatch.setattr(index_kernel, "_assert_output_parity", lambda *_args: None)
    monkeypatch.setattr(index_kernel, "_warm_specializations", lambda *_args: None)
    monkeypatch.setattr(
        index_kernel,
        "_measure_pairs",
        lambda *_args: ([1.0, 3.0], [2.0, 4.0]),
    )

    index_kernel.run_index_kernel_dtype_benchmark(
        runner,
        MethodBenchmarkRunnerConfig(
            num_iterations=2,
            num_rounds=2,
            num_instances=20,
            num_joints=4,
            device="cuda:0",
        ),
    )

    assert tuple(phase for phase, _measurement in runner.measurements) == (
        "index_kernel_5pct",
        "index_kernel_5pct",
        "index_kernel_5pct",
        "index_kernel_5pct",
        "index_kernel_95pct",
        "index_kernel_95pct",
        "index_kernel_95pct",
        "index_kernel_95pct",
        "index_kernel_100pct",
        "index_kernel_100pct",
        "index_kernel_100pct",
        "index_kernel_100pct",
    )
    for offset in range(0, 12, 4):
        measurements = [measurement for _phase, measurement in runner.measurements[offset : offset + 4]]
        assert tuple(measurement.name for measurement in measurements) == (
            "int32",
            "int64",
            "absolute_delta",
            "percentage_delta",
        )
        assert measurements[0].mean == pytest.approx(2.0)
        assert measurements[1].mean == pytest.approx(3.0)
        assert measurements[2].value == pytest.approx(1.0)
        assert measurements[3].value == pytest.approx(50.0)


def test_index_kernel_skips_cuda_event_timing_for_cpu(monkeypatch) -> None:
    """A CPU benchmark request should not prepare or time the CUDA-only auxiliary phase."""
    runner = _MeasurementRunner()

    def fail_prepare(*_args):
        raise AssertionError("launch preparation should not run")

    monkeypatch.setattr(index_kernel, "_prepare_launch", fail_prepare)

    index_kernel.run_index_kernel_dtype_benchmark(
        runner,
        MethodBenchmarkRunnerConfig(
            num_iterations=1,
            num_rounds=1,
            num_instances=4,
            num_joints=3,
            device="cpu",
        ),
    )

    assert runner.measurements == []


def test_index_kernel_timing_records_events_on_selected_device_stream(monkeypatch) -> None:
    """Raw timing should not create or record events on Warp's default CUDA device."""
    events: list[object] = []
    recordings: list[object] = []
    stream = SimpleNamespace(record_event=lambda event: recordings.append(event))
    device = SimpleNamespace(stream=stream)

    def create_event(*, device=None, enable_timing=False):
        event = SimpleNamespace(device=device, enable_timing=enable_timing)
        events.append(event)
        return event

    monkeypatch.setattr(index_kernel.wp, "Event", create_event)
    monkeypatch.setattr(index_kernel.wp, "record_event", lambda event: recordings.append(("default", event)))
    monkeypatch.setattr(index_kernel.wp, "synchronize_event", lambda _event: None)
    monkeypatch.setattr(index_kernel.wp, "get_event_elapsed_time", lambda _start, _end: 0.5)
    monkeypatch.setattr(index_kernel, "_launch_once", lambda _prepared, _device: None)

    elapsed = index_kernel._measure_launch(SimpleNamespace(), device, num_iterations=2)

    assert elapsed == pytest.approx(250.0)
    assert [event.device for event in events] == [device, device]
    assert recordings == events
