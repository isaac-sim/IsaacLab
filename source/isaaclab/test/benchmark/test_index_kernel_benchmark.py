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


def test_index_kernel_measures_one_batch_per_dtype_and_fill(monkeypatch) -> None:
    """Every fill phase should time one batch per width and record their latency deltas."""
    runner = _MeasurementRunner()
    fake_device = SimpleNamespace(is_cuda=True)
    measured: list[tuple[object, int]] = []
    monkeypatch.setattr(index_kernel, "_resolve_cuda_device", lambda _device: fake_device)
    monkeypatch.setattr(index_kernel, "_prepare_launch", lambda *args: args[2])
    monkeypatch.setattr(index_kernel, "_assert_output_parity", lambda *_args: None)
    monkeypatch.setattr(index_kernel, "_warm_specializations", lambda *_args: None)

    def measure(prepared, _device, num_iterations):
        measured.append((prepared, num_iterations))
        return 1.0 if prepared == wp.int32 else 1.5

    monkeypatch.setattr(index_kernel, "_measure_launch", measure)

    index_kernel.run_index_kernel_dtype_benchmark(
        runner,
        MethodBenchmarkRunnerConfig(
            num_iterations=7,
            num_instances=20,
            num_joints=4,
            device="cuda:0",
        ),
    )

    assert measured == [(wp.int32, 7), (wp.int64, 7)] * 3
    assert [phase for phase, _ in runner.measurements] == [
        f"index_kernel_{suffix}" for suffix in ("5pct", "95pct", "100pct") for _ in range(4)
    ]
    for offset in range(0, 12, 4):
        measurements = [measurement for _phase, measurement in runner.measurements[offset : offset + 4]]
        assert tuple(measurement.name for measurement in measurements) == (
            "int32",
            "int64",
            "absolute_delta",
            "percentage_delta",
        )
        assert measurements[0].value == pytest.approx(1.0)
        assert measurements[1].value == pytest.approx(1.5)
        assert measurements[2].value == pytest.approx(0.5)
        assert measurements[3].value == pytest.approx(50.0)
