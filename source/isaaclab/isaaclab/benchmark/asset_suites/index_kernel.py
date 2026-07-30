# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Raw selector-width benchmark for a production indexed articulation writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from isaaclab.assets.articulation.ordering_kernels import (
    write_joint_state_user_to_backend_with_indices_kernel,
)

from ..measurements import SingleMeasurement
from ..method_benchmark import MethodBenchmarkRunnerConfig
from .generators import FILL_RATIOS

if TYPE_CHECKING:
    from ..method_benchmark import MethodBenchmarkRunner


@dataclass(frozen=True)
class PreparedLaunch:
    """Inputs, outputs, and specialization prepared outside the timing interval."""

    kernel: wp.Kernel
    launch_dim: tuple[int, int]
    inputs: list[Any]
    outputs: list[wp.array]


def _selector(values: np.ndarray, dtype: type, device: wp.context.Device) -> wp.array:
    numpy_dtype = np.int32 if dtype == wp.int32 else np.int64
    return wp.array(values.astype(numpy_dtype), dtype=dtype, device=device)


def _prepare_launch(
    env_values: np.ndarray,
    joint_values: np.ndarray,
    index_dtype: type,
    device: wp.context.Device,
    num_envs: int,
    num_joints: int,
) -> PreparedLaunch:
    """Prepare one production-kernel specialization and its buffers."""
    env_ids = _selector(env_values, index_dtype, device)
    joint_ids = _selector(joint_values, index_dtype, device)
    compact_shape = (env_values.size, joint_values.size)
    position_values = np.arange(np.prod(compact_shape), dtype=np.float32).reshape(compact_shape)
    velocity_values = position_values + np.float32(0.25)
    position = wp.array(position_values, dtype=wp.float32, device=device)
    velocity = wp.array(velocity_values, dtype=wp.float32, device=device)
    user_to_backend_values = np.roll(np.arange(num_joints, dtype=np.int32), 1)
    user_to_backend = wp.array(user_to_backend_values, dtype=wp.int32, device=device)
    outputs = [
        wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device),
        wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device),
        wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device),
        wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device),
        wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device),
        wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device),
    ]
    return PreparedLaunch(
        kernel=write_joint_state_user_to_backend_with_indices_kernel(env_ids, joint_ids),
        launch_dim=compact_shape,
        inputs=[position, velocity, env_ids, joint_ids, user_to_backend, True, False],
        outputs=outputs,
    )


def _launch_once(prepared: PreparedLaunch, device: wp.context.Device) -> None:
    wp.launch(
        prepared.kernel,
        dim=prepared.launch_dim,
        inputs=prepared.inputs,
        outputs=prepared.outputs,
        device=device,
    )


def _assert_output_parity(
    int32_launch: PreparedLaunch,
    int64_launch: PreparedLaunch,
    device: wp.context.Device,
) -> None:
    _launch_once(int32_launch, device)
    _launch_once(int64_launch, device)
    wp.synchronize_device(device)
    for int32_output, int64_output in zip(int32_launch.outputs, int64_launch.outputs, strict=True):
        np.testing.assert_array_equal(int32_output.numpy(), int64_output.numpy())


def _warm_specializations(
    int32_launch: PreparedLaunch,
    int64_launch: PreparedLaunch,
    device: wp.context.Device,
    num_warmup_launches: int,
) -> None:
    for _ in range(num_warmup_launches):
        _launch_once(int32_launch, device)
        _launch_once(int64_launch, device)
    wp.synchronize_device(device)


def _measure_launch(
    prepared: PreparedLaunch,
    device: wp.context.Device,
    num_iterations: int,
) -> float:
    start = wp.Event(device=device, enable_timing=True)
    end = wp.Event(device=device, enable_timing=True)
    stream = device.stream
    stream.record_event(start)
    for _ in range(num_iterations):
        _launch_once(prepared, device)
    stream.record_event(end)
    wp.synchronize_event(end)
    return 1000.0 * wp.get_event_elapsed_time(start, end) / num_iterations


def _record_measurements(
    runner: MethodBenchmarkRunner,
    phase: str,
    int32_mean: float,
    int64_mean: float,
) -> None:
    delta_us = int64_mean - int32_mean
    percentage_delta = 100.0 * delta_us / int32_mean
    runner.add_measurement(
        phase,
        SingleMeasurement(name="int32", value=int32_mean, unit="us"),
    )
    runner.add_measurement(
        phase,
        SingleMeasurement(name="int64", value=int64_mean, unit="us"),
    )
    runner.add_measurement(
        phase,
        SingleMeasurement(name="absolute_delta", value=abs(delta_us), unit="us"),
    )
    runner.add_measurement(
        phase,
        SingleMeasurement(name="percentage_delta", value=percentage_delta, unit="%"),
    )


def _resolve_cuda_device(device_name: str) -> wp.context.Device | None:
    device = wp.get_device(device_name)
    return device if device.is_cuda else None


def run_index_kernel_dtype_benchmark(
    runner: MethodBenchmarkRunner,
    config: MethodBenchmarkRunnerConfig,
) -> None:
    """Record raw int32/int64 index-kernel timing in an articulation method artifact."""
    device = _resolve_cuda_device(config.device)
    if device is None or config.num_instances < 2 or config.num_joints < 2:
        return

    joint_values = np.roll(np.arange(config.num_joints, dtype=np.int64), 1)
    for suffix, fill_ratio in FILL_RATIOS:
        num_selected_envs = max(1, int(config.num_instances * fill_ratio))
        env_values = np.roll(np.arange(config.num_instances, dtype=np.int64), 1)[:num_selected_envs]
        int32_launch = _prepare_launch(
            env_values,
            joint_values,
            wp.int32,
            device,
            config.num_instances,
            config.num_joints,
        )
        int64_launch = _prepare_launch(
            env_values,
            joint_values,
            wp.int64,
            device,
            config.num_instances,
            config.num_joints,
        )
        _assert_output_parity(int32_launch, int64_launch, device)
        _warm_specializations(
            int32_launch,
            int64_launch,
            device,
            config.warmup_steps,
        )
        int32_mean = _measure_launch(int32_launch, device, config.num_iterations)
        int64_mean = _measure_launch(int64_launch, device, config.num_iterations)
        _record_measurements(
            runner,
            f"index_kernel_{suffix}",
            int32_mean,
            int64_mean,
        )
