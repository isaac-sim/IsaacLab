# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare signed selector widths for a production indexed articulation writer.

This benchmark reports raw kernel timings. It does not characterize end-to-end
articulation method performance or infer a speedup from timing noise.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from isaaclab.assets.articulation.ordering_kernels import (
    write_joint_state_user_to_backend_with_indices_kernel,
)

FILL_RATIOS = (0.05, 0.95, 1.0)


@dataclass(frozen=True)
class PreparedLaunch:
    """Inputs, outputs, and specialization prepared outside the timing interval."""

    kernel: wp.Kernel
    launch_dim: tuple[int, int]
    inputs: list[Any]
    outputs: list[wp.array]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="CUDA device used for the benchmark")
    parser.add_argument("--num_envs", type=int, default=4096, help="Total number of environments")
    parser.add_argument("--num_joints", type=int, default=12, help="Number of selected joints")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Kernel launches per timing sample")
    parser.add_argument("--num_rounds", type=int, default=15, help="Alternating paired timing rounds")
    args = parser.parse_args()
    if args.num_envs < 2:
        parser.error("--num_envs must be at least 2 to construct a nonidentity selector")
    if args.num_joints < 2:
        parser.error("--num_joints must be at least 2 to construct a nonidentity selector")
    if args.num_iterations < 1:
        parser.error("--num_iterations must be positive")
    if args.num_rounds < 1:
        parser.error("--num_rounds must be positive")
    return args


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
    selected_kernel = write_joint_state_user_to_backend_with_indices_kernel(env_ids, joint_ids)
    return PreparedLaunch(
        kernel=selected_kernel,
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
    int32_launch: PreparedLaunch, int64_launch: PreparedLaunch, device: wp.context.Device
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
    num_warmup_launches: int = 10,
) -> None:
    for _ in range(num_warmup_launches):
        _launch_once(int32_launch, device)
        _launch_once(int64_launch, device)
    wp.synchronize_device(device)


def _measure_launch(prepared: PreparedLaunch, device: wp.context.Device, num_iterations: int) -> float:
    start = wp.Event(enable_timing=True)
    end = wp.Event(enable_timing=True)
    wp.record_event(start)
    for _ in range(num_iterations):
        wp.launch(
            prepared.kernel,
            dim=prepared.launch_dim,
            inputs=prepared.inputs,
            outputs=prepared.outputs,
            device=device,
        )
    wp.record_event(end)
    wp.synchronize_event(end)
    elapsed_us = 1000.0 * wp.get_event_elapsed_time(start, end) / num_iterations
    return elapsed_us


def _measure_pairs(
    int32_launch: PreparedLaunch,
    int64_launch: PreparedLaunch,
    device: wp.context.Device,
    num_iterations: int,
    num_rounds: int,
) -> tuple[list[float], list[float]]:
    samples = {"int32": [], "int64": []}
    launches = {"int32": int32_launch, "int64": int64_launch}
    for round_index in range(num_rounds):
        order = ("int32", "int64") if round_index % 2 == 0 else ("int64", "int32")
        for label in order:
            samples[label].append(_measure_launch(launches[label], device, num_iterations))
    return samples["int32"], samples["int64"]


def _mean_and_std(samples: list[float]) -> tuple[float, float]:
    mean = statistics.mean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, std


def _print_results(
    fill_ratio: float, num_selected_envs: int, int32_samples: list[float], int64_samples: list[float]
) -> None:
    int32_mean, int32_std = _mean_and_std(int32_samples)
    int64_mean, int64_std = _mean_and_std(int64_samples)
    delta_us = int64_mean - int32_mean
    absolute_delta_us = abs(delta_us)
    delta_percent = 100.0 * delta_us / int32_mean
    print(f"fill={fill_ratio:.0%} selected_envs={num_selected_envs}")
    print(f"  int32: mean={int32_mean:.6f} us  std={int32_std:.6f} us")
    print(f"  int64: mean={int64_mean:.6f} us  std={int64_std:.6f} us")
    print(f"  absolute delta: {absolute_delta_us:.6f} us")
    print(f"  percentage delta (int64 - int32): {delta_percent:+.3f}%")


def main() -> None:
    args = _parse_args()
    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise ValueError(f"CUDA event timing requires a CUDA device, got {device}.")

    joint_values = np.roll(np.arange(args.num_joints, dtype=np.int64), 1)
    with wp.ScopedDevice(device):
        for fill_ratio in FILL_RATIOS:
            num_selected_envs = max(1, int(args.num_envs * fill_ratio))
            env_values = np.roll(np.arange(args.num_envs, dtype=np.int64), 1)[:num_selected_envs]
            int32_launch = _prepare_launch(env_values, joint_values, wp.int32, device, args.num_envs, args.num_joints)
            int64_launch = _prepare_launch(env_values, joint_values, wp.int64, device, args.num_envs, args.num_joints)
            _assert_output_parity(int32_launch, int64_launch, device)
            _warm_specializations(int32_launch, int64_launch, device)
            int32_samples, int64_samples = _measure_pairs(
                int32_launch,
                int64_launch,
                device,
                args.num_iterations,
                args.num_rounds,
            )
            _print_results(fill_ratio, num_selected_envs, int32_samples, int64_samples)


if __name__ == "__main__":
    main()
