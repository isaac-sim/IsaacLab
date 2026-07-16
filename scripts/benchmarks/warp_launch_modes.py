# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark Warp launch modes on a mixed production-like kernel sequence.

The sequence models the recurring action, observation, reward, and termination
kernel roles in the experimental Warp Cartpole task. It compares eager launches,
bare recorded commands, typed and documented no-conversion command updates,
the Isaac Lab ``WarpLaunchCache`` wrapper, and CUDA graphs. One-time
recording/capture costs, host submission, synchronized wall time, and GPU
activity are reported separately.

Example::

    ./isaaclab.sh -p scripts/benchmarks/warp_launch_modes.py \
        --device cuda:0 --threads 256,1024,4096,16384 \
        --stage_repeats 1,4,16,64 --output_dir results/warp_launch_modes
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import hashlib
import itertools
import json
import math
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

import warp as wp

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ISAACLAB_SOURCE = _REPOSITORY_ROOT / "source" / "isaaclab"
if str(_ISAACLAB_SOURCE) not in sys.path:
    sys.path.insert(0, str(_ISAACLAB_SOURCE))

from isaaclab.utils.warp import WarpLaunchCache  # noqa: E402


@wp.kernel
def update_actions(
    input_actions: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    action_scale: wp.float32,
    cart_dof_idx: wp.int32,
):
    """Apply one persistent action buffer to a Cartpole-shaped output."""
    env_index = wp.tid()
    actions[env_index, cart_dof_idx] = action_scale * input_actions[env_index, 0]


@wp.kernel
def get_observations(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    observations: wp.array(dtype=wp.vec4f),
):
    """Gather four Cartpole-shaped state values."""
    env_index = wp.tid()
    observations[env_index][0] = joint_pos[env_index, pole_dof_idx]
    observations[env_index][1] = joint_vel[env_index, pole_dof_idx]
    observations[env_index][2] = joint_pos[env_index, cart_dof_idx]
    observations[env_index][3] = joint_vel[env_index, cart_dof_idx]


@wp.kernel
def compute_rewards(
    rew_scale_alive: wp.float32,
    rew_scale_terminated: wp.float32,
    rew_scale_pole_pos: wp.float32,
    rew_scale_cart_vel: wp.float32,
    rew_scale_pole_vel: wp.float32,
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    reset_terminated: wp.array(dtype=wp.bool),
    reward: wp.array(dtype=wp.float32),
):
    """Compute a representative multi-input reward kernel."""
    env_index = wp.tid()
    alive = wp.where(reset_terminated[env_index], wp.float32(0.0), rew_scale_alive)
    terminated = wp.where(reset_terminated[env_index], rew_scale_terminated, wp.float32(0.0))
    pole_pos = joint_pos[env_index, pole_dof_idx]
    reward[env_index] = (
        alive
        + terminated
        + rew_scale_pole_pos * pole_pos * pole_pos
        + rew_scale_cart_vel * wp.abs(joint_vel[env_index, cart_dof_idx])
        + rew_scale_pole_vel * wp.abs(joint_vel[env_index, pole_dof_idx])
    )


@wp.kernel
def get_dones(
    joint_pos: wp.array2d(dtype=wp.float32),
    episode_length: wp.array(dtype=wp.int32),
    cart_dof_idx: wp.int32,
    pole_dof_idx: wp.int32,
    max_episode_length: wp.int32,
    max_cart_pos: wp.float32,
    out_of_bounds: wp.array(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
    reset: wp.array(dtype=wp.bool),
):
    """Compute representative termination flags."""
    env_index = wp.tid()
    out_of_bounds[env_index] = (wp.abs(joint_pos[env_index, cart_dof_idx]) > max_cart_pos) or (
        wp.abs(joint_pos[env_index, pole_dof_idx]) > wp.pi / 2.0
    )
    time_out[env_index] = episode_length[env_index] >= max_episode_length - 1
    reset[env_index] = out_of_bounds[env_index] or time_out[env_index]


_BASE_NODE_COUNT = 4
_ALL_MODES = (
    "eager",
    "cache_eager",
    "bare_replay",
    "bare_dynamic_normal",
    "bare_dynamic_ctype_constructed",
    "bare_dynamic_ctype_prepacked",
    "cache_static",
    "graph_eager",
    "graph_replay",
)
_CSV_FIELDS = (
    "timestamp_utc",
    "case_id",
    "status",
    "message",
    "sample_index",
    "mode",
    "threads",
    "stage_repeats",
    "cycles_per_step",
    "nodes_per_step",
    "dynamic_nodes_per_step",
    "dynamic_updates_per_step",
    "setter_method",
    "warmup_executions",
    "record_setup_ms",
    "capture_setup_ms",
    "first_execution_batch_ms",
    "cpu_submit_batch_ms",
    "cpu_submit_us_per_node",
    "synchronized_wall_batch_ms",
    "synchronized_wall_us_per_node",
    "gpu_kernel_batch_ms",
    "gpu_graph_batch_ms",
    "gpu_activity_batch_ms",
    "gpu_activity_us_per_node",
    "gpu_activity_type",
    "gpu_activity_count",
    "action_checksum",
    "observation_checksum",
    "reward_checksum",
    "reset_count",
    "dynamic_probe_first_action_checksum",
    "dynamic_probe_second_action_checksum",
    "semantic_validation",
    "device",
)


@dataclasses.dataclass(frozen=True)
class _Case:
    """One point in the mixed-sequence benchmark matrix."""

    mode: str
    threads: int
    stage_repeats: int

    @property
    def nodes_per_step(self) -> int:
        return _BASE_NODE_COUNT * self.stage_repeats

    @property
    def case_id(self) -> str:
        return f"{self.mode}__threads_{self.threads}__nodes_{self.nodes_per_step}"


@dataclasses.dataclass
class _Node:
    """One retained kernel call in the production-like sequence."""

    kernel: wp.Kernel
    arguments: list[Any]
    dynamic_index: int | None = None


@dataclasses.dataclass
class _Workload:
    """Buffers and nodes retained for one benchmark case."""

    nodes: list[_Node]
    action_banks: tuple[wp.array, wp.array]
    actions: wp.array
    reward: wp.array
    observations: wp.array
    reset_mask: wp.array


@dataclasses.dataclass
class _PreparedMode:
    """Prepared steady-state callback and one-time setup metadata."""

    execute: Callable[[], None]
    record_setup_ms: float
    capture_setup_ms: float
    dynamic_nodes_per_step: int
    dynamic_updates_per_step: int
    setter_method: str


def run(argv: Sequence[str] | None = None) -> int:
    """Run the selected matrix and return a process exit code."""
    args = _parse_args(argv)
    cases = [
        _Case(mode=mode, threads=threads, stage_repeats=stage_repeats)
        for mode, threads, stage_repeats in itertools.product(args.modes, args.threads, args.stage_repeats)
    ]
    if args.case_order == "randomized":
        random.Random(args.case_seed).shuffle(cases)
    if args.dry_run:
        for case in cases:
            print(case.case_id)
        print(f"Matrix cases: {len(cases)}")
        return 0

    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("This benchmark requires a CUDA device because two modes use CUDA graphs.")

    started_utc = _utc_now()
    run_name = args.run_name or f"warp_launch_modes_{started_utc.replace('-', '').replace(':', '')[:15]}Z"
    _validate_run_name(run_name)
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_name}.csv"
    json_path = output_dir / f"{run_name}.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing benchmark output for run {run_name!r}.")

    kernel_source = pathlib.Path(__file__).resolve()
    cache_source = pathlib.Path(sys.modules[WarpLaunchCache.__module__].__file__).resolve()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_utc": started_utc,
        "completed_utc": None,
        "command": [sys.executable, str(pathlib.Path(__file__).resolve()), *sys.argv[1:]],
        "arguments": vars(args),
        "matrix": {"total_cases": len(cases), "completed_cases": 0, "failed_cases": 0},
        "software": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "warp_path": str(pathlib.Path(wp.__file__).resolve()),
            "git_commit": _git_commit(),
        },
        "sources": {
            "benchmark": _file_metadata(pathlib.Path(__file__)),
            "warp_launch_cache": _file_metadata(cache_source),
        },
        "device": _device_metadata(device),
        "kernel_sequence": {
            "source": str(kernel_source),
            "source_sha256": hashlib.sha256(kernel_source.read_bytes()).hexdigest(),
            "base_nodes": ["update_actions", "get_observations", "compute_rewards", "get_dones"],
        },
        "metric_notes": {
            "cpu_submit_batch_ms": "Host time to submit one complete sequence; trailing synchronization excluded.",
            "synchronized_wall_batch_ms": "Wall time with device synchronization before and after the sequence.",
            "gpu_activity_batch_ms": "Warp CUDA activity attributed to kernels or the graph during the sequence.",
            "record_setup_ms": (
                "Host time to record every command/replay-cache entry; replay-cache setup also submits its first call."
            ),
            "capture_setup_ms": "Host wall time to construct the CUDA graph after kernel precompilation.",
            "dynamic_updates_per_step": (
                "Actual setter calls expected in steady state; unchanged cache tokens are elided."
            ),
            "bare_dynamic_ctype_constructed": (
                "Constructs an array descriptor inside every update, then uses Warp's documented "
                "set_param_at_index_from_ctype no-conversion API."
            ),
            "bare_dynamic_ctype_prepacked": (
                "Reuses retained array descriptors constructed before steady-state timing, then uses Warp's "
                "documented set_param_at_index_from_ctype no-conversion API."
            ),
        },
        "rows": [],
    }
    _write_outputs(csv_path, json_path, manifest, [])

    _precompile(device)
    rows: list[dict[str, Any]] = []
    failed_cases = 0
    for case_index, case in enumerate(cases, start=1):
        print(f"[{case_index}/{len(cases)}] {case.case_id}", flush=True)
        try:
            case_rows = _run_case(case, args.warmup_steps, args.warmup_seconds, args.repeats, device)
        except Exception as exc:
            failed_cases += 1
            case_rows = [
                _make_row(
                    case,
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                    sample_index=None,
                    device=device.alias,
                )
            ]
            print(f"FAILED {case.case_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        rows.extend(case_rows)
        manifest["rows"] = rows
        manifest["matrix"]["completed_cases"] = case_index - failed_cases
        manifest["matrix"]["failed_cases"] = failed_cases
        _write_outputs(csv_path, json_path, manifest, rows)

    manifest["status"] = "completed" if failed_cases == 0 else "completed_with_failures"
    manifest["completed_utc"] = _utc_now()
    _write_outputs(csv_path, json_path, manifest, rows)
    print(f"Cases: {len(cases) - failed_cases} completed, {failed_cases} failed")
    print(f"CSV: {csv_path}")
    return 0 if failed_cases == 0 else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a mixed sequence of recurring Warp Cartpole kernels.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=_positive_int_list, default=(256, 1024, 4096, 16384))
    parser.add_argument("--stage_repeats", type=_positive_int_list, default=(1, 4, 16, 64))
    parser.add_argument("--modes", type=_mode_list, default=_ALL_MODES)
    parser.add_argument("--warmup_steps", type=_nonnegative_int, default=5)
    parser.add_argument(
        "--warmup_seconds",
        type=_nonnegative_float,
        default=1.0,
        help="Minimum synchronized warm-up duration per case to stabilize adaptive clocks.",
    )
    parser.add_argument("--repeats", type=_positive_int, default=10)
    parser.add_argument("--case_order", choices=("grouped", "randomized"), default="randomized")
    parser.add_argument("--case_seed", type=int, default=271828)
    parser.add_argument("--output_dir", default="results/warp_launch_modes")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _positive_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(dict.fromkeys(_positive_int(item.strip()) for item in value.split(",") if item.strip()))
    if not parsed:
        raise argparse.ArgumentTypeError("at least one positive integer is required")
    return parsed


def _mode_list(value: str) -> tuple[str, ...]:
    parsed = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = [mode for mode in parsed if mode not in _ALL_MODES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown modes {unknown}; choose from {', '.join(_ALL_MODES)}")
    if not parsed:
        raise argparse.ArgumentTypeError("at least one mode is required")
    return parsed


def _precompile(device: wp.Device) -> None:
    """Compile every production kernel before setup timing."""
    workload = _make_workload(1, 1, device)
    for node in workload.nodes:
        wp.launch(node.kernel, dim=1, inputs=node.arguments, device=device)
    wp.synchronize_device(device)


def _make_workload(threads: int, stage_repeats: int, device: wp.Device) -> _Workload:
    """Allocate Cartpole-shaped buffers and construct the repeated node sequence."""
    joint_pos = wp.full((threads, 2), 0.1, dtype=wp.float32, device=device)
    joint_vel = wp.full((threads, 2), 0.2, dtype=wp.float32, device=device)
    action_banks = (
        wp.full((threads, 1), 0.25, dtype=wp.float32, device=device),
        wp.full((threads, 1), -0.25, dtype=wp.float32, device=device),
    )
    actions = wp.zeros((threads, 2), dtype=wp.float32, device=device)
    observations = wp.zeros(threads, dtype=wp.vec4f, device=device)
    episode_length = wp.zeros(threads, dtype=wp.int32, device=device)
    out_of_bounds = wp.zeros(threads, dtype=wp.bool, device=device)
    time_out = wp.zeros(threads, dtype=wp.bool, device=device)
    reset_mask = wp.zeros(threads, dtype=wp.bool, device=device)
    reward = wp.zeros(threads, dtype=wp.float32, device=device)

    nodes: list[_Node] = []
    for _ in range(stage_repeats):
        nodes.extend(
            (
                _Node(
                    update_actions,
                    [action_banks[0], actions, 1.0, 0],
                    dynamic_index=0,
                ),
                _Node(
                    get_observations,
                    [joint_pos, joint_vel, 0, 1, observations],
                ),
                _Node(
                    compute_rewards,
                    [1.0, -2.0, -1.0, -0.01, -0.005, joint_pos, joint_vel, 0, 1, out_of_bounds, reward],
                ),
                _Node(
                    get_dones,
                    [joint_pos, episode_length, 0, 1, 500, 3.0, out_of_bounds, time_out, reset_mask],
                ),
            )
        )
    return _Workload(
        nodes=nodes,
        action_banks=action_banks,
        actions=actions,
        reward=reward,
        observations=observations,
        reset_mask=reset_mask,
    )


def _record_commands(workload: _Workload, threads: int, device: wp.Device) -> tuple[list[wp.Launch], float]:
    wp.synchronize_device(device)
    start_ns = time.perf_counter_ns()
    commands = [
        wp.launch(node.kernel, dim=threads, inputs=node.arguments, device=device, record_cmd=True)
        for node in workload.nodes
    ]
    record_setup_ms = (time.perf_counter_ns() - start_ns) / 1.0e6
    if any(command is None for command in commands):
        raise RuntimeError("wp.launch(record_cmd=True) did not return a command for every sequence node.")
    return commands, record_setup_ms  # type: ignore[return-value]


def _prepare_mode(case: _Case, workload: _Workload, device: wp.Device) -> _PreparedMode:
    """Prepare one execution path and time its one-time setup."""
    record_setup_ms = 0.0
    capture_setup_ms = 0.0
    dynamic_nodes = sum(node.dynamic_index is not None for node in workload.nodes)

    if case.mode == "eager":

        def execute() -> None:
            for node in workload.nodes:
                wp.launch(node.kernel, dim=case.threads, inputs=node.arguments, device=device)

        return _PreparedMode(execute, 0.0, 0.0, 0, 0, "none")

    if case.mode in {
        "bare_replay",
        "bare_dynamic_normal",
        "bare_dynamic_ctype_constructed",
        "bare_dynamic_ctype_prepacked",
        "graph_replay",
    }:
        commands, record_setup_ms = _record_commands(workload, case.threads, device)
    else:
        commands = []

    if case.mode == "bare_replay":

        def execute() -> None:
            for command in commands:
                command.launch()

        return _PreparedMode(execute, record_setup_ms, 0.0, 0, 0, "none")

    if case.mode in {
        "bare_dynamic_normal",
        "bare_dynamic_ctype_constructed",
        "bare_dynamic_ctype_prepacked",
    }:
        execution_index = 0
        action_descriptors = tuple(action.__ctype__() for action in workload.action_banks)

        def execute() -> None:
            nonlocal execution_index
            action_index = (execution_index + 1) & 1
            action = workload.action_banks[action_index]
            for node, command in zip(workload.nodes, commands):
                if node.dynamic_index is not None:
                    if case.mode == "bare_dynamic_normal":
                        command.set_param_at_index(node.dynamic_index, action)
                    elif case.mode == "bare_dynamic_ctype_constructed":
                        command.set_param_at_index_from_ctype(node.dynamic_index, action.__ctype__())
                    else:
                        command.set_param_at_index_from_ctype(node.dynamic_index, action_descriptors[action_index])
                command.launch()
            execution_index += 1

        setter = {
            "bare_dynamic_normal": "set_param_at_index",
            "bare_dynamic_ctype_constructed": "set_param_at_index_from_ctype + array.__ctype__",
            "bare_dynamic_ctype_prepacked": "set_param_at_index_from_ctype (prepacked)",
        }[case.mode]
        return _PreparedMode(execute, record_setup_ms, 0.0, dynamic_nodes, dynamic_nodes, setter)

    if case.mode in {"cache_eager", "cache_static"}:
        cache = WarpLaunchCache(
            mode="eager" if case.mode == "cache_eager" else "replay",
            debug=False,
            device=device,
        )
        if case.mode != "cache_eager":
            wp.synchronize_device(device)
            start_ns = time.perf_counter_ns()
            for node in workload.nodes:
                cache.launch(
                    node.kernel,
                    dim=case.threads,
                    inputs=node.arguments,
                )
            record_setup_ms = (time.perf_counter_ns() - start_ns) / 1.0e6

        def execute() -> None:
            for node in workload.nodes:
                cache.launch(
                    node.kernel,
                    dim=case.threads,
                    inputs=node.arguments,
                )

        return _PreparedMode(execute, record_setup_ms, 0.0, 0, 0, "none")

    if case.mode in {"graph_eager", "graph_replay"}:
        wp.synchronize_device(device)
        start_ns = time.perf_counter_ns()
        with wp.ScopedCapture(device=device) as capture:
            if case.mode == "graph_eager":
                for node in workload.nodes:
                    wp.launch(node.kernel, dim=case.threads, inputs=node.arguments, device=device)
            else:
                for command in commands:
                    command.launch()
        capture_setup_ms = (time.perf_counter_ns() - start_ns) / 1.0e6
        graph = capture.graph

        def execute() -> None:
            wp.capture_launch(graph)

        return _PreparedMode(execute, record_setup_ms, capture_setup_ms, 0, 0, "none")

    raise ValueError(f"Unknown mode: {case.mode}")


def _run_case(
    case: _Case,
    warmup_steps: int,
    warmup_seconds: float,
    repeats: int,
    device: wp.Device,
) -> list[dict[str, Any]]:
    workload = _make_workload(case.threads, case.stage_repeats, device)
    prepared = _prepare_mode(case, workload, device)

    with wp.ScopedTimer("first_execution", print=False, synchronize=True) as first_timer:
        prepared.execute()
    warmup_executions = _warm_up(prepared.execute, warmup_steps, warmup_seconds, device)

    rows: list[dict[str, Any]] = []
    for sample_index in range(repeats):
        wp.synchronize_device(device)
        submit_start_ns = time.perf_counter_ns()
        prepared.execute()
        cpu_submit_batch_ms = (time.perf_counter_ns() - submit_start_ns) / 1.0e6
        wp.synchronize_device(device)

        with wp.ScopedTimer(
            "synchronized_sequence",
            print=False,
            synchronize=True,
            cuda_filter=wp.TIMING_KERNEL | wp.TIMING_GRAPH,
        ) as synchronized_timer:
            prepared.execute()

        kernel_results = [timing for timing in synchronized_timer.timing_results if timing.filter & wp.TIMING_KERNEL]
        graph_results = [timing for timing in synchronized_timer.timing_results if timing.filter & wp.TIMING_GRAPH]
        (
            gpu_kernel_batch_ms,
            gpu_graph_batch_ms,
            gpu_activity_batch_ms,
            gpu_activity_type,
            gpu_activity_count,
        ) = _select_gpu_activity(case, kernel_results, graph_results)
        expected_activity_count = (
            1 if case.mode.startswith("graph_") and gpu_activity_type == "graph" else case.nodes_per_step
        )
        if gpu_activity_count != expected_activity_count:
            raise RuntimeError(
                f"GPU activity count mismatch for {case.case_id}: measured {gpu_activity_count}, "
                f"expected {expected_activity_count}."
            )

        rows.append(
            _make_row(
                case,
                status="ok",
                sample_index=sample_index,
                dynamic_nodes_per_step=prepared.dynamic_nodes_per_step,
                dynamic_updates_per_step=prepared.dynamic_updates_per_step,
                setter_method=prepared.setter_method,
                warmup_executions=warmup_executions,
                record_setup_ms=prepared.record_setup_ms,
                capture_setup_ms=prepared.capture_setup_ms,
                first_execution_batch_ms=first_timer.elapsed,
                cpu_submit_batch_ms=cpu_submit_batch_ms,
                cpu_submit_us_per_node=cpu_submit_batch_ms * 1_000.0 / case.nodes_per_step,
                synchronized_wall_batch_ms=synchronized_timer.elapsed,
                synchronized_wall_us_per_node=synchronized_timer.elapsed * 1_000.0 / case.nodes_per_step,
                gpu_kernel_batch_ms=gpu_kernel_batch_ms,
                gpu_graph_batch_ms=gpu_graph_batch_ms,
                gpu_activity_batch_ms=gpu_activity_batch_ms,
                gpu_activity_us_per_node=(
                    gpu_activity_batch_ms * 1_000.0 / case.nodes_per_step if gpu_activity_batch_ms is not None else None
                ),
                gpu_activity_type=gpu_activity_type,
                gpu_activity_count=gpu_activity_count,
                device=device.alias,
            )
        )

    action_checksum = float(workload.actions.numpy().sum())
    observation_checksum = float(workload.observations.numpy().sum())
    reward_checksum = float(workload.reward.numpy().sum())
    reset_count = int(workload.reset_mask.numpy().sum())
    total_executions = 1 + warmup_executions + 2 * repeats
    expected = _semantic_expectations(case, total_executions)
    actual = {
        "action_checksum": action_checksum,
        "observation_checksum": observation_checksum,
        "reward_checksum": reward_checksum,
        "reset_count": reset_count,
    }
    if not all(math.isfinite(value) for value in (action_checksum, observation_checksum, reward_checksum)):
        raise RuntimeError(f"Non-finite semantic checksum: {actual}")
    if (
        not math.isclose(action_checksum, expected["action_checksum"], rel_tol=1.0e-5, abs_tol=1.0e-5)
        or not math.isclose(observation_checksum, expected["observation_checksum"], rel_tol=1.0e-5, abs_tol=1.0e-5)
        or not math.isclose(reward_checksum, expected["reward_checksum"], rel_tol=1.0e-5, abs_tol=1.0e-5)
        or reset_count != expected["reset_count"]
    ):
        raise RuntimeError(f"Semantic validation failed: actual={actual}, expected={expected}")

    dynamic_probe_checksums: tuple[float | None, float | None] = (None, None)
    if _is_changed_dynamic_mode(case.mode):
        probe_checksums = []
        for _ in range(2):
            prepared.execute()
            wp.synchronize_device(device)
            probe_checksums.append(float(workload.actions.numpy().sum()))
        _validate_dynamic_probe(case, probe_checksums)
        dynamic_probe_checksums = (probe_checksums[0], probe_checksums[1])
    for row in rows:
        row.update(actual)
        row["dynamic_probe_first_action_checksum"] = dynamic_probe_checksums[0]
        row["dynamic_probe_second_action_checksum"] = dynamic_probe_checksums[1]
        row["semantic_validation"] = "passed"
    return rows


def _warm_up(execute: Callable[[], None], minimum_steps: int, minimum_seconds: float, device: wp.Device) -> int:
    """Warm up a case until both count and elapsed-time thresholds are met."""
    executions = 0
    start = time.perf_counter()
    while executions < minimum_steps or time.perf_counter() - start < minimum_seconds:
        execute()
        executions += 1
        if executions % 64 == 0:
            wp.synchronize_device(device)
    wp.synchronize_device(device)
    return executions


def _select_gpu_activity(
    case: _Case,
    kernel_results: Sequence[Any],
    graph_results: Sequence[Any],
) -> tuple[float | None, float | None, float | None, str, int]:
    """Select the GPU activity representation appropriate for an execution mode."""
    gpu_kernel_batch_ms = sum(timing.elapsed for timing in kernel_results) if kernel_results else None
    gpu_graph_batch_ms = sum(timing.elapsed for timing in graph_results) if graph_results else None
    if case.mode.startswith("graph_") and gpu_graph_batch_ms is not None:
        return gpu_kernel_batch_ms, gpu_graph_batch_ms, gpu_graph_batch_ms, "graph", len(graph_results)
    if gpu_kernel_batch_ms is not None:
        return gpu_kernel_batch_ms, gpu_graph_batch_ms, gpu_kernel_batch_ms, "kernel", len(kernel_results)
    if gpu_graph_batch_ms is not None:
        return gpu_kernel_batch_ms, gpu_graph_batch_ms, gpu_graph_batch_ms, "graph", len(graph_results)
    return None, None, None, "none", 0


def _semantic_expectations(case: _Case, total_executions: int) -> dict[str, float | int]:
    """Return analytical checksums for the final state of a benchmark case."""
    final_action_sign = -1.0 if _is_changed_dynamic_mode(case.mode) and total_executions % 2 else 1.0
    return {
        "action_checksum": final_action_sign * 0.25 * case.threads,
        "observation_checksum": 0.6 * case.threads,
        "reward_checksum": 0.987 * case.threads,
        "reset_count": 0,
    }


def _is_changed_dynamic_mode(mode: str) -> bool:
    """Return whether an execution mode alternates its dynamic action pointer."""
    return mode in {
        "bare_dynamic_normal",
        "bare_dynamic_ctype_constructed",
        "bare_dynamic_ctype_prepacked",
    }


def _validate_dynamic_probe(case: _Case, checksums: Sequence[float]) -> None:
    """Validate that two explicit post-timing executions used opposite action banks."""
    expected_magnitude = 0.25 * case.threads
    if len(checksums) != 2 or not (
        math.isclose(checksums[0], -checksums[1], rel_tol=1.0e-5, abs_tol=1.0e-5)
        and math.isclose(abs(checksums[0]), expected_magnitude, rel_tol=1.0e-5, abs_tol=1.0e-5)
    ):
        raise RuntimeError(
            f"Dynamic action probe failed for {case.case_id}: measured {checksums}, "
            f"expected alternating +/-{expected_magnitude}."
        )


def _make_row(case: _Case, **updates: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(_CSV_FIELDS)
    row.update(
        {
            "timestamp_utc": _utc_now(),
            "case_id": case.case_id,
            "mode": case.mode,
            "threads": case.threads,
            "stage_repeats": case.stage_repeats,
            "cycles_per_step": case.stage_repeats,
            "nodes_per_step": case.nodes_per_step,
        }
    )
    row.update(updates)
    return row


def _write_outputs(
    csv_path: pathlib.Path,
    json_path: pathlib.Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    csv_temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    json_temporary = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    with csv_temporary.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_temporary, csv_path)
    with json_temporary.open("w", encoding="utf-8") as json_file:
        json.dump(manifest, json_file, indent=2, sort_keys=True, allow_nan=False)
        json_file.write("\n")
    os.replace(json_temporary, json_path)


def _device_metadata(device: wp.Device) -> dict[str, Any]:
    return {
        "alias": device.alias,
        "name": device.name,
        "arch": getattr(device, "arch", None),
        "ordinal": getattr(device, "ordinal", None),
        "uuid": str(getattr(device, "uuid", "")),
        "is_cuda": device.is_cuda,
    }


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _file_metadata(path: pathlib.Path) -> dict[str, str]:
    """Return an absolute source path and its content digest."""
    path = path.resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _validate_run_name(run_name: str) -> None:
    if not run_name or pathlib.Path(run_name).name != run_name:
        raise ValueError("run_name must be a non-empty filename stem without path separators")


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
