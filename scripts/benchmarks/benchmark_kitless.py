# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less benchmark harness for synthetic workloads.

This script runs without Kit/Isaac Sim and emits benchmark JSON compatible with
the existing metrics backends (OmniPerf/JSON/Osmo/Summary).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure local IsaacLab sources are importable (avoid relying on installed package)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SOURCE_ROOT = os.path.join(REPO_ROOT, "source/isaaclab")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, SingleMeasurement

from scripts.benchmarks.utils import (
    get_backend_type,
    log_python_imports_time,
    log_runtime_step_times,
    log_task_start_time,
    log_total_start_time,
)


def _cpu_spin(iterations: int) -> float:
    """Simple CPU-bound loop to simulate work."""
    acc = 0.0
    for i in range(iterations):
        acc += (i % 97) * 0.0001
    return acc


def _prepare_torch_workload(dim: int, device: str):
    import torch

    a = torch.randn((dim, dim), device=device)
    b = torch.randn((dim, dim), device=device)
    return a, b


def _run_torch_workload(a, b, device: str):
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    _ = a @ b
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kit-less benchmark harness.")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to run.")
    parser.add_argument("--num_envs", type=int, default=1, help="Logical env count for throughput math.")
    parser.add_argument(
        "--workload",
        type=str,
        default="sleep",
        choices=["sleep", "cpu_spin", "torch_matmul"],
        help="Synthetic workload type.",
    )
    parser.add_argument("--sleep_ms", type=float, default=1.0, help="Sleep time per step (ms).")
    parser.add_argument("--cpu_iterations", type=int, default=250000, help="CPU spin iterations per step.")
    parser.add_argument("--torch_dim", type=int, default=512, help="Matrix dimension for torch_matmul.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for torch workload (cpu/cuda:0).")
    parser.add_argument(
        "--benchmark_backend",
        type=str,
        default="omniperf",
        choices=[
            "json",
            "osmo",
            "omniperf",
            "summary",
            "LocalLogMetrics",
            "JSONFileMetrics",
            "OsmoKPIFile",
            "OmniPerfKPIFile",
        ],
        help="Benchmarking backend options, defaults omniperf.",
    )
    parser.add_argument("--output_path", type=str, default=".", help="Path to output benchmark results.")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="benchmark_kitless",
        help="Output filename prefix (without extension).",
    )
    parser.add_argument(
        "--monitor_interval",
        type=float,
        default=1.0,
        help="Recorder update interval in seconds; set 0 to disable background monitor.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    script_start_ns = time.perf_counter_ns()
    imports_time_begin = time.perf_counter_ns()

    # Lazy imports for optional workloads
    if args.workload == "torch_matmul":
        import torch  # noqa: F401

    imports_time_end = time.perf_counter_ns()

    backend_type = get_backend_type(args.benchmark_backend)
    benchmark = BaseIsaacLabBenchmark(
        benchmark_name="benchmark_kitless",
        backend_type=backend_type,
        output_path=args.output_path,
        use_recorders=True,
        frametime_recorders=False,
        output_prefix=args.output_prefix,
        workflow_metadata={
            "metadata": [
                {"name": "kitless", "data": True},
                {"name": "workload", "data": args.workload},
                {"name": "num_steps", "data": args.num_steps},
                {"name": "num_envs", "data": args.num_envs},
                {"name": "sleep_ms", "data": args.sleep_ms},
                {"name": "cpu_iterations", "data": args.cpu_iterations},
                {"name": "torch_dim", "data": args.torch_dim},
                {"name": "device", "data": args.device},
            ]
        },
    )

    workload_setup_begin = time.perf_counter_ns()
    torch_buffers = None
    if args.workload == "torch_matmul":
        torch_buffers = _prepare_torch_workload(args.torch_dim, args.device)
    workload_setup_end = time.perf_counter_ns()

    step_times_ns: list[int] = []

    def _run_step() -> None:
        if args.workload == "sleep":
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
        elif args.workload == "cpu_spin":
            _cpu_spin(args.cpu_iterations)
        elif args.workload == "torch_matmul":
            if torch_buffers is None:
                raise RuntimeError("Torch workload requested but buffers are not initialized.")
            _run_torch_workload(torch_buffers[0], torch_buffers[1], args.device)

    if args.monitor_interval > 0:
        monitor_ctx = BenchmarkMonitor(benchmark, interval=args.monitor_interval)
    else:
        monitor_ctx = None

    if monitor_ctx:
        monitor_ctx.start()
    try:
        for _ in range(args.num_steps):
            step_begin = time.perf_counter_ns()
            _run_step()
            step_end = time.perf_counter_ns()
            step_times_ns.append(step_end - step_begin)
    finally:
        if monitor_ctx:
            monitor_ctx.stop()

    # Final recorder update after loop completes
    benchmark.update_manual_recorders()

    step_times_ms = [t / 1e6 for t in step_times_ns]
    fps = [1000.0 / max(t, 1e-9) for t in step_times_ms]
    effective_fps = [value * args.num_envs for value in fps]

    step_metrics = {
        "Kitless step times": step_times_ms,
        "Kitless step FPS": fps,
        "Kitless step effective FPS": effective_fps,
    }

    loop_start_ns = workload_setup_end
    loop_end_ns = time.perf_counter_ns()

    log_python_imports_time(benchmark, (imports_time_end - imports_time_begin) / 1e6)
    log_task_start_time(benchmark, (workload_setup_end - workload_setup_begin) / 1e6)
    log_total_start_time(benchmark, (loop_start_ns - script_start_ns) / 1e6)
    log_runtime_step_times(benchmark, step_metrics, compute_stats=True)

    # Add total runtime measurement for convenience
    benchmark.add_measurement(
        "runtime",
        measurement=SingleMeasurement(
            name="Kitless loop duration",
            value=(loop_end_ns - loop_start_ns) / 1e6,
            unit="ms",
        ),
    )

    benchmark._finalize_impl()


if __name__ == "__main__":
    main()
