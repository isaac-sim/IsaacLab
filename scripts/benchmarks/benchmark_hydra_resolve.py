# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark task config resolution through IsaacLab's Hydra preset layer.

This measures the pre-Kit path used by training scripts to load an env cfg,
resolve ``PresetCfg`` selections, register the plain Hydra config, run Hydra
scalar overrides, and return the resolved env/agent cfg objects.

The benchmark prints a local summary table and writes per-case measurements to
the standard Isaac Lab benchmark backend. It does not create environments and
does not require a GPU.

Usage::

    ./isaaclab.sh -p scripts/benchmarks/benchmark_hydra_resolve.py
    ./isaaclab.sh -p scripts/benchmarks/benchmark_hydra_resolve.py --suite broad
    ./isaaclab.sh -p scripts/benchmarks/benchmark_hydra_resolve.py --iterations 100
    ./isaaclab.sh -p scripts/benchmarks/benchmark_hydra_resolve.py \
        --case cartpole:Isaac-Cartpole-v0:: \
        --case anymal:Isaac-Velocity-Rough-Anymal-C-v0::env.scene.num_envs=256

Case format is ``name:task:agent_entry:arg[,arg...]``. Leave ``agent_entry`` or
``arg`` empty when not needed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import statistics
import sys
import time
import warnings
from dataclasses import dataclass

import gymnasium

from isaaclab.test.benchmark import BaseIsaacLabBenchmark, SingleMeasurement

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.utils.hydra import resolve_task_config

from scripts.benchmarks.utils import get_backend_type


@dataclass(frozen=True)
class Case:
    name: str
    task: str
    agent_entry: str | None = None
    args: tuple[str, ...] = ()


QUICK_CASES = (
    Case("cartpole_manager", "Isaac-Cartpole-v0"),
    Case("cartpole_camera_presets", "Isaac-Cartpole-Camera-Presets-Direct-v0", "rl_games_cfg_entry_point"),
    Case("anymal_rough", "Isaac-Velocity-Rough-Anymal-C-v0"),
    Case("franka_lift_cube", "Isaac-Lift-Cube-Franka-v0"),
    Case(
        "cartpole_camera_newton_ovrtx",
        "Isaac-Cartpole-Camera-Presets-Direct-v0",
        "rl_games_cfg_entry_point",
        ("presets=newton_mjwarp,ovrtx_renderer",),
    ),
    Case("anymal_rough_scalar", "Isaac-Velocity-Rough-Anymal-C-v0", None, ("env.scene.num_envs=256",)),
)


BROAD_CASES = (
    *QUICK_CASES,
    Case("cartpole_direct", "Isaac-Cartpole-Direct-v0"),
    Case("cartpole_rgb_direct", "Isaac-Cartpole-RGB-Camera-Direct-v0"),
    Case("ant_manager", "Isaac-Ant-v0"),
    Case("humanoid_manager", "Isaac-Humanoid-v0", "rsl_rl_cfg_entry_point"),
    Case("franka_reach", "Isaac-Reach-Franka-v0"),
    Case("franka_lift_cube_agent", "Isaac-Lift-Cube-Franka-v0", "sb3_cfg_entry_point"),
    Case("kuka_allegro_lift", "Isaac-Dexsuite-Kuka-Allegro-Lift-v0", "rsl_rl_cfg_entry_point"),
    Case(
        "kuka_allegro_lift_single_camera",
        "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
        "rsl_rl_cfg_entry_point",
        ("presets=single_camera,rgb128",),
    ),
    Case(
        "kuka_allegro_lift_duo_camera",
        "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
        "rsl_rl_cfg_entry_point",
        ("presets=duo_camera,rgb128",),
    ),
    Case(
        "kuka_allegro_lift_scalar",
        "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
        "rsl_rl_cfg_entry_point",
        ("env.scene.num_envs=256",),
    ),
    Case(
        "cartpole_camera_hydra_force",
        "Isaac-Cartpole-Camera-Presets-Direct-v0",
        "rl_games_cfg_entry_point",
        ("++env.scene.num_envs=256",),
    ),
)

SUITES = {"quick": QUICK_CASES, "broad": BROAD_CASES}


def _parse_case(spec: str) -> Case:
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("case must have format name:task:agent_entry:arg[,arg...]")
    name, task, agent_entry, args = parts
    if not name or not task:
        raise argparse.ArgumentTypeError("case name and task must be non-empty")
    return Case(
        name=name,
        task=task,
        agent_entry=agent_entry or None,
        args=tuple(arg for arg in args.split(",") if arg),
    )


def _resolve_once(case: Case, *, verbose: bool = False) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *case.args]
        if verbose:
            resolve_task_config(case.task, case.agent_entry)
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                resolve_task_config(case.task, case.agent_entry)
    finally:
        sys.argv = old_argv


def _benchmark_case(case: Case, iterations: int, warmup: int, *, verbose: bool) -> list[float]:
    for _ in range(warmup):
        _resolve_once(case, verbose=verbose)

    times_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        _resolve_once(case, verbose=verbose)
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1_000_000)
    return times_ms


def _print_results(results: dict[Case, list[float]]) -> None:
    print(f"\n{'Case':<32} {'median ms':>10} {'mean ms':>10} {'stdev ms':>10} {'min ms':>10} {'max ms':>10}  argv")
    print(f"{'-' * 32} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}  {'-' * 24}")
    for case, times in results.items():
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        argv = " ".join(case.args) if case.args else "(none)"
        print(
            f"{case.name:<32} {statistics.median(times):>10.2f} {statistics.mean(times):>10.2f}"
            f" {stdev:>10.2f} {min(times):>10.2f} {max(times):>10.2f}  {argv}"
        )


def _log_results(benchmark: BaseIsaacLabBenchmark, results: dict[Case, list[float]]) -> None:
    for case, times in results.items():
        stats = {
            "Median Resolve Task Config Time": statistics.median(times),
            "Mean Resolve Task Config Time": statistics.mean(times),
            "Stdev Resolve Task Config Time": statistics.stdev(times) if len(times) > 1 else 0.0,
            "Min Resolve Task Config Time": min(times),
            "Max Resolve Task Config Time": max(times),
        }
        for name, value in stats.items():
            benchmark.add_measurement(
                "task_config",
                measurement=SingleMeasurement(name=f"{case.name} {name}", value=value, unit="ms"),
            )

    benchmark.update_manual_recorders()
    benchmark._finalize_impl()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark env_cfg + Hydra preset resolution.")
    parser.add_argument("--iterations", type=int, default=50, help="Timed iterations per case.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per case.")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="quick",
        help="Named benchmark suite to run when --case is not provided.",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        default=None,
        help="Benchmark case in format name:task:agent_entry:arg[,arg...]. May be repeated.",
    )
    parser.add_argument(
        "--benchmark_backend",
        type=str,
        default="summary",
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
        help="Benchmarking backend options, defaults summary.",
    )
    parser.add_argument("--output_path", type=str, default=".", help="Path to output benchmark results.")
    parser.add_argument("--verbose", action="store_true", help="Keep per-iteration resolver output.")
    args = parser.parse_args()

    cases = tuple(args.case) if args.case else SUITES[args.suite]
    valid_cases = tuple(case for case in cases if case.task in gymnasium.registry)
    skipped = [case.task for case in cases if case.task not in gymnasium.registry]
    if skipped:
        print(f"[WARN] Skipping unregistered task(s): {skipped}")
    if not valid_cases:
        print("[ERROR] No valid benchmark cases.")
        return 1

    print("Benchmarking resolve_task_config()")
    print(f"Iterations: {args.iterations}, warmup: {args.warmup}")

    results = {case: _benchmark_case(case, args.iterations, args.warmup, verbose=args.verbose) for case in valid_cases}
    _print_results(results)

    benchmark = BaseIsaacLabBenchmark(
        benchmark_name="benchmark_hydra_resolve",
        backend_type=get_backend_type(args.benchmark_backend),
        output_path=args.output_path,
        use_recorders=True,
        output_prefix="benchmark_hydra_resolve",
        workflow_metadata={
            "metadata": [
                {"name": "suite", "data": args.suite if not args.case else "custom"},
                {"name": "iterations", "data": args.iterations},
                {"name": "warmup", "data": args.warmup},
            ]
        },
    )
    _log_results(benchmark, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
