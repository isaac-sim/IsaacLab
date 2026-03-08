# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark lazy_export import performance via parse_env_cfg.

Measures the cold end-to-end time to construct environment configs — the real
user-facing cost that includes gym registry walking, lazy_export stub parsing,
fallback resolution, module imports, and config class instantiation.

Each iteration fully purges task modules and the gym registration guard so
that the next ``import isaaclab_tasks`` re-walks every task package and
re-registers every gym environment, matching a fresh-process cold start.

The report separates **package loading** (``import isaaclab_tasks`` — registry
walk + gym registrations) from **config construction**
(``load_cfg_from_registry`` — module import + class instantiation).

This script does NOT require Isaac Sim or a GPU.

Usage::

    ./isaaclab.sh -p scripts/benchmarks/benchmark_lazy_export.py
    ./isaaclab.sh -p scripts/benchmarks/benchmark_lazy_export.py --iterations 20
    ./isaaclab.sh -p scripts/benchmarks/benchmark_lazy_export.py --tasks Isaac-Velocity-Flat-Anymal-D-v0
"""

from __future__ import annotations

import argparse
import builtins
import io
import statistics
import sys
import time
import warnings

import gymnasium

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_REPRESENTATIVE_TASKS = [
    "Isaac-Cartpole-v0",
    "Isaac-Humanoid-v0",
    "Isaac-Velocity-Flat-Anymal-D-v0",
    "Isaac-Reach-Franka-v0",
    "Isaac-Lift-Cube-Franka-v0",
    "Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
    "Isaac-Navigation-Flat-Anymal-C-v0",
    "Isaac-Stack-Cube-Franka-v0",
]


def _purge_all() -> None:
    """Fully purge task modules, gym registrations, and the registration guard."""
    to_delete = [k for k in sys.modules if k.startswith("isaaclab_tasks.")]
    for k in to_delete:
        del sys.modules[k]
    del sys.modules["isaaclab_tasks"]

    builtins._isaaclab_tasks_registered = False

    isaac_ids = [name for name in gymnasium.registry if name.startswith("Isaac-")]
    for name in isaac_ids:
        del gymnasium.registry[name]


def _short_name(task: str) -> str:
    return task.replace("Isaac-", "").replace("-v0", "")


def benchmark(
    tasks: list[str], iterations: int
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    """Benchmark with package loading and config construction timed separately.

    Returns:
        Three dicts keyed by task name, each mapping to a list of per-iteration
        times in ms: (pkg_load_times, cfg_construct_times, total_times).
    """
    pkg_results: dict[str, list[float]] = {t: [] for t in tasks}
    cfg_results: dict[str, list[float]] = {t: [] for t in tasks}
    total_results: dict[str, list[float]] = {t: [] for t in tasks}

    for _ in range(iterations):
        for task in tasks:
            _purge_all()

            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    t0 = time.perf_counter_ns()
                    import isaaclab_tasks  # noqa: F811
                    t1 = time.perf_counter_ns()
                    load_cfg_from_registry(task, "env_cfg_entry_point")
                    t2 = time.perf_counter_ns()
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

            pkg_ms = (t1 - t0) / 1_000_000
            cfg_ms = (t2 - t1) / 1_000_000
            pkg_results[task].append(pkg_ms)
            cfg_results[task].append(cfg_ms)
            total_results[task].append(pkg_ms + cfg_ms)

    return pkg_results, cfg_results, total_results


def _print_table(title: str, results: dict[str, list[float]], unit: str = "ms") -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Task':<40} {'median':>10} {'mean':>10} {'stdev':>10}  ({unit})")
    print(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")

    all_medians: list[float] = []
    for task, times in results.items():
        short = _short_name(task)
        med = statistics.median(times)
        avg = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        all_medians.append(med)
        print(f"  {short:<40} {med:>10.2f} {avg:>10.2f} {std:>10.2f}")

    total_med = sum(all_medians)
    avg_med = total_med / len(all_medians) if all_medians else 0
    print(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"  {'TOTAL (sum of medians)':<40} {total_med:>10.2f}")
    print(f"  {'AVERAGE (per task)':<40} {avg_med:>10.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark lazy_export via cold parse_env_cfg.")
    parser.add_argument("--iterations", type=int, default=20, help="Iterations per benchmark.")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task IDs to benchmark. Defaults to a representative set.",
    )
    args = parser.parse_args()
    n = args.iterations

    tasks = args.tasks or _REPRESENTATIVE_TASKS
    valid = [t for t in tasks if t in gymnasium.registry]
    skipped = [t for t in tasks if t not in gymnasium.registry]
    if skipped:
        print(f"[WARN] Skipping unregistered tasks: {skipped}")
    tasks = valid

    if not tasks:
        print("[ERROR] No valid tasks found.")
        return

    print(f"Benchmarking cold parse_env_cfg with {n} iterations")
    print(f"Tasks ({len(tasks)}): {[_short_name(t) for t in tasks]}")

    pkg, cfg, total = benchmark(tasks, n)

    _print_table("Package loading (import isaaclab_tasks — registry walk)", pkg)
    _print_table("Config construction (load_cfg_from_registry)", cfg)
    _print_table("Total (package loading + config construction)", total)


if __name__ == "__main__":
    main()
