# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local end-to-end benchmark runner without Docker/Github actions

Mirrors the structure of the three-phase CI workflow:

    Phase 1  bench        tasks.json → matrix → benchmark_non_rl.py (per task)
    Phase 2  post-bench   build_bench_result.py (per task, plain Python)
    Phase 3  aggregate    aggregate.py + oracle → verdict table

Usage::

    # Preview the full task matrix without running anything
    python3 tools/perf_regression_gate/local_runner.py --dry_run

    # Seed/build baselines (run a few times on stable main)
    python3 tools/perf_regression_gate/local_runner.py --allow_baseline_update

    # Check a branch against baselines (no update)
    python3 tools/perf_regression_gate/local_runner.py

    # Run only "always"-tagged tasks on L40S label
    python3 tools/perf_regression_gate/local_runner.py --tags always --gpu_model L40S

Preconditions:
    IsaacSim must be accessible via ./isaaclab.sh -p.
    Activate venv before running
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_TOOLS_DIR = _MODULE_DIR.parent
_REPO_ROOT = _TOOLS_DIR.parent

if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from task_config import TaskConfig, load_tasks  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local end-to-end benchmark runner (no Docker required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tags",
        nargs="+",
        default=["always"],
        help="Run only tasks whose tag list overlaps this set (default: always)",
    )
    p.add_argument("--gpu_model", default="L40S", help="GPU model label for baseline bucket (default: L40S)")
    p.add_argument(
        "--artifacts_dir",
        type=Path,
        default=_MODULE_DIR / "artifacts",
        help="Root directory for per-task benchmark artifacts (default: tools/perf_regression_gate/artifacts/)",
    )
    p.add_argument(
        "--baselines_dir",
        type=Path,
        default=_MODULE_DIR / "local_baselines",
        help="Flat-file baseline directory (default: tools/perf_regression_gate/local_baselines/)",
    )
    p.add_argument(
        "--allow_baseline_update",
        action="store_true",
        help="Append this run's FPS to the baseline window (use when building baselines on stable main)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the expanded task matrix and exit without running anything",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip tasks whose bench_result.json already exists in --artifacts_dir",
    )
    p.add_argument(
        "--gate_config",
        type=Path,
        default=_MODULE_DIR / "gate_config.json",
        help="Path to gate_config.json (controls advisory vs blocking mode)",
    )
    p.add_argument(
        "--cache-dir",
        dest="cache_dir",
        default=None,
        help="Persist Warp/CUDA JIT caches here for warm runs (local stand-in for the S3 "
        "sidecar; in CI this dir is restored from / saved to object storage). Omit to run "
        "cold -- purely additive, never changes the verdict.",
    )
    return p.parse_args()


def _cache_env(cache_dir: str | None) -> dict[str, str] | None:
    """Build the JIT-cache env overlay for a warm run, or None to run cold.

    Persisting ``WARP_CACHE_PATH`` / ``CUDA_CACHE_PATH`` across runs is the whole of
    the sidecar mechanism: the first run populates the dir (cold), later runs reuse it
    (warm). The dominant cold-start cost is Newton/Warp JIT compilation.
    """
    if not cache_dir:
        return None
    root = Path(cache_dir).resolve()
    warp_dir = root / "warp"
    cuda_dir = root / "nv"
    warp_dir.mkdir(parents=True, exist_ok=True)
    cuda_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["WARP_CACHE_PATH"] = str(warp_dir)
    env["CUDA_CACHE_PATH"] = str(cuda_dir)
    return env


# ---------------------------------------------------------------------------
# Phase 1 helpers: matrix expansion and benchmark execution
# ---------------------------------------------------------------------------


def _print_matrix(tasks: list[TaskConfig], tags: list[str]) -> None:
    print(f"\n{'=' * 68}")
    print(f"  Phase 1 — Task Matrix  ({len(tasks)} entries, tags={tags})")
    print(f"{'=' * 68}")
    col = "{:<42} {:<14} {:>8} {:>8} {:>5}"
    print(col.format("task_id", "backend_key", "num_envs", "frames", "tmin"))
    print("-" * 68)
    for t in tasks:
        print(col.format(t.task_id, t.backend_key, t.num_envs, t.num_frames, t.timeout_minutes))
    print()


# Physics Hydra group value per backend. Launching with an explicit ``physics=``
# (rather than folding Newton into a preset bundle) lets the run echo the backend
# back via benchmark_info.physics, which the config-drift guard verifies.
_PHYSICS_TOKEN = {"physx": "physx", "newton": "newton_mjwarp"}


def _isaaclab_cmd(bench_script: Path, task: TaskConfig, artifact_dir: Path) -> list[str]:
    """Build the ./isaaclab.sh -p benchmark_non_rl.py command for one task/backend."""
    cmd = [
        str(_REPO_ROOT / "isaaclab.sh"),
        "-p",
        str(bench_script),
        "--task",
        task.task_id,
        "--num_envs",
        str(task.num_envs),
        "--num_frames",
        str(task.num_frames),
        "--benchmark_backend",
        "json",
        "--output_path",
        str(artifact_dir),
    ]
    if task.enable_cameras:
        cmd.append("--enable_cameras")
    if task.seed is not None:
        cmd.extend(["--seed", str(task.seed)])
    # Hydra overrides (appended after the argparse flags).
    cmd.append(f"physics={_PHYSICS_TOKEN.get(task.physics_backend, task.physics_backend)}")
    if task.render_backend:
        cmd.append(f"presets={task.render_backend}")
    if task.camera_resolution:
        w, h = task.camera_resolution
        cmd.extend([f"env.tiled_camera.width={w}", f"env.tiled_camera.height={h}"])
    return cmd


def _run_benchmark(
    task: TaskConfig, artifact_dir: Path, bench_script: Path, cache_env: dict[str, str] | None = None
) -> tuple[int, float]:
    """Run benchmark_non_rl.py for one task, streaming output live + writing to log.

    Returns (exit_code, wall_time_s).
    """
    log_file = artifact_dir / "benchmark.log"
    cmd = _isaaclab_cmd(bench_script, task, artifact_dir)

    print(f"  cmd: {' '.join(cmd)}")
    print(f"  log: {log_file}")
    print()

    start = time.monotonic()
    with open(log_file, "w") as log_fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=cache_env)
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace")
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fh.write(line)
        proc.wait()
    wall_time = time.monotonic() - start
    return proc.returncode, wall_time


# ---------------------------------------------------------------------------
# Phase 2 helper: build_bench_result.py
# ---------------------------------------------------------------------------


def _run_build_bench_result(
    task: TaskConfig,
    artifact_dir: Path,
    exit_code: int,
    wall_time: float,
    *,
    was_retried: bool = False,
    attempt: int = 1,
) -> None:
    """Run build_bench_result.py for one task (plain Python, no IsaacSim needed)"""
    build_script = _MODULE_DIR / "build_bench_result.py"
    cmd = [
        sys.executable,
        str(build_script),
        "--task_id",
        task.task_id,
        "--physics_backend",
        task.physics_backend,
        "--render_backend",
        task.render_backend or "",
        "--artifact_dir",
        str(artifact_dir),
        "--exit_code",
        str(exit_code),
        "--wall_time_s",
        f"{wall_time:.1f}",
        "--timeout_s",
        str(task.timeout_minutes * 60),
        "--log_file",
        str(artifact_dir / "benchmark.log"),
        "--attempt",
        str(attempt),
    ]
    if was_retried:
        cmd.append("--was_retried")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Phase 3 helper: aggregate.py
# ---------------------------------------------------------------------------


def _run_aggregate(
    artifacts_dir: Path, baselines_dir: Path, gpu_model: str, gate_config: Path, allow_baseline_update: bool
) -> int:
    """Run aggregate.py over all artifacts and return its exit code."""
    agg_script = _MODULE_DIR / "aggregate.py"
    cmd = [
        sys.executable,
        str(agg_script),
        "--artifacts_dir",
        str(artifacts_dir),
        "--gpu_model",
        gpu_model,
        "--baselines_dir",
        str(baselines_dir),
        "--gate_config",
        str(gate_config),
        "--allow_baseline_update",
        "true" if allow_baseline_update else "false",
    ]
    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()

    # Expand matrix from tasks.json
    all_tasks = load_tasks()
    tag_set = frozenset(args.tags)
    tasks = [t for t in all_tasks if tag_set.intersection(frozenset(t.tags))]

    if not tasks:
        print(f"[local_runner] No tasks match tags {args.tags}.")
        return 1

    _print_matrix(tasks, args.tags)

    if args.dry_run:
        print("[local_runner] --dry_run: exiting without running benchmarks.")
        return 0

    bench_script = _REPO_ROOT / "scripts" / "benchmarks" / "benchmark_non_rl.py"
    if not bench_script.exists():
        print(f"[local_runner] ERROR: benchmark script not found at {bench_script}")
        return 1

    cache_env = _cache_env(args.cache_dir)
    if cache_env is not None:
        print(f"[local_runner] warm-cache overlay: {args.cache_dir}")

    # -----------------------------------------------------------------------
    # Phase 1 + 2: benchmark → build_bench_result, one task at a time
    # -----------------------------------------------------------------------
    print(f"{'=' * 68}")
    print("  Phase 1+2 — Benchmark + Post-process")
    print(f"{'=' * 68}\n")

    for i, task in enumerate(tasks, 1):
        artifact_dir = args.artifacts_dir / task.task_id / task.backend_key
        artifact_dir.mkdir(parents=True, exist_ok=True)

        bench_result_path = artifact_dir / "bench_result.json"
        if args.skip_existing and bench_result_path.exists():
            print(f"[{i}/{len(tasks)}] SKIP (bench_result.json exists): {task.task_id} / {task.backend_key}")
            continue

        print(f"[{i}/{len(tasks)}] {task.task_id} / {task.backend_key}")
        print(f"  envs={task.num_envs}  frames={task.num_frames}  timeout={task.timeout_minutes}m")

        exit_code, wall_time = _run_benchmark(task, artifact_dir, bench_script, cache_env)

        was_retried = False
        if exit_code != 0:
            print(f"\n[{i}/{len(tasks)}] first attempt failed (exit={exit_code}), retrying once...")
            # Remove any partial perf output from the failed attempt so the retry is clean
            stale = artifact_dir / "perf_regression_gate_info.json"
            if stale.exists():
                stale.unlink()
            exit_code, wall_time = _run_benchmark(task, artifact_dir, bench_script, cache_env)
            was_retried = True

        attempt = 2 if was_retried else 1
        print(f"\n[{i}/{len(tasks)}] exit={exit_code}  wall={wall_time:.0f}s  attempt={attempt}  → build_bench_result")
        _run_build_bench_result(task, artifact_dir, exit_code, wall_time, was_retried=was_retried, attempt=attempt)
        print()

    # -----------------------------------------------------------------------
    # Phase 3: aggregate + oracle
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 68}")
    print("  Phase 3 — Aggregate + Oracle Verdict")
    print(f"{'=' * 68}\n")

    if args.allow_baseline_update:
        print("[local_runner] --allow_baseline_update: PASS/WARN results will extend the baseline window.\n")
    else:
        print("[local_runner] Read-only baseline run (pass --allow_baseline_update to extend window).\n")

    rc = _run_aggregate(
        args.artifacts_dir,
        args.baselines_dir,
        args.gpu_model,
        args.gate_config,
        args.allow_baseline_update,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
