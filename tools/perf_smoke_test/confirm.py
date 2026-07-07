# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Confirm-on-BLOCK reruns for the performance smoke gate.

When the oracle initially flags a cell as ``BLOCK``, only that cell is re-run a
few more times. The per-attempt gate FPS (initial draw + reruns) is recorded into
the cell artifact's ``confirmation_fps_attempts`` field, and the oracle then
compares the *median* of attempts. A single slow draw can therefore no longer
finalize a block.

Two rerun backends share one interface so the decision logic is identical in CI
and locally:

* :func:`make_local_rerun` runs ``benchmark_non_rl.py`` directly through
  ``./isaaclab.sh`` (no Docker); used by ``local_runner.py``.
* :func:`make_docker_rerun` runs the benchmark inside the prebuilt CI image; used
  by the GitHub Actions aggregate job.

A rerun backend returns the attempt's gate-mean FPS, or ``None`` when the attempt
failed to produce usable FPS. A failed rerun is never silently dropped: the
oracle surfaces it as ``block_unconfirmed(reruns_failed)`` and keeps the blocking
verdict (fail-safe).
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

try:
    from .oracle import apply_excluded_frames
except ImportError:  # pragma: no cover - supports direct script imports
    from oracle import apply_excluded_frames

# A rerun backend: given the cell's bench_result and an output directory for this
# attempt, run the benchmark and return the attempt's gate-mean FPS (or None).
RerunFn = Callable[[dict, Path, int], "float | None"]


def _extract_fps_series(perf_info: list[dict]) -> list[float]:
    """Return the per-frame effective FPS series from a perf info payload."""
    for phase in perf_info:
        if phase.get("phase_name") == "runtime":
            for measurement in phase.get("measurements", []):
                value = measurement.get("value", {})
                if measurement.get("name", "").endswith("Step Frametimes") and isinstance(value, dict):
                    return list(value.get("Environment step effective FPS", []))
    return []


def _excluded_frames(launch_config: dict) -> frozenset[int]:
    """Expand ``excluded_frames_raw`` (indices and inclusive ranges) to a set."""
    indices: set[int] = set()
    for entry in launch_config.get("excluded_frames_raw") or []:
        if isinstance(entry, list):
            indices.update(range(int(entry[0]), int(entry[1]) + 1))
        else:
            indices.add(int(entry))
    return frozenset(indices)


def gate_mean_fps(info_path: Path, excluded_frames: frozenset[int]) -> float | None:
    """Return the gate FPS (mean over kept frames) for one perf info artifact.

    Mirrors the oracle's measured-FPS computation so a rerun attempt is scored on
    exactly the same basis as the initial draw. Returns ``None`` when no usable
    FPS samples remain after warmup filtering.
    """
    with info_path.open() as fh:
        series = _extract_fps_series(json.load(fh))
    filtered = apply_excluded_frames(series, excluded_frames)
    if not filtered:
        return None
    return statistics.mean(filtered)


def _latest_perf_info(attempt_dir: Path) -> Path | None:
    """Find the perf info artifact a benchmark attempt wrote into ``attempt_dir``."""
    canonical = attempt_dir / "perf_smoke_test_info.json"
    if canonical.exists():
        return canonical
    matches = sorted(attempt_dir.glob("benchmark_non_rl_*.json"))
    return matches[-1] if matches else None


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value)


def make_local_rerun(repo_root: Path, bench_script: Path) -> RerunFn:
    """Build a rerun backend that runs the benchmark directly via ``./isaaclab.sh``.

    Args:
        repo_root: IsaacLab repository root (the directory containing ``isaaclab.sh``).
        bench_script: Path to ``scripts/benchmarks/benchmark_non_rl.py``.

    Returns:
        A rerun callable suitable for :func:`confirm_block_cell`.
    """
    isaaclab = repo_root / "isaaclab.sh"

    def _rerun(bench_result: dict, attempt_dir: Path, attempt: int) -> float | None:
        launch_config = bench_result.get("launch_config") or {}
        task_id = bench_result["task_id"]
        backend_key = bench_result.get("backend_key") or bench_result.get("backend")
        attempt_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(isaaclab),
            "-p",
            str(bench_script),
            "--task",
            task_id,
            "--num_envs",
            str(launch_config["num_envs"]),
            "--num_frames",
            str(launch_config["num_frames"]),
            "--benchmark_backend",
            "json",
            "--output_path",
            str(attempt_dir),
        ]
        if launch_config.get("seed") is not None:
            cmd.extend(["--seed", str(launch_config["seed"])])
        cmd.extend(str(arg) for arg in launch_config.get("hydra_args") or [])

        log_file = attempt_dir / "benchmark.log"
        print(f"[confirm] {task_id}/{backend_key} local rerun attempt {attempt}")
        with log_file.open("wb") as log_fh:
            proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            print(f"[confirm] {task_id}/{backend_key} attempt {attempt} failed (exit={proc.returncode})")
            return None
        info = _latest_perf_info(attempt_dir)
        if info is None:
            print(f"[confirm] {task_id}/{backend_key} attempt {attempt} produced no perf info")
            return None
        fps = gate_mean_fps(info, _excluded_frames(launch_config))
        if fps is not None:
            print(f"[confirm] {task_id}/{backend_key} attempt {attempt}: {fps:.1f} FPS")
        return fps

    return _rerun


def make_docker_rerun(workspace: Path, ci_image_tag: str) -> RerunFn:
    """Build a rerun backend that runs the benchmark inside the prebuilt CI image.

    The Docker invocation mirrors the workflow's bench step (mounts, caches, env)
    so a rerun runs in the same environment as the original draw. ``attempt_dir``
    is always resolved to an absolute path before mounting, because Docker treats a
    relative ``-v`` source as a named volume rather than a host bind mount (the bug
    that previously caused reruns to silently record no FPS).

    Args:
        workspace: GitHub Actions workspace root (holds the checkout and caches).
        ci_image_tag: Local Docker tag of the pulled/built CI image.

    Returns:
        A rerun callable suitable for :func:`confirm_block_cell`.
    """
    workspace = workspace.resolve()

    def _rerun(bench_result: dict, attempt_dir: Path, attempt: int) -> float | None:
        launch_config = bench_result.get("launch_config") or {}
        task_id = bench_result["task_id"]
        backend_key = bench_result.get("backend_key") or bench_result.get("backend")
        attempt_dir = attempt_dir.resolve()
        attempt_dir.mkdir(parents=True, exist_ok=True)
        timeout_s = int(launch_config.get("timeout_minutes", 12)) * 60
        container = f"perf-confirm-{_safe_name(task_id)}-{_safe_name(str(backend_key))}-{int(time.time())}-{attempt}"

        hydra_args = " ".join(str(arg) for arg in launch_config.get("hydra_args") or [])
        seed = launch_config.get("seed")
        seed_arg = f"--seed {seed}" if seed is not None else ""
        inner = "\n".join(
            [
                "set -e",
                "cd /workspace/isaaclab",
                "rm -f _isaac_sim",
                "ln -s /isaac-sim _isaac_sim",
                "./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py "
                f"--task '{task_id}' "
                f"--num_envs {launch_config['num_envs']} "
                f"--num_frames {launch_config['num_frames']} "
                "--benchmark_backend json "
                "--output_path /tmp/bench_out "
                f"{seed_arg} "
                f"{hydra_args}",
            ]
        )
        docker_cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            container,
            "--init",
            "--stop-timeout",
            "10",
            "--entrypoint",
            "bash",
            "--gpus",
            "all",
            "--network=host",
            "--security-opt=no-new-privileges:true",
            "--ulimit",
            "nofile=65536:65536",
            "--ulimit",
            "nproc=4096:4096",
            "-e",
            "OMNI_KIT_ACCEPT_EULA=yes",
            "-e",
            "ACCEPT_EULA=Y",
            "-e",
            "OMNI_KIT_DISABLE_CUP=1",
            "-e",
            "ISAAC_SIM_HEADLESS=1",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "WARP_CACHE_PATH=/tmp/jit-cache/warp",
            "-e",
            "CUDA_CACHE_PATH=/tmp/jit-cache/nv",
            "-v",
            f"{attempt_dir}:/tmp/bench_out",
            "-v",
            f"{workspace / 'jit-cache'}:/tmp/jit-cache",
            "-v",
            f"{workspace / 'kit-cache'}:/isaac-sim/kit/cache",
            "-v",
            f"{workspace}:/workspace/isaaclab",
            ci_image_tag,
            "-c",
            inner,
        ]

        subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[confirm] {task_id}/{backend_key} docker rerun attempt {attempt}")
        subprocess.run(docker_cmd, check=True)
        exit_code = 1
        wait = subprocess.run(
            ["timeout", str(timeout_s), "docker", "wait", container],
            capture_output=True,
            text=True,
            check=False,
        )
        if wait.returncode == 0:
            exit_code = int((wait.stdout or "1").strip() or "1")
        else:
            subprocess.run(["docker", "kill", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logs = subprocess.run(["docker", "logs", container], capture_output=True, check=False)
        (attempt_dir / "benchmark.log").write_bytes((logs.stdout or b"") + (logs.stderr or b""))
        subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if exit_code != 0:
            print(f"[confirm] {task_id}/{backend_key} attempt {attempt} failed (exit={exit_code})")
            return None
        info = _latest_perf_info(attempt_dir)
        if info is None:
            print(f"[confirm] {task_id}/{backend_key} attempt {attempt} produced no perf info")
            return None
        fps = gate_mean_fps(info, _excluded_frames(launch_config))
        if fps is not None:
            print(f"[confirm] {task_id}/{backend_key} attempt {attempt}: {fps:.1f} FPS")
        return fps

    return _rerun


def confirm_block_cell(
    bench_result: dict,
    artifact_dir: Path,
    result_path: Path,
    excluded_frames: frozenset[int],
    rerun_fn: RerunFn,
    reruns: int,
) -> list[float]:
    """Re-run one initially-blocking cell and persist its confirmation attempts.

    The attempts list is seeded with the cell's initial gate FPS, then up to
    ``reruns`` additional attempts are appended (failed reruns are skipped). The
    list and a small ``confirmation_policy`` record are written back into the
    cell's ``perf_smoke_test_result.json`` so the re-aggregation oracle can apply
    the median policy.

    Args:
        bench_result: The cell's parsed ``perf_smoke_test_result.json`` contents.
        artifact_dir: Directory holding the cell's artifacts.
        result_path: Path to the cell's ``perf_smoke_test_result.json``.
        excluded_frames: Warmup frame indices to drop when scoring each attempt.
        rerun_fn: A rerun backend from :func:`make_local_rerun` / :func:`make_docker_rerun`.
        reruns: Number of additional attempts to run.

    Returns:
        The list of recorded attempt FPS values (initial draw first).
    """
    task_id = bench_result["task_id"]
    backend_key = bench_result.get("backend_key") or bench_result.get("backend")
    info_path = artifact_dir / "perf_smoke_test_info.json"
    initial = gate_mean_fps(info_path, excluded_frames)
    attempts: list[float] = [] if initial is None else [initial]
    if initial is not None:
        print(f"[confirm] confirming {task_id}/{backend_key}; initial={initial:.1f} FPS")

    for offset in range(reruns):
        attempt_dir = (artifact_dir / f"confirm_attempt_{offset + 2}").resolve()
        fps = rerun_fn(bench_result, attempt_dir, offset + 2)
        if fps is not None:
            attempts.append(fps)

    bench_result["confirmation_fps_attempts"] = attempts
    bench_result["confirmation_policy"] = {
        "trigger": "initial_block",
        "requested_reruns": reruns,
        "completed_attempts": len(attempts),
    }
    with result_path.open("w") as fh:
        json.dump(bench_result, fh, indent=2)
        fh.write("\n")
    return attempts
