# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-benchmark script: normalizes benchmark output and writes perf_smoke_test_result.json.

Locates the timestamped runtime bundle written by perf_runtime.py, copies it
to the canonical ``perf_smoke_test_info.json``, classifies the failure phase from the
captured log, and writes ``perf_smoke_test_result.json`` for the aggregate job.

Usage::

    python3 tools/perf_smoke_test/build_bench_result.py \\
        --task_id Isaac-Cartpole-Direct-v0 \\
        --artifact_dir artifacts/Isaac-Cartpole-Direct-v0 \\
        --exit_code 0 \\
        --wall_time_s 48.3 \\
        --timeout_s 600 \\
        --log_file artifacts/Isaac-Cartpole-Direct-v0/benchmark.log
"""

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_TOOLS_DIR = _MODULE_DIR.parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from backend_identity import (  # noqa: E402
    backend_identity_from_benchmark_info,
    backend_identity_from_launch_config,
    identity_from_parts,
    make_backend_key,
    normalize_physics_backend,
    normalize_render_backend,
)
from benchmark_result_adapter import load_info, project_runtime  # noqa: E402
from contracts import BenchResult  # noqa: E402
from gate_config import load_gate_config  # noqa: E402
from gate_types import FailurePhase  # noqa: E402
from gpu_identity import normalize_gpu_fields  # noqa: E402
from launch_config import fallback_launch_config, load_launch_config  # noqa: E402
from runtime_contract import build_runtime_contract, build_runtime_publish_info  # noqa: E402
from subprocess_runner import classify_failure_phase  # noqa: E402
from task_config import get_task  # noqa: E402


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _config_drift(benchmark_info: dict, launch_config: dict) -> str | None:
    """Return a compact mismatch string when the actual run differs from launch intent."""
    if not benchmark_info:
        return None
    mismatches: list[str] = []

    wanted_task = launch_config.get("task_id")
    ran_task = benchmark_info.get("task")
    if isinstance(ran_task, str) and ran_task and wanted_task and ran_task != wanted_task:
        mismatches.append(f"task(ran={ran_task},want={wanted_task})")

    for field in ("num_envs", "seed"):
        wanted = _coerce_int(launch_config.get(field))
        ran = _coerce_int(benchmark_info.get(field))
        if wanted is not None and ran is not None and ran != wanted:
            mismatches.append(f"{field}(ran={ran},want={wanted})")

    wanted_frames = _coerce_int(launch_config.get("num_frames"))
    ran_frames = _coerce_int(benchmark_info.get("num_frames"))
    if wanted_frames is not None and ran_frames is not None and ran_frames < wanted_frames:
        mismatches.append(f"num_frames(ran={ran_frames},want>={wanted_frames})")

    wanted_backend = backend_identity_from_launch_config(launch_config)
    ran_backend = backend_identity_from_benchmark_info(benchmark_info)
    if wanted_backend is not None and ran_backend is not None and wanted_backend.backend_key != ran_backend.backend_key:
        mismatches.append(f"backend(ran={ran_backend.backend_key},want={wanted_backend.backend_key})")

    return " ".join(mismatches) if mismatches else None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build perf_smoke_test_result.json from a benchmark run")
    p.add_argument("--task_id", required=True)
    p.add_argument("--physics_backend", required=True, help="Physics backend used (e.g. physx, newton)")
    p.add_argument("--render_backend", default="", help="Render backend used (e.g. rtx, warp, ovrtx); empty = none")
    p.add_argument("--artifact_dir", required=True, type=Path)
    p.add_argument("--exit_code", required=True, type=int)
    p.add_argument("--wall_time_s", required=True, type=float)
    p.add_argument("--timeout_s", required=True, type=float)
    p.add_argument("--log_file", type=Path, default=None)
    p.add_argument(
        "--launch_config",
        type=Path,
        default=None,
        help="Path to launch_config.json (default: artifact_dir/launch_config.json)",
    )
    p.add_argument("--gate_config", type=Path, default=_MODULE_DIR / "gate_config.json")
    p.add_argument("--attempt", type=int, default=1, help="Attempt number (1 = first run, 2 = after one retry)")
    p.add_argument(
        "--was_retried", action="store_true", help="Set when this result comes from a retry of a failed first attempt"
    )
    return p.parse_args()


def _normalize_benchmark_output(artifact_dir: Path, task_id: str) -> bool:
    """Copy the timestamped runtime bundle to ``perf_smoke_test_info.json``.

    ``perf_runtime.py`` writes ``benchmark_runtime_{task_id}_{timestamp}.json``
    (a schema-v1 :class:`~isaaclab.test.benchmark.schema.RuntimeBundle`). The gate
    reads the canonical ``perf_smoke_test_info.json``; this bridges the gap.

    Returns True if perf_smoke_test_info.json exists after call
    """
    perf_smoke_test_info = artifact_dir / "perf_smoke_test_info.json"
    if perf_smoke_test_info.exists():
        return True
    # Primary pattern: exact task_id match
    matches = sorted(glob.glob(str(artifact_dir / f"benchmark_runtime_{task_id}_*.json")))
    if not matches:
        # Fallback: any benchmark_runtime_*.json in the artifact dir
        matches = sorted(glob.glob(str(artifact_dir / "benchmark_runtime_*.json")))
    if not matches:
        return False
    shutil.copy(matches[-1], perf_smoke_test_info)
    return True


def main() -> int:
    args = _parse_args()
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    gate_config = load_gate_config(args.gate_config)
    runtime_policy = gate_config.get("runtime_compatibility", {})

    cli_physics_backend = normalize_physics_backend(args.physics_backend)
    if cli_physics_backend is None:
        raise ValueError("--physics_backend must name a concrete backend")
    cli_render_backend = normalize_render_backend(args.render_backend)
    cli_backend_key = make_backend_key(cli_physics_backend, cli_render_backend)

    launch_config = load_launch_config(artifact_dir, args.launch_config)
    if launch_config is None:
        try:
            task = get_task(args.task_id, cli_backend_key)
        except KeyError:
            print(
                f"[build_bench_result] Warning: ({args.task_id!r}, {cli_backend_key!r}) not found in "
                "tasks.json; using defaults"
            )
            task = None
        launch_config = fallback_launch_config(
            task_id=args.task_id,
            physics_backend=cli_physics_backend,
            render_backend=cli_render_backend,
            backend_key=cli_backend_key,
            timeout_s=args.timeout_s,
            task=task,
        )
    launch_config = dict(launch_config)

    expected_backend = backend_identity_from_launch_config(launch_config) or identity_from_parts(
        cli_physics_backend, cli_render_backend
    )
    if expected_backend is None:
        raise ValueError("launch_config must define a concrete backend identity")
    task_id = str(launch_config.get("task_id") or args.task_id)
    physics_backend = expected_backend.physics_backend
    render_backend = expected_backend.render_backend
    backend_key = expected_backend.backend_key

    phase2_mismatches: list[str] = []
    if args.task_id != task_id:
        phase2_mismatches.append(f"phase2_task_arg(arg={args.task_id},want={task_id})")
    if cli_backend_key != backend_key:
        phase2_mismatches.append(f"phase2_backend_arg(arg={cli_backend_key},want={backend_key})")
    phase2_arg_mismatch = " ".join(phase2_mismatches) if phase2_mismatches else None

    gpu_fields = normalize_gpu_fields(launch_config.get("gpu_model_raw") or launch_config.get("gpu_model"))
    launch_config["task_id"] = task_id
    launch_config["backend_key"] = backend_key
    launch_config["backend"] = backend_key
    launch_config["physics_backend"] = physics_backend
    launch_config["render_backend"] = render_backend
    launch_config["gpu_model"] = gpu_fields["gpu_model"]
    launch_config["gpu_model_raw"] = gpu_fields["gpu_model_raw"]

    num_envs = launch_config.get("num_envs", 0)
    num_frames = launch_config.get("num_frames", 0)
    warmup_frames = launch_config.get("warmup_frames", 0)
    timeout_minutes = launch_config.get("timeout_minutes", int(args.timeout_s / 60))
    preset = launch_config.get("preset", "default")
    tags = launch_config.get("tags", ["always"])
    seed = launch_config.get("seed")

    # Read combined stdout/stderr log for failure classification
    log_text = ""
    if args.log_file and args.log_file.exists():
        log_text = args.log_file.read_text(errors="replace")

    perf_smoke_test_info_present = _normalize_benchmark_output(artifact_dir, task_id)

    failure_phase = classify_failure_phase(
        stdout=log_text,
        stderr="",
        exit_code=args.exit_code,
        wall_time_s=args.wall_time_s,
        timeout_s=args.timeout_s,
    )

    # Read the runtime bundle (schema v1) and project it into a typed RuntimeSample.
    sample = None
    benchmark_info: dict = {}
    config_mismatch: str | None = None
    observed_backend = None
    runtime_contract = None
    runtime_contract_hash = None
    runtime_info = None
    if perf_smoke_test_info_present:
        info_path = artifact_dir / "perf_smoke_test_info.json"
        bundle = load_info(info_path)
        sample = project_runtime(bundle) if bundle is not None else None
        if sample is not None:
            benchmark_info = sample.benchmark_info()
            observed_backend = backend_identity_from_benchmark_info(benchmark_info)
            runtime_contract, runtime_contract_hash = build_runtime_contract(
                provenance=sample.provenance,
                runtime_resources=sample.runtime_resources,
                backend=expected_backend,
                policy=runtime_policy,
            )
            runtime_info = build_runtime_publish_info(
                provenance=sample.provenance,
                runtime_resources=sample.runtime_resources,
                policy=runtime_policy,
            )
            config_mismatch = _config_drift(benchmark_info, launch_config)
        else:
            # File exists but is not a valid schema-v1 bundle (corrupt/truncated).
            perf_smoke_test_info_present = False
    config_mismatch = " ".join(part for part in (phase2_arg_mismatch, config_mismatch) if part) or None
    if config_mismatch and failure_phase is None:
        failure_phase = FailurePhase.CONFIG_MISMATCH.value

    bench_result = BenchResult(
        task_id=task_id,
        backend=backend_key,
        physics_backend=physics_backend,
        render_backend=render_backend,
        backend_key=backend_key,
        preset=preset,
        attempt=args.attempt,
        was_retried=args.was_retried,
        exit_code=args.exit_code,
        failure_phase=failure_phase,
        stdout_tail=log_text[-2000:] if len(log_text) > 2000 else log_text,
        wall_time_s=args.wall_time_s,
        startup_time_s=sample.startup_time_s if sample else None,
        perf_smoke_test_info_present=perf_smoke_test_info_present,
        raw_fps_mean=sample.fps_mean if sample else None,
        raw_fps_std=sample.fps_std if sample else None,
        raw_fps_min=sample.fps_min if sample else None,
        raw_fps_max=sample.fps_max if sample else None,
        benchmark_info=benchmark_info,
        observed_backend=observed_backend.to_dict() if observed_backend else None,
        config_mismatch=config_mismatch,
        runtime_contract=runtime_contract,
        runtime_contract_hash=runtime_contract_hash,
        runtime_info=runtime_info,
        runtime_resources=(sample.runtime_resources or None) if sample else None,
        provenance=sample.provenance if sample else None,
        launch_config=launch_config,
        launch_config_hash=launch_config.get("launch_config_hash"),
        benchmark_contract_hash=launch_config.get("benchmark_contract_hash"),
        baseline_epoch=launch_config.get("baseline_epoch", 1),
        task_config_snapshot={
            "task_id": task_id,
            "backend": backend_key,
            "physics_backend": physics_backend,
            "render_backend": render_backend,
            "backend_key": backend_key,
            "preset": preset,
            "num_envs": num_envs,
            "num_frames": num_frames,
            "warmup_frames": warmup_frames,
            "timeout_minutes": timeout_minutes,
            "tags": tags,
            "seed": seed,
            "launch_config_hash": launch_config.get("launch_config_hash"),
            "benchmark_contract_hash": launch_config.get("benchmark_contract_hash"),
            "runtime_contract_hash": runtime_contract_hash,
            "baseline_epoch": launch_config.get("baseline_epoch", 1),
        },
    )

    out = artifact_dir / "perf_smoke_test_result.json"
    out.write_text(json.dumps(bench_result.to_dict(), indent=2))

    status = (
        f"failure_phase={failure_phase!r}, perf_smoke_test_info_present={perf_smoke_test_info_present}, "
        f"exit_code={args.exit_code}, config_mismatch={config_mismatch!r}"
    )
    print(f"[build_bench_result] {task_id}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
