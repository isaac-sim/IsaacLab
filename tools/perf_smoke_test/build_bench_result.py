# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-benchmark script: normalizes benchmark output and writes perf_smoke_test_result.json.

Locates the timestamped benchmark output file written by benchmark_non_rl.py, renames it
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
import statistics
import subprocess
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
from gate_config import load_gate_config  # noqa: E402
from gate_types import FailurePhase  # noqa: E402
from gpu_identity import normalize_gpu_fields  # noqa: E402
from launch_config import fallback_launch_config, load_launch_config  # noqa: E402
from runtime_contract import build_runtime_contract, build_runtime_publish_info  # noqa: E402
from subprocess_runner import classify_failure_phase  # noqa: E402
from task_config import get_task  # noqa: E402


def _percentile(sorted_data: list[float], p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list"""
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def _strip_phase_prefix(name: str, phase_name: str) -> str:
    """Strip the '{task_name} {phase_name} ' prefix added by JSONFileMetrics.finalize()"""
    marker = f" {phase_name} "
    idx = name.find(marker)
    return name[idx + len(marker) :] if idx >= 0 else name


def _duration_to_seconds(value: float, unit: str | None) -> float:
    normalized = str(unit or "s").strip().lower()
    if normalized in {"ms", "millisecond", "milliseconds"}:
        return value / 1000.0
    if normalized in {"us", "microsecond", "microseconds"}:
        return value / 1_000_000.0
    return value


def _memory_to_mb(value: float, unit: str | None) -> float:
    normalized = str(unit or "MB").strip().lower()
    if normalized in {"b", "byte", "bytes"}:
        return value / (1024.0 * 1024.0)
    if normalized in {"kb", "kib", "kilobyte", "kilobytes"}:
        return value / 1024.0
    if normalized in {"gb", "gib", "gigabyte", "gigabytes"}:
        return value * 1024.0
    return value


def _system_memory_measurement_mb(name: str, value: object, unit: str | None) -> float | None:
    """Return a system-memory measurement in MB when a runtime recorder emits one."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    normalized = name.lower()
    markers = (
        "system ram used",
        "system memory used",
        "cpu memory used",
        "process rss",
        "resident set size",
        "rss memory",
    )
    if any(marker in normalized for marker in markers):
        return _memory_to_mb(float(value), unit)
    return None


def _gpu_driver_version() -> str | None:
    """Return the GPU driver version string from nvidia-smi, or None if unavailable"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip().splitlines()[0].strip()
            return version if version else None
    except Exception:
        pass
    return None


def _extract_info_provenance(info_path: Path) -> dict:
    """Parse ``perf_smoke_test_info.json`` and return provenance + FPS fields

    Extracts hardware metadata (GPU name/memory/CUDA, CPU, RAM), software versions,
    git provenance, FPS distribution statistics, GPU memory used at runtime,
    optional system memory usage, and startup time.  All fields are best-effort;
    missing data is omitted.

    Returns:
        Dict with a subset of the following keys:

        - ``raw_fps_{mean,std,min,max,median,p5,p95}``: float
        - ``startup_time_s``: float
        - ``gpu_diag``: dict with gpu_name, gpu_total_memory_gb, cuda_version,
          nvidia_driver_version, gpu_mem_used_mb
        - ``memory_diag``: dict with optional system_ram_used_mb
        - ``provenance``: dict with ``hardware``, ``software``, ``git`` sub-dicts
    """
    try:
        phases: list[dict] = json.loads(info_path.read_text())
    except Exception:
        return {}

    hardware: dict = {}
    software: dict = {}
    git: dict = {}
    fps_stats: dict = {}
    gpu_mem_used_gb: float | None = None
    system_ram_used_mb: float | None = None
    startup_time_s: float | None = None

    for phase in phases:
        pname: str = phase.get("phase_name", "")
        metadata_map = {
            _strip_phase_prefix(m["name"], pname): m["data"]
            for m in phase.get("metadata", [])
            if "name" in m and "data" in m
        }

        if pname == "hardware_info":
            hardware["cpu_name"] = metadata_map.get("cpu_name")
            hardware["cpu_physical_cores"] = metadata_map.get("physical_cores")
            hardware["total_ram_gb"] = metadata_map.get("total_ram_gb")
            hardware["gpu_device_count"] = metadata_map.get("gpu_device_count")
            hardware["cuda_version"] = metadata_map.get("cuda_version")
            gpu_devices = metadata_map.get("gpu_devices")
            if isinstance(gpu_devices, dict):
                current = str(metadata_map.get("gpu_current_device", 0))
                dev = gpu_devices.get(current) or next(iter(gpu_devices.values()), {})
                hardware["gpu_name"] = dev.get("name")
                hardware["gpu_total_memory_gb"] = dev.get("total_memory_gb")
                hardware["gpu_compute_capability"] = dev.get("compute_capability")
                hardware["gpu_multi_processor_count"] = dev.get("multi_processor_count")

        elif pname == "version_info":
            dev_data: dict = metadata_map.pop("dev", {}) or {}
            for k, v in metadata_map.items():
                # strip trailing "_version" suffix added by VersionInfoRecorder
                key = k[: -len("_version")] if k.endswith("_version") else k
                if v is not None:
                    software[key] = v
            git = {
                k: dev_data[k]
                for k in ("commit_hash", "commit_hash_short", "branch", "commit_date", "dirty")
                if k in dev_data
            }

        elif pname == "runtime":
            for m in phase.get("measurements", []):
                name: str = m.get("name", "")
                value = m.get("value")
                # FPS series: DictMeasurement written by BenchmarkMonitor
                if name.endswith("Step Frametimes") and isinstance(value, dict):
                    fps_series: list[float] = value.get("Environment step effective FPS", [])
                    if fps_series:
                        sorted_fps = sorted(fps_series)
                        n = len(sorted_fps)
                        fps_stats = {
                            "raw_fps_mean": statistics.mean(fps_series),
                            "raw_fps_std": statistics.stdev(fps_series) if n > 1 else 0.0,
                            "raw_fps_min": sorted_fps[0],
                            "raw_fps_max": sorted_fps[-1],
                            "raw_fps_median": _percentile(sorted_fps, 50.0),
                            "raw_fps_p5": _percentile(sorted_fps, 5.0),
                            "raw_fps_p95": _percentile(sorted_fps, 95.0),
                        }
                # GPU memory used (mean over run, in GB):  SingleMeasurement from GPUInfoRecorder
                elif name.endswith("GPU Memory Used") and isinstance(value, (int, float)):
                    gpu_mem_used_gb = float(value)
                else:
                    measurement_mb = _system_memory_measurement_mb(name, value, m.get("unit"))
                    if measurement_mb is not None:
                        system_ram_used_mb = measurement_mb

        elif pname == "startup":
            for m in phase.get("measurements", []):
                if m.get("name", "").endswith("Total Start Time (Launch to Train)"):
                    val = m.get("value")
                    if isinstance(val, (int, float)):
                        startup_time_s = _duration_to_seconds(float(val), m.get("unit"))
                    break

    gpu_diag: dict = {
        k: v
        for k, v in {
            "gpu_name": hardware.get("gpu_name"),
            "gpu_total_memory_gb": hardware.get("gpu_total_memory_gb"),
            "cuda_version": hardware.get("cuda_version"),
            "nvidia_driver_version": _gpu_driver_version(),
            "gpu_mem_used_mb": round(gpu_mem_used_gb * 1024, 2) if gpu_mem_used_gb is not None else None,
        }.items()
        if v is not None
    }
    memory_diag: dict = {
        k: v
        for k, v in {
            "system_ram_used_mb": round(system_ram_used_mb, 2) if system_ram_used_mb is not None else None,
        }.items()
        if v is not None
    }

    result: dict = {}
    result.update(fps_stats)
    if startup_time_s is not None:
        result["startup_time_s"] = startup_time_s
    if gpu_diag:
        result["gpu_diag"] = gpu_diag
    if memory_diag:
        result["memory_diag"] = memory_diag
    result["provenance"] = {
        "hardware": {k: v for k, v in hardware.items() if v is not None},
        "software": software,
        "git": git,
    }
    return result


# ---------------------------------------------------------------------------
# Launch/run provenance guard + step-time debug KPIs
# ---------------------------------------------------------------------------

_OUTLIER_FACTOR = 2.0
_WARMUP_GUARD_FACTOR = 3.0
_MAX_REPORTED_OUTLIERS = 8


def _extract_benchmark_info(info_path: Path) -> dict:
    """Return the run's self-reported benchmark_info metadata, if present."""
    try:
        phases: list[dict] = json.loads(info_path.read_text())
    except Exception:
        return {}
    for phase in phases:
        if phase.get("phase_name") != "benchmark_info":
            continue
        out = {}
        for item in phase.get("metadata", []):
            if "name" in item and "data" in item:
                out[_strip_phase_prefix(item["name"], "benchmark_info")] = item["data"]
        return out
    return {}


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


def _expand_excluded_frames(raw: list) -> frozenset[int]:
    indices: set[int] = set()
    for entry in raw or []:
        if isinstance(entry, list):
            indices.update(range(int(entry[0]), int(entry[1]) + 1))
        else:
            indices.add(int(entry))
    return frozenset(indices)


def _extract_debug_kpis(info_path: Path, excluded_frames: frozenset[int]) -> dict:
    """Return post-warm-up step-time diagnostics for aggregate summaries."""
    try:
        phases: list[dict] = json.loads(info_path.read_text())
    except Exception:
        return {}
    steps: list[float] = []
    for phase in phases:
        if phase.get("phase_name") != "runtime":
            continue
        for measurement in phase.get("measurements", []):
            value = measurement.get("value")
            if measurement.get("name", "").endswith("Step Frametimes") and isinstance(value, dict):
                raw = value.get("Environment step times", [])
                steps = [float(v) for v in raw if isinstance(v, (int, float)) and not isinstance(v, bool)]
        break
    if not steps:
        return {}
    steady = [value for idx, value in enumerate(steps) if idx not in excluded_frames]
    if len(steady) < 2:
        return {}
    ordered = sorted(steady)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    p99 = ordered[min(n - 1, int(round(0.99 * (n - 1))))]
    out: dict = {"steady_frames": n}
    if median > 0:
        out["p99_over_median"] = round(p99 / median, 3)
        outliers = [(idx, value) for idx, value in enumerate(steady) if value > _OUTLIER_FACTOR * median]
        out["outlier_count"] = len(outliers)
        if outliers:
            out["outlier_idx"] = ",".join(str(idx) for idx, _ in outliers[:_MAX_REPORTED_OUTLIERS])
            out["outlier_mag_x"] = ",".join(f"{value / median:.2g}" for _, value in outliers[:_MAX_REPORTED_OUTLIERS])
        if steady[0] > _WARMUP_GUARD_FACTOR * median:
            out["warmup_flag"] = f"first_kept_frame={steady[0] / median:.1f}x_median"
    return out


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
    """Rename the timestamped benchmark JSON to ``perf_smoke_test_info.json``

    benchmark_non_rl.py writes ``benchmark_non_rl_{task_id}_{timestamp}.json``.
    The oracle reads ``perf_smoke_test_info.json``.  This function bridges the gap.

    Returns True if perf_smoke_test_info.json exists after call
    """
    perf_smoke_test_info = artifact_dir / "perf_smoke_test_info.json"
    if perf_smoke_test_info.exists():
        return True
    # Primary pattern: exact task_id match
    matches = sorted(glob.glob(str(artifact_dir / f"benchmark_non_rl_{task_id}_*.json")))
    if not matches:
        # Fallback: any benchmark_non_rl_*.json in the artifact dir
        matches = sorted(glob.glob(str(artifact_dir / "benchmark_non_rl_*.json")))
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
    excluded_frames_raw = launch_config.get("excluded_frames_raw", [])
    excluded_frames = _expand_excluded_frames(excluded_frames_raw)
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

    # Extract FPS stats, startup time, GPU diag, run config, and full provenance.
    info_provenance: dict = {}
    benchmark_info: dict = {}
    debug_kpis: dict = {}
    config_mismatch: str | None = None
    observed_backend = None
    runtime_contract = None
    runtime_contract_hash = None
    runtime_info = None
    if perf_smoke_test_info_present:
        info_path = artifact_dir / "perf_smoke_test_info.json"
        info_provenance = _extract_info_provenance(info_path)
        benchmark_info = _extract_benchmark_info(info_path)
        observed_backend = backend_identity_from_benchmark_info(benchmark_info)
        debug_kpis = _extract_debug_kpis(info_path, excluded_frames)
        runtime_contract, runtime_contract_hash = build_runtime_contract(
            provenance=info_provenance.get("provenance"),
            gpu_diag=info_provenance.get("gpu_diag"),
            backend=expected_backend,
            policy=runtime_policy,
        )
        runtime_info = build_runtime_publish_info(
            provenance=info_provenance.get("provenance"),
            gpu_diag=info_provenance.get("gpu_diag"),
            policy=runtime_policy,
        )
        config_mismatch = _config_drift(benchmark_info, launch_config)
    config_mismatch = " ".join(part for part in (phase2_arg_mismatch, config_mismatch) if part) or None
    if config_mismatch and failure_phase is None:
        failure_phase = FailurePhase.CONFIG_MISMATCH.value

    bench_result = {
        "task_id": task_id,
        "backend": backend_key,
        "physics_backend": physics_backend,
        "render_backend": render_backend,
        "backend_key": backend_key,
        "preset": preset,
        "attempt": args.attempt,
        "was_retried": args.was_retried,
        "exit_code": args.exit_code,
        "failure_phase": failure_phase,
        "stdout_tail": log_text[-2000:] if len(log_text) > 2000 else log_text,
        "wall_time_s": args.wall_time_s,
        "startup_time_s": info_provenance.get("startup_time_s"),
        "perf_smoke_test_info_present": perf_smoke_test_info_present,
        "raw_fps_mean": info_provenance.get("raw_fps_mean"),
        "raw_fps_std": info_provenance.get("raw_fps_std"),
        "raw_fps_min": info_provenance.get("raw_fps_min"),
        "raw_fps_max": info_provenance.get("raw_fps_max"),
        "raw_fps_median": info_provenance.get("raw_fps_median"),
        "raw_fps_p5": info_provenance.get("raw_fps_p5"),
        "raw_fps_p95": info_provenance.get("raw_fps_p95"),
        "p99_over_median": debug_kpis.get("p99_over_median"),
        "outlier_count": debug_kpis.get("outlier_count"),
        "debug_kpis": debug_kpis,
        "benchmark_info": benchmark_info,
        "observed_backend": observed_backend.to_dict() if observed_backend else None,
        "config_mismatch": config_mismatch,
        "runtime_contract": runtime_contract,
        "runtime_contract_hash": runtime_contract_hash,
        "runtime_info": runtime_info,
        "gpu_diag": info_provenance.get("gpu_diag"),
        "memory_diag": info_provenance.get("memory_diag"),
        "provenance": info_provenance.get("provenance"),
        "launch_config": launch_config,
        "launch_config_hash": launch_config.get("launch_config_hash"),
        "benchmark_contract_hash": launch_config.get("benchmark_contract_hash"),
        "baseline_epoch": launch_config.get("baseline_epoch", 1),
        "task_config_snapshot": {
            "task_id": task_id,
            "backend": backend_key,
            "physics_backend": physics_backend,
            "render_backend": render_backend,
            "backend_key": backend_key,
            "preset": preset,
            "num_envs": num_envs,
            "num_frames": num_frames,
            "excluded_frames_raw": excluded_frames_raw,
            "timeout_minutes": timeout_minutes,
            "tags": tags,
            "seed": seed,
            "launch_config_hash": launch_config.get("launch_config_hash"),
            "benchmark_contract_hash": launch_config.get("benchmark_contract_hash"),
            "runtime_contract_hash": runtime_contract_hash,
            "baseline_epoch": launch_config.get("baseline_epoch", 1),
        },
    }

    out = artifact_dir / "perf_smoke_test_result.json"
    out.write_text(json.dumps(bench_result, indent=2))

    status = (
        f"failure_phase={failure_phase!r}, perf_smoke_test_info_present={perf_smoke_test_info_present}, "
        f"exit_code={args.exit_code}, config_mismatch={config_mismatch!r}"
    )
    print(f"[build_bench_result] {task_id}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
