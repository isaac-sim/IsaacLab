# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a reproducible 3.0-vs-2.3.2 performance evidence bundle from CI artifacts.

The script accepts one or more labeled artifact roots, searches for raw
``benchmark_non_rl.py`` JSON outputs and normalized ``perf_smoke_test_result.json``
files, then writes a JSON summary and a Markdown table with:

* steady-state FPS (same excluded-frame convention as the gate),
* within-run and run-to-run standard deviation,
* GPU memory and system RAM usage when available,
* CPU/GPU hardware identity and software provenance.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_EXCLUDED = frozenset(range(0, 101))
_MEM_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)?")
_TASK_ALIASES = {
    "Isaac-Cartpole-Direct-v0": "Isaac-Cartpole-Direct",
    "Isaac-Factory-GearMesh-Direct-v0": "Isaac-Factory-GearMesh-Direct",
    "Isaac-Velocity-Flat-G1-v0": "Isaac-Velocity-Flat-G1",
}


@dataclass(frozen=True)
class ArtifactInput:
    label: str
    root: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize perf regression evidence artifacts.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Artifact root to scan. Repeat for each release/build label.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("tools/perf_smoke_test/docs/regression_evidence_summary.json"),
        help="Path to write machine-readable evidence summary.",
    )
    parser.add_argument(
        "--markdown_out",
        type=Path,
        default=Path("tools/perf_smoke_test/docs/3_0_vs_2_3_2_regression_evidence.md"),
        help="Path to write the shareable Markdown report.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _strip_phase_prefix(name: str, phase_name: str) -> str:
    marker = f" {phase_name} "
    idx = name.find(marker)
    return name[idx + len(marker) :] if idx >= 0 else name


def _phase_metadata(phases: list[dict], phase_name: str) -> dict[str, Any]:
    for phase in phases:
        if phase.get("phase_name") != phase_name:
            continue
        return {
            _strip_phase_prefix(str(item.get("name", "")), phase_name): item.get("data")
            for item in phase.get("metadata", [])
            if "name" in item and "data" in item
        }
    return {}


def _phase_measurements(phases: list[dict], phase_name: str) -> list[dict]:
    for phase in phases:
        if phase.get("phase_name") == phase_name:
            return [m for m in phase.get("measurements", []) if isinstance(m, dict)]
    return []


def _find_series(obj: Any, key: str = "Environment step effective FPS") -> list[float]:
    if isinstance(obj, dict):
        for k, value in obj.items():
            if k == key and isinstance(value, list):
                return [float(v) for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]
            nested = _find_series(value, key)
            if nested:
                return nested
    elif isinstance(obj, list):
        for item in obj:
            nested = _find_series(item, key)
            if nested:
                return nested
    return []


def _expand_excluded(raw: Any) -> frozenset[int]:
    if raw is None:
        return _DEFAULT_EXCLUDED
    if isinstance(raw, str):
        indices: set[int] = set()
        for token in raw.replace(",", " ").split():
            if "-" in token:
                lo, hi = token.split("-", 1)
                indices.update(range(int(lo), int(hi) + 1))
            else:
                indices.add(int(token))
        return frozenset(indices)
    indices = set()
    for entry in raw or []:
        if isinstance(entry, list):
            indices.update(range(int(entry[0]), int(entry[1]) + 1))
        else:
            indices.add(int(entry))
    return frozenset(indices)


def _excluded_for(path: Path, result: dict | None) -> frozenset[int]:
    if result:
        launch_config = result.get("launch_config") or {}
        snapshot = result.get("task_config_snapshot") or {}
        raw = launch_config.get("excluded_frames_raw", snapshot.get("excluded_frames_raw"))
        if raw is not None:
            return _expand_excluded(raw)
    for parent in (path.parent, *path.parents):
        excluded_path = parent / "excluded.txt"
        if excluded_path.exists():
            return _expand_excluded(excluded_path.read_text(encoding="utf-8").strip())
    return _DEFAULT_EXCLUDED


def _memory_to_mb(value: float, unit: str | None) -> float:
    normalized = str(unit or "MB").strip().lower()
    if normalized in {"b", "byte", "bytes"}:
        return value / (1024.0 * 1024.0)
    if normalized in {"kb", "kib", "kilobyte", "kilobytes"}:
        return value / 1024.0
    if normalized in {"gb", "gib", "gigabyte", "gigabytes"}:
        return value * 1024.0
    if normalized in {"tb", "tib", "terabyte", "terabytes"}:
        return value * 1024.0 * 1024.0
    return value


def _parse_memory_token(token: str) -> float | None:
    match = _MEM_RE.match(token)
    if not match:
        return None
    return _memory_to_mb(float(match.group(1)), match.group(2))


def _docker_stats_memory(stats_path: Path) -> dict[str, float]:
    values: list[float] = []
    cpu_values: list[float] = []
    if not stats_path.exists():
        return {}
    for line in stats_path.read_text(encoding="utf-8", errors="replace").splitlines():
        obj = _load_json_from_text(line)
        if not isinstance(obj, dict):
            continue
        mem_usage = obj.get("MemUsage")
        if isinstance(mem_usage, str):
            current = _parse_memory_token(mem_usage.split("/", 1)[0])
            if current is not None:
                values.append(current)
        cpu_perc = obj.get("CPUPerc")
        if isinstance(cpu_perc, str):
            with contextlib.suppress(ValueError):
                cpu_values.append(float(cpu_perc.strip().rstrip("%")))
    out: dict[str, float] = {}
    if values:
        out["system_ram_mean_mb"] = statistics.mean(values)
        out["system_ram_peak_mb"] = max(values)
    if cpu_values:
        out["docker_cpu_mean_pct"] = statistics.mean(cpu_values)
        out["docker_cpu_peak_pct"] = max(cpu_values)
    return out


def _nvidia_smi_memory(samples_path: Path) -> dict[str, float]:
    values: list[float] = []
    if not samples_path.exists():
        return {}
    for line in samples_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("timestamp"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        used = _parse_memory_token(parts[4])
        if used is not None and used > 0:
            values.append(used)
    if not values:
        return {}
    return {
        "gpu_mem_mean_mb": statistics.mean(values),
        "gpu_mem_peak_mb": max(values),
    }


def _load_json_from_text(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _runtime_memory(phases: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for measurement in _phase_measurements(phases, "runtime"):
        name = str(measurement.get("name", "")).lower()
        value = measurement.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if name.endswith("gpu memory used"):
            out["gpu_mem_used_mb"] = _memory_to_mb(float(value), measurement.get("unit") or "GB")
        elif any(marker in name for marker in ("system ram", "system memory", "process rss", "resident set size")):
            out["system_ram_used_mb"] = _memory_to_mb(float(value), measurement.get("unit"))
    return out


def _hardware_from_info(phases: list[dict]) -> dict[str, Any]:
    metadata = _phase_metadata(phases, "hardware_info")
    gpu_devices = metadata.get("gpu_devices")
    gpu = {}
    if isinstance(gpu_devices, dict):
        current = str(metadata.get("gpu_current_device", 0))
        gpu = gpu_devices.get(current) or next(iter(gpu_devices.values()), {})
    return {
        "cpu_name": metadata.get("cpu_name"),
        "cpu_physical_cores": metadata.get("physical_cores"),
        "total_ram_gb": metadata.get("total_ram_gb"),
        "gpu_name": gpu.get("name"),
        "gpu_total_memory_gb": gpu.get("total_memory_gb"),
        "gpu_compute_capability": gpu.get("compute_capability"),
        "gpu_multi_processor_count": gpu.get("multi_processor_count"),
        "cuda_version": metadata.get("cuda_version"),
    }


def _software_from_info(phases: list[dict]) -> dict[str, Any]:
    metadata = _phase_metadata(phases, "version_info")
    software = {}
    for key, value in metadata.items():
        if key == "dev" or value is None:
            continue
        software[key[: -len("_version")] if key.endswith("_version") else key] = value
    return software


def _benchmark_info_from_info(phases: list[dict]) -> dict[str, Any]:
    return _phase_metadata(phases, "benchmark_info")


def _read_task_id_near(path: Path) -> str | None:
    for parent in (path.parent, *path.parents):
        task_id_path = parent / "task_id.txt"
        if task_id_path.exists():
            task_id = task_id_path.read_text(encoding="utf-8").strip()
            if task_id:
                return task_id
    return None


def _run_log_number(path: Path, field: str) -> int | None:
    run_log = path.parent / "run.log"
    if not run_log.exists():
        return None
    names = [field, field.replace("_", " ")]
    if field == "num_envs":
        names.extend(["Number of environments", "num envs"])
    elif field == "num_frames":
        names.append("num frames")
    patterns = [re.compile(rf"^\s*{re.escape(name)}\s*:\s*([0-9]+)\s*$", re.IGNORECASE) for name in names]
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                return int(match.group(1))
    return None


def _sample_id(path: Path) -> str:
    for parent in (path.parent, *path.parents):
        if parent.name.startswith(("sample_", "run_")):
            return parent.name
    return path.parent.name


def _canonical_task_id(task_id: Any) -> str:
    value = str(task_id)
    return _TASK_ALIASES.get(value, value)


def _candidate_files(root: Path) -> list[Path]:
    names = {
        "perf_smoke_test_result.json",
        "perf_smoke_test_info.json",
        "benchmark_output.json",
    }
    files = [path for path in root.rglob("*.json") if path.name in names or path.name.startswith("benchmark_non_rl_")]
    # Prefer normalized result files over their adjacent raw info file to avoid double-counting.
    result_dirs = {path.parent for path in files if path.name == "perf_smoke_test_result.json"}
    raw_names = {"perf_smoke_test_info.json"} | {name for name in names if name != "perf_smoke_test_result.json"}
    return [path for path in sorted(files) if not (path.parent in result_dirs and path.name in raw_names)]


def _sample_from_file(label: str, path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    if payload is None:
        return None
    result = payload if isinstance(payload, dict) and path.name == "perf_smoke_test_result.json" else None
    info_path = path.parent / "perf_smoke_test_info.json" if result else path
    if result and not info_path.exists():
        info_path = path
    info_payload = _load_json(info_path)
    phases = info_payload if isinstance(info_payload, list) else []
    benchmark_info = result.get("benchmark_info") if result else {}
    if not benchmark_info and phases:
        benchmark_info = _benchmark_info_from_info(phases)
    launch_config = result.get("launch_config") if result else {}
    snapshot = result.get("task_config_snapshot") if result else {}

    fps_series = _find_series(phases)
    excluded = _excluded_for(path, result)
    steady_fps = [value for idx, value in enumerate(fps_series) if idx not in excluded]
    measured_fps = statistics.mean(steady_fps) if steady_fps else result.get("raw_fps_mean") if result else None
    if measured_fps is None:
        return None

    hardware = (result.get("provenance") or {}).get("hardware") if result else {}
    if not hardware and phases:
        hardware = _hardware_from_info(phases)
    software = (result.get("provenance") or {}).get("software") if result else {}
    if not software and phases:
        software = _software_from_info(phases)
    gpu_diag = result.get("gpu_diag") if result else {}
    memory_diag = result.get("memory_diag") if result else {}
    runtime_memory = _runtime_memory(phases) if phases else {}
    docker_memory = _docker_stats_memory(path.parent / "docker_stats.jsonl")
    nvidia_memory = _nvidia_smi_memory(path.parent / "nvidia_smi_samples.csv")

    backend = (
        result.get("backend_key")
        if result
        else launch_config.get("backend_key")
        or launch_config.get("backend")
        or str(benchmark_info.get("physics") or "physx").strip()
        or "physx"
    )
    task_id = (
        result.get("task_id")
        if result
        else launch_config.get("task_id")
        or benchmark_info.get("task")
        or _read_task_id_near(path)
        or path.parent.parent.name
    )

    gpu_mem_used_mb = (
        (gpu_diag or {}).get("gpu_mem_used_mb")
        or runtime_memory.get("gpu_mem_used_mb")
        or nvidia_memory.get("gpu_mem_peak_mb")
    )
    system_ram_used_mb = (
        (memory_diag or {}).get("system_ram_used_mb")
        or runtime_memory.get("system_ram_used_mb")
        or docker_memory.get("system_ram_mean_mb")
    )

    within_run_std_fps = (
        statistics.stdev(steady_fps) if len(steady_fps) > 1 else result.get("raw_fps_std") if result else None
    )
    raw_fps_mean = result.get("raw_fps_mean") if result else statistics.mean(fps_series) if fps_series else None
    raw_fps_std = result.get("raw_fps_std") if result else statistics.stdev(fps_series) if len(fps_series) > 1 else None

    return {
        "label": label,
        "task_id": _canonical_task_id(task_id),
        "raw_task_id": task_id,
        "backend": backend,
        "sample_id": _sample_id(path),
        "artifact_path": str(path),
        "measured_fps": measured_fps,
        "within_run_std_fps": within_run_std_fps,
        "raw_fps_mean": raw_fps_mean,
        "raw_fps_std": raw_fps_std,
        "num_envs": launch_config.get("num_envs")
        or snapshot.get("num_envs")
        or benchmark_info.get("num_envs")
        or _run_log_number(path, "num_envs"),
        "num_frames": launch_config.get("num_frames")
        or snapshot.get("num_frames")
        or benchmark_info.get("num_frames")
        or _run_log_number(path, "num_frames"),
        "excluded_frames": sorted(excluded),
        "gpu_mem_used_mb": gpu_mem_used_mb,
        "gpu_mem_mean_mb": nvidia_memory.get("gpu_mem_mean_mb"),
        "gpu_mem_peak_mb": nvidia_memory.get("gpu_mem_peak_mb") or gpu_mem_used_mb,
        "system_ram_used_mb": system_ram_used_mb,
        "system_ram_mean_mb": docker_memory.get("system_ram_mean_mb") or system_ram_used_mb,
        "system_ram_peak_mb": docker_memory.get("system_ram_peak_mb"),
        "docker_cpu_mean_pct": docker_memory.get("docker_cpu_mean_pct"),
        "docker_cpu_peak_pct": docker_memory.get("docker_cpu_peak_pct"),
        "cpu_name": hardware.get("cpu_name"),
        "cpu_physical_cores": hardware.get("cpu_physical_cores") or hardware.get("physical_cores"),
        "total_ram_gb": hardware.get("total_ram_gb"),
        "gpu_name": (gpu_diag or {}).get("gpu_name") or hardware.get("gpu_name"),
        "gpu_total_memory_gb": (gpu_diag or {}).get("gpu_total_memory_gb") or hardware.get("gpu_total_memory_gb"),
        "cuda_version": (gpu_diag or {}).get("cuda_version") or hardware.get("cuda_version"),
        "nvidia_driver_version": (gpu_diag or {}).get("nvidia_driver_version"),
        "software": software,
    }


def _mean(values: list[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return statistics.mean(filtered) if filtered else None


def _std(values: list[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return statistics.stdev(filtered) if len(filtered) > 1 else 0.0 if len(filtered) == 1 else None


def _median(values: list[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return statistics.median(filtered) if filtered else None


def _first(values: list[Any]) -> Any:
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def _aggregate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for sample in samples:
        key = (sample["label"], sample["task_id"], sample["backend"])
        groups.setdefault(key, []).append(sample)

    rows = []
    for (label, task_id, backend), group in sorted(groups.items()):
        fps_values = [sample.get("measured_fps") for sample in group]
        rows.append(
            {
                "label": label,
                "task_id": task_id,
                "backend": backend,
                "sample_count": len(group),
                "mean_fps": _mean(fps_values),
                "median_fps": _median(fps_values),
                "run_to_run_std_fps": _std(fps_values),
                "min_fps": min(float(v) for v in fps_values if v is not None),
                "max_fps": max(float(v) for v in fps_values if v is not None),
                "mean_within_run_std_fps": _mean([sample.get("within_run_std_fps") for sample in group]),
                "mean_gpu_mem_used_mb": _mean([sample.get("gpu_mem_used_mb") for sample in group]),
                "mean_gpu_mem_mean_mb": _mean([sample.get("gpu_mem_mean_mb") for sample in group]),
                "peak_gpu_mem_used_mb": max(
                    [float(v) for v in [sample.get("gpu_mem_peak_mb") for sample in group] if v is not None],
                    default=None,
                ),
                "mean_system_ram_used_mb": _mean([sample.get("system_ram_mean_mb") for sample in group]),
                "peak_system_ram_used_mb": max(
                    [float(v) for v in [sample.get("system_ram_peak_mb") for sample in group] if v is not None],
                    default=None,
                ),
                "mean_docker_cpu_pct": _mean([sample.get("docker_cpu_mean_pct") for sample in group]),
                "peak_docker_cpu_pct": max(
                    [float(v) for v in [sample.get("docker_cpu_peak_pct") for sample in group] if v is not None],
                    default=None,
                ),
                "num_envs": _first([sample.get("num_envs") for sample in group]),
                "num_frames": _first([sample.get("num_frames") for sample in group]),
                "cpu_name": _first([sample.get("cpu_name") for sample in group]),
                "cpu_physical_cores": _first([sample.get("cpu_physical_cores") for sample in group]),
                "total_ram_gb": _first([sample.get("total_ram_gb") for sample in group]),
                "gpu_name": _first([sample.get("gpu_name") for sample in group]),
                "gpu_total_memory_gb": _first([sample.get("gpu_total_memory_gb") for sample in group]),
                "cuda_version": _first([sample.get("cuda_version") for sample in group]),
                "nvidia_driver_version": _first([sample.get("nvidia_driver_version") for sample in group]),
            }
        )
    return rows


def _fmt(value: Any, *, decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.{decimals}f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, Any]], inputs: list[ArtifactInput]) -> None:
    lines = [
        "# 3.0 vs 2.3.2 Regression Evidence",
        "",
        "This report is generated from downloaded benchmark artifacts. FPS uses the same steady-state "
        "convention as the gate: mean `Environment step effective FPS` after excluded warm-up frames.",
        "",
        "## Inputs",
        "",
    ]
    for item in inputs:
        lines.append(f"- `{item.label}`: `{item.root}`")
    lines += [
        "",
        "## Summary",
        "",
        "| Label | Task | Backend | Env count | Samples | Mean FPS | Median FPS | Run-to-run std | "
        "Avg within-run std | Mean VRAM MB | Peak VRAM MB | Mean system RAM MB | Peak system RAM MB | CPU | GPU |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        cpu = row.get("cpu_name") or "N/A"
        if row.get("cpu_physical_cores") is not None:
            cpu = f"{cpu} ({row['cpu_physical_cores']} physical cores)"
        gpu = row.get("gpu_name") or "N/A"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["task_id"]),
                    str(row["backend"]),
                    _fmt(row.get("num_envs"), decimals=0),
                    _fmt(row["sample_count"], decimals=0),
                    _fmt(row.get("mean_fps")),
                    _fmt(row.get("median_fps")),
                    _fmt(row.get("run_to_run_std_fps")),
                    _fmt(row.get("mean_within_run_std_fps")),
                    _fmt(row.get("mean_gpu_mem_mean_mb")),
                    _fmt(row.get("peak_gpu_mem_used_mb")),
                    _fmt(row.get("mean_system_ram_used_mb")),
                    _fmt(row.get("peak_system_ram_used_mb")),
                    cpu,
                    gpu,
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `Run-to-run std` is computed across repeated benchmark samples for the same label/task/backend.",
        "- `Avg within-run std` is computed from per-frame steady-state FPS inside each sample, then averaged "
        "across samples.",
        "- `Mean VRAM MB` and `Peak VRAM MB` are populated from `nvidia-smi` samples when available. If an "
        "artifact only reports benchmark-level GPU memory, that value is used as peak VRAM.",
        "- `Mean system RAM MB` and `Peak system RAM MB` are populated from `docker stats` samples emitted by "
        "the evidence workflow.",
        "- Nsight Systems traces are uploaded separately as workflow artifacts and should be copied to Google "
        "Drive manually.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_inputs(raw_inputs: list[str]) -> list[ArtifactInput]:
    inputs = []
    for raw in raw_inputs:
        label, sep, path = raw.partition("=")
        if not sep or not label.strip() or not path.strip():
            raise ValueError(f"--input must be LABEL=PATH, got {raw!r}")
        inputs.append(ArtifactInput(label=label.strip(), root=Path(path).expanduser()))
    if not inputs:
        raise ValueError("At least one --input LABEL=PATH is required.")
    return inputs


def main() -> int:
    args = _parse_args()
    inputs = _parse_inputs(args.input)
    samples: list[dict[str, Any]] = []
    for item in inputs:
        for path in _candidate_files(item.root):
            sample = _sample_from_file(item.label, path)
            if sample is not None:
                samples.append(sample)
    rows = _aggregate(samples)
    payload = {
        "inputs": [{"label": item.label, "root": str(item.root)} for item in inputs],
        "samples": samples,
        "summary": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(args.markdown_out, rows, inputs)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
