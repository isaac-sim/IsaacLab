# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter: schema-v1 ``RuntimeBundle`` JSON -> perf-gate normalized fields.

``perf_runtime.py`` emits a
:class:`~isaaclab.test.benchmark.schema.RuntimeBundle` (Isaac Lab benchmark
refactor Part 1, PR #6197) serialized by
:func:`~isaaclab.test.benchmark.serialize.write_bundle_file`.  This module is the
single point that reads that JSON and projects it into the flat
``provenance`` / ``runtime_resources`` / fps / ``benchmark_info`` shapes the gate's
:mod:`oracle` and :mod:`build_bench_result` consume, replacing the legacy
phase-array parsing.

``raw_fps_min`` is *recovered* from the slowest steady-state step
(``num_envs / iteration_time_s.peak``); ``raw_fps_{mean,std,max}`` come straight
from ``runtime.total_fps.{mean,std,peak}``.  Warmup exclusion is applied at the
source by ``perf_runtime.py`` (``--warmup_frames``), so ``total_fps.mean`` is
already steady-state.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from contracts import RuntimeSample

# Map the schema's rendering-backend vocabulary to the gate's render-preset
# tokens used in tasks.json / backend_identity.  ``"none"`` (headless, no camera)
# maps to ``None``.
_SCHEMA_RENDER_TO_GATE: dict[str | None, str | None] = {
    None: None,
    "": None,
    "none": None,
    "newton": "newton_renderer",
    "ovrtx": "ovrtx_renderer",
    "isaacsim_rtx": "rtx_renderer",
}


def _as_dict(value: Any) -> dict:
    """Return ``value`` if it is a dict, else an empty dict.

    Bundle sections are always dicts in a valid schema-v1 file; this keeps a
    malformed/hand-edited bundle (e.g. a list where a dict is expected) from
    crashing the projection so the gate degrades gracefully instead of the
    result builder aborting with no output.
    """
    return value if isinstance(value, dict) else {}


def steady_state_slice(step_times: list[float], warmup_frames: int) -> tuple[list[float], int]:
    """Drop the leading ``warmup_frames`` cold-start steps, keeping >=1 frame.

    Producer-side helper used by ``perf_runtime.py`` to exclude warmup at the
    source before aggregation. If ``warmup_frames`` would leave nothing, it is
    clamped to ``len(step_times) - 1`` so the aggregate never silently falls back
    to the full, cold-start-inclusive series (which would misreport non-steady
    numbers as steady-state).

    Args:
        step_times: Per-step wall times [s], in order.
        warmup_frames: Requested number of leading steps to discard.

    Returns:
        ``(measured_step_times, warmup_applied)`` where ``warmup_applied`` is the
        number of leading steps actually discarded (may be clamped below the
        request).
    """
    n = len(step_times)
    if n == 0:
        return [], 0
    warmup = max(0, min(warmup_frames, n - 1))
    return list(step_times[warmup:]), warmup


def load_info(info_path: Path) -> dict | None:
    """Load the benchmark info JSON, or ``None`` if it is missing/unreadable."""
    try:
        data = json.loads(Path(info_path).read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def is_runtime_bundle(data: Any) -> bool:
    """Return True when ``data`` looks like a schema-v1 runtime/training bundle."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("run"), dict)
        and isinstance(data.get("runtime"), dict)
        and "schema_version" in data
    )


def gpu_driver_version() -> str | None:
    """Return the GPU driver version from nvidia-smi, or ``None`` if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip().splitlines()[0].strip()
            return version or None
    except Exception:
        pass
    return None


def _current_gpu(bundle: dict) -> dict:
    """Return the first GPU device dict from the hardware snapshot (or ``{}``)."""
    devices = _as_dict(bundle.get("hardware")).get("gpu_devices") or []
    return devices[0] if devices and isinstance(devices[0], dict) else {}


def render_backend(bundle: dict) -> str | None:
    """Return the gate render token for the bundle's rendering backend."""
    schema_render = _as_dict(_as_dict(bundle.get("run")).get("config")).get("rendering_backend")
    key = schema_render.lower() if isinstance(schema_render, str) else schema_render
    return _SCHEMA_RENDER_TO_GATE.get(key, key or None)


def fps_stats(bundle: dict) -> dict:
    """Return ``raw_fps_{mean,std,min,max}`` from the bundle's runtime aggregates.

    ``mean``/``std``/``max`` come from ``total_fps``; ``min`` (worst steady-state
    frame) is recovered from the slowest step time.  Percentile fields are not
    available in the schema and are intentionally omitted.
    """
    runtime = _as_dict(bundle.get("runtime"))
    total_fps = _as_dict(runtime.get("total_fps"))
    iter_time = _as_dict(runtime.get("iteration_time_s"))
    steps_per_iter = runtime.get("steps_per_iteration") or _as_dict(bundle.get("run")).get("num_envs")

    out: dict = {}
    if isinstance(total_fps.get("mean"), (int, float)):
        out["raw_fps_mean"] = float(total_fps["mean"])
    if isinstance(total_fps.get("std"), (int, float)):
        out["raw_fps_std"] = float(total_fps["std"])
    if isinstance(total_fps.get("peak"), (int, float)):
        out["raw_fps_max"] = float(total_fps["peak"])
    peak_step = iter_time.get("peak")
    if isinstance(peak_step, (int, float)) and peak_step > 0 and steps_per_iter:
        out["raw_fps_min"] = float(steps_per_iter) / float(peak_step)
    return out


def startup_seconds(bundle: dict) -> float | None:
    """Return total launch-to-first-step wall time [s] (sum of startup phases)."""
    startup = _as_dict(_as_dict(bundle.get("runtime")).get("startup_time_s"))
    values = [v for v in startup.values() if isinstance(v, (int, float))]
    return float(sum(values)) if values else None


def provenance(bundle: dict) -> dict:
    """Return ``{hardware, software, git}`` for the runtime-compatibility contract.

    ``software`` is the bundle's typed ``versions`` map verbatim (its field names
    — ``isaaclab``/``isaacsim``/``torch``/``warp``/``isaaclab_physx``/
    ``isaaclab_newton``/``newton``/``isaaclab_ov`` — match the contract policy
    paths, so the ``runtime_contract_hash`` is preserved across the migration).
    """
    versions = _as_dict(bundle.get("versions"))
    hardware_snapshot = _as_dict(bundle.get("hardware"))
    gpu = _current_gpu(bundle)

    software = {k: v for k, v in versions.items() if v is not None and not k.startswith("git_")}
    hardware = {
        k: v
        for k, v in {
            "cpu_name": hardware_snapshot.get("cpu_name"),
            "cpu_physical_cores": hardware_snapshot.get("cpu_count"),
            "total_ram_gb": hardware_snapshot.get("ram_gb"),
            "gpu_device_count": len(hardware_snapshot.get("gpu_devices") or []) or None,
            "gpu_name": gpu.get("name"),
            "gpu_total_memory_gb": gpu.get("mem_gb"),
            "gpu_compute_capability": gpu.get("compute_cap"),
        }.items()
        if v is not None
    }
    git = {
        gate_key: versions[schema_key]
        for gate_key, schema_key in (
            ("commit_hash", "git_commit"),
            ("branch", "git_branch"),
            ("dirty", "git_dirty"),
        )
        if versions.get(schema_key) is not None
    }
    return {"hardware": hardware, "software": software, "git": git}


def _gb_to_mb(value: Any) -> float | None:
    """Return ``value`` GB converted to MB (2 dp), or ``None`` if not numeric."""
    return round(float(value) * 1024, 2) if isinstance(value, (int, float)) else None


def _pct(value: Any) -> float | None:
    """Return ``value`` as a utilisation percent (2 dp), or ``None`` if not numeric."""
    return round(float(value), 2) if isinstance(value, (int, float)) else None


def runtime_resources(bundle: dict) -> dict:
    """Return the GPU-diagnostics + resource-utilisation block (non-gating publish info).

    Combines GPU identity (name / total memory / CUDA / driver) with the run's
    measured resource utilisation from the bundle's ``resources`` section: VRAM
    (mean + peak), system RAM (mean + peak), and GPU/CPU utilisation (mean). Every
    field here is informational — it is published for humans but never feeds the
    gate verdict or the ``runtime_contract_hash``.

    Memory is reported in MB (the schema stores GB); utilisation in percent. Two
    semantic caveats worth remembering when reading these values:

    * ``gpu_mem_*`` is **device-wide** VRAM (``nvidia-smi memory.used`` includes
      any other process on the GPU) — accurate on a 1-benchmark-per-GPU runner.
    * ``system_ram_*`` is the benchmark **process** resident set size (psutil
      ``memory_info().rss``), not whole-host RAM and excluding child processes.

    Absent/malformed sub-sections drop their fields (``None`` filtered out) so a
    partial bundle degrades gracefully instead of crashing the projection.
    """
    gpu = _current_gpu(bundle)
    resources = _as_dict(bundle.get("resources"))
    gpu_mem = _as_dict(resources.get("gpu_mem_gb"))
    ram = _as_dict(resources.get("ram_gb"))
    gpu_util = _as_dict(resources.get("gpu_util_pct"))
    cpu_util = _as_dict(resources.get("cpu_util_pct"))
    # schema-v1 Hardware has no CUDA-runtime field; use the CUDA bindings version
    # (Versions.cuda_bindings) as the closest available proxy for display.
    cuda_version = _as_dict(bundle.get("versions")).get("cuda_bindings")
    diag = {
        "gpu_name": gpu.get("name"),
        "gpu_total_memory_gb": gpu.get("mem_gb"),
        "cuda_version": cuda_version,
        "nvidia_driver_version": gpu_driver_version(),
        "gpu_mem_used_mb": _gb_to_mb(gpu_mem.get("mean")),
        "gpu_mem_peak_mb": _gb_to_mb(gpu_mem.get("peak")),
        "gpu_util_pct": _pct(gpu_util.get("mean")),
        "system_ram_used_mb": _gb_to_mb(ram.get("mean")),
        "system_ram_peak_mb": _gb_to_mb(ram.get("peak")),
        "cpu_util_pct": _pct(cpu_util.get("mean")),
    }
    return {k: v for k, v in diag.items() if v is not None}


def benchmark_info(bundle: dict) -> dict:
    """Return the run's self-reported identity for launch/run drift checks."""
    run = _as_dict(bundle.get("run"))
    config = _as_dict(run.get("config"))
    extra = _as_dict(bundle.get("extra"))
    info = {
        "task": run.get("task"),
        "num_envs": run.get("num_envs"),
        "seed": run.get("seed"),
        "num_frames": extra.get("num_frames"),
        "warmup_frames": extra.get("warmup_frames"),
        "status": run.get("status"),
        "physics_backend": config.get("physics_backend"),
        "render_backend": render_backend(bundle),
        "presets": config.get("presets") or [],
    }
    return {k: v for k, v in info.items() if v is not None}


def project_runtime(bundle: dict) -> RuntimeSample | None:
    """Project a runtime bundle into a typed :class:`~contracts.RuntimeSample`.

    Returns ``None`` when ``bundle`` is not a valid schema-v1 runtime bundle, so the
    caller can degrade to a HARD_FAILURE (missing benchmark output).
    """
    if not is_runtime_bundle(bundle):
        return None
    stats = fps_stats(bundle)
    info = benchmark_info(bundle)
    return RuntimeSample(
        fps_mean=stats.get("raw_fps_mean"),
        fps_std=stats.get("raw_fps_std"),
        fps_min=stats.get("raw_fps_min"),
        fps_max=stats.get("raw_fps_max"),
        startup_time_s=startup_seconds(bundle),
        task=info.get("task"),
        num_envs=info.get("num_envs"),
        seed=info.get("seed"),
        num_frames=info.get("num_frames"),
        warmup_frames=info.get("warmup_frames"),
        status=info.get("status"),
        physics_backend=info.get("physics_backend"),
        render_backend=info.get("render_backend"),
        presets=info.get("presets") or [],
        provenance=provenance(bundle),
        runtime_resources=runtime_resources(bundle),
    )
