# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch configuration artifact helpers for the performance smoke test

``launch_config.json`` is the durable contract between the matrix builder, the
benchmark runner, post-processing, and aggregate.  Phase 2 treats this artifact
as the run intent instead of re-reading ``tasks.json``, so it can catch workflow
command bugs and self-hosted-runner handoff issues.
"""

import json
from pathlib import Path
from typing import Any

try:
    from .backend_identity import make_backend_key, normalize_render_backend
    from .gate_types import FpsMeanThreshold
    from .gpu_identity import normalize_gpu_fields
    from .hashing import stable_hash
    from .task_config import TaskConfig
except ImportError:  # pragma: no cover (for direct scripting execution/import)
    from backend_identity import make_backend_key, normalize_render_backend
    from gate_types import FpsMeanThreshold
    from gpu_identity import normalize_gpu_fields
    from hashing import stable_hash
    from task_config import TaskConfig

LAUNCH_CONFIG_SCHEMA_VERSION = 1
BENCHMARK_CONTRACT_VERSION = 1
DEFAULT_BASELINE_EPOCH = 1
LAUNCH_CONFIG_FILENAME = "launch_config.json"


def workload_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Return the workload-defining subset used for launch_config_hash."""
    keys = (
        "task_id",
        "backend_key",
        "physics_backend",
        "render_backend",
        "preset",
        "num_envs",
        "num_frames",
        "seed",
        "warmup_frames",
        "camera_resolution",
        "benchmark_formatter",
        "hydra_args",
    )
    return {key: config.get(key) for key in keys}


def hydra_args_for_task(task: TaskConfig) -> list[str]:
    """Return Hydra args used by the current local/CI benchmark launch path."""
    presets: list[str] = []
    if task.physics_backend == "newton":
        presets.append("newton_mjwarp")
    if task.render_backend:
        presets.append(task.render_backend)
        if task.render_backend == "newton_renderer":
            presets.append("rgb")
    return [f"presets={','.join(presets)}"] if presets else []


def task_to_launch_config(
    task: TaskConfig,
    *,
    fps_mean_thresholds: list[FpsMeanThreshold],
    gpu_model: str | None = None,
    hydra_args: list[str] | None = None,
    benchmark_formatter: str = "schema",
) -> dict[str, Any]:
    """Build a serializable launch config for one task/backend job"""
    gpu_fields = normalize_gpu_fields(gpu_model)
    config: dict[str, Any] = {
        "schema_version": LAUNCH_CONFIG_SCHEMA_VERSION,
        "task_id": task.task_id,
        "backend_key": make_backend_key(task.physics_backend, task.render_backend),
        "physics_backend": task.physics_backend,
        "render_backend": normalize_render_backend(task.render_backend),
        "preset": task.preset,
        "num_envs": task.num_envs,
        "num_frames": task.num_frames,
        "seed": task.seed,
        "warmup_frames": task.warmup_frames,
        "camera_resolution": list(task.camera_resolution) if task.camera_resolution else None,
        "timeout_minutes": task.timeout_minutes,
        "tags": list(task.tags),
        "gpu_model": gpu_fields["gpu_model"],
        "gpu_model_raw": gpu_fields["gpu_model_raw"],
        "benchmark_formatter": benchmark_formatter,
        "hydra_args": list(hydra_args or []),
        "fps_mean_thresholds": [t.to_dict() for t in fps_mean_thresholds],
        "baseline_epoch": int(getattr(task, "baseline_epoch", DEFAULT_BASELINE_EPOCH)),
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
    }
    config["launch_config_hash"] = stable_hash(workload_contract(config))
    config["benchmark_contract_hash"] = stable_hash(
        {
            "benchmark_contract_version": config["benchmark_contract_version"],
            "warmup_frames": config["warmup_frames"],
            "benchmark_formatter": config["benchmark_formatter"],
        }
    )
    return config


def fallback_launch_config(
    *,
    task_id: str,
    physics_backend: str,
    render_backend: str | None,
    backend_key: str,
    timeout_s: float,
    task: TaskConfig | None = None,
) -> dict[str, Any]:
    """Build a launch config for legacy/manual Phase 2 calls without an artifact"""
    normalized_backend_key = make_backend_key(physics_backend, render_backend)
    gpu_fields = normalize_gpu_fields(None)

    if task is None:
        config: dict[str, Any] = {
            "schema_version": LAUNCH_CONFIG_SCHEMA_VERSION,
            "task_id": task_id,
            "backend_key": normalized_backend_key,
            "physics_backend": physics_backend,
            "render_backend": normalize_render_backend(render_backend),
            "preset": "default",
            "num_envs": 0,
            "num_frames": 0,
            "seed": None,
            "warmup_frames": 0,
            "camera_resolution": None,
            "timeout_minutes": int(timeout_s / 60),
            "tags": ["always"],
            "gpu_model": gpu_fields["gpu_model"],
            "gpu_model_raw": gpu_fields["gpu_model_raw"],
            "benchmark_formatter": "schema",
            "hydra_args": [],
            "fps_mean_thresholds": [],
            "baseline_epoch": DEFAULT_BASELINE_EPOCH,
            "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        }
        config["launch_config_hash"] = stable_hash(workload_contract(config))
        config["benchmark_contract_hash"] = stable_hash(
            {
                "benchmark_contract_version": config["benchmark_contract_version"],
                "warmup_frames": config["warmup_frames"],
                "benchmark_formatter": config["benchmark_formatter"],
            }
        )
        return config

    return task_to_launch_config(
        task,
        fps_mean_thresholds=task.thresholds_for("L40S"),
        hydra_args=[],
    )


def load_launch_config(artifact_dir: Path, explicit_path: Path | None = None) -> dict[str, Any] | None:
    path = explicit_path or artifact_dir / LAUNCH_CONFIG_FILENAME
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def write_launch_config(artifact_dir: Path, config: dict[str, Any]) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / LAUNCH_CONFIG_FILENAME
    path.write_text(json.dumps(config, indent=2, sort_keys=True))
    return path
