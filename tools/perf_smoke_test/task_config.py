# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .backend_identity import make_backend_key, normalize_physics_backend, normalize_render_backend
    from .gate_types import FpsMeanThreshold
    from .gpu_identity import gpu_model_config_keys
except ImportError:  # pragma: no cover - supports direct script imports
    from backend_identity import make_backend_key, normalize_physics_backend, normalize_render_backend
    from gate_types import FpsMeanThreshold
    from gpu_identity import gpu_model_config_keys

_DEFAULT_TASKS_JSON = Path(__file__).parent / "tasks.json"

# Maps backend name to list of cache identifiers that CI needs to pull; absent key defaults to no caches
_BACKEND_CACHES: dict[str, list[str]] = {
    "newton": ["mjwarp_jit"],
}


def parse_fps_mean_thresholds(raw) -> dict[str, dict[str, list[FpsMeanThreshold]]]:
    """Parse the nested ``{gpu_model: {backend_key: [entries]}}`` threshold config.

    Validation (mandatory names, gating-verdict enum, value-less skips) happens
    here via :meth:`FpsMeanThreshold.from_list`, so malformed ``tasks.json`` fails
    fast at load time.

    Args:
        raw: The raw ``fps_mean_thresholds`` value from ``tasks.json`` (may be empty).

    Returns:
        Nested mapping of GPU model to backend key to parsed thresholds.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("fps_mean_thresholds must be an object keyed by GPU model")
    parsed: dict[str, dict[str, list[FpsMeanThreshold]]] = {}
    for gpu_key, backends in raw.items():
        if not isinstance(backends, dict):
            raise TypeError(f"fps_mean_thresholds[{gpu_key!r}] must be an object keyed by backend")
        parsed[gpu_key] = {
            backend_key: FpsMeanThreshold.from_list(entries, context=f"{gpu_key}/{backend_key}")
            for backend_key, entries in backends.items()
        }
    return parsed


def caches_for_backend(backend: str) -> list[str]:
    """Return cache identifiers required before benchmarking with a given physics backend.

    The returned identifiers are consumed by the CI cache-pull step to locate and
    restore named cache artifacts by name or pattern.  An empty list means no
    pre-run cache restoration is needed.

    Currently defined identifiers:
        ``"mjwarp_jit"``: Newton MJWarp JIT compilation cache.

    Args:
        backend: Backend name (e.g. ``"newton"``, ``"physx"``).

    Returns:
        List of cache identifier strings.
    """
    return list(_BACKEND_CACHES.get(backend, []))


@dataclass
class TaskConfig:
    """Configuration for a single benchmark task and backend combination."""

    task_id: str
    physics_backend: str
    render_backend: str | None
    preset: str
    num_envs: int
    num_frames: int
    warmup_frames: int
    camera_resolution: tuple[int, int] | None
    timeout_minutes: int
    fps_mean_thresholds: dict[str, dict[str, list[FpsMeanThreshold]]]
    caches: list[str]
    tags: list[str] = field(default_factory=lambda: ["always"])
    task_type: str = "benchmark"
    runs_on: str = "gpu-l40s"
    seed: int | None = None
    baseline_epoch: int = 1
    noise_floor_pct: dict = field(default_factory=dict)

    @property
    def backend_key(self) -> str:
        """Composite key identifying the backend combination.

        Returns f"{physics_backend}_{render_backend}" when render_backend is set,
        otherwise returns physics_backend.
        """
        return make_backend_key(self.physics_backend, self.render_backend)

    def thresholds_for(self, gpu_model: str) -> list[FpsMeanThreshold]:
        """Return configured FPS thresholds for this task/backend on ``gpu_model``.

        Normalizes ``gpu_model`` to its canonical key (e.g. ``l40s``) and returns
        the first matching backend entry, or an empty list if none apply.

        Args:
            gpu_model: GPU model string (canonical slug, display name, or raw name).

        Returns:
            List of :class:`~gate_types.FpsMeanThreshold` (possibly empty).
        """
        for key in gpu_model_config_keys(gpu_model):
            backends = self.fps_mean_thresholds.get(key)
            if backends and self.backend_key in backends:
                return backends[self.backend_key]
        return []


def _load_tasks_json(path: Path) -> tuple[dict, list[dict]]:
    with open(path) as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        defaults = raw_data.get("defaults", {})
        raw_list = raw_data.get("tasks", [])
        if not isinstance(raw_list, list):
            raise TypeError(f"'tasks' field in {path} must be a list")
    elif isinstance(raw_data, list):
        defaults = {}
        raw_list = raw_data
    else:
        raise TypeError(f"{path} must contain a JSON list or an object with a top-level 'tasks' list")

    if not isinstance(defaults, dict):
        raise TypeError(f"'defaults' field in {path} must be an object")

    return defaults, raw_list


def load_tasks(tasks_json_path: Path | str | None = None) -> list[TaskConfig]:
    """Load all benchmark tasks from tasks.json, producing a TaskConfig for each backend combination.

    Args:
        tasks_json_path: Path to tasks.json. Defaults to the tasks.json next to this module.

    Returns:
        List of TaskConfig objects, one per (task_id, backend) combination.
    """
    path = Path(tasks_json_path) if tasks_json_path is not None else _DEFAULT_TASKS_JSON
    defaults, raw_list = _load_tasks_json(path)

    tasks: list[TaskConfig] = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            raise TypeError(f"task entry in {path} must be an object")
        merged = {**defaults, **raw}

        camera_raw = merged.get("camera_resolution")
        camera_resolution: tuple[int, int] | None = (
            tuple(camera_raw) if camera_raw is not None else None  # type: ignore[assignment]
        )
        fps_mean_thresholds = parse_fps_mean_thresholds(merged.get("fps_mean_thresholds", {}))
        noise_floor_pct: dict = merged.get("noise_floor_pct", {})
        backends: list[dict] = merged.get("backends", [])

        for backend_entry in backends:
            physics = normalize_physics_backend(backend_entry["physics"])
            if physics is None:
                raise ValueError(f"backend entry in {path} must define a non-default physics backend")
            render = normalize_render_backend(backend_entry.get("render"))
            tasks.append(
                TaskConfig(
                    task_id=merged["task_id"],
                    physics_backend=physics,
                    render_backend=render,
                    preset=merged["preset"],
                    num_envs=int(merged["num_envs"]),
                    num_frames=int(merged["num_frames"]),
                    warmup_frames=int(merged.get("warmup_frames", 0)),
                    camera_resolution=camera_resolution,
                    timeout_minutes=int(merged["timeout_minutes"]),
                    fps_mean_thresholds=fps_mean_thresholds,
                    noise_floor_pct=noise_floor_pct,
                    caches=caches_for_backend(physics),
                    tags=merged["tags"],
                    task_type=merged["type"],
                    runs_on=merged["runs_on"],
                    seed=int(merged["seed"]) if merged.get("seed") is not None else None,
                    baseline_epoch=int(merged.get("baseline_epoch", 1)),
                )
            )
    return tasks


def get_task(
    task_id: str,
    backend_key: str,
    tasks_json_path: Path | str | None = None,
) -> TaskConfig:
    """Return the TaskConfig for the given task_id and backend_key combination.

    Args:
        task_id: The task identifier to look up.
        backend_key: The backend key (e.g. "physx", "newton", "physx_rtx").
        tasks_json_path: Optional path to tasks.json.

    Returns:
        The matching TaskConfig.

    Raises:
        KeyError: If no task with the given (task_id, backend_key) exists.
    """
    tasks = load_tasks(tasks_json_path)
    for task in tasks:
        if task.task_id == task_id and task.backend_key == backend_key:
            return task
    raise KeyError(f"Task not found: task_id={task_id!r} backend_key={backend_key!r}")
