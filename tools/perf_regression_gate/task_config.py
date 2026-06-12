# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task matrix loader -- the single source of truth for the perf gate.

``tasks.json`` defines each benchmark task and its backend combinations. Most
fields are inherited from a top-level ``defaults`` block; a per-backend entry may
override ``num_envs``, ``excluded_frames`` (warm-up differs by backend -- e.g.
Newton JIT needs more warm-up frames than PhysX), and carries a per-GPU
``ref_fps`` calibration used for the relative catastrophic floor.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_TASKS_JSON = Path(__file__).parent / "tasks.json"

# Maps backend name to list of cache identifiers that CI needs to pull; absent key defaults to no caches
_BACKEND_CACHES: dict[str, list[str]] = {
    "newton": ["mjwarp_jit"],
}


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
    excluded_frames_raw: list[int | list[int]]
    camera_resolution: tuple[int, int] | None
    timeout_minutes: int
    ref_fps: dict[str, float]  # per-GPU calibrated reference FPS (for the relative hard floor)
    fps_floor_pct: float  # catastrophic floor as % of ref_fps (0 disables it)
    caches: list[str]
    tags: list[str] = field(default_factory=lambda: ["always"])
    task_type: str = "benchmark"
    runs_on: str = "gpu-rtx6000"
    seed: int | None = None
    enable_cameras: bool = False  # pass --enable_cameras (camera/rendering tasks)

    @property
    def backend_key(self) -> str:
        """Composite key identifying the backend combination.

        Returns f"{physics_backend}_{render_backend}" when render_backend is set,
        otherwise returns physics_backend.
        """
        if self.render_backend:
            return f"{self.physics_backend}_{self.render_backend}"
        return self.physics_backend

    @property
    def excluded_frames(self) -> frozenset[int]:
        """Expand raw excluded_frames entries (single index or inclusive range) to a
        frozenset of integer frame indices.
        """
        indices: set[int] = set()
        for entry in self.excluded_frames_raw:
            if isinstance(entry, list):
                if len(entry) != 2:
                    raise ValueError(f"excluded_frames range entry must have exactly 2 elements, got {entry!r}")
                start, end = entry[0], entry[1]
                if start > end:
                    raise ValueError(f"excluded_frames range start must be <= end, got [{start}, {end}]")
                indices.update(range(start, end + 1))
            else:
                indices.add(int(entry))
        return frozenset(indices)

    def fps_floor(self, gpu_model: str) -> float:
        """Absolute catastrophic FPS floor for this task on ``gpu_model``.

        Derived as ``fps_floor_pct%`` of the per-GPU calibrated ``ref_fps``. Returns
        ``0.0`` (disabled) when no reference is calibrated for the GPU. Expressing the
        floor relative to the reference keeps it meaningful across tasks whose
        effective FPS spans several orders of magnitude.
        """
        ref = self.ref_fps.get(gpu_model)
        if ref is None:
            # tolerate substring GPU keys (e.g. "NVIDIA L40S" vs "L40S")
            for key, val in self.ref_fps.items():
                if key in gpu_model or gpu_model in key:
                    ref = val
                    break
        if ref is None or self.fps_floor_pct <= 0:
            return 0.0
        return self.fps_floor_pct / 100.0 * float(ref)


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

    Field resolution is ``defaults`` < task entry < backend entry, so a backend may
    override ``num_envs`` and ``excluded_frames`` (warm-up) and carry its own ``ref_fps``.

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
        fps_floor_pct = float(merged.get("fps_floor_pct", 0.0))
        backends: list[dict] = merged.get("backends", [])

        for backend_entry in backends:
            physics = backend_entry["physics"]
            render = backend_entry.get("render")
            # Backend-level overrides fall back to the task/default value.
            excluded = backend_entry.get("excluded_frames", merged["excluded_frames"])
            num_envs = int(backend_entry.get("num_envs", merged["num_envs"]))
            ref_fps = {k: float(v) for k, v in (backend_entry.get("ref_fps", {}) or {}).items()}
            tasks.append(
                TaskConfig(
                    task_id=merged["task_id"],
                    physics_backend=physics,
                    render_backend=render,
                    preset=merged["preset"],
                    num_envs=num_envs,
                    num_frames=merged["num_frames"],
                    excluded_frames_raw=excluded,
                    camera_resolution=camera_resolution,
                    timeout_minutes=int(merged["timeout_minutes"]),
                    ref_fps=ref_fps,
                    fps_floor_pct=fps_floor_pct,
                    caches=caches_for_backend(physics),
                    tags=merged["tags"],
                    task_type=merged["type"],
                    runs_on=merged["runs_on"],
                    seed=merged.get("seed"),
                    enable_cameras=bool(merged.get("enable_cameras", False)),
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
