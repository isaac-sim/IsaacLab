# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Derive per-task benchmark metadata from a completed dispatch.

A first dispatch runs every task at its shipped agent-config default and records
what actually happened. This module reads those bundles back and distils them
into ``task_metadata.yaml``: the resolved sizing, a wall-clock budget, and a
reward baseline. Later dispatches read that file as an overlay on the seed list,
so the sizing table is measured rather than guessed.

The comparable upstream table is
``source/isaaclab_tasks/test/benchmarking/configs.yaml``, which is the natural
home for these values once they have settled.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DEFAULT_TIMEOUT_HEADROOM",
    "HarvestError",
    "TaskMetadata",
    "harvest_dispatch",
    "read_bundle",
    "write_task_metadata",
]


class HarvestError(ValueError):
    """Raised when a dispatch cannot be harvested into trustworthy metadata."""


# Multiplier applied to the slowest observed run when deriving a wall-clock
# budget. Generous because an under-budgeted row is killed mid-training, which
# costs a whole rerun, whereas an over-budgeted one merely finishes early.
DEFAULT_TIMEOUT_HEADROOM = 2.0

_BUNDLE_GLOB = "benchmark_*.json"


@dataclass(frozen=True)
class TaskMetadata:
    """Measured metadata for one ``(task, rl_library, physics)`` combination.

    Args:
        task_id: Gym task id.
        rl_library: RL library the runs used.
        physics: Physics backend the runs used.
        renderer: Rendering backend the runs used, or ``none``.
        presets: Comma-joined domain presets the runs used.
        num_envs: Resolved parallel environment count.
        max_iterations: Resolved training iteration budget.
        duration_s_max: Slowest observed wall-clock duration [s].
        timeout_s: Derived wall-clock budget [s].
        reward_final_ema_mean: Mean final EMA reward across seeds, or ``None``
            when no run reported one.
        reward_final_ema_min: Lowest final EMA reward across seeds, or ``None``.
        seeds: Seeds that contributed, ascending.
    """

    task_id: str
    rl_library: str
    physics: str
    renderer: str
    presets: str
    num_envs: int | None
    max_iterations: int | None
    duration_s_max: float
    timeout_s: int
    reward_final_ema_mean: float | None
    reward_final_ema_min: float | None
    seeds: list[int]


def read_bundle(row_dir: Path) -> dict[str, Any] | None:
    """Return the parsed schema bundle in *row_dir*, or ``None``.

    Args:
        row_dir: Directory holding one row's results.

    Returns:
        The first readable bundle mapping that declares a ``schema_version``, or
        ``None`` when the directory holds none.
    """
    if not row_dir.is_dir():
        return None
    for candidate in sorted(row_dir.glob(_BUNDLE_GLOB)):
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version"):
            return payload
    return None


def _duration_of(bundle: dict[str, Any]) -> float:
    """Return a run's wall-clock duration [s], preferring the runtime total."""
    runtime = bundle.get("runtime") or {}
    total = runtime.get("total_wall_time_s")
    if isinstance(total, (int, float)) and total > 0:
        return float(total)
    return float((bundle.get("run") or {}).get("duration_s") or 0.0)


def _reward_of(bundle: dict[str, Any]) -> float | None:
    """Return a run's final EMA reward, or ``None`` when absent."""
    reward = ((bundle.get("learning") or {}).get("reward") or {}).get("final_ema")
    return float(reward) if isinstance(reward, (int, float)) else None


def harvest_dispatch(dispatch_dir: Path, *, timeout_headroom: float = DEFAULT_TIMEOUT_HEADROOM) -> list[TaskMetadata]:
    """Distil every readable bundle under *dispatch_dir* into per-task metadata.

    Runs whose recorded status is not ``completed`` are skipped: a crashed or
    interrupted run's duration and reward describe the failure, not the task.

    Args:
        dispatch_dir: Dispatch directory holding one sub-directory per row.
        timeout_headroom: Multiplier applied to the slowest observed duration
            when deriving ``timeout_s``.

    Returns:
        One entry per ``(task_id, rl_library, physics)``, sorted by that key.

    Raises:
        HarvestError: If *timeout_headroom* is not greater than zero, or the
            dispatch's ``dispatch.json`` exists but cannot be read.
    """
    if timeout_headroom <= 0:
        raise HarvestError(f"timeout_headroom must be > 0, got {timeout_headroom}")

    # An A/B dispatch holds two commits' results side by side. Harvesting both
    # into one baseline would silently average them, so side B is excluded.
    # An unreadable state file is fatal rather than empty: treating it as "no
    # side B" is exactly the silent averaging this guards against.
    side_b_rows: set[str] = set()
    state_path = dispatch_dir / "dispatch.json"
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text())
        except (OSError, ValueError) as exc:
            raise HarvestError(
                f"could not read {state_path}: {exc}. Without it an A/B dispatch's two sides would be averaged "
                "into one baseline."
            ) from exc
        side_b_rows = {
            str(job.get("row_key")) for job in payload.get("jobs") or [] if job.get("side") not in (None, "a")
        }

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row_dir in sorted(p for p in dispatch_dir.iterdir() if p.is_dir()):
        if row_dir.name in side_b_rows:
            continue
        bundle = read_bundle(row_dir)
        if bundle is None:
            continue
        run = bundle.get("run") or {}
        if run.get("status") != "completed":
            continue
        task_id = run.get("task")
        rl_library = run.get("framework")
        config = run.get("config") or {}
        physics = config.get("physics_backend")
        if not (task_id and rl_library and physics):
            continue
        # Renderer and presets change what a run measures, so they must not be
        # averaged together: a depth rollout is not the same workload as an RGB
        # one on the same task and backend.
        renderer = str(config.get("rendering_backend") or "none")
        presets = ",".join(sorted(config.get("presets") or []))
        grouped.setdefault((str(task_id), str(rl_library), str(physics), renderer, presets), []).append(bundle)

    entries: list[TaskMetadata] = []
    for (task_id, rl_library, physics, renderer, presets), bundles in sorted(grouped.items()):
        durations = [_duration_of(bundle) for bundle in bundles]
        rewards = [value for value in (_reward_of(bundle) for bundle in bundles) if value is not None]
        first_run = bundles[0].get("run") or {}
        duration_max = max(durations) if durations else 0.0
        entries.append(
            TaskMetadata(
                task_id=task_id,
                rl_library=rl_library,
                physics=physics,
                renderer=renderer,
                presets=presets,
                num_envs=_optional_int(first_run.get("num_envs")),
                max_iterations=_optional_int(first_run.get("max_iterations")),
                duration_s_max=round(duration_max, 1),
                timeout_s=max(1, int(duration_max * timeout_headroom)),
                reward_final_ema_mean=round(statistics.fmean(rewards), 4) if rewards else None,
                reward_final_ema_min=round(min(rewards), 4) if rewards else None,
                seeds=sorted({int(bundle["run"]["seed"]) for bundle in bundles if "seed" in (bundle.get("run") or {})}),
            )
        )
    return entries


def write_task_metadata(path: Path, entries: list[TaskMetadata]) -> None:
    """Write harvested metadata to *path* as YAML.

    Args:
        path: Destination file; parent directories are created.
        entries: Harvested metadata.
    """
    payload = {
        "tasks": [
            {
                "task_id": entry.task_id,
                "rl_library": entry.rl_library,
                "physics": entry.physics,
                "renderer": entry.renderer,
                "presets": entry.presets,
                "num_envs": entry.num_envs,
                "max_iterations": entry.max_iterations,
                "timeout_s": entry.timeout_s,
                "observed": {
                    "duration_s_max": entry.duration_s_max,
                    "reward_final_ema_mean": entry.reward_final_ema_mean,
                    "reward_final_ema_min": entry.reward_final_ema_min,
                    "seeds": entry.seeds,
                },
            }
            for entry in entries
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by tools/odin/harvest.py from a completed dispatch.\n"
        "# 'timeout_s' is the slowest observed run scaled by the harvest headroom;\n"
        "# 'observed' records what the measurement was derived from.\n"
    )
    path.write_text(header + yaml.safe_dump(payload, sort_keys=False))


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
