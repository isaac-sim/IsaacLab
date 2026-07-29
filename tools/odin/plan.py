# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expand the seed task list into dispatchable rows.

A seed entry carries only ``(task_id, rl_library, physics)``. Upstream defaults
``--num_envs`` and ``--max_iterations`` to ``None`` and falls back to the task's
shipped agent config, so a first dispatch runs every task at its shipped size
and the resolved values are harvested back out of the emitted bundles.

This module is also the seam the upstream task-discovery API replaces: when
``isaaclab tasks list --json`` lands, :func:`load_task_rows` swaps its source and
nothing downstream of :class:`PlannedRow` changes.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.odin.workflow import osmo_safe_task_name

__all__ = [
    "PlanError",
    "PlannedRow",
    "apply_metadata",
    "build_row_key",
    "normalize_presets",
    "chunk_rows",
    "load_task_rows",
    "plan_rows",
    "uv_extras_for_profile",
]

# Sizing fields a harvested metadata entry may contribute to a seed entry.
_OVERLAY_FIELDS = ("num_envs", "max_iterations", "timeout_s")

# Extras per profile. Mirrors image.PROFILES; kept here rather than imported to
# keep plan.py free of the image-build layer.
#
# A single ``full`` profile holds every backend, so a row's physics preset no
# longer has to be resolved to a Kit/kitless runtime before it can be dispatched
# -- which matters because a preset token does not determine the backend:
# `physics=physx` resolves to OvPhysX on 39 tasks and to Kit PhysX on 5.
PROFILE_EXTRAS: dict[str, tuple[str, ...]] = {
    "full": ("isaacsim", "ovphysx", "ovrtx", "rsl-rl", "skrl", "rl-games", "sb3", "rerun"),
}

DEFAULT_PROFILE = "full"

# Frames recorded during a chained play rollout.
DEFAULT_VIDEO_LENGTH = 200

_RL_LIBRARIES = frozenset({"rsl_rl", "rl_games", "skrl", "sb3"})

# ``physics`` is deliberately absent: 35 of the 87 usable tasks declare no
# physics preset at all and hard-fail on any ``physics=`` token.
_REQUIRED_FIELDS = ("task_id", "rl_library")


class PlanError(ValueError):
    """Raised when the task list cannot be expanded into rows."""


@dataclass(frozen=True)
class PlannedRow:
    """One dispatchable unit of work.

    Args:
        row_key: Deterministic identity, ``{rl_library}_{physics}[_{renderer}]_{task_id}_seed{seed}``.
        task_id: Gym task id.
        rl_library: One of ``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``.
        physics: Physics preset token passed as ``physics=<value>``, or ``None``
            for tasks that declare no physics preset.
        renderer: Renderer preset token, or ``None`` to run headless.
        presets: Domain preset tokens, emitted as a comma-separated
            ``presets=a,b``. Empty when none are selected.
        play: Whether to chain a play rollout after training, in the same
            task, reading the checkpoint training wrote.
        keep_checkpoint: Whether to copy the trained checkpoint into the
            uploaded output. RL libraries write it under ``logs/``, which
            OSMO does not collect.
        video_length: Frames to record during the chained play rollout.
        seed: Environment seed.
        num_envs: Parallel environment count, or ``None`` to use the task's
            shipped agent-config default.
        max_iterations: Training iterations, or ``None`` to use the task's
            shipped agent-config default.
        timeout_s: Wall-clock budget [s] driving chunk ordering, or ``None`` to
            inherit the workflow-level ceiling.
        uv_extras: ``--extra`` tokens for the task's ``uv run`` invocation.
        profile: uv profile the row runs under; ``full`` holds every backend.
        osmo_task_name: DNS-1123-safe OSMO task name derived from *row_key*.
    """

    row_key: str
    task_id: str
    rl_library: str
    physics: str | None
    renderer: str | None
    presets: tuple[str, ...]
    play: bool
    keep_checkpoint: bool
    video_length: int
    seed: int
    num_envs: int | None
    max_iterations: int | None
    timeout_s: int | None
    uv_extras: tuple[str, ...]
    profile: str
    osmo_task_name: str


def normalize_presets(value: Any) -> tuple[str, ...]:
    """Normalise a task entry's ``presets`` field into a token tuple.

    Hand-written task lists may give a single token, a comma-separated string,
    or a list. Discovery only ever emits one token because domain presets that
    target the same field conflict, but the executor imposes no such limit — a
    human who knows two presets compose can ask for both.

    Args:
        value: Raw ``presets`` value: ``None``, a string, or a sequence.

    Returns:
        Preset tokens in the given order, with blanks removed.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def build_row_key(
    rl_library: str,
    physics: str | None,
    task_id: str,
    seed: int,
    renderer: str | None = None,
    presets: tuple[str, ...] = (),
) -> str:
    """Return the deterministic identity for one row.

    The renderer and preset segments are only present when selected, so keys for
    plain headless rows are unchanged. They cannot be omitted when they *are*
    selected: a camera task expands across several renderers and domain presets
    on the same physics, and those rows would otherwise collide.

    Args:
        rl_library: RL library token.
        physics: Physics preset token, or ``None``.
        task_id: Gym task id.
        seed: Environment seed.
        renderer: Renderer preset token, or ``None`` for headless.
        presets: Domain preset tokens; empty when none are selected.

    Returns:
        ``{rl_library}_{physics}[_{renderer}][_{presets}]_{task_id}_seed{seed}``.
    """
    optional = "".join(f"{part}_" for part in (renderer, "-".join(presets)) if part)
    return f"{rl_library}_{physics or 'default'}_{optional}{task_id}_seed{seed}"


def uv_extras_for_profile(profile: str) -> tuple[str, ...]:
    """Return the full extras set for a uv profile.

    Args:
        profile: Profile name; ``full`` holds every backend.

    Returns:
        Every extra baked into that profile's virtualenv.

    Raises:
        PlanError: If *profile* is not a known profile.
    """
    if profile not in PROFILE_EXTRAS:
        raise PlanError(f"unknown profile {profile!r}; expected one of {sorted(PROFILE_EXTRAS)}")
    return PROFILE_EXTRAS[profile]


def load_task_rows(path: Path) -> list[dict[str, Any]]:
    """Load the seed task list.

    Args:
        path: Path to ``tasks.yaml``.

    Returns:
        Raw task entries, unvalidated; :func:`plan_rows` validates them.

    Raises:
        PlanError: If the file is unreadable, unparsable, or lacks a ``tasks``
            list.
    """
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise PlanError(f"could not read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PlanError(f"could not parse {path}: {exc}") from exc
    tasks = (raw or {}).get("tasks")
    if not isinstance(tasks, list):
        raise PlanError(f"{path} must contain a top-level 'tasks' list")
    return tasks


def _overlay_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    """Return the identity an entry is matched on when overlaying metadata.

    Mirrors :func:`~tools.odin.harvest.harvest_dispatch`'s grouping key exactly.
    Renderer and presets belong in it: a camera row and a headless row of the
    same task and backend measure different workloads, so one must not inherit
    the other's budget.
    """
    return (
        entry.get("task_id"),
        entry.get("rl_library"),
        entry.get("physics"),
        entry.get("renderer") or "none",
        ",".join(sorted(normalize_presets(entry.get("presets")))),
    )


def apply_metadata(task_rows: list[dict[str, Any]], metadata_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overlay harvested sizing onto seed entries.

    Entries are matched on ``(task_id, rl_library, physics, renderer, presets)``.
    A value already present on the seed entry wins, so a hand-set override is
    never silently replaced by a measurement. Metadata for combinations absent
    from the seed list is ignored.

    Args:
        task_rows: Seed entries from :func:`load_task_rows`.
        metadata_rows: Harvested entries, from ``task_metadata.yaml``'s
            ``tasks`` list.

    Returns:
        New seed entries with sizing filled in where it was missing.
    """
    by_key = {_overlay_key(entry): entry for entry in metadata_rows}
    merged: list[dict[str, Any]] = []
    for entry in task_rows:
        overlay = by_key.get(_overlay_key(entry))
        combined = dict(entry)
        if overlay is not None:
            for field_name in _OVERLAY_FIELDS:
                if combined.get(field_name) is None and overlay.get(field_name) is not None:
                    combined[field_name] = overlay[field_name]
        merged.append(combined)
    return merged


def _sort_key(row: PlannedRow) -> tuple[int, str, str, str, str, int]:
    """Order rows cheapest-first, with unbudgeted rows last.

    Rows without a harvested budget sort after budgeted ones so a chunk does not
    mix a known-short task with an unknown-length one. The renderer and preset
    segments make the order total: without them, a task's preset variants tie
    and their relative order follows input order rather than being fixed.
    """
    return (
        row.timeout_s if row.timeout_s is not None else 2**31 - 1,
        row.task_id,
        row.physics or "",
        row.renderer or "",
        ",".join(row.presets),
        row.seed,
    )


def plan_rows(
    *,
    task_rows: list[dict[str, Any]],
    seeds: list[int],
    include: str | None = None,
) -> list[PlannedRow]:
    """Expand task entries across seeds into dispatchable rows.

    Args:
        task_rows: Entries from :func:`load_task_rows`.
        seeds: Seeds to expand each entry across.
        include: Optional glob filter applied to ``task_id``.

    Returns:
        Rows sorted by ``(timeout_s, task_id, physics, renderer, presets, seed)``
        so reruns and resumes observe an identical layout.

    Raises:
        PlanError: On a missing field, an unknown RL library, or a duplicate row
            key.
    """
    rows: list[PlannedRow] = []
    seen: set[str] = set()

    for entry in task_rows:
        task_id = str(entry.get("task_id", "<unknown>"))
        if include is not None and not fnmatch.fnmatch(task_id, include):
            continue
        for field_name in _REQUIRED_FIELDS:
            if entry.get(field_name) is None:
                raise PlanError(f"task {task_id!r} is missing required field {field_name!r}")

        rl_library = str(entry["rl_library"])
        if rl_library not in _RL_LIBRARIES:
            raise PlanError(f"unknown rl_library {rl_library!r}; expected one of {sorted(_RL_LIBRARIES)}")
        physics = entry.get("physics")
        physics = str(physics) if physics else None
        renderer = entry.get("renderer")
        presets = normalize_presets(entry.get("presets"))
        play = bool(entry.get("play", False))
        keep_checkpoint = bool(entry.get("keep_checkpoint", False))
        video_length = int(entry.get("video_length", DEFAULT_VIDEO_LENGTH))
        profile = str(entry.get("profile") or DEFAULT_PROFILE)
        extras = uv_extras_for_profile(profile)

        for seed in seeds:
            row_key = build_row_key(rl_library, physics, task_id, seed, renderer, presets)
            if row_key in seen:
                raise PlanError(f"duplicate row key {row_key!r}; check tasks.yaml for repeated entries")
            seen.add(row_key)
            rows.append(
                PlannedRow(
                    row_key=row_key,
                    task_id=task_id,
                    rl_library=rl_library,
                    physics=physics,
                    renderer=renderer,
                    presets=presets,
                    play=play,
                    keep_checkpoint=keep_checkpoint,
                    video_length=video_length,
                    seed=int(seed),
                    num_envs=_optional_int(entry.get("num_envs")),
                    max_iterations=_optional_int(entry.get("max_iterations")),
                    timeout_s=_optional_int(entry.get("timeout_s")),
                    uv_extras=extras,
                    profile=profile,
                    osmo_task_name=osmo_safe_task_name(row_key),
                )
            )

    rows.sort(key=_sort_key)
    return rows


def chunk_rows(rows: list[PlannedRow], chunk_size: int) -> list[tuple[int, list[PlannedRow]]]:
    """Split rows into per-workflow chunks, cheapest rows first.

    Rows are sorted before chunking so a workflow groups rows of comparable cost
    and reruns observe an identical layout.

    Args:
        rows: Planned rows.
        chunk_size: Maximum rows per chunk.

    Returns:
        ``(chunk_index, chunk_rows)`` pairs with ascending *chunk_index*.

    Raises:
        PlanError: If *chunk_size* is less than one.
    """
    if chunk_size < 1:
        raise PlanError(f"chunk_size must be >= 1, got {chunk_size}")
    if not rows:
        return []
    ordered = sorted(rows, key=_sort_key)
    return [
        (index, ordered[start : start + chunk_size]) for index, start in enumerate(range(0, len(ordered), chunk_size))
    ]


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
