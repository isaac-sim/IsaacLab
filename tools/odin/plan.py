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
    "chunk_rows",
    "load_task_rows",
    "plan_rows",
    "uv_extras_for",
]

# Sizing fields a harvested metadata entry may contribute to a seed entry.
_OVERLAY_FIELDS = ("num_envs", "max_iterations", "timeout_s")

# pyproject extra names differ from the --rl_library token: underscores in the
# CLI token, hyphens in the extra.
_RL_LIBRARY_EXTRAS: dict[str, str] = {
    "rsl_rl": "rsl-rl",
    "rl_games": "rl-games",
    "skrl": "skrl",
    "sb3": "sb3",
}

# Legacy preset aliases, mirroring PresetTarget.all_legacy_aliases(). Normalised
# before profile selection so ``physics=newton`` and ``physics=newton_mjwarp``
# resolve identically.
_PRESET_ALIASES: dict[str, str] = {
    "newton": "newton_mjwarp",
    "kamino": "newton_kamino",
    "ovrtx_renderer": "ovrtx",
    "isaacsim_rtx_renderer": "isaacsim_rtx",
}

# Physics presets that require Isaac Sim in the virtualenv. Both ``physx`` and
# its explicit ``isaacsim_physx`` spelling are registered across the task tree;
# they are distinct preset names, not aliases of each other.
_ISAACSIM_PHYSICS = frozenset({"physx", "isaacsim_physx"})
# Renderers that require Isaac Sim in the virtualenv. Note the separate ``rtx``
# preset (5 tasks) is deliberately absent: it has not been confirmed to need
# Isaac Sim, and guessing wrong either breaks the venv or silently drops it.
_ISAACSIM_RENDERERS = frozenset({"isaacsim_rtx"})
# Extras added alongside the RL library when Isaac Sim is required. Nothing from
# [tool.uv].conflicts' isaacsim row (all, test, mimic, teleop, ovphysx, viser)
# may ever join this set.
_ISAACSIM_COMPANIONS: tuple[str, ...] = ("isaacsim",)
# Extras added when Isaac Sim is not required.
_NEWTON_COMPANIONS: tuple[str, ...] = ("rerun",)
# Omniverse tokens that are only resolvable outside the isaacsim fork.
_OV_TOKENS = frozenset({"ovphysx", "ovrtx"})

_REQUIRED_FIELDS = ("task_id", "rl_library", "physics")


class PlanError(ValueError):
    """Raised when the task list cannot be expanded into rows."""


@dataclass(frozen=True)
class PlannedRow:
    """One dispatchable unit of work.

    Args:
        row_key: Deterministic identity, ``{rl_library}_{physics}_{task_id}_seed{seed}``.
        task_id: Gym task id.
        rl_library: One of ``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``.
        physics: Physics preset token passed as ``physics=<value>``.
        renderer: Renderer preset token, or ``None`` to run headless.
        seed: Environment seed.
        num_envs: Parallel environment count, or ``None`` to use the task's
            shipped agent-config default.
        max_iterations: Training iterations, or ``None`` to use the task's
            shipped agent-config default.
        timeout_s: Wall-clock budget [s] driving chunk ordering, or ``None`` to
            inherit the workflow-level ceiling.
        uv_extras: ``--extra`` tokens for the task's ``uv run`` invocation.
        osmo_task_name: DNS-1123-safe OSMO task name derived from *row_key*.
    """

    row_key: str
    task_id: str
    rl_library: str
    physics: str
    renderer: str | None
    seed: int
    num_envs: int | None
    max_iterations: int | None
    timeout_s: int | None
    uv_extras: tuple[str, ...]
    osmo_task_name: str


def build_row_key(rl_library: str, physics: str, task_id: str, seed: int) -> str:
    """Return the deterministic identity for one row.

    Args:
        rl_library: RL library token.
        physics: Physics preset token.
        task_id: Gym task id.
        seed: Environment seed.

    Returns:
        ``{rl_library}_{physics}_{task_id}_seed{seed}``.
    """
    return f"{rl_library}_{physics}_{task_id}_seed{seed}"


def uv_extras_for(rl_library: str, physics: str, renderer: str | None) -> tuple[str, ...]:
    """Return the ``uv`` extras needed to run one row.

    Isaac Sim conflicts with the ``all`` aggregate under ``[tool.uv].conflicts``,
    so RL-library extras are always selected individually rather than via ``all``.

    Args:
        rl_library: RL library token, e.g. ``rsl_rl``.
        physics: Physics preset token, e.g. ``physx``.
        renderer: Renderer preset token, or ``None``.

    Returns:
        Extras in a stable, de-duplicated order, suitable for ``--extra`` flags.

    Raises:
        PlanError: If *rl_library* is not a known library.
    """
    if rl_library not in _RL_LIBRARY_EXTRAS:
        raise PlanError(f"unknown rl_library {rl_library!r}; expected one of {sorted(_RL_LIBRARY_EXTRAS)}")

    physics = _PRESET_ALIASES.get(physics, physics)
    renderer = _PRESET_ALIASES.get(renderer or "", renderer or "")

    extras: list[str] = [_RL_LIBRARY_EXTRAS[rl_library]]
    if physics in _ISAACSIM_PHYSICS or renderer in _ISAACSIM_RENDERERS:
        extras.extend(_ISAACSIM_COMPANIONS)
    else:
        extras.extend(_NEWTON_COMPANIONS)
        extras.extend(token for token in (physics, renderer) if token in _OV_TOKENS)
    return tuple(dict.fromkeys(extras))


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


def apply_metadata(task_rows: list[dict[str, Any]], metadata_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overlay harvested sizing onto seed entries.

    Entries are matched on ``(task_id, rl_library, physics)``. A value already
    present on the seed entry wins, so a hand-set override is never silently
    replaced by a measurement. Metadata for combinations absent from the seed
    list is ignored.

    Args:
        task_rows: Seed entries from :func:`load_task_rows`.
        metadata_rows: Harvested entries, from ``task_metadata.yaml``'s
            ``tasks`` list.

    Returns:
        New seed entries with sizing filled in where it was missing.
    """
    by_key = {(entry.get("task_id"), entry.get("rl_library"), entry.get("physics")): entry for entry in metadata_rows}
    merged: list[dict[str, Any]] = []
    for entry in task_rows:
        overlay = by_key.get((entry.get("task_id"), entry.get("rl_library"), entry.get("physics")))
        combined = dict(entry)
        if overlay is not None:
            for field_name in _OVERLAY_FIELDS:
                if combined.get(field_name) is None and overlay.get(field_name) is not None:
                    combined[field_name] = overlay[field_name]
        merged.append(combined)
    return merged


def _sort_key(row: PlannedRow) -> tuple[int, str, str, int]:
    """Order rows cheapest-first, with unbudgeted rows last.

    Rows without a harvested budget sort after budgeted ones so a chunk does not
    mix a known-short task with an unknown-length one.
    """
    return (row.timeout_s if row.timeout_s is not None else 2**31 - 1, row.task_id, row.physics, row.seed)


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
        Rows sorted by ``(timeout_s, task_id, physics, seed)`` so reruns and
        resumes observe an identical layout.

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
        physics = str(entry["physics"])
        renderer = entry.get("renderer")
        extras = uv_extras_for(rl_library, physics, renderer)

        for seed in seeds:
            row_key = build_row_key(rl_library, physics, task_id, seed)
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
                    seed=int(seed),
                    num_envs=_optional_int(entry.get("num_envs")),
                    max_iterations=_optional_int(entry.get("max_iterations")),
                    timeout_s=_optional_int(entry.get("timeout_s")),
                    uv_extras=extras,
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
        ValueError: If *chunk_size* is less than one.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if not rows:
        return []
    ordered = sorted(rows, key=_sort_key)
    return [
        (index, ordered[start : start + chunk_size]) for index, start in enumerate(range(0, len(ordered), chunk_size))
    ]


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
