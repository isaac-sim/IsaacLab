# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the dispatch task list from the Gym registry.

Discovery is the default row source; a hand-written list remains possible as an
override via ``--tasks-yaml``. The registry walk lives in :func:`discover_tasks`
and needs Isaac Lab importable; everything else is pure and testable offline.

The emitted file is the same shape :func:`~tools.odin.plan.load_task_rows`
already consumes, so nothing downstream of ``PlannedRow`` changes.

Sizing is deliberately never invented here. ``num_envs``, ``max_iterations`` and
``timeout_s`` come from the harvested ``task_metadata.yaml`` overlay, measured
from a real run.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "POLICIES",
    "RL_LIBRARY_PRIORITY",
    "DiscoveredTask",
    "DiscoveryError",
    "discover_tasks",
    "filter_rows",
    "rows_for_policy",
    "write_task_list",
]

# Preferred order when a policy picks one library per task.
RL_LIBRARY_PRIORITY: tuple[str, ...] = ("rsl_rl", "rl_games", "skrl", "sb3")

# Physics presets never dispatched. ``newton_mjwarp_vbd_proxy`` is a proxy
# variant rather than a backend under test.
_SKIP_PHYSICS = frozenset({"newton_mjwarp_vbd_proxy"})

POLICIES: tuple[str, ...] = ("standard", "core-only", "all-libraries", "cross-backend")

_CROSS_BACKEND_PHYSICS = ("newton_mjwarp", "ovphysx")


class DiscoveryError(RuntimeError):
    """Raised when the registry cannot be walked."""


@dataclass(frozen=True)
class DiscoveredTask:
    """One registered training task, as discovered from the Gym registry.

    Args:
        task_id: Gym task id.
        scope: ``core`` or ``contrib``.
        rl_libraries: Odin-supported RL libraries the task declares, in
            :data:`RL_LIBRARY_PRIORITY` order.
        physics: Physics preset names the task declares, excluding names that
            are not valid ``physics=`` selectors.
        renderers: Renderer preset names the task declares.
    """

    task_id: str
    scope: str
    rl_libraries: tuple[str, ...]
    physics: tuple[str, ...]
    renderers: tuple[str, ...]


def _canonical_physics(names: tuple[str, ...]) -> tuple[str, ...]:
    """Return the physics presets worth dispatching for one task.

    ``physx`` is dropped when ``ovphysx`` is also declared: headless, ``physx``
    resolves to OvPhysX on most tasks, so running both is an exact duplicate.
    """
    kept = [name for name in names if name not in _SKIP_PHYSICS]
    if "ovphysx" in kept and "physx" in kept:
        kept.remove("physx")
    return tuple(kept)


def rows_for_policy(tasks: list[DiscoveredTask], policy: str) -> list[dict[str, Any]]:
    """Expand discovered tasks into dispatch rows under a named policy.

    Args:
        tasks: Discovered tasks.
        policy: One of :data:`POLICIES`.

    Returns:
        Row dicts sorted by ``(task_id, rl_library, physics)``, each carrying
        ``task_id``, ``scope``, ``rl_library`` and — unless the task declares no
        physics preset — ``physics``.

    Raises:
        DiscoveryError: If *policy* is unknown.
    """
    if policy not in POLICIES:
        raise DiscoveryError(f"unknown policy {policy!r}; expected one of {list(POLICIES)}")

    rows: list[dict[str, Any]] = []
    for task in tasks:
        if not task.rl_libraries:
            continue
        if policy == "core-only" and task.scope != "core":
            continue

        libraries = task.rl_libraries if policy == "all-libraries" else task.rl_libraries[:1]
        physics = _canonical_physics(task.physics)
        if policy == "cross-backend":
            physics = tuple(name for name in physics if name in _CROSS_BACKEND_PHYSICS)
            if not physics:
                continue

        for library in libraries:
            if not physics:
                # 35 tasks declare no physics preset and reject any physics=
                # token, so they get exactly one row with the field absent.
                rows.append({"task_id": task.task_id, "scope": task.scope, "rl_library": library})
                continue
            for name in physics:
                rows.append(
                    {"task_id": task.task_id, "scope": task.scope, "rl_library": library, "physics": name}
                )

    rows.sort(key=lambda row: (row["task_id"], row["rl_library"], row.get("physics") or ""))
    return rows


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    include: str | None = None,
    exclude: str | None = None,
    libraries: list[str] | None = None,
    physics: list[str] | None = None,
    scope: str | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Apply post-filters to policy rows.

    Args:
        rows: Rows from :func:`rows_for_policy`.
        include: Glob a ``task_id`` must match.
        exclude: Glob a ``task_id`` must not match.
        libraries: Restrict to these RL libraries.
        physics: Restrict to these physics presets. Rows with no physics token
            are kept only when ``"default"`` is listed.
        scope: ``core``, ``contrib``, or ``all``.
        max_rows: Deterministic head of the sorted order, as a cost valve.

    Returns:
        The filtered rows, order preserved.
    """
    result = rows
    if include is not None:
        result = [row for row in result if fnmatch.fnmatch(row["task_id"], include)]
    if exclude is not None:
        result = [row for row in result if not fnmatch.fnmatch(row["task_id"], exclude)]
    if libraries:
        result = [row for row in result if row["rl_library"] in libraries]
    if physics:
        wanted = set(physics)
        result = [row for row in result if (row.get("physics") or "default") in wanted]
    if scope and scope != "all":
        result = [row for row in result if row.get("scope") == scope]
    if max_rows is not None:
        result = result[:max_rows]
    return result


def discover_tasks() -> list[DiscoveredTask]:
    """Walk the Gym registry and return every dispatchable training task.

    Imports Isaac Lab, so it needs the project environment. Contrib tasks are
    included when ``isaaclab_tasks_experimental`` is importable.

    Returns:
        Discovered tasks sorted by ``task_id``.

    Raises:
        DiscoveryError: If the task packages cannot be imported.
    """
    import contextlib

    try:
        import gymnasium as gym

        import isaaclab_tasks  # noqa: F401

        from isaaclab_tasks.utils.preset_cli import enumerate_task_presets
        from isaaclab_tasks.utils.preset_target import PresetTarget
    except ImportError as exc:
        raise DiscoveryError(f"could not import the Isaac Lab task packages: {exc}") from exc

    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    tasks: list[DiscoveredTask] = []
    for spec in gym.registry.values():
        if not _is_training_task(spec.id) or spec.kwargs.get("deprecated"):
            continue
        libraries = _rl_libraries_from_kwargs(spec.kwargs)
        if not libraries:
            continue
        preset_map = enumerate_task_presets(spec.id)
        physics = tuple(sorted(preset_map.get(PresetTarget.PHYSICS, []))) if preset_map else ()
        renderers = tuple(sorted(preset_map.get(PresetTarget.RENDERER, []))) if preset_map else ()
        tasks.append(
            DiscoveredTask(
                task_id=spec.id,
                scope="contrib" if spec.id.startswith("IsaacContrib-") else "core",
                rl_libraries=libraries,
                physics=physics,
                renderers=renderers,
            )
        )
    tasks.sort(key=lambda task: task.task_id)
    return tasks


def _is_training_task(task_id: str) -> bool:
    """Return whether *task_id* is a trainable Isaac task."""
    if "Isaac" not in task_id:
        return False
    return not task_id.endswith("-Eval") and "-Benchmark-" not in task_id


def _rl_libraries_from_kwargs(kwargs: dict[str, Any]) -> tuple[str, ...]:
    """Return the Odin-supported RL libraries a registration declares."""
    declared = set()
    for key in kwargs:
        if not key.endswith("_cfg_entry_point") or key == "env_cfg_entry_point":
            continue
        stem = key[: -len("_cfg_entry_point")]
        for candidate in RL_LIBRARY_PRIORITY:
            if stem == candidate or stem.startswith(f"{candidate}_"):
                declared.add(candidate)
                break
    return tuple(name for name in RL_LIBRARY_PRIORITY if name in declared)


def write_task_list(path: Path, rows: list[dict[str, Any]], *, meta: dict[str, Any]) -> None:
    """Write discovered rows as a task list.

    Args:
        path: Destination file; parent directories are created.
        rows: Rows to write.
        meta: Provenance recorded under a ``discovery`` header, which
            :func:`~tools.odin.plan.load_task_rows` ignores.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by tools/odin/discover.py. Do not edit by hand -- regenerate,\n"
        "# or pass a hand-written file to `dispatch --tasks-yaml` to override.\n"
        "#\n"
        "# Sizing is absent by design: it is harvested from a real run into\n"
        "# task_metadata.yaml and applied as an overlay.\n"
    )
    path.write_text(header + yaml.safe_dump({"discovery": meta, "tasks": rows}, sort_keys=False))
