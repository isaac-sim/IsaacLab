# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the dispatch task list from the Gym registry.

Discovery is the default row source; a hand-written list remains possible as an
override via ``--tasks_yaml``. Expansion is total and filtering is the only way
to narrow, so there is one vocabulary rather than two. The registry walk lives in :func:`discover_tasks`
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
    "RL_LIBRARY_PRIORITY",
    "DiscoveredTask",
    "DiscoveryError",
    "discover_tasks",
    "filter_rows",
    "expand_rows",
    "write_task_list",
]

# Stable ordering for the RL library axis.
RL_LIBRARY_PRIORITY: tuple[str, ...] = ("rsl_rl", "rl_games", "skrl", "sb3")

# Physics presets never dispatched. ``newton_mjwarp_vbd_proxy`` is a proxy
# variant rather than a backend under test.
_SKIP_PHYSICS = frozenset({"newton_mjwarp_vbd_proxy"})


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
        modes: The task's **legal** physics/renderer combinations. Callers never
            reconstruct the cross product or reason about which pairings are
            rejected; discovery has already resolved that.
    """

    @dataclass(frozen=True)
    class Mode:
        """One legal way to run a task.

        Args:
            physics: Physics preset token, or ``None`` for tasks that declare
                none and reject any ``physics=`` selector.
            renderer: Renderer preset token, or ``None`` to run headless.
        """

        physics: str | None
        renderer: str | None

    task_id: str
    scope: str
    rl_libraries: tuple[str, ...]
    modes: tuple[DiscoveredTask.Mode, ...]


def _canonical_physics(names: tuple[str, ...]) -> tuple[str, ...]:
    """Return the physics presets worth dispatching for one task.

    ``physx`` is dropped when ``ovphysx`` is also declared: headless, ``physx``
    resolves to OvPhysX on most tasks, so running both is an exact duplicate.
    """
    kept = [name for name in names if name not in _SKIP_PHYSICS]
    if "ovphysx" in kept and "physx" in kept:
        kept.remove("physx")
    return tuple(kept)


def expand_rows(tasks: list[DiscoveredTask]) -> list[dict[str, Any]]:
    """Expand discovered tasks into every dispatchable row.

    Expansion is total across the RL library axis and the task's legal modes.
    Narrowing is :func:`filter_rows`' job, so there is one mechanism for "what to
    keep" rather than a second vocabulary of named row-set shapes.

    Args:
        tasks: Discovered tasks.

    Returns:
        Row dicts sorted by ``(task_id, rl_library, physics, renderer)``. The
        ``physics`` and ``renderer`` keys are omitted when the mode leaves them
        unset, because an empty ``physics=`` token is rejected by the task.
    """
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for library in task.rl_libraries:
            for mode in task.modes:
                row: dict[str, Any] = {
                    "task_id": task.task_id,
                    "scope": task.scope,
                    "rl_library": library,
                }
                if mode.physics is not None:
                    row["physics"] = mode.physics
                if mode.renderer is not None:
                    row["renderer"] = mode.renderer
                rows.append(row)

    rows.sort(key=lambda row: (row["task_id"], row["rl_library"], row.get("physics") or "", row.get("renderer") or ""))
    return rows


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    include: str | None = None,
    exclude: str | None = None,
    libraries: list[str] | None = None,
    physics: list[str] | None = None,
    renderers: list[str] | None = None,
    scope: str | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Narrow expanded rows.

    Args:
        rows: Rows from :func:`expand_rows`.
        include: Glob a ``task_id`` must match.
        exclude: Glob a ``task_id`` must not match.
        libraries: Restrict to these RL libraries.
        physics: Restrict to these physics presets. Rows with no physics token
            are kept only when ``"default"`` is listed.
        renderers: Restrict to these renderer presets. Rows are headless unless
            a renderer was requested, so ``"none"`` keeps headless rows.
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
    if renderers:
        wanted_renderers = set(renderers)
        result = [row for row in result if (row.get("renderer") or "none") in wanted_renderers]
    if scope and scope != "all":
        result = [row for row in result if row.get("scope") == scope]
    if max_rows is not None:
        result = result[:max_rows]
    return result


def discover_tasks(*, validate_modes: bool = True) -> list[DiscoveredTask]:
    """Walk the Gym registry and return every dispatchable training task.

    Imports Isaac Lab, so it needs the project environment. Contrib tasks are
    included when ``isaaclab_tasks_experimental`` is importable.

    For tasks that declare renderers, every physics/renderer pairing is checked
    against the runtime validator and only the legal ones become modes. That
    matters because the cross product is not all legal: OVRTX is kitless and
    cannot share a process with Kit physics, so ``isaacsim_physx + ovrtx`` is
    rejected. Discovering that here costs one config resolution per pairing;
    discovering it on a GPU costs a whole dispatch.

    Args:
        validate_modes: Check renderer pairings against the runtime validator.
            Disabling it emits the full cross product, which is faster but will
            include combinations that fail at startup.

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
        physics = _canonical_physics(tuple(sorted(preset_map.get(PresetTarget.PHYSICS, [])))) if preset_map else ()
        renderers = tuple(sorted(preset_map.get(PresetTarget.RENDERER, []))) if preset_map else ()
        tasks.append(
            DiscoveredTask(
                task_id=spec.id,
                scope="contrib" if spec.id.startswith("IsaacContrib-") else "core",
                rl_libraries=libraries,
                modes=_legal_modes(spec.id, physics, renderers, validate=validate_modes),
            )
        )
    tasks.sort(key=lambda task: task.task_id)
    return tasks


def _legal_modes(
    task_id: str, physics: tuple[str, ...], renderers: tuple[str, ...], *, validate: bool
) -> tuple[DiscoveredTask.Mode, ...]:
    """Return the legal physics/renderer combinations for one task.

    A task declaring renderers is always expanded across them: benchmarking a
    camera task headless measures everything except the thing under test.
    """
    physics_options: tuple[str | None, ...] = physics or (None,)
    if not renderers:
        return tuple(DiscoveredTask.Mode(physics=name, renderer=None) for name in physics_options)

    modes: list[DiscoveredTask.Mode] = []
    for physics_name in physics_options:
        for renderer in renderers:
            if validate and not _mode_resolves(task_id, physics_name, renderer):
                continue
            modes.append(DiscoveredTask.Mode(physics=physics_name, renderer=renderer))
    return tuple(modes)


def _mode_resolves(task_id: str, physics: str | None, renderer: str | None) -> bool:
    """Return whether a physics/renderer pairing resolves and passes validation.

    Any failure — an unknown preset, an unloadable config, or a rejected backend
    combination — makes the pairing undispatchable, so all of them are treated
    the same.
    """
    import argparse
    import sys

    from isaaclab.app.sim_launcher import _validate_runtime, scan

    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--agent", default=None)
    argv = ["--task", task_id]
    if physics is not None:
        argv.append(f"physics={physics}")
    if renderer is not None:
        argv.append(f"renderer={renderer}")

    original_argv = list(sys.argv)
    try:
        args, remaining = setup_preset_cli(parser, argv)
        sys.argv = [sys.argv[0]] + remaining
        env_cfg, _ = resolve_task_config(args.task, args.agent)
        _validate_runtime(scan(env_cfg, args), args)
        return True
    except Exception:  # noqa: BLE001 - any failure means undispatchable
        return False
    finally:
        sys.argv = original_argv


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
