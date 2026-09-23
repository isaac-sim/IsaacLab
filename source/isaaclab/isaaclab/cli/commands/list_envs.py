# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""List registered Isaac Lab environments."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
from pathlib import Path
from typing import Any

import tomllib


def command_list_envs(args: list[str] | None = None) -> None:
    """List registered environments and their optional presets.

    When invoked from a project that declares an ``isaaclab.tasks`` entry point,
    the command shows that project's tasks by default. Pass ``--keyword`` to use
    an explicit task-id filter or ``--all`` to show every registered task.

    Args:
        args: Command-line arguments excluding the executable name.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--keyword", help="Substring used to filter task ids.")
    filter_group.add_argument("--all", action="store_true", help="Show every registered Isaac Lab task.")
    parser.add_argument("--show_presets", action="store_true", help="Show physics, renderer, and domain presets.")
    parsed_args = parser.parse_args(args)

    import gymnasium as gym
    from prettytable import PrettyTable

    import isaaclab_tasks.registry  # noqa: F401

    # PLACEHOLDER: Extension template (do not remove this comment)
    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    for entry_point in importlib.metadata.entry_points(group="isaaclab.tasks"):
        entry_point.load()

    project_modules = () if parsed_args.all or parsed_args.keyword else _find_project_task_modules(Path.cwd())
    task_specs = [
        spec
        for spec in gym.registry.values()
        if "env_cfg_entry_point" in spec.kwargs
        and not spec.kwargs.get("deprecated")
        and (parsed_args.keyword is None or parsed_args.keyword in spec.id)
        and (not project_modules or _belongs_to_project(spec, project_modules))
    ]

    columns = ["S. No.", "Task Name", "Entry Point", "Config"]
    if parsed_args.show_presets:
        columns.append("Presets")
    table = PrettyTable(columns)
    table.title = "Available Environments in Isaac Lab"
    for column in columns[1:]:
        table.align[column] = "l"

    if parsed_args.show_presets:
        from isaaclab_tasks.utils.preset_cli import enumerate_task_presets

    for index, spec in enumerate(task_specs, start=1):
        row = [index, spec.id, spec.entry_point, spec.kwargs["env_cfg_entry_point"]]
        if parsed_args.show_presets:
            row.append(_format_presets(enumerate_task_presets(spec.id)))
        table.add_row(row)
    print(table)


def _find_project_task_modules(start: Path) -> tuple[str, ...]:
    """Return task modules declared by the nearest parent ``pyproject.toml``."""
    for directory in (start, *start.parents):
        pyproject_path = directory / "pyproject.toml"
        if not pyproject_path.is_file():
            continue
        with pyproject_path.open("rb") as file:
            project = tomllib.load(file).get("project", {})
        entry_points = project.get("entry-points", {}).get("isaaclab.tasks", {})
        if isinstance(entry_points, dict):
            return tuple(value for value in entry_points.values() if isinstance(value, str))
        return ()
    return ()


def _belongs_to_project(spec: Any, task_modules: tuple[str, ...]) -> bool:
    """Return whether a Gym specification uses one of the project's task modules."""
    references = (spec.entry_point, spec.kwargs.get("env_cfg_entry_point"))
    module_prefixes = tuple(f"{module}." for module in task_modules)
    return any(
        isinstance(reference, str)
        and ((module := reference.partition(":")[0]) in task_modules or module.startswith(module_prefixes))
        for reference in references
    )


def _format_presets(preset_map: dict | None) -> str:
    """Format the available preset selectors for one task."""
    if preset_map is None:
        return "(unavailable)"

    from isaaclab_tasks.utils.preset_target import PresetTarget

    labels = {
        PresetTarget.PHYSICS: "physics",
        PresetTarget.RENDERER: "renderer",
        PresetTarget.DOMAIN: "domain",
    }
    lines = [f"{label}: {', '.join(preset_map[target])}" for target, label in labels.items() if preset_map[target]]
    return "\n".join(lines) if lines else "(none)"
