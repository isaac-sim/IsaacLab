# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""List the environments registered by the generated project."""

import argparse
import importlib

import gymnasium as gym
from prettytable import PrettyTable

from isaaclab_tasks.utils.preset_cli import enumerate_task_presets
from isaaclab_tasks.utils.preset_target import PresetTarget

importlib.import_module("{{ name }}.tasks")


def _format_presets(preset_map: dict | None) -> str:
    """Format the available preset selectors for one task."""
    if preset_map is None:
        return "(unavailable)"

    labels = {
        PresetTarget.PHYSICS: "physics",
        PresetTarget.RENDERER: "renderer",
        PresetTarget.DOMAIN: "domain",
    }
    lines = [f"{label}: {', '.join(preset_map[target])}" for target, label in labels.items() if preset_map[target]]
    return "\n".join(lines) if lines else "(none)"


def main() -> None:
    """Print the generated project's registered environments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyword", default="{{ task_id_prefix }}-", help="Substring used to filter task ids.")
    parser.add_argument("--show_presets", action="store_true", help="Show physics, renderer, and domain presets.")
    args = parser.parse_args()

    task_specs = [
        spec for spec in gym.registry.values() if args.keyword in spec.id and not spec.kwargs.get("deprecated")
    ]
    columns = ["S. No.", "Task Name", "Entry Point", "Config"]
    if args.show_presets:
        columns.append("Presets")

    table = PrettyTable(columns)
    table.title = "Available {{ name }} Environments"
    for column in columns[1:]:
        table.align[column] = "l"

    for index, spec in enumerate(task_specs, start=1):
        row = [index, spec.id, spec.entry_point, spec.kwargs["env_cfg_entry_point"]]
        if args.show_presets:
            row.append(_format_presets(enumerate_task_presets(spec.id)))
        table.add_row(row)
    print(table)


if __name__ == "__main__":
    main()
