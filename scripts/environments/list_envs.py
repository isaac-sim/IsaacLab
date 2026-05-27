# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `isaaclab_tasks` extension. They start
with `Isaac` in their name.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="List Isaac Lab environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
parser.add_argument(
    "--show_presets",
    action="store_true",
    default=False,
    help=(
        "Show available preset selectors for each environment. "
        "Presets are grouped by selector type: physics (physics=NAME), "
        "renderer (renderer=NAME), and domain (presets=NAME)."
    ),
)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import isaaclab_tasks  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _format_presets(preset_map: dict | None) -> str:
    """Format a preset map returned by :func:`enumerate_task_presets` into a human-readable string.

    Args:
        preset_map: Mapping of :class:`~isaaclab_tasks.utils.preset_target.PresetTarget`
            to sorted preset name lists, or ``None`` when the env cfg could not be loaded.

    Returns:
        A multi-line string with one line per non-empty selector category, or a
        short placeholder when no presets are available or the cfg failed to load.
    """
    if preset_map is None:
        return "(unavailable)"
    from isaaclab_tasks.utils.preset_target import PresetTarget

    lines = []
    labels = {
        PresetTarget.PHYSICS: "physics",
        PresetTarget.RENDERER: "renderer",
        PresetTarget.DOMAIN: "domain",
    }
    for target, label in labels.items():
        names = preset_map.get(target, [])
        if names:
            lines.append(f"{label}: {', '.join(names)}")
    return "\n".join(lines) if lines else "(none)"


def main():
    """Print all environments registered in `isaaclab_tasks` extension."""
    # Collect matching task specs first so we can enumerate presets in one pass.
    task_specs = [
        spec
        for spec in gym.registry.values()
        if "Isaac" in spec.id and (args_cli.keyword is None or args_cli.keyword in spec.id)
    ]

    if args_cli.show_presets:
        from isaaclab_tasks.utils.preset_cli import enumerate_task_presets

        table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config", "Presets"])
        table.title = "Available Environments in Isaac Lab"
        table.align["Task Name"] = "l"
        table.align["Entry Point"] = "l"
        table.align["Config"] = "l"
        table.align["Presets"] = "l"

        for index, spec in enumerate(task_specs):
            preset_map = enumerate_task_presets(spec.id)
            table.add_row(
                [
                    index + 1,
                    spec.id,
                    spec.entry_point,
                    spec.kwargs["env_cfg_entry_point"],
                    _format_presets(preset_map),
                ]
            )
    else:
        table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
        table.title = "Available Environments in Isaac Lab"
        table.align["Task Name"] = "l"
        table.align["Entry Point"] = "l"
        table.align["Config"] = "l"

        for index, spec in enumerate(task_specs):
            table.add_row([index + 1, spec.id, spec.entry_point, spec.kwargs["env_cfg_entry_point"]])

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
