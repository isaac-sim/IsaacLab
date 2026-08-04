# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the unified teleoperation console entry point."""

import re
import sys
from pathlib import Path
from unittest import mock

import pytest
import tomllib

import isaaclab.cli as cli

pytestmark = pytest.mark.unit

# Each teleop workflow and the repository-relative script it forwards to.
TELEOP_WORKFLOWS = {
    "run": ("scripts", "environments", "teleoperation", "teleop_se3_agent.py"),
    "record": ("scripts", "tools", "record_demos.py"),
    "replay": ("scripts", "tools", "replay_demos.py"),
}


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_dispatches_to_an_existing_script(command, script_parts):
    """``isaaclab teleop`` forwards the remaining arguments to a script that exists."""
    args = [command, "--task", "IsaacContrib-PickPlace-Locomanipulation-G1-Abs", "--xr"]

    with (
        mock.patch.object(sys, "argv", ["isaaclab", "teleop", *args]),
        mock.patch("isaaclab.cli.run_python_command") as run_python,
    ):
        cli.cli()

    script = cli.ISAACLAB_ROOT.joinpath(*script_parts)
    run_python.assert_called_once_with(script, args[1:], check=True)
    assert script.is_file()


def _requirement_names(requirements: list[str]) -> set[str]:
    """Return the importable module names for a list of requirement strings."""
    names = set()
    for requirement in requirements:
        name = re.split(r"[\s<>=!~\[;@]", requirement, maxsplit=1)[0]
        if name.startswith("isaaclab"):
            names.add(name.replace("-", "_"))
    return names


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_workflow_isaaclab_imports_are_covered_by_the_extras(command, script_parts):
    """Every ``isaaclab_*`` package a teleop script imports must ship with the extras.

    ``record_demos.py`` imports ``isaaclab_mimic`` at module level, so an environment built
    from ``--extra teleop`` alone used to die with ``ModuleNotFoundError`` only after Isaac Sim
    had finished booting. This catches that class of gap without needing an install.
    """
    with (cli.ISAACLAB_ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    project = pyproject["project"]

    available = _requirement_names(project["dependencies"])
    available |= _requirement_names(project["optional-dependencies"]["teleop"])

    source = Path(cli.ISAACLAB_ROOT).joinpath(*script_parts).read_text(encoding="utf-8")
    imported = set(re.findall(r"^\s*(?:import|from)\s+(isaaclab_\w+)", source, re.MULTILINE))

    missing = imported - available
    assert not missing, f"'isaaclab teleop {command}' imports {sorted(missing)}, absent from the teleop extra"


def test_teleop_rejects_unknown_workflow():
    """An unknown workflow name fails at the sub-parser instead of launching a script."""
    with (
        mock.patch.object(sys, "argv", ["isaaclab", "teleop", "bogus"]),
        mock.patch("isaaclab.cli.run_python_command") as run_python,
        pytest.raises(SystemExit),
    ):
        cli.cli()

    run_python.assert_not_called()
