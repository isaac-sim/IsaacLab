# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the teleoperation and demonstration console entry points."""

import sys
from unittest import mock

import pytest

import isaaclab.cli as cli

# Each verb and the repository-relative script it forwards to.
TELEOP_COMMANDS = {
    "teleop": ("scripts", "environments", "teleoperation", "teleop_se3_agent.py"),
    "record": ("scripts", "tools", "record_demos.py"),
    "replay": ("scripts", "tools", "replay_demos.py"),
}


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_COMMANDS.items())
def test_teleop_command_dispatches_to_its_script(command, script_parts):
    """Each teleop verb forwards its arguments to the matching script, like ``train``/``play``."""
    args = ["--task", "IsaacContrib-PickPlace-Locomanipulation-G1-Abs", "--xr"]

    with (
        mock.patch.object(sys, "argv", ["isaaclab", command, *args]),
        mock.patch("isaaclab.cli.run_python_command") as run_python,
    ):
        cli.cli()

    run_python.assert_called_once_with(cli.ISAACLAB_ROOT.joinpath(*script_parts), args, check=True)


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_COMMANDS.items())
def test_teleop_command_script_exists(command, script_parts):
    """The dispatched scripts must exist so the documented commands are runnable."""
    assert cli.ISAACLAB_ROOT.joinpath(*script_parts).is_file()
