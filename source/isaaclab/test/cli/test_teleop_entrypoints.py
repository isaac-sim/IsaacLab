# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the unified teleoperation console entry point."""

import sys
from unittest import mock

import pytest

import isaaclab.cli as cli

# Each teleop workflow and the repository-relative script it forwards to.
TELEOP_WORKFLOWS = {
    "run": ("scripts", "environments", "teleoperation", "teleop_se3_agent.py"),
    "record": ("scripts", "tools", "record_demos.py"),
    "replay": ("scripts", "tools", "replay_demos.py"),
}


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_dispatches_to_requested_script(command, script_parts):
    """The ``isaaclab teleop`` command forwards arguments to the requested script."""
    args = [command, "--task", "IsaacContrib-PickPlace-Locomanipulation-G1-Abs", "--xr"]

    with (
        mock.patch.object(sys, "argv", ["isaaclab", "teleop", *args]),
        mock.patch("isaaclab.cli.run_python_command") as run_python,
    ):
        cli.cli()

    run_python.assert_called_once_with(cli.ISAACLAB_ROOT.joinpath(*script_parts), args[1:], check=True)


@pytest.mark.parametrize("script_parts", TELEOP_WORKFLOWS.values())
def test_teleop_workflow_script_exists(script_parts):
    """The dispatched scripts must exist so the documented commands are runnable."""
    assert cli.ISAACLAB_ROOT.joinpath(*script_parts).is_file()


def test_teleop_rejects_unknown_workflow():
    """An unknown workflow name fails at the sub-parser instead of launching a script."""
    with (
        mock.patch.object(sys, "argv", ["isaaclab", "teleop", "bogus"]),
        mock.patch("isaaclab.cli.run_python_command") as run_python,
        pytest.raises(SystemExit),
    ):
        cli.cli()

    run_python.assert_not_called()
