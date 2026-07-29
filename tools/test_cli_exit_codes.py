# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that CLI subcommands propagate the exit code of the commands they run."""

import os
import sys

import pytest

# Make the CLI importable straight from the source tree so this test also runs
# outside an environment where the isaaclab package is installed.
ISAACLAB_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ISAACLAB_PATH, "source", "isaaclab"))

from isaaclab.cli import utils  # noqa: E402
from isaaclab.cli.commands import misc  # noqa: E402


def test_run_command_check_propagates_exit_code():
    """A failing child with check=True exits the process with the child's code."""
    fail = [sys.executable, "-c", "import sys; sys.exit(3)"]
    with pytest.raises(SystemExit) as excinfo:
        utils.run_command(fail, check=True, capture_output=True)
    assert excinfo.value.code == 3


def test_run_command_no_check_swallows_exit_code():
    """check=False is the opt-out: the caller sees the code but the process continues."""
    fail = [sys.executable, "-c", "import sys; sys.exit(3)"]
    result = utils.run_command(fail, check=False, capture_output=True)
    assert result.returncode == 3


def test_misc_commands_request_checked_execution(monkeypatch):
    """Every python child launched by the misc subcommands must pass check=True.

    This is the regression guard for the -t/--new/--vscode/--docker paths
    reporting success while their child failed.
    """
    calls = []

    def record(script_or_module, args, **kwargs):
        calls.append((str(script_or_module), kwargs.get("check", False)))

    monkeypatch.setattr(misc, "run_python_command", record)

    misc.command_new([])
    misc.command_test([])
    misc.command_vscode_settings()
    misc.command_run_docker([])

    assert len(calls) == 5  # command_new launches pip install + the generator
    unchecked = [script for script, checked in calls if not checked]
    assert not unchecked, f"child commands launched without check=True: {unchecked}"
