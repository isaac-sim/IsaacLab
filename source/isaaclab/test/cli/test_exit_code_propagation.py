# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that CLI subcommands propagate the exit code of the commands they run."""

import sys

import pytest

from isaaclab.cli import utils
from isaaclab.cli.commands import misc


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


def test_run_python_command_forwards_check(monkeypatch):
    """check=True must survive the delegation from run_python_command to run_command."""
    seen = {}

    def fake_run_command(cmd, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(utils, "extract_python_exe", lambda: sys.executable)
    monkeypatch.setattr(utils, "run_command", fake_run_command)
    utils.run_python_command("script.py", [], check=True)
    assert seen.get("check") is True


# command_vscode_settings is deliberately absent: settings generation is
# best-effort (it also runs inside the install flow) and must not abort the
# invoking command on failure.
@pytest.mark.parametrize(
    "invoke",
    [
        pytest.param(lambda: misc.command_new([]), id="new"),
        pytest.param(lambda: misc.command_test([]), id="test"),
        pytest.param(lambda: misc.command_run_docker([]), id="docker"),
    ],
)
def test_misc_commands_request_checked_execution(monkeypatch, invoke):
    """Regression guard for -t/--new/--docker reporting success while their child failed."""
    calls = []

    def record(script_or_module, args, **kwargs):
        calls.append((str(script_or_module), kwargs.get("check", False)))

    monkeypatch.setattr(misc, "run_python_command", record)
    invoke()
    assert calls, "command did not launch a python child"
    unchecked = [script for script, checked in calls if not checked]
    assert not unchecked, f"child commands launched without check=True: {unchecked}"
