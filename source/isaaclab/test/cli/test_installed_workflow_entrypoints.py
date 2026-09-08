# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for workflow commands exposed by an installed ``isaaclab`` package."""

from __future__ import annotations

import subprocess
import sys
from unittest import mock

import pytest

import isaaclab.cli as cli

pytestmark = pytest.mark.unit


def test_cli_import_does_not_require_runtime_dependencies():
    """The installation CLI must load before core runtime dependencies are installed."""
    result = subprocess.run(
        [sys.executable, "-c", 'import sys; sys.modules["lazy_loader"] = None; import isaaclab.cli'],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_editor_option_uses_cli_dispatcher():
    """The installed CLI must forward editor-specific arguments to the editor command."""
    with (
        mock.patch.object(sys, "argv", ["isaaclab", "--editor", "--isaac_path", "/sim", "--verbose"]),
        mock.patch.object(cli, "command_editor") as editor,
    ):
        cli.cli()

    editor.assert_called_once_with(["--isaac_path", "/sim", "--verbose"])


@pytest.mark.parametrize("option", ["--vscode", "--generate-vscode-settings"])
def test_removed_editor_options_are_rejected(option):
    """Removed editor setup options must not remain as hidden compatibility paths."""
    with mock.patch.object(sys, "argv", ["isaaclab", option]), pytest.raises(SystemExit, match="2"):
        cli.cli()


@pytest.mark.parametrize(
    ("command", "runner"),
    [
        (cli.train, "run_train_cli"),
        (cli.play, "run_play_cli"),
        (cli.train_multigpu, "run_train_multigpu_cli"),
        (cli.zero_agent, "run_zero_agent_cli"),
        (cli.random_agent, "run_random_agent_cli"),
    ],
)
def test_workflow_commands_dispatch_to_installed_entrypoints(command, runner):
    """Workflow commands must not depend on scripts from a source checkout."""
    args = ["--task", "Example"]
    with mock.patch(f"isaaclab_rl.entrypoints.{runner}", return_value=0) as run:
        command(args)

    run.assert_called_once_with(args)


def test_workflow_command_propagates_failure_status():
    """A nonzero in-process result must remain the console command's exit status."""
    with mock.patch("isaaclab_rl.entrypoints.run_train_cli", return_value=2), pytest.raises(SystemExit, match="2"):
        cli.train([])


def test_cli_loads_downstream_tasks_before_benchmark():
    """Benchmarking must discover tasks from installed projects."""
    task_entry_point = mock.Mock()
    with (
        mock.patch.object(cli.importlib.metadata, "entry_points", return_value=[task_entry_point]) as entry_points,
        mock.patch.object(cli, "benchmark") as benchmark,
        mock.patch.object(sys, "argv", ["isaaclab", "benchmark", "runtime", "--task", "Example"]),
    ):
        cli.cli()

    entry_points.assert_called_once_with(group="isaaclab.tasks")
    task_entry_point.load.assert_called_once_with()
    benchmark.assert_called_once_with(["runtime", "--task", "Example"])
