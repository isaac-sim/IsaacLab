# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for workflow commands exposed by an installed ``isaaclab`` package."""

from __future__ import annotations

import sys
from unittest import mock

import pytest

import isaaclab.cli as cli

pytestmark = pytest.mark.unit


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
    args = ["--task", "Example-v0"]
    with mock.patch(f"isaaclab_rl.entrypoints.{runner}", return_value=0) as run:
        command(args)

    run.assert_called_once_with(args)


def test_workflow_command_propagates_failure_status():
    """A nonzero in-process result must remain the console command's exit status."""
    with mock.patch("isaaclab_rl.entrypoints.run_train_cli", return_value=2), pytest.raises(SystemExit, match="2"):
        cli.train([])


@pytest.mark.parametrize(
    "command_name",
    ["train", "play", "train_multigpu", "zero_agent", "random_agent", "benchmark"],
)
def test_cli_loads_downstream_task_entrypoints_before_dispatch(command_name):
    """Installed project task packages must register before a task-aware workflow is dispatched."""
    task_entry_point = mock.Mock()
    with (
        mock.patch.object(cli.importlib.metadata, "entry_points", return_value=[task_entry_point]) as entry_points,
        mock.patch.object(cli, command_name) as command,
        mock.patch.object(sys, "argv", ["isaaclab", command_name, "--task", "Example-v0"]),
    ):
        cli.cli()

    entry_points.assert_called_once_with(group="isaaclab.tasks")
    task_entry_point.load.assert_called_once_with()
    command.assert_called_once_with(["--task", "Example-v0"])
