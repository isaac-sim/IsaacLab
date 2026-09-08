# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the installed LEAPP commands."""

import sys
from unittest import mock

import pytest

import isaaclab.cli as cli

pytestmark = pytest.mark.unit


def test_export_dispatches_in_process():
    """``isaaclab leapp export`` forwards arguments to the library dispatcher."""
    args = ["--rl_library", "rsl_rl", "--task", "Isaac-Cartpole"]

    with (
        mock.patch.object(sys, "argv", ["isaaclab", "leapp", "export", *args]),
        mock.patch("isaaclab_rl.entrypoints.run_export_cli", return_value=0) as run_export,
    ):
        cli.cli()

    run_export.assert_called_once_with(args)


def test_export_propagates_nonzero_dispatch_status():
    """Export failures become the CLI process status."""
    with mock.patch("isaaclab_rl.entrypoints.run_export_cli", return_value=1):
        with pytest.raises(SystemExit) as exc_info:
            cli.leapp(["export", "--rl_library", "rsl_rl"])

    assert exc_info.value.code == 1


def test_deploy_dispatches_in_process():
    """``isaaclab leapp deploy`` forwards LEAPP deployment arguments."""
    args = [
        "--task",
        "Isaac-Cartpole",
        "--pipeline",
        "exported/Isaac-Cartpole.yaml",
        "physics=newton_mjwarp",
    ]

    with (
        mock.patch.object(sys, "argv", ["isaaclab", "leapp", "deploy", *args]),
        mock.patch("isaaclab.cli.command_deploy_leapp", return_value=0) as deploy,
    ):
        cli.cli()

    deploy.assert_called_once_with(args)


def test_deploy_propagates_nonzero_status():
    """Deployment failures become the CLI process status."""
    with mock.patch("isaaclab.cli.command_deploy_leapp", return_value=3):
        with pytest.raises(SystemExit) as exc_info:
            cli.leapp(["deploy", "--task", "Isaac-Cartpole", "--pipeline", "policy.yaml"])

    assert exc_info.value.code == 3
