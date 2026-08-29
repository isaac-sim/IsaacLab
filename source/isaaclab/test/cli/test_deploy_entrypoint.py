# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the installed deployment command."""

import sys
from unittest import mock

import pytest

import isaaclab.cli as cli

pytestmark = pytest.mark.unit


def test_deploy_leapp_dispatches_in_process():
    """``isaaclab deploy_leapp`` forwards LEAPP deployment arguments."""
    args = [
        "--task",
        "Isaac-Cartpole",
        "--pipeline",
        "exported/Isaac-Cartpole.yaml",
        "physics=newton_mjwarp",
    ]

    with (
        mock.patch.object(sys, "argv", ["isaaclab", "deploy_leapp", *args]),
        mock.patch("isaaclab.cli.command_deploy_leapp", return_value=0) as deploy_leapp,
    ):
        cli.cli()

    deploy_leapp.assert_called_once_with(args)


def test_deploy_leapp_propagates_nonzero_status():
    """Deployment failures become the CLI process status."""
    with mock.patch("isaaclab.cli.command_deploy_leapp", return_value=3):
        with pytest.raises(SystemExit) as exc_info:
            cli.deploy_leapp(["--task", "Isaac-Cartpole", "--pipeline", "policy.yaml"])

    assert exc_info.value.code == 3
