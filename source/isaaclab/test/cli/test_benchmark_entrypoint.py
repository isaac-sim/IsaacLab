# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the unified benchmark console entry point."""

import sys
from unittest import mock

import pytest

import isaaclab.cli as cli


@pytest.mark.parametrize("command", ["runtime", "startup", "training", "play"])
def test_benchmark_dispatches_to_requested_script(command):
    """The ``isaaclab benchmark`` command forwards arguments to the requested script."""
    args = [command, "--help"]

    with (
        mock.patch.object(sys, "argv", ["isaaclab", "benchmark", *args]),
        mock.patch("isaaclab.cli.run_python_command") as run_python,
    ):
        cli.cli()

    run_python.assert_called_once_with(
        cli.ISAACLAB_ROOT / "scripts" / "benchmarks" / f"{command}.py",
        args[1:],
        check=True,
    )
