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
def test_benchmark_dispatches_in_process(command):
    """Test that ``isaaclab benchmark`` forwards arguments to the library dispatcher."""
    args = [command, "--help"]

    with (
        mock.patch.object(sys, "argv", ["isaaclab", "benchmark", *args]),
        mock.patch("isaaclab.benchmark.run_benchmark_cli", return_value=0) as run_benchmark,
    ):
        cli.cli()

    run_benchmark.assert_called_once_with(args)


def test_benchmark_propagates_nonzero_dispatch_status():
    """Test that a benchmark dispatcher failure becomes the CLI exit status."""
    with mock.patch("isaaclab.benchmark.run_benchmark_cli", return_value=2):
        with pytest.raises(SystemExit) as exc_info:
            cli.benchmark(["training"])

    assert exc_info.value.code == 2
