# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for miscellaneous Isaac Lab CLI commands."""

from unittest import mock

import pytest

import isaaclab.cli.commands.misc as misc

pytestmark = pytest.mark.unit


def test_build_docs_runs_sphinx_with_the_uv_test_extra():
    """The docs command must build through UV instead of an unpinned pip install."""
    docs_dir = misc.ISAACLAB_ROOT / "docs"
    output_dir = docs_dir / "_build" / "current"

    with (
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch.object(misc, "run_command") as run_command,
    ):
        misc.command_build_docs()

    run_command.assert_called_once_with(
        [
            "/usr/bin/uv",
            "run",
            "--isolated",
            "--extra",
            "test",
            "--",
            "python",
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-j",
            "auto",
            "-b",
            "html",
            "-d",
            "_build/doctrees",
            ".",
            str(output_dir),
        ],
        cwd=docs_dir,
    )


def test_build_docs_explains_how_to_install_uv():
    """The docs command must fail with actionable guidance when UV is unavailable."""
    with (
        mock.patch("shutil.which", return_value=None),
        mock.patch.object(misc, "print_error") as print_error,
        pytest.raises(SystemExit, match="1"),
    ):
        misc.command_build_docs()

    assert print_error.call_args_list == [
        mock.call("uv could not be found. Please install uv and try again."),
        mock.call("https://docs.astral.sh/uv/getting-started/installation/"),
    ]
