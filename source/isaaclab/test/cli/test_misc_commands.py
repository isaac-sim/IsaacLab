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


def test_build_isaacsim_links_incremental_build_without_packaging(tmp_path):
    """The source workflow must link the live build without creating Python wheels."""
    isaacsim_root = tmp_path / "IsaacSim"
    build_script = isaacsim_root / "build.sh"
    build_script.parent.mkdir()
    build_script.touch()
    release_dir = isaacsim_root / "_build" / "linux-x86_64" / "release"
    release_dir.mkdir(parents=True)
    (release_dir / "python.sh").touch()

    workspace = tmp_path / "IsaacLab"
    workspace.mkdir()

    with (
        mock.patch.object(misc, "ISAACLAB_ROOT", workspace),
        mock.patch.object(misc, "run_command") as run_command,
        mock.patch.object(misc, "_repoint_source_build_prebundles") as repoint_prebundles,
        mock.patch.object(misc.sys, "platform", "linux"),
        mock.patch.object(misc.platform, "machine", return_value="x86_64"),
    ):
        misc.command_build_isaacsim(str(isaacsim_root))

    run_command.assert_called_once_with([str(build_script)], cwd=isaacsim_root)
    repoint_prebundles.assert_called_once_with()
    assert (workspace / "_isaac_sim").resolve() == release_dir
    assert (release_dir / ".isaaclab_source_build").is_file()


@pytest.mark.parametrize(
    ("sys_platform", "machine", "target"),
    [
        ("linux", "x86_64", "linux-x86_64"),
        ("linux", "aarch64", "linux-aarch64"),
        ("win32", "AMD64", "windows-x86_64"),
    ],
)
def test_build_isaacsim_resolves_release_directory(tmp_path, sys_platform, machine, target):
    """The source workflow must select the current platform's live release tree."""
    with (
        mock.patch.object(misc.sys, "platform", sys_platform),
        mock.patch.object(misc.platform, "machine", return_value=machine),
    ):
        result = misc._resolve_isaacsim_release_dir(tmp_path)

    assert result == tmp_path / "_build" / target / "release"


def test_build_isaacsim_rejects_unsupported_platform(tmp_path):
    """The source workflow must reject platforms Isaac Sim cannot build."""
    with (
        mock.patch.object(misc.sys, "platform", "darwin"),
        mock.patch.object(misc.platform, "machine", return_value="arm64"),
        pytest.raises(SystemExit, match="1"),
    ):
        misc._resolve_isaacsim_release_dir(tmp_path)
