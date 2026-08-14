# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for miscellaneous Isaac Lab CLI commands."""

from unittest import mock

import pytest
import tomllib

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


def test_build_isaacsim_runs_incremental_build_for_existing_checkout(tmp_path):
    """The source-build workflow must update an existing Isaac Sim build incrementally."""
    isaacsim_root = tmp_path / "IsaacSim"
    build_script = isaacsim_root / "build.sh"
    build_script.parent.mkdir()
    build_script.touch()
    (isaacsim_root / "repo.sh").touch()
    wheel_dir = isaacsim_root / "_build" / "packages" / "dist"
    wheel_dir.mkdir(parents=True)
    (wheel_dir / "isaacsim-6.0.1+local-py3-none-any.whl").touch()

    workspace = tmp_path / "IsaacLab"
    workspace.mkdir()

    with (
        mock.patch.object(misc, "ISAACLAB_ROOT", workspace),
        mock.patch.object(misc, "run_command") as run_command,
        mock.patch.object(misc, "_set_uv_find_links"),
        mock.patch.object(misc, "_pin_isaacsim_local_extra"),
        mock.patch("shutil.which", return_value=None),
    ):
        misc.command_build_isaacsim(str(isaacsim_root))

    assert run_command.call_args_list == [
        mock.call([str(build_script)], cwd=isaacsim_root),
        mock.call([str(isaacsim_root / "repo.sh"), "python_package", "--create"], cwd=isaacsim_root),
        mock.call([str(isaacsim_root / "repo.sh"), "comment_archive_deps"], cwd=isaacsim_root),
        mock.call([str(isaacsim_root / "repo.sh"), "python_package", "--wheel"], cwd=isaacsim_root),
    ]


def test_build_isaacsim_preserves_multiline_find_links(tmp_path):
    """The source-build workflow must retain existing uv wheel indexes."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.uv]\nfind-links = [\n    "https://example.com/wheels",\n    "local-wheels",\n]\n',
        encoding="utf-8",
    )

    with mock.patch.object(misc, "ISAACLAB_ROOT", tmp_path):
        misc._set_uv_find_links("_isaac_sim_wheels")
        misc._set_uv_find_links("_isaac_sim_wheels")

    assert tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["uv"]["find-links"] == [
        "https://example.com/wheels",
        "local-wheels",
        "_isaac_sim_wheels",
    ]
