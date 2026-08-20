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
        mock.patch.object(misc, "_set_uv_environment"),
        mock.patch.object(misc, "_pin_isaacsim_local_extra"),
        mock.patch.object(misc, "_add_isaacsim_local_conflicts"),
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


def test_build_isaacsim_creates_and_updates_local_extra(tmp_path):
    """The source-build workflow must generate its local-only optional dependency."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project.optional-dependencies]\nisaacsim = ["isaacsim[all,extscache]==6.0.1.0"]\n\n[tool.uv]\n',
        encoding="utf-8",
    )

    with mock.patch.object(misc, "ISAACLAB_ROOT", tmp_path):
        misc._pin_isaacsim_local_extra("6.0.1rc7+develop.0.local")
        misc._pin_isaacsim_local_extra("6.0.1rc8+develop.1.local")

    extras = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert extras == {
        "isaacsim": ["isaacsim[all,extscache]==6.0.1.0"],
        "isaacsim-local": ["isaacsim[all,extscache]==6.0.1rc8+develop.1.local"],
    }


@pytest.mark.parametrize(
    ("sys_platform", "machine", "expected"),
    [
        ("linux", "x86_64", "sys_platform == 'linux' and platform_machine == 'x86_64'"),
        ("linux", "aarch64", "sys_platform == 'linux' and platform_machine == 'aarch64'"),
        ("win32", "AMD64", "sys_platform == 'win32' and platform_machine == 'AMD64'"),
    ],
)
def test_build_isaacsim_limits_uv_resolution_to_current_platform(tmp_path, sys_platform, machine, expected):
    """The local lock must only resolve for the platform that produced the wheels."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """[tool.uv]
# Platforms supported by the committed lock.
environments = [
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
    "sys_platform == 'linux' and platform_machine == 'aarch64'",
    "sys_platform == 'win32' and platform_machine == 'AMD64'",
]

[tool.other]
environments = ["preserve-me"]
""",
        encoding="utf-8",
    )

    with (
        mock.patch.object(misc, "ISAACLAB_ROOT", tmp_path),
        mock.patch.object(misc.sys, "platform", sys_platform),
        mock.patch.object(misc.platform, "machine", return_value=machine),
    ):
        misc._set_uv_environment()
        misc._set_uv_environment()

    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert config["tool"]["uv"]["environments"] == [expected]
    assert config["tool"]["other"]["environments"] == ["preserve-me"]
    assert pyproject.read_text(encoding="utf-8").count("Local-only, do not commit.") == 1


def test_build_isaacsim_rejects_unsupported_uv_platform(tmp_path):
    """The source-build workflow must reject platforms without supported Isaac Sim wheels."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.uv]\n", encoding="utf-8")

    with (
        mock.patch.object(misc, "ISAACLAB_ROOT", tmp_path),
        mock.patch.object(misc.sys, "platform", "darwin"),
        mock.patch.object(misc.platform, "machine", return_value="arm64"),
        pytest.raises(SystemExit, match="1"),
    ):
        misc._set_uv_environment()

    assert pyproject.read_text(encoding="utf-8") == "[tool.uv]\n"


def test_build_isaacsim_adds_local_conflicts_without_duplicates(tmp_path):
    """A local Isaac Sim pin must split from extras using the published wheel."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.uv]\nconflicts = [\n    [{ extra = "other" }, { extra = "extra" }],\n]\n',
        encoding="utf-8",
    )

    with mock.patch.object(misc, "ISAACLAB_ROOT", tmp_path):
        misc._add_isaacsim_local_conflicts()
        misc._add_isaacsim_local_conflicts()

    assert tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["uv"]["conflicts"] == [
        [{"extra": "other"}, {"extra": "extra"}],
        [{"extra": "isaacsim-local"}, {"extra": "isaacsim"}],
        [{"extra": "isaacsim-local"}, {"extra": "teleop"}],
        [{"extra": "isaacsim-local"}, {"extra": "all"}],
    ]
