# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for miscellaneous Isaac Lab CLI commands."""

import sys
from unittest import mock

import pytest

import isaaclab.cli as cli
import isaaclab.cli.commands.misc as misc

pytestmark = pytest.mark.unit


def test_checkout_command_rejects_wheel_installation(tmp_path, monkeypatch, capsys):
    """Wheel users get checkout guidance before a development command launches or installs tools."""
    monkeypatch.setattr(cli, "ISAACLAB_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["isaaclab", "--format"])

    with mock.patch.object(cli, "command_format") as format_command, pytest.raises(SystemExit) as error:
        cli.cli()

    assert error.value.code == 2
    assert "requires an Isaac Lab source checkout" in capsys.readouterr().err
    format_command.assert_not_called()


def test_sim_command_propagates_failure(monkeypatch):
    """A failed simulator process must make the CLI fail with the same exit code."""
    monkeypatch.setattr(sys, "argv", ["isaaclab", "--sim"])
    monkeypatch.setattr(misc, "extract_isaacsim_exe", lambda: [sys.executable, "-c", "raise SystemExit(7)"])

    with pytest.raises(SystemExit) as error:
        cli.cli()

    assert error.value.code == 7


def test_python_subcommands_propagate_failures():
    """Python-based CLI subcommands must propagate failures from their child process."""
    with mock.patch.object(misc, "run_python_command") as run_python_command:
        misc.command_new(["--help"])
        misc.command_test(["-q"])
        misc.command_run_docker(["--help"])

    assert run_python_command.call_args_list == [
        mock.call(misc.ISAACLAB_ROOT / "tools" / "template" / "cli.py", ["--help"], check=True),
        mock.call("-m", ["pytest", str(misc.ISAACLAB_ROOT / "tools"), "-q"], check=True),
        mock.call(misc.ISAACLAB_ROOT / "docker" / "container.py", ["--help"], check=True),
    ]


def test_build_docs_runs_sphinx_with_the_uv_dev_extra(tmp_path, monkeypatch):
    """The docs command must use the UV extra that provides Sphinx."""
    monkeypatch.setattr(misc, "ISAACLAB_ROOT", tmp_path)
    docs_dir = tmp_path / "docs"
    output_dir = docs_dir / "_build" / "current"
    output_dir.mkdir(parents=True)
    (output_dir / "removed.html").touch()
    (output_dir / ".doctrees").mkdir()
    (output_dir / ".doctrees" / "environment.pickle").touch()
    other_version = docs_dir / "_build" / "v3.0.0-EA"
    other_version.mkdir()
    (other_version / "index.html").touch()

    def check_clean_output(*args, **kwargs):
        assert not output_dir.exists()
        assert (other_version / "index.html").is_file()

    with (
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch.object(misc, "run_command", side_effect=check_clean_output) as run_command,
    ):
        misc.command_build_docs()

    run_command.assert_called_once_with(
        [
            "/usr/bin/uv",
            "run",
            "--extra",
            "dev",
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
            str(output_dir / ".doctrees"),
            ".",
            str(output_dir),
        ],
        cwd=docs_dir,
    )


def test_build_docs_multi_redirects_to_selected_ref(tmp_path, monkeypatch):
    """The CLI must write its root redirect to the selected built version."""
    monkeypatch.setattr(misc, "ISAACLAB_ROOT", tmp_path)
    monkeypatch.setenv("DOCS_DEFAULT_REF", "develop")
    monkeypatch.setattr("sys.argv", ["isaaclab", "--docs_multi"])
    docs_dir = tmp_path / "docs"
    output_dir = docs_dir / "_build"
    (output_dir / "develop").mkdir(parents=True)
    (output_dir / "develop" / "index.html").write_text("built docs", encoding="utf-8")
    (docs_dir / "_redirect").mkdir()
    (docs_dir / "_redirect" / "index.html").write_text(
        '<meta http-equiv="refresh" content="0; url=./__DOCS_DEFAULT_REF__/index.html">', encoding="utf-8"
    )

    with (
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch.object(misc, "run_command") as run_command,
    ):
        cli.cli()

    assert (output_dir / "index.html").read_text(encoding="utf-8") == (
        '<meta http-equiv="refresh" content="0; url=./develop/index.html">'
    )
    assert (output_dir / "develop" / "index.html").read_text(encoding="utf-8") == "built docs"
    run_command.assert_called_once_with(
        ["/usr/bin/uv", "run", "--extra", "dev", "--", "sphinx-multiversion", ".", str(output_dir), "--jobs=auto"],
        cwd=docs_dir,
    )


def test_build_docs_multi_rejects_missing_default_ref(tmp_path, monkeypatch):
    """An unbuilt redirect target must fail with guidance and leave any redirect intact."""
    monkeypatch.setattr(misc, "ISAACLAB_ROOT", tmp_path)
    monkeypatch.delenv("DOCS_DEFAULT_REF", raising=False)
    output_dir = tmp_path / "docs" / "_build"
    output_dir.mkdir(parents=True)
    redirect = output_dir / "index.html"
    redirect.write_text("previous redirect", encoding="utf-8")

    with (
        mock.patch("shutil.which", return_value="/usr/bin/uv"),
        mock.patch.object(misc, "run_command"),
        mock.patch.object(misc, "print_error") as print_error,
        pytest.raises(SystemExit, match="1"),
    ):
        misc.command_build_docs(multi_version=True)

    print_error.assert_called_once_with(
        "Default docs ref 'v3.0.0-EA' was not built. Fetch the Git refs or set DOCS_DEFAULT_REF."
    )
    assert redirect.read_text(encoding="utf-8") == "previous redirect"


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
        mock.patch.object(misc, "repoint_prebundle_packages") as repoint_prebundles,
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
        # linux-x86_64 is covered end to end by test_build_isaacsim_links_incremental_build_without_packaging.
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
