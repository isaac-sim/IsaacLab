# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the interpreter, Isaac Sim, and subprocess helpers shared by the CLI commands."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from isaaclab.cli import utils

pytestmark = pytest.mark.unit


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


@pytest.fixture
def local_sim(monkeypatch, tmp_path):
    """A downloaded Isaac Sim package linked at ``_isaac_sim``, with the launcher's Python resolved to a venv."""
    sim = tmp_path / "_isaac_sim"
    _touch(sim / "python.sh")
    monkeypatch.setattr(utils, "DEFAULT_ISAAC_SIM_PATH", sim)
    monkeypatch.setattr(utils, "extract_python_exe", lambda: str(tmp_path / ".venv" / "bin" / "python"))
    return sim


@pytest.fixture
def run_command(monkeypatch):
    """Capture the command ``run_python_command`` hands to ``run_command``."""
    calls = []
    monkeypatch.setattr(utils, "run_command", lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    return calls


def test_run_command_retries_a_failed_process(monkeypatch):
    """A command-level retry reruns a failed package-manager process after the configured delay."""
    failure = subprocess.CalledProcessError(returncode=1, cmd=["pip"])
    success = subprocess.CompletedProcess(args=["pip"], returncode=0)
    attempts = iter([failure, success])
    sleeps = []

    def _run(*_args, **_kwargs):
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(utils.subprocess, "run", _run)
    monkeypatch.setattr(utils.time, "sleep", sleeps.append)

    assert utils.run_command(["pip"], retry_attempts=3, retry_delay_seconds=3.0) is success
    assert sleeps == [3.0]


def test_run_python_command_wraps_live_source_build(local_sim, run_command, monkeypatch, tmp_path):
    """Direct uv launches combine the live source build with the active Python through ``python.sh``."""
    (local_sim / ".isaaclab_source_build").touch()
    monkeypatch.setattr(os, "environ", {})

    utils.run_python_command("train.py", ["--task", "Cartpole"])

    ((command, kwargs),) = run_command
    assert command[0] == str(local_sim / "python.sh")
    assert Path(command[1]).name == "train.py" and command[2:] == ["--task", "Cartpole"]
    assert kwargs["env"]["PYTHONEXE"] == str(tmp_path / ".venv" / "bin" / "python")


def test_run_python_command_skips_wrapping_when_isaac_sim_env_is_active(local_sim, run_command, monkeypatch):
    """The legacy wrapper path must not source the same Isaac Sim environment twice."""
    (local_sim / ".isaaclab_source_build").touch()
    monkeypatch.setattr(os, "environ", {"ISAAC_PATH": str(local_sim)})

    utils.run_python_command("script.py", [])

    assert run_command[0][0] == [utils.extract_python_exe(), "script.py"]


@pytest.mark.parametrize("venv_home", [None, "/usr/bin"], ids=["no-venv-metadata", "foreign-python"])
def test_run_python_command_rejects_downloaded_isaac_sim_with_virtual_environment(
    local_sim, run_command, monkeypatch, tmp_path, venv_home
):
    """Downloaded Isaac Sim packages must not run through a virtual environment built on another interpreter."""
    venv = tmp_path / ".venv"
    venv.mkdir()
    if venv_home is not None:
        (venv / "pyvenv.cfg").write_text(f"home = {venv_home}\n")
    monkeypatch.setattr(os, "environ", {"VIRTUAL_ENV": str(venv)})

    with pytest.raises(SystemExit, match="1"):
        utils.run_python_command("train.py", [])
    assert run_command == []


def test_run_python_command_accepts_virtual_environment_on_bundled_python(
    local_sim, run_command, monkeypatch, tmp_path
):
    """A virtual environment created on the package's own Python runs that exact interpreter."""
    bundled_python = _touch(local_sim / "kit" / "python" / "bin" / "python3")
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(bundled_python)
    (venv / "pyvenv.cfg").write_text(f"home = {bundled_python.parent}\n")
    monkeypatch.setattr(os, "environ", {"VIRTUAL_ENV": str(venv)})

    utils.run_python_command("script.py", [])

    assert run_command[0][0][0] == str(local_sim / "python.sh")


@pytest.mark.parametrize(
    ("environ", "uv_available", "expected"),
    [
        ({"VIRTUAL_ENV": "/venv"}, True, ["uv", "pip"]),
        ({"VIRTUAL_ENV": "/venv"}, False, ["/venv/bin/python", "-m", "pip"]),
        ({"CONDA_PREFIX": "/conda"}, False, ["/venv/bin/python", "-m", "pip"]),
    ],
    ids=["uv-venv", "venv-without-uv", "conda-without-uv"],
)
def test_get_pip_command(monkeypatch, environ, uv_available, expected):
    """``uv pip`` is used inside a virtual environment when uv is installed; otherwise the interpreter's pip."""
    monkeypatch.setattr(os, "environ", dict(environ))
    monkeypatch.setattr(utils.shutil, "which", lambda _name: "/usr/bin/uv" if uv_available else None)

    assert utils.get_pip_command(python_exe="/venv/bin/python") == expected


@pytest.mark.parametrize(
    ("variable", "relative_python"),
    [("VIRTUAL_ENV", ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python"))]
    + [("CONDA_PREFIX", ("python.exe",) if sys.platform == "win32" else ("bin", "python"))],
    ids=["venv", "conda"],
)
def test_extract_python_exe_prefers_active_environment(monkeypatch, tmp_path, variable, relative_python):
    """The active virtual or conda environment's interpreter wins over every other candidate."""
    python = _touch(tmp_path.joinpath(*relative_python))
    monkeypatch.setattr(os, "environ", {variable: str(tmp_path)})

    assert Path(utils.extract_python_exe()) == python


@pytest.mark.parametrize("required", [True, False])
def test_extract_isaacsim_path_without_isaac_sim(monkeypatch, required):
    """A missing Isaac Sim exits the process when required and returns ``None`` otherwise."""
    monkeypatch.setattr(utils, "DEFAULT_ISAAC_SIM_PATH", Path("/nonexistent/_isaac_sim"))
    monkeypatch.setattr(utils.subprocess, "run", lambda *_, **__: subprocess.CompletedProcess(args=[], returncode=1))

    if required:
        with pytest.raises(SystemExit):
            utils.extract_isaacsim_path(required=True)
    else:
        assert utils.extract_isaacsim_path(required=False) is None


def test_extract_isaacsim_path_uses_the_symlink(monkeypatch, tmp_path):
    """The repo-local ``_isaac_sim`` directory is used when it exists."""
    monkeypatch.setattr(utils, "DEFAULT_ISAAC_SIM_PATH", tmp_path)
    assert utils.extract_isaacsim_path() == tmp_path


@pytest.mark.parametrize(
    ("version_file", "metadata_version", "expected"),
    [
        ("5.0.0", None, "3.11"),
        ("6.0.0", None, "3.12"),
        (None, "5.1.0", "3.11"),
        (None, None, "3.12"),
        ("99.0.0", None, RuntimeError),
    ],
    ids=["sim-5", "sim-6", "metadata", "no-sim", "unknown"],
)
def test_determine_python_version(monkeypatch, tmp_path, version_file, metadata_version, expected):
    """The Python version follows the Isaac Sim major version, defaulting to 3.12 without Isaac Sim."""
    if version_file is not None:
        (tmp_path / "VERSION").write_text(version_file)
    monkeypatch.setattr(utils, "extract_isaacsim_path", lambda **_: tmp_path)

    def _version(_name):
        if metadata_version is None:
            raise Exception("not found")
        return metadata_version

    monkeypatch.setattr("importlib.metadata.version", _version)

    if expected is RuntimeError:
        with pytest.raises(RuntimeError, match="Unsupported Isaac Sim version"):
            utils.determine_python_version()
    else:
        assert utils.determine_python_version() == expected
