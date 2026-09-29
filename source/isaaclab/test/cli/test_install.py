# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for CLI utility functions used by the uv installation path."""

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from isaaclab.cli.utils import (
    extract_isaacsim_path,
    run_command,
    run_python_command,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------


def test_run_command_retries_a_failed_process():
    """A command-level retry reruns a failed package-manager process."""
    failure = subprocess.CalledProcessError(returncode=1, cmd=["pip", "install", "example"])
    success = subprocess.CompletedProcess(args=["pip", "install", "example"], returncode=0)

    with (
        mock.patch("isaaclab.cli.utils.subprocess.run", side_effect=[failure, success]) as subprocess_run,
        mock.patch("isaaclab.cli.utils.time.sleep") as sleep,
    ):
        result = run_command(
            ["pip", "install", "example"],
            retry_attempts=3,
            retry_delay_seconds=3.0,
        )

    assert result is success
    assert subprocess_run.call_count == 2
    sleep.assert_called_once_with(3.0)


def test_run_python_command_uses_live_isaac_sim_with_active_python(tmp_path):
    """Direct uv launches must combine the live source build with the active Python."""
    local_sim = tmp_path / "_isaac_sim"
    local_sim.mkdir()
    python_launcher = local_sim / "python.sh"
    python_launcher.touch()
    (local_sim / ".isaaclab_source_build").touch()
    active_python = str(tmp_path / ".venv" / "bin" / "python")

    with (
        mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", local_sim),
        mock.patch("isaaclab.cli.utils.extract_python_exe", return_value=active_python),
        mock.patch("isaaclab.cli.utils.run_command") as run,
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        run_python_command("train.py", ["--task", "Cartpole"])

    command = run.call_args.args[0]
    assert command[0] == str(python_launcher)
    assert Path(command[1]).name == "train.py"
    assert command[2:] == ["--task", "Cartpole"]
    assert run.call_args.kwargs["env"]["PYTHONEXE"] == active_python


def test_run_python_command_accepts_virtual_environment_on_bundled_python(tmp_path):
    """A virtual environment created on a downloaded package's Python runs that interpreter."""
    local_sim = tmp_path / "_isaac_sim"
    local_sim.mkdir()
    (local_sim / "python.sh").touch()
    bundled_python = local_sim / "kit" / "python" / "bin" / "python3"
    bundled_python.parent.mkdir(parents=True)
    bundled_python.touch()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(bundled_python)
    (venv / "pyvenv.cfg").write_text(f"home = {bundled_python.parent}\n")

    with (
        mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", local_sim),
        mock.patch("isaaclab.cli.utils.extract_python_exe", return_value=str(venv / "bin" / "python")),
        mock.patch("isaaclab.cli.utils.run_command") as run,
        mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(venv)}, clear=True),
    ):
        run_python_command("script.py", [])

    assert run.call_args is not None


def test_run_python_command_rejects_virtual_environment_on_foreign_python(tmp_path):
    """A virtual environment built on another interpreter stays rejected."""
    local_sim = tmp_path / "_isaac_sim"
    local_sim.mkdir()
    (local_sim / "python.sh").touch()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")

    with (
        mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", local_sim),
        mock.patch("isaaclab.cli.utils.extract_python_exe", return_value=str(venv / "bin" / "python")),
        mock.patch("isaaclab.cli.utils.run_command"),
        mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(venv)}, clear=True),
        pytest.raises(SystemExit),
    ):
        run_python_command("script.py", [])


def test_run_python_command_does_not_wrap_an_active_isaac_sim_environment(tmp_path):
    """The legacy wrapper path must not source the same Isaac Sim environment twice."""
    local_sim = tmp_path / "_isaac_sim"
    local_sim.mkdir()
    (local_sim / "python.sh").touch()
    (local_sim / ".isaaclab_source_build").touch()
    active_python = str(tmp_path / ".venv" / "bin" / "python")

    with (
        mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", local_sim),
        mock.patch("isaaclab.cli.utils.extract_python_exe", return_value=active_python),
        mock.patch("isaaclab.cli.utils.run_command") as run,
        mock.patch.dict(os.environ, {"ISAAC_PATH": str(local_sim)}, clear=True),
    ):
        run_python_command("script.py", [])

    assert run.call_args.args[0] == [active_python, "script.py"]


def test_run_python_command_rejects_downloaded_isaac_sim_with_virtual_environment(tmp_path):
    """Downloaded Isaac Sim packages must not run through a virtual environment."""
    local_sim = tmp_path / "_isaac_sim"
    local_sim.mkdir()
    (local_sim / "python.sh").touch()
    active_python = str(tmp_path / ".venv" / "bin" / "python")

    with (
        mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", local_sim),
        mock.patch("isaaclab.cli.utils.extract_python_exe", return_value=active_python),
        mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(tmp_path / ".venv")}, clear=True),
        pytest.raises(SystemExit, match="1"),
    ):
        run_python_command("train.py", ["--task", "Cartpole"])


# ---------------------------------------------------------------------------
# extract_isaacsim_path
# ---------------------------------------------------------------------------


class TestExtractIsaacsimPath:
    """Tests for :func:`extract_isaacsim_path`."""

    def test_returns_none_when_not_required(self):
        """When required=False and Isaac Sim is not found, return None."""
        with (
            mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", Path("/nonexistent/path")),
            mock.patch(
                "isaaclab.cli.utils.subprocess.run",
                return_value=subprocess.CompletedProcess(args=[], returncode=1),
            ),
        ):
            result = extract_isaacsim_path(required=False)
            assert result is None

    def test_exits_when_required(self):
        """When required=True and Isaac Sim is not found, sys.exit."""
        with (
            mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", Path("/nonexistent/path")),
            mock.patch(
                "isaaclab.cli.utils.subprocess.run",
                return_value=subprocess.CompletedProcess(args=[], returncode=1),
            ),
            pytest.raises(SystemExit),
        ):
            extract_isaacsim_path(required=True)

    def test_returns_path_when_symlink_exists(self, tmp_path):
        """When the default path exists, return it."""
        fake_sim = tmp_path / "_isaac_sim"
        fake_sim.mkdir()

        with mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", fake_sim):
            result = extract_isaacsim_path(required=True)
            assert result == fake_sim


@pytest.mark.parametrize("option", ["--install", "--conda", "--uv"])
def test_removed_installer_options_fail(option: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "isaaclab", option],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


@pytest.mark.parametrize(
    ("installation", "configured", "delegated"),
    [("source", False, True), ("source", True, False), ("container", False, True), ("stale", False, False)],
)
def test_runtime_dispatch_loads_local_kit_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installation: str, configured: bool, delegated: bool
) -> None:
    import isaaclab.cli as cli

    local_sim = tmp_path / "_isaac_sim"
    local_sim.mkdir()
    (local_sim / ("python.bat" if os.name == "nt" else "python.sh")).touch()
    if installation == "source":
        (local_sim / ".isaaclab_source_build").touch()
    elif installation == "container":
        monkeypatch.setattr(sys, "executable", str(local_sim / "kit/python/bin/python3"))
    monkeypatch.setattr(cli, "DEFAULT_ISAAC_SIM_PATH", local_sim)
    monkeypatch.setattr(sys, "argv", ["isaaclab", "demo", "--help"])
    if configured:
        monkeypatch.setenv("ISAAC_PATH", str(local_sim))
    else:
        monkeypatch.delenv("ISAAC_PATH", raising=False)
    with mock.patch.object(cli, "run_python_command") as launch, mock.patch.object(cli, "demo") as demo:
        cli.cli()
    if not delegated:
        launch.assert_not_called()
        demo.assert_called_once_with(["--help"])
    else:
        launch.assert_called_once_with("-m", ["isaaclab", "demo", "--help"], check=True)
        demo.assert_not_called()
