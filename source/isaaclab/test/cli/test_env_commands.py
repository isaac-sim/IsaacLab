# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for virtual environment setup commands."""

import os
import shutil
import subprocess
from unittest import mock

import pytest

import isaaclab.cli.commands.envs as envs

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("command", "environment_type"),
    [
        (envs.command_setup_conda, "conda"),
        (envs.command_setup_uv, "uv"),
    ],
)
def test_environment_setup_rejects_downloaded_isaac_sim(tmp_path, command, environment_type):
    """Downloaded Isaac Sim packages must use their bundled Python."""
    isaaclab_root = tmp_path / "IsaacLab"
    isaaclab_root.mkdir()
    isaac_sim_path = tmp_path / "isaacsim"
    isaac_sim_path.mkdir()
    local_sim = isaaclab_root / "_isaac_sim"
    if envs.is_windows():
        # Creating directory symlinks requires elevated privileges on some Windows runners.
        local_sim.mkdir()
    else:
        local_sim.symlink_to(isaac_sim_path, target_is_directory=True)

    with (
        mock.patch.object(envs, "ISAACLAB_ROOT", isaaclab_root),
        mock.patch.object(envs, "print_error") as print_error,
        pytest.raises(SystemExit, match="1"),
    ):
        command("env_isaaclab")

    assert print_error.call_args_list[0] == mock.call(
        f"Downloaded Isaac Sim packages cannot be combined with a {environment_type} environment."
    )


def test_environment_setup_accepts_marked_source_build(tmp_path):
    """Live source builds remain compatible with virtual environments."""
    isaac_sim_path = tmp_path / "_isaac_sim"
    isaac_sim_path.mkdir()
    (isaac_sim_path / ".isaaclab_source_build").touch()

    with mock.patch.object(envs, "ISAACLAB_ROOT", tmp_path):
        envs._reject_downloaded_isaac_sim("uv")


def test_launcher_rejects_downloaded_isaac_sim_with_active_environment(tmp_path):
    """Platform launchers must reject an active environment before selecting its Python."""
    launcher_name = "isaaclab.bat" if envs.is_windows() else "isaaclab.sh"
    launcher = tmp_path / launcher_name
    shutil.copy2(envs.ISAACLAB_ROOT / launcher_name, launcher)
    (tmp_path / "_isaac_sim").mkdir()

    environment = os.environ.copy()
    environment["VIRTUAL_ENV"] = str(tmp_path / "virtual-env")
    environment.pop("CONDA_PREFIX", None)
    command = ["cmd.exe", "/c", str(launcher), "-h"] if envs.is_windows() else ["bash", str(launcher), "-h"]

    result = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)

    assert result.returncode == 1
    assert "Downloaded Isaac Sim packages cannot be combined" in result.stderr


@pytest.mark.skipif(envs.is_windows(), reason="Linux launcher behavior")
def test_launcher_uses_bundled_python_with_inactive_default_environment(tmp_path):
    """An inactive default environment must not override the bundled Python."""
    launcher = tmp_path / "isaaclab.sh"
    shutil.copy2(envs.ISAACLAB_ROOT / "isaaclab.sh", launcher)
    bundled_python = tmp_path / "_isaac_sim" / "python.sh"
    bundled_python.parent.mkdir()
    bundled_python.write_text("#!/usr/bin/env bash\necho bundled-python\n")
    bundled_python.chmod(0o755)
    inactive_python = tmp_path / "env_isaaclab" / "bin" / "python"
    inactive_python.parent.mkdir(parents=True)
    inactive_python.write_text("#!/usr/bin/env bash\necho inactive-python\n")
    inactive_python.chmod(0o755)

    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("CONDA_PREFIX", None)
    result = subprocess.run(["bash", str(launcher), "-h"], capture_output=True, text=True, check=False, env=environment)

    assert result.returncode == 0
    assert "bundled-python" in result.stdout
    assert "inactive-python" not in result.stdout


@pytest.mark.skipif(envs.is_windows(), reason="Linux launcher behavior")
def test_launcher_allows_relinking_unmarked_source_build(tmp_path):
    """The source-build command must bypass downloaded-package environment rejection."""
    launcher = tmp_path / "isaaclab.sh"
    shutil.copy2(envs.ISAACLAB_ROOT / "isaaclab.sh", launcher)
    (tmp_path / "_isaac_sim").mkdir()
    active_python = tmp_path / "virtual-env" / "bin" / "python"
    active_python.parent.mkdir(parents=True)
    active_python.write_text("#!/usr/bin/env bash\necho active-python\n")
    active_python.chmod(0o755)

    environment = os.environ.copy()
    environment["VIRTUAL_ENV"] = str(tmp_path / "virtual-env")
    environment.pop("CONDA_PREFIX", None)
    result = subprocess.run(
        ["bash", str(launcher), "--isaacsim_source", str(tmp_path / "IsaacSim")],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0
    assert "active-python" in result.stdout
