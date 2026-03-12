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
    determine_python_version,
    extract_isaacsim_path,
    extract_python_exe,
    get_pip_command,
)


def _python_in_venv(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _python_for_conda(base: Path) -> Path:
    if sys.platform == "win32":
        return base / "python.exe"
    return base / "bin" / "python"


# ---------------------------------------------------------------------------
# get_pip_command
# ---------------------------------------------------------------------------


class TestGetPipCommand:
    """Tests for :func:`get_pip_command`."""

    def test_returns_uv_pip_in_venv_without_pip_module(self, tmp_path):
        """When VIRTUAL_ENV is set, uv is on PATH, and pip module is missing, return uv pip."""
        fake_python = str(tmp_path / "python")

        with (
            mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(tmp_path)}),
            mock.patch("isaaclab.cli.utils.shutil.which", return_value="/usr/bin/uv"),
            mock.patch(
                "isaaclab.cli.utils.subprocess.run",
                return_value=subprocess.CompletedProcess(args=[], returncode=1),
            ),
        ):
            result = get_pip_command(python_exe=fake_python)
            assert result == ["uv", "pip"]

    def test_returns_uv_pip_in_venv_with_uv(self, tmp_path):
        """When VIRTUAL_ENV is set and uv is on PATH, always return uv pip."""
        fake_python = str(tmp_path / "python")

        with (
            mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(tmp_path)}),
            mock.patch("isaaclab.cli.utils.shutil.which", return_value="/usr/bin/uv"),
        ):
            result = get_pip_command(python_exe=fake_python)
            assert result == ["uv", "pip"]

    def test_returns_python_pip_without_uv(self, tmp_path):
        """When uv is not installed, always return python -m pip."""
        fake_python = str(tmp_path / "python")

        with (
            mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(tmp_path)}),
            mock.patch("isaaclab.cli.utils.shutil.which", return_value=None),
        ):
            result = get_pip_command(python_exe=fake_python)
            assert result == [fake_python, "-m", "pip"]

    def test_returns_python_pip_in_conda_without_uv(self, tmp_path):
        """When in a conda env and uv is not available, return python -m pip."""
        fake_python = str(tmp_path / "python")

        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env["CONDA_PREFIX"] = str(tmp_path)
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("isaaclab.cli.utils.shutil.which", return_value=None),
        ):
            result = get_pip_command(python_exe=fake_python)
            assert result == [fake_python, "-m", "pip"]


# ---------------------------------------------------------------------------
# extract_python_exe
# ---------------------------------------------------------------------------


class TestExtractPythonExe:
    """Tests for :func:`extract_python_exe`."""

    def test_uses_virtual_env_when_set(self, tmp_path):
        """Should return the venv Python when VIRTUAL_ENV is set."""
        venv_python = _python_in_venv(tmp_path)
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.touch()

        with mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(tmp_path)}, clear=False):
            result = extract_python_exe()
            assert Path(result) == venv_python

    def test_uses_conda_prefix_when_no_venv(self, tmp_path):
        """Should return conda Python when CONDA_PREFIX is set and no VIRTUAL_ENV."""
        conda_python = _python_for_conda(tmp_path)
        conda_python.parent.mkdir(parents=True, exist_ok=True)
        conda_python.touch()

        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env["CONDA_PREFIX"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            result = extract_python_exe()
            assert Path(result) == conda_python


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


# ---------------------------------------------------------------------------
# determine_python_version
# ---------------------------------------------------------------------------


class TestDeterminePythonVersion:
    """Tests for :func:`determine_python_version`."""

    def test_defaults_to_3_12_when_no_sim(self):
        """Without Isaac Sim, should default to python 3.12 (Isaac Sim 6.x requirement)."""
        with (
            mock.patch("isaaclab.cli.utils.extract_isaacsim_path", return_value=None),
            mock.patch("importlib.metadata.version", side_effect=Exception("not found")),
        ):
            result = determine_python_version()
            assert result == "3.12"

    def test_returns_3_11_for_sim_5(self, tmp_path):
        """Isaac Sim 5.x should map to Python 3.11."""
        version_file = tmp_path / "VERSION"
        version_file.write_text("5.0.0")

        with mock.patch("isaaclab.cli.utils.extract_isaacsim_path", return_value=tmp_path):
            result = determine_python_version()
            assert result == "3.11"

    def test_returns_3_12_for_sim_6(self, tmp_path):
        """Isaac Sim 6.x should map to Python 3.12."""
        version_file = tmp_path / "VERSION"
        version_file.write_text("6.0.0")

        with mock.patch("isaaclab.cli.utils.extract_isaacsim_path", return_value=tmp_path):
            result = determine_python_version()
            assert result == "3.12"

    def test_raises_for_unknown_version(self, tmp_path):
        """Unknown Isaac Sim version should raise RuntimeError."""
        version_file = tmp_path / "VERSION"
        version_file.write_text("99.0.0")

        with (
            mock.patch("isaaclab.cli.utils.extract_isaacsim_path", return_value=tmp_path),
            pytest.raises(RuntimeError, match="Unsupported Isaac Sim version"),
        ):
            determine_python_version()

    def test_uses_package_metadata_when_no_version_file(self, tmp_path):
        """Should fall back to importlib.metadata when VERSION file doesn't exist."""
        # tmp_path exists but has no VERSION file
        with (
            mock.patch("isaaclab.cli.utils.extract_isaacsim_path", return_value=tmp_path),
            mock.patch("importlib.metadata.version", return_value="5.1.0"),
        ):
            result = determine_python_version()
            assert result == "3.11"
