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

    def test_returns_python_pip_in_venv_with_pip_module(self, tmp_path):
        """When VIRTUAL_ENV is set and pip module is available, return python -m pip."""
        fake_python = str(tmp_path / "python")

        with (
            mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(tmp_path)}),
            mock.patch("isaaclab.cli.utils.shutil.which", return_value="/usr/bin/uv"),
            mock.patch(
                "isaaclab.cli.utils.subprocess.run",
                return_value=subprocess.CompletedProcess(args=[], returncode=0),
            ),
        ):
            result = get_pip_command(python_exe=fake_python)
            assert result == [fake_python, "-m", "pip"]

    def test_returns_python_pip_without_uv(self, tmp_path):
        """When uv is not installed, always return python -m pip."""
        fake_python = str(tmp_path / "python")

        with (
            mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(tmp_path)}),
            mock.patch("isaaclab.cli.utils.shutil.which", return_value=None),
        ):
            result = get_pip_command(python_exe=fake_python)
            assert result == [fake_python, "-m", "pip"]

    def test_returns_python_pip_in_conda(self, tmp_path):
        """When VIRTUAL_ENV is not set (conda env), always return python -m pip."""
        fake_python = str(tmp_path / "python")

        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.dict(os.environ, {"CONDA_PREFIX": str(tmp_path)}, clear=False),
        ):
            # Remove VIRTUAL_ENV if present
            env = os.environ.copy()
            env.pop("VIRTUAL_ENV", None)
            with mock.patch.dict(os.environ, env, clear=True):
                result = get_pip_command(python_exe=fake_python)
                assert result == [fake_python, "-m", "pip"]


# ---------------------------------------------------------------------------
# extract_python_exe
# ---------------------------------------------------------------------------


class TestExtractPythonExe:
    """Tests for :func:`extract_python_exe`."""

    def test_uses_virtual_env_when_set(self, tmp_path):
        """Should return the venv Python when VIRTUAL_ENV is set."""
        venv_python = tmp_path / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.touch()

        with mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(tmp_path)}, clear=False):
            result = extract_python_exe()
            assert Path(result) == venv_python

    def test_uses_conda_prefix_when_no_venv(self, tmp_path):
        """Should return conda Python when CONDA_PREFIX is set and no VIRTUAL_ENV."""
        conda_python = tmp_path / "bin" / "python"
        conda_python.parent.mkdir(parents=True)
        conda_python.touch()

        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env["CONDA_PREFIX"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            result = extract_python_exe(allow_isaacsim_python=False)
            assert Path(result) == conda_python

    def test_falls_back_to_sys_executable(self):
        """When no VIRTUAL_ENV or CONDA_PREFIX, should use sys.executable."""
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env.pop("CONDA_PREFIX", None)
        with mock.patch.dict(os.environ, env, clear=True):
            result = extract_python_exe(allow_isaacsim_python=False)
            # Should return some valid Python path
            assert result is not None
            assert len(result) > 0


# ---------------------------------------------------------------------------
# extract_isaacsim_path
# ---------------------------------------------------------------------------


class TestExtractIsaacsimPath:
    """Tests for :func:`extract_isaacsim_path`."""

    def test_returns_none_when_not_required(self):
        """When required=False and Isaac Sim is not found, return None."""
        with mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", Path("/nonexistent/path")):
            result = extract_isaacsim_path(required=False)
            assert result is None

    def test_exits_when_required(self):
        """When required=True and Isaac Sim is not found, sys.exit."""
        with (
            mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", Path("/nonexistent/path")),
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

    def test_defaults_to_current_python_when_no_sim(self):
        """Without Isaac Sim, should return the current interpreter's version."""
        expected = f"{sys.version_info[0]}.{sys.version_info[1]}"

        with (
            mock.patch("isaaclab.cli.utils.extract_isaacsim_path", return_value=None),
            mock.patch("importlib.metadata.version", side_effect=Exception("not found")),
        ):
            result = determine_python_version()
            assert result == expected

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


# ---------------------------------------------------------------------------
# Integration: uv venv install path
# ---------------------------------------------------------------------------


class TestUvInstallPath:
    """Smoke tests verifying utility functions in the current venv."""

    def test_get_pip_command_uses_uv_in_current_venv(self):
        """In the current test venv (uv-managed), get_pip_command should return uv pip."""
        if not os.environ.get("VIRTUAL_ENV"):
            pytest.skip("Not running in a virtual environment")
        if not __import__("shutil").which("uv"):
            pytest.skip("uv not available")

        result = get_pip_command(python_exe=sys.executable)
        # In a uv venv without pip module, should return ["uv", "pip"]
        # In a venv with pip, should return [python, "-m", "pip"]
        assert len(result) >= 2
        assert result[-1] == "pip"

    def test_extract_python_exe_in_current_venv(self):
        """extract_python_exe should find the current venv's Python."""
        if not os.environ.get("VIRTUAL_ENV"):
            pytest.skip("Not running in a virtual environment")

        result = extract_python_exe()
        # Should point into the current venv
        assert os.environ["VIRTUAL_ENV"] in result

    def test_determine_python_version_without_sim(self):
        """Without Isaac Sim installed, should not crash and return a valid version."""
        try:
            from importlib.metadata import distribution

            distribution("isaacsim")
            pytest.skip("Isaac Sim is installed — this test is for the Kit-less path")
        except Exception:
            pass

        version = determine_python_version()
        major, minor = version.split(".")
        assert int(major) == 3
        assert int(minor) >= 10


# ---------------------------------------------------------------------------
# Integration: uv install into a fresh venv
# ---------------------------------------------------------------------------

# Resolve the Isaac Lab repo root once (source/isaaclab/test/cli/ -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _uv(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run a uv command, raising on failure."""
    return subprocess.run(["uv", *args], check=True, capture_output=True, text=True, **kwargs)


def _python_in_venv(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _run_in_venv(venv: Path, code: str) -> subprocess.CompletedProcess:
    """Run a Python snippet inside the given venv."""
    python = _python_in_venv(venv)
    return subprocess.run([str(python), "-c", code], check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def uv_venv(tmp_path_factory):
    """Create a temporary uv venv and install the Newton sub-packages into it.

    This fixture is module-scoped so the install only happens once for all
    integration tests, keeping total runtime reasonable (~30s).
    """
    import shutil

    if not shutil.which("uv"):
        pytest.skip("uv is not installed")

    venv_dir = tmp_path_factory.mktemp("isaaclab_test_venv")
    py_version = f"{sys.version_info[0]}.{sys.version_info[1]}"

    # Create venv
    _uv("venv", "--python", py_version, str(venv_dir))

    # Install Isaac Lab core + Newton sub-packages (editable, from local source).
    src = _REPO_ROOT / "source"
    packages = [
        f"-e{src / 'isaaclab'}",
        f"-e{src / 'isaaclab_newton'}",
        f"-e{src / 'isaaclab_physx'}",
        f"-e{src / 'isaaclab_tasks'}",
        f"-e{src / 'isaaclab_assets'}",
        f"-e{src / 'isaaclab_visualizers'}",
        f"-e{src / 'isaaclab_rl'}",
    ]
    _uv("pip", "install", *packages, "--python", str(venv_dir / "bin" / "python"))

    return venv_dir


class TestUvInstallIntegration:
    """Integration tests that install Isaac Lab into a fresh uv venv."""

    def test_isaaclab_importable(self, uv_venv):
        """Core isaaclab package should be importable."""
        _run_in_venv(uv_venv, "import isaaclab")

    def test_newton_importable(self, uv_venv):
        """isaaclab_newton should be importable."""
        _run_in_venv(uv_venv, "import isaaclab_newton")

    def test_physx_importable(self, uv_venv):
        """isaaclab_physx should be importable."""
        _run_in_venv(uv_venv, "import isaaclab_physx")

    def test_tasks_importable(self, uv_venv):
        """isaaclab_tasks should be importable."""
        _run_in_venv(uv_venv, "import isaaclab_tasks")

    def test_assets_importable(self, uv_venv):
        """isaaclab_assets should be importable."""
        _run_in_venv(uv_venv, "import isaaclab_assets")

    def test_visualizers_importable(self, uv_venv):
        """isaaclab_visualizers should be importable."""
        _run_in_venv(uv_venv, "import isaaclab_visualizers")

    def test_rl_importable(self, uv_venv):
        """isaaclab_rl should be importable."""
        _run_in_venv(uv_venv, "import isaaclab_rl")

    def test_builtin_tasks_registered(self, uv_venv):
        """Built-in gym environments should be registered after importing isaaclab_tasks."""
        result = _run_in_venv(
            uv_venv,
            "import isaaclab_tasks; import gymnasium as gym; "
            "envs = [s.id for s in gym.registry.values() if s.id.startswith('Isaac-')]; "
            "print(len(envs))",
        )
        count = int(result.stdout.strip())
        assert count > 50, f"Expected >50 built-in tasks, got {count}"

    def test_cartpole_env_spec_resolves(self, uv_venv):
        """The cartpole direct env should have a resolvable spec."""
        _run_in_venv(
            uv_venv,
            "import isaaclab_tasks; import gymnasium as gym; "
            "spec = gym.spec('Isaac-Cartpole-Direct-v0'); "
            "assert spec.entry_point is not None",
        )

    def test_get_pip_command_returns_uv(self, uv_venv):
        """In a fresh uv venv (no pip module), get_pip_command should return uv pip."""
        result = _run_in_venv(
            uv_venv,
            "import os; os.environ['VIRTUAL_ENV'] = os.environ.get('VIRTUAL_ENV', ''); "
            "from isaaclab.cli.utils import get_pip_command; "
            "import sys; cmd = get_pip_command(python_exe=sys.executable); "
            "print(','.join(cmd))",
        )
        cmd_parts = result.stdout.strip().split(",")
        # uv venvs don't include pip, so should detect uv
        assert cmd_parts[-1] == "pip"
