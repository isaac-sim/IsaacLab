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

pytestmark = pytest.mark.unit


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


# ---------------------------------------------------------------------------
# Prebundled-torch shadowing invariant (regression: nvbugs 6343978)
# ---------------------------------------------------------------------------


class TestEnsureNewton:
    """Tests for :func:`~isaaclab.cli.commands.install._ensure_newton`.

    Isaac Sim bundles ``newton[sim]==1.2.0``; the install CLI must force the pinned
    Newton git build (sourced from ``[tool.uv].override-dependencies``) over it.
    """

    @staticmethod
    def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    def test_installs_pinned_git_build_when_absent(self):
        """When the pinned commit is not installed, uninstall newton then install the git build."""
        from isaaclab.cli.commands import install

        commit = install._pinned_version("newton")
        calls = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return self._completed(stdout="numpy==2.0.0\n") if cmd[-1] == "freeze" else self._completed()

        with (
            mock.patch.object(install, "extract_python_exe", return_value="python"),
            mock.patch.object(install, "get_pip_command", return_value=["uv", "pip"]),
            mock.patch.object(install, "run_command", side_effect=fake_run),
        ):
            install._ensure_newton()

        assert any("uninstall" in cmd for cmd in calls), "old Newton should be uninstalled first"
        install_cmds = [cmd for cmd in calls if "install" in cmd]
        assert install_cmds, "expected a pip install call"
        install_args = install_cmds[-1]
        assert any(arg.startswith("newton[sim,importers]") and arg.endswith(commit) for arg in install_args)
        assert any(arg.startswith("newton-usd-schemas") for arg in install_args), "schemas must be forced too"

    def test_skips_when_commit_already_installed(self):
        """When freeze already reports the pinned commit, do not reinstall."""
        from isaaclab.cli.commands import install

        commit = install._pinned_version("newton")
        calls = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if cmd[-1] == "freeze":
                stdout = f"newton @ git+https://github.com/newton-physics/newton.git@{commit}\n"
                return self._completed(stdout=stdout)
            return self._completed()

        with (
            mock.patch.object(install, "extract_python_exe", return_value="python"),
            mock.patch.object(install, "get_pip_command", return_value=["uv", "pip"]),
            mock.patch.object(install, "run_command", side_effect=fake_run),
        ):
            install._ensure_newton()

        assert not any("install" in cmd for cmd in calls), "should not install when commit already present"
        assert not any("uninstall" in cmd for cmd in calls), "should not uninstall when commit already present"


def test_no_shadowing_prebundled_torch_in_isaac_sim():
    """A prebundled torch must not shadow the pip-installed torch.

    Regression test for nvbugs 6343978: Isaac Sim 6.0 ships a prebundled PyTorch under
    the deprecated ``omni.isaac.ml_archive`` extension whose ``libtorch_cuda.so``
    requires an NCCL symbol the co-bundled NCCL does not export. Launch paths that do
    not import :mod:`isaaclab` (e.g. ``isaac-sim.streaming.sh`` / ``runheadless.sh``)
    bypass the ``sys.path`` deprioritization and import this broken copy, crashing with
    ``undefined symbol: ncclDevCommCreate``. After install/build, every
    ``pip_prebundle/torch`` under Isaac Sim must therefore be either removed or a
    symlink into the active environment, never a real shadowing directory.
    """
    isaacsim_path = extract_isaacsim_path(required=False)
    if isaacsim_path is None or not isaacsim_path.exists():
        pytest.skip("Isaac Sim installation not found; skipping prebundle-shadow invariant check")

    shadowing = [
        prebundled_torch
        for prebundled_torch in isaacsim_path.rglob("pip_prebundle/torch")
        if prebundled_torch.is_dir() and not prebundled_torch.is_symlink()
    ]
    assert not shadowing, (
        "Found prebundled torch directories that shadow the pip-installed torch (nvbugs 6343978). "
        "They must be removed at image build time or repointed to the active environment:\n  "
        + "\n  ".join(str(p) for p in shadowing)
    )


# ---------------------------------------------------------------------------
# Pink IK stack derivation (single-source pins)
# ---------------------------------------------------------------------------


class TestPinkIkStack:
    """Tests for :func:`~isaaclab.cli.commands.install._pink_ik_stack`.

    The Pink IK pins live only in the root ``pyproject.toml``
    ``[project.dependencies]``; the install CLI derives its force-install
    stack from there instead of mirroring the versions.
    """

    def test_stack_derived_from_root_pyproject_pins(self):
        """The derived stack covers every stack package, exactly pinned, markers stripped."""
        from isaaclab.cli.commands import install

        stack = install._pink_ik_stack()
        assert [install._requirement_name(r) for r in stack] == list(install._PINK_IK_PACKAGES)
        assert any(r.startswith("pin-pink==") for r in stack), "pin-pink must stay exactly pinned"
        assert any(r.startswith("daqp==") for r in stack), "daqp must stay exactly pinned"
        assert all(";" not in r for r in stack), "environment markers must be stripped"
