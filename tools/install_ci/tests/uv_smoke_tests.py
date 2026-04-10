# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test uv-based installation scenarios for isaaclab."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

import pytest
from utils import run_cmd

_IS_WINDOWS = platform.system() == "Windows"


class Test_UV_Smoke:
    """Test uv-based installation scenarios."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    def _create_uv_env(self, isaaclab_root: Path, env_name: str) -> None:
        """Create a uv environment and store info on self.

        Sets ``self.env_path``, ``self.python``, ``self.cli_script``,
        and ``self.env`` (activation env vars).
        """
        self.env_path = isaaclab_root / env_name
        self.cli_script = isaaclab_root / ("isaaclab.bat" if _IS_WINDOWS else "isaaclab.sh")

        if self.env_path.exists():
            shutil.rmtree(self.env_path)

        result = run_cmd([str(self.cli_script), "-u", env_name], cwd=isaaclab_root, check=False)
        assert result.returncode == 0, f"uv env creation failed:\n{result.stdout}\n{result.stderr}"
        assert self.env_path.exists(), f"Expected env directory {self.env_path} was not created"

        self.python = (
            (self.env_path / "Scripts" / "python.exe") if _IS_WINDOWS else (self.env_path / "bin" / "python")
        )
        assert self.python.exists(), f"Python executable not found at {self.python}"

        bin_dir = str(self.env_path / "Scripts") if _IS_WINDOWS else str(self.env_path / "bin")
        self.env = {"VIRTUAL_ENV": str(self.env_path), "PATH": bin_dir + os.pathsep + os.environ.get("PATH", "")}

    def _destroy_uv_env(self) -> None:
        """Remove the uv environment directory if it exists."""
        if hasattr(self, "env_path") and self.env_path.exists():
            shutil.rmtree(self.env_path)

    @pytest.mark.uv
    @pytest.mark.timeout(10)
    def test_isaaclab_sh_uv_creates_env_with_python_312(self, isaaclab_root):
        """Run ./isaaclab.x -u and verify the created env has Python 3.12."""

        env_name = f"_installci_uvenv_{os.urandom(4).hex()}"
        try:
            self._create_uv_env(isaaclab_root, env_name)
            version_output = run_cmd([str(self.python), "--version"], check=False).stdout.strip()
            assert "3.12" in version_output, f"Expected Python 3.12, got: {version_output}"
        finally:
            self._destroy_uv_env()

    @pytest.mark.uv
    @pytest.mark.timeout(200)
    def test_isaaclab_install_assets(self, isaaclab_root):
        """Run ./isaaclab.x -i 'assets' and verify isaaclab_assets is importable."""

        env_name = f"install_ci_uvenv_{os.urandom(4).hex()}"
        try:
            self._create_uv_env(isaaclab_root, env_name)

            result = run_cmd(
                [str(self.cli_script), "-i", "assets"], cwd=isaaclab_root, env=self.env, check=False
            )
            assert result.returncode == 0, f"isaaclab -i assets failed:\n{result.stdout}\n{result.stderr}"

            result = run_cmd(
                [str(self.python), "-c", "import isaaclab_assets; print(isaaclab_assets.__version__)"],
                check=False,
            )
            assert result.returncode == 0, f"import isaaclab_assets failed:\n{result.stdout}\n{result.stderr}"
        finally:
            self._destroy_uv_env()

    @pytest.mark.uv
    @pytest.mark.timeout(300)
    def test_isaaclab_newton_installs_isaaclab_physx(self, isaaclab_root):
        """Run ./isaaclab.x -i 'newton' and verify isaaclab_physx is importable."""

        env_name = f"install_ci_uvenv_newton_{os.urandom(4).hex()}"
        try:
            self._create_uv_env(isaaclab_root, env_name)

            result = run_cmd(
                [str(self.cli_script), "-i", "newton"], cwd=isaaclab_root, env=self.env, check=False
            )
            assert result.returncode == 0, f"isaaclab -i newton failed:\n{result.stdout}\n{result.stderr}"

            result = run_cmd(
                [str(self.python), "-c", "import isaaclab_physx; print(isaaclab_physx.__version__)"],
                check=False,
            )
            assert result.returncode == 0, f"import isaaclab_physx failed:\n{result.stdout}\n{result.stderr}"
        finally:
            self._destroy_uv_env()
