# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test uv-based installation scenarios for isaaclab."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
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

    def _create_uv_env(self, isaaclab_root: Path, env_name: str = "") -> None:
        """Create a uv environment and store info on self.

        Sets ``self.env_path``, ``self.python``, and ``self.cli_script``.
        """
        
        # gen random env name to avoid conflicts with other tests and ensure cleanup
        env_name = f"_isaaclab_install_ci_{os.urandom(4).hex()}" if not env_name else env_name
        
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

    def _destroy_uv_env(self) -> None:
        """Remove the uv environment directory if it exists."""
        if hasattr(self, "env_path") and self.env_path.exists():
            shutil.rmtree(self.env_path)

    def _run_in_env(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a command inside the activated venv by sourcing the activate script."""
        escaped = " ".join(shlex.quote(str(a)) for a in cmd)
        if _IS_WINDOWS:
            activate = str(self.env_path / "Scripts" / "activate.bat")
            shell_cmd = f'call "{activate}" && {escaped}'
            return run_cmd(["cmd", "/c", shell_cmd], **kwargs)
        else:
            activate = shlex.quote(str(self.env_path / "bin" / "activate"))
            shell_cmd = f"source {activate} && {escaped}"
            return run_cmd(["bash", "-c", shell_cmd], **kwargs)

    @pytest.mark.uv
    @pytest.mark.timeout(10)
    def test_isaaclab_sh_uv_creates_env_with_python_312(self, isaaclab_root):
        """Run ./isaaclab.x -u and verify the created env has Python 3.12."""

        try:
            self._create_uv_env(isaaclab_root)
            version_output = self._run_in_env(["python", "--version"], check=False).stdout.strip()
            assert "3.12" in version_output, f"Expected Python 3.12, got: {version_output}"
        finally:
            self._destroy_uv_env()

    @pytest.mark.uv
    @pytest.mark.timeout(200)
    def test_isaaclab_install_assets(self, isaaclab_root):
        """Run ./isaaclab.x -i 'assets' and verify isaaclab_assets is importable."""

        try:
            self._create_uv_env(isaaclab_root)

            result = self._run_in_env(
                [str(self.cli_script), "-i", "assets"], cwd=isaaclab_root, check=False
            )
            assert result.returncode == 0, f"isaaclab -i assets failed:\n{result.stdout}\n{result.stderr}"

            result = self._run_in_env(
                ["python", "-c", "import isaaclab_assets; print(isaaclab_assets.__version__)"],
                check=False,
            )
            assert result.returncode == 0, f"import isaaclab_assets failed:\n{result.stdout}\n{result.stderr}"
        finally:
            self._destroy_uv_env()

    @pytest.mark.uv
    @pytest.mark.timeout(300)
    def test_isaaclab_newton_installs_isaaclab_physx(self, isaaclab_root):
        """Run ./isaaclab.x -i 'newton' and verify isaaclab_physx is importable."""

        try:
            self._create_uv_env(isaaclab_root)

            result = self._run_in_env(
                [str(self.cli_script), "-i", "newton"], cwd=isaaclab_root, check=False
            )
            assert result.returncode == 0, f"isaaclab -i newton failed:\n{result.stdout}\n{result.stderr}"

            result = self._run_in_env(
                ["python", "-c", "import isaaclab_physx; print(isaaclab_physx.__version__)"],
                check=False,
            )
            assert result.returncode == 0, f"import isaaclab_physx failed:\n{result.stdout}\n{result.stderr}"
        finally:
            self._destroy_uv_env()
