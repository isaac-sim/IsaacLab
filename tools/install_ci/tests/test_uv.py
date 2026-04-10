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


@pytest.mark.uv
class Test_UV:
    """Test uv-based installation scenarios."""

    @pytest.mark.timeout(10)
    def test_isaaclab_sh_uv_creates_env_with_python_312(self, isaaclab_root, tmp_path):
        """Run ./isaaclab.x -u and verify the created env has Python 3.12."""

        if not shutil.which("uv"):
            pytest.skip("uv is not available")

        is_windows = platform.system() == "Windows"
        env_name = "_installci_uvenv_"
        env_path = isaaclab_root / env_name

        # Clean up any leftover env from a previous run
        if env_path.exists():
            shutil.rmtree(env_path)

        # Pick the right CLI script for the platform
        cli_script = isaaclab_root / ("isaaclab.bat" if is_windows else "isaaclab.sh")

        try:
            # Create the uv environment via the CLI
            result = run_cmd(
                [str(cli_script), "-u", env_name],
                cwd=isaaclab_root,
                check=False,
            )

            assert result.returncode == 0, f"{cli_script.name} -u failed:\n{result.stdout}\n{result.stderr}"
            assert env_path.exists(), f"Expected env directory {env_path} was not created"

            # Locate the Python executable in the new env
            if is_windows:
                python = env_path / "Scripts" / "python.exe"
            else:
                python = env_path / "bin" / "python"
            assert python.exists(), f"Python executable not found at {python}"

            # Verify Python version is 3.12

            ver_result = run_cmd([str(python), "--version"], check=False)
            assert ver_result.returncode == 0, f"python --version failed:\n{ver_result.stderr}"

            version_output = ver_result.stdout.strip()
            assert "3.12" in version_output, f"Expected Python 3.12, got: {version_output}"

        finally:
            # Clean up the created environment
            if env_path.exists():
                shutil.rmtree(env_path)

    @pytest.mark.timeout(200)
    def test_isaaclab_install_assets(self, isaaclab_root):
        """Run ./isaaclab.x -i 'assets' and verify isaaclab_assets is importable."""

        if not shutil.which("uv"):
            pytest.skip("uv is not available")

        is_windows = platform.system() == "Windows"
        cli_script = isaaclab_root / ("isaaclab.bat" if is_windows else "isaaclab.sh")
        env_name = "install_ci_uvenv"
        env_path = isaaclab_root / env_name

        # Clean up any leftover env from a previous run
        if env_path.exists():
            shutil.rmtree(env_path)

        try:
            # Create uv environment first
            result = run_cmd(
                [str(cli_script), "-u", env_name],
                cwd=isaaclab_root,
                check=False,
            )
            assert result.returncode == 0, f"uv env creation failed:\n{result.stdout}\n{result.stderr}"

            # Activate the env by setting VIRTUAL_ENV and prepending to PATH
            if is_windows:
                python = env_path / "Scripts" / "python.exe"
                bin_dir = str(env_path / "Scripts")
            else:
                python = env_path / "bin" / "python"
                bin_dir = str(env_path / "bin")

            env = {"VIRTUAL_ENV": str(env_path), "PATH": bin_dir + ":" + os.environ.get("PATH", "")}

            # Install isaaclab core + assets
            result = run_cmd(
                [str(cli_script), "-i", "assets"],
                cwd=isaaclab_root,
                env=env,
                timeout=600,
                check=False,
            )
            assert result.returncode == 0, f"isaaclab -i assets failed:\n{result.stdout}\n{result.stderr}"

            # Verify isaaclab_assets is importable
            result = run_cmd(
                [str(python), "-c", "import isaaclab_assets; print(isaaclab_assets.__version__)"],
                check=False,
            )
            assert result.returncode == 0, f"import isaaclab_assets failed:\n{result.stdout}\n{result.stderr}"

        finally:
            if env_path.exists():
                shutil.rmtree(env_path)
