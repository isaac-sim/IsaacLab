# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test uv-based installation scenarios for isaaclab."""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin


class Test_UV_Env_Smoke(UV_Mixin):
    """Test ./isaaclab.x -u, then validate with some quick checks."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.uv
    @pytest.mark.timeout(10)
    def test_isaaclab_sh_uv_creates_env_with_python_312(self, isaaclab_root):
        """Run ./isaaclab.x -u and verify the created env has Python 3.12."""

        try:
            self.create_uv_env(isaaclab_root)
            # python --version
            version_output = self.run_in_uv_env(["python", "--version"]).stdout.strip()
            assert "3.12" in version_output, f"Expected Python 3.12, got: {version_output}"
        finally:
            self.destroy_uv_env()

    @pytest.mark.uv
    @pytest.mark.timeout(200)
    def test_isaaclab_install_assets(self, isaaclab_root):
        """Run ./isaaclab.x -i 'assets' and verify isaaclab_assets is importable."""

        try:
            self.create_uv_env(isaaclab_root)

            # ./isaaclab.x -i assets
            result = self.run_in_uv_env([str(self.cli_script), "-i", "assets"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i assets failed:\n{result.stdout}\n{result.stderr}"

            # import isaaclab_assets
            result = self.run_in_uv_env(["python", "-c", "import isaaclab_assets; print(isaaclab_assets.__version__)"])
            assert result.returncode == 0, f"import isaaclab_assets failed:\n{result.stdout}\n{result.stderr}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.uv
    @pytest.mark.timeout(300)
    def test_isaaclab_newton_installs_isaaclab_newton(self, isaaclab_root):
        """Run ./isaaclab.x -i 'newton' and verify isaaclab_newton is importable."""

        try:
            self.create_uv_env(isaaclab_root)

            # ./isaaclab.x -i newton
            result = self.run_in_uv_env([str(self.cli_script), "-i", "newton"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i newton failed:\n{result.stdout}\n{result.stderr}"

            # import isaaclab_newton
            result = self.run_in_uv_env(["python", "-c", "import isaaclab_newton; print(isaaclab_newton.__version__)"])
            assert result.returncode == 0, f"import isaaclab_newton failed:\n{result.stdout}\n{result.stderr}"

        finally:
            self.destroy_uv_env()
