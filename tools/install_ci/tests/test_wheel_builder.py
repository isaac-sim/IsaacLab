# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test wheel build and install scenarios for isaaclab."""

from __future__ import annotations

import glob
import shutil

import pytest
from utils import run_cmd
from uv_utils import UV_Utils


class Test_Wheel_Builder(UV_Utils):
    """Test building the isaaclab wheel and installing it in a uv environment."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(600)
    def test_build_wheel_and_import_isaaclab(self, isaaclab_root):
        """Build wheel locally, install it in UV env, verify 'isaaclab' is importable."""

        build_script = isaaclab_root / "tools" / "wheel_builder" / "build.sh"
        dist_dir = isaaclab_root / "tools" / "wheel_builder" / "build" / "dist"

        try:
            # Build the wheel
            result = run_cmd(["bash", str(build_script)], cwd=isaaclab_root, check=False)
            assert result.returncode == 0, f"build.sh failed:\n{result.stdout}\n{result.stderr}"

            # Find the built wheel
            wheels = glob.glob(str(dist_dir / "isaaclab-*.whl"))
            assert len(wheels) == 1, f"Expected exactly 1 wheel in {dist_dir}, found: {wheels}"
            wheel_path = wheels[0]

            # Create uv environment and install the wheel (without isaacsim extra to avoid dep conflicts)
            self.create_uv_env(isaaclab_root)
            result = self.run_in_env(["uv", "pip", "install", wheel_path], check=False)
            assert result.returncode == 0, f"uv pip install wheel failed:\n{result.stdout}\n{result.stderr}"

            # Verify isaaclab is importable
            result = self.run_in_env(
                ["python", "-c", "import isaaclab; print(isaaclab.__version__)"],
                check=False,
            )
            assert result.returncode == 0, f"import isaaclab failed:\n{result.stdout}\n{result.stderr}"
        finally:
            self.destroy_uv_env()
