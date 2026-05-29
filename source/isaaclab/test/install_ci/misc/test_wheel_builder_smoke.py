# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - bash tools/wheel_builder/build.sh
    - ./isaaclab.sh -u
    - uv pip install <wheel>[all]
Tests:
    - import isaaclab -> verify importable
    - from isaaclab import __version__ -> verify version matches wheel filename
    - from isaaclab.app import AppLauncher -> verify importable
    - from isaaclab.envs import ViewerCfg -> verify importable
    - from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG -> verify importable
    - from isaaclab.scene import InteractiveSceneCfg -> verify importable
    - python -m isaaclab --help -> verify CLI functional
    - import pinocchio -> verify importable
"""

from __future__ import annotations

import glob
import shutil

import pytest
from utils import UV_Mixin, run_cmd


class Test_Wheel_Builder_Smoke(UV_Mixin):
    """Test building the isaaclab wheel and installing it in a uv environment."""

    _wheel: str = ""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.fixture(autouse=True, scope="class")
    def _build_and_install_wheel(self, isaaclab_root):
        """Build the wheel and install it in a uv environment once for all tests."""

        cls = self.__class__
        build_script = isaaclab_root / "tools" / "wheel_builder" / "build.sh"
        dist_dir = isaaclab_root / "tools" / "wheel_builder" / "build" / "dist"

        # Build the wheel (capture output silently to avoid spamming the test log with 10k+
        # setuptools/pip lines; the captured output is included in the assertion if it fails).
        result = run_cmd(["bash", str(build_script)], cwd=isaaclab_root, stream=False)
        assert result.returncode == 0, f"build.sh failed:\n{result.stdout}\n{result.stderr}"

        # Find the built wheel
        wheels = glob.glob(str(dist_dir / "isaaclab-*.whl"))
        assert len(wheels) == 1, f"Expected exactly 1 wheel in {dist_dir}, found: {wheels}"
        cls._wheel = wheels[0]

        # Create uv environment and install the wheel
        self.create_uv_env(isaaclab_root)

        # Share env state with all test instances via the class
        cls.env_path = self.env_path
        cls.python = self.python
        cls.cli_script = self.cli_script
        result = self.run_in_uv_env(["uv", "pip", "install", cls._wheel + "[all]"])
        assert result.returncode == 0, f"uv pip install wheel failed:\n{result.stdout}\n{result.stderr}"

        yield

        self.destroy_uv_env()

    # import isaaclab
    def test_isaaclab_importable(self):
        """Verify 'isaaclab' is importable."""
        result = self.run_in_uv_env(["python", "-c", "import isaaclab;"])
        assert result.returncode == 0, f"import isaaclab failed:\n{result.stdout}\n{result.stderr}"

    # from isaaclab import __version__; print(__version__)
    def test_isaaclab_version_matches_wheel(self):
        """Verify isaaclab.__version__ matches the wheel version."""
        result = self.run_in_uv_env(["python", "-c", "from isaaclab import __version__; print(__version__)"])
        imported_version = result.stdout.strip()
        expected_version = self._wheel.split("/")[-1].split("-")[1]
        assert imported_version == expected_version, (
            f"isaaclab.__version__ mismatch: expected {expected_version}, got {imported_version}"
        )

    # from isaaclab.app import AppLauncher
    def test_isaaclab_app_importable(self):
        """Verify isaaclab.app and AppLauncher are importable."""
        result = self.run_in_uv_env(["python", "-c", "from isaaclab.app import AppLauncher"])
        assert result.returncode == 0, f"import isaaclab.app failed:\n{result.stdout}\n{result.stderr}"

    # from isaaclab.envs import ViewerCfg
    def test_isaaclab_envs_importable(self):
        """Verify isaaclab.envs is importable."""
        result = self.run_in_uv_env(["python", "-c", "from isaaclab.envs import ViewerCfg"])
        assert result.returncode == 0, f"import isaaclab.envs failed:\n{result.stdout}\n{result.stderr}"

    # from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG
    def test_isaaclab_assets_importable(self):
        """Verify isaaclab_assets is importable."""
        result = self.run_in_uv_env(["python", "-c", "from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG"])
        assert result.returncode == 0, f"import isaaclab_assets failed:\n{result.stdout}\n{result.stderr}"

    # from isaaclab.scene import InteractiveSceneCfg
    def test_isaaclab_scene_importable(self):
        """Verify isaaclab.scene and InteractiveSceneCfg are importable."""
        result = self.run_in_uv_env(["python", "-c", "from isaaclab.scene import InteractiveSceneCfg"])
        assert result.returncode == 0, f"import isaaclab.scene failed:\n{result.stdout}\n{result.stderr}"

    # python -m isaaclab --help
    def test_python_m_isaaclab_help_works(self):
        """Verify the isaaclab CLI is functional."""
        result = self.run_in_uv_env(["python", "-m", "isaaclab", "--help"])
        assert result.returncode == 0, f"isaaclab CLI help failed:\n{result.stdout}\n{result.stderr}"

    # import pinocchio as pin; print(pin.__version__)
    def test_pinocchio_importable(self):
        """Verify pinocchio is importable and has the expected version."""
        result = self.run_in_uv_env(["python", "-c", "import pinocchio as pin; print(pin.__version__)"])
        assert result.returncode == 0, f"import pinocchio failed:\n{result.stdout}\n{result.stderr}"
