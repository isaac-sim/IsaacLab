# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - bash tools/wheel_builder/build.sh
    - ./isaaclab.sh -u
    - uv pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
    - uv pip install <wheel>[isaacsim] --extra-index-url https://pypi.nvidia.com
        --index-strategy unsafe-best-match --prerelease=allow
    - (aarch64 only) export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
Tests:
    - python -c "from isaaclab.app import AppLauncher" -> verify AppLauncher importable
    - python -c "from isaaclab.sim import SimulationContext" -> verify pxr-dependent imports resolve
"""

from __future__ import annotations

import glob
import shutil

import pytest
from utils import UV_Mixin, aarch64_isaacsim_env, run_cmd


@pytest.mark.install_path_uv_pip
class Test_Uv_Pip_Install_Isaaclab_Isaacsim_Imports_Simulation_Context(UV_Mixin):
    """Build the wheel, ``uv pip install <wheel>[isaacsim]`` from the NVIDIA index, verify pxr-dependent imports."""

    _wheel: str = ""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.fixture(autouse=True, scope="class")
    def _build_and_install_wheel(self, isaaclab_root):
        cls = self.__class__
        build_script = isaaclab_root / "tools" / "wheel_builder" / "build.sh"
        dist_dir = isaaclab_root / "tools" / "wheel_builder" / "build" / "dist"

        # 1. Build the wheel.
        result = run_cmd(["bash", str(build_script)], cwd=isaaclab_root, timeout=600)
        assert result.returncode == 0, f"build.sh failed:\n{result.stdout}\n{result.stderr}"

        wheels = glob.glob(str(dist_dir / "isaaclab-*.whl"))
        assert len(wheels) == 1, f"Expected exactly 1 wheel in {dist_dir}, found: {wheels}"
        cls._wheel = wheels[0]

        # 2. Create the uv env.
        self.create_uv_env(isaaclab_root)
        cls.env_path = self.env_path
        cls.python = self.python
        cls.cli_script = self.cli_script

        # 3. Pre-install CUDA-matched torch (mirrors docs/source/setup/installation/pip_installation.rst).
        result = self.run_in_uv_env(
            [
                "uv",
                "pip",
                "install",
                "-U",
                "torch==2.10.0",
                "torchvision==0.25.0",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            cwd=isaaclab_root,
            timeout=1800,
        )
        assert result.returncode == 0, f"uv pip install torch failed:\n{result.stdout}\n{result.stderr}"

        # 4. Install isaaclab with the isaacsim extra from the NVIDIA index.
        result = self.run_in_uv_env(
            [
                "uv",
                "pip",
                "install",
                f"{cls._wheel}[isaacsim]",
                "--extra-index-url",
                "https://pypi.nvidia.com",
                "--index-strategy",
                "unsafe-best-match",
                "--prerelease=allow",
            ],
            cwd=isaaclab_root,
            timeout=1800,
        )
        assert result.returncode == 0, (
            f"uv pip install {cls._wheel}[isaacsim] failed:\n{result.stdout}\n{result.stderr}"
        )

        yield

        self.destroy_uv_env()

    @pytest.mark.docker
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_install_isaacsim_makes_isaaclab_app_importable(self):
        """``from isaaclab.app import AppLauncher`` succeeds after ``uv pip install <wheel>[isaacsim]``."""
        result = self.run_in_uv_env(
            ["python", "-c", "from isaaclab.app import AppLauncher"],
            env=aarch64_isaacsim_env(),
        )
        assert result.returncode == 0, f"import isaaclab.app failed:\n{result.stdout}\n{result.stderr}"

    @pytest.mark.docker
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_install_isaacsim_makes_simulation_context_importable(self):
        """``from isaaclab.sim import SimulationContext`` resolves pxr from the isaacsim extra."""
        result = self.run_in_uv_env(
            ["python", "-c", "from isaaclab.sim import SimulationContext"],
            env=aarch64_isaacsim_env(),
        )
        assert result.returncode == 0, f"import isaaclab.sim failed:\n{result.stdout}\n{result.stderr}"
