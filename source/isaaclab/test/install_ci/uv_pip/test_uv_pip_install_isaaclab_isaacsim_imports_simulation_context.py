# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (wheel supplied by runner: tools/run_install_ci.py --build-wheel or --wheel <path>)
    - ./isaaclab.sh -u
    - uv pip install <wheel>[isaacsim] --extra-index-url https://pypi.nvidia.com
        --index-strategy unsafe-best-match --prerelease=allow
    - uv pip install --reinstall-package torch --reinstall-package torchvision
        torch==2.10.0 torchvision==0.25.0 --index-url <cu128|cu130>
        (cu128 on x86_64, cu130 on aarch64; per docs/source/setup/installation/pip_installation.rst.
         Reinstall AFTER the wheel install: unsafe-best-match re-resolves torch from PyPI to CPU.)
    - (aarch64 only) export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
Tests:
    - python -c "from isaaclab.app import AppLauncher" -> verify AppLauncher importable
    - python -c "from isaaclab.sim import SimulationContext" -> verify pxr-dependent imports resolve
"""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin, aarch64_isaacsim_env, cuda_torch_index_url


@pytest.mark.install_path_uv_pip
class Test_Uv_Pip_Install_Isaaclab_Isaacsim_Imports_Simulation_Context(UV_Mixin):
    """``uv pip install <wheel>[isaacsim]`` from the NVIDIA index; verify pxr-dependent imports."""

    _wheel: str = ""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.fixture(autouse=True, scope="class")
    def _install_wheel(self, isaaclab_root, wheel):
        cls = self.__class__
        cls._wheel = str(wheel)

        # 1. Create the uv env.
        self.create_uv_env(isaaclab_root)
        cls.env_path = self.env_path
        cls.python = self.python
        cls.cli_script = self.cli_script

        # 2. Install isaaclab with the isaacsim extra from the NVIDIA index.
        #    NOTE: --index-strategy unsafe-best-match re-resolves torch from PyPI (CPU build),
        #    so install isaaclab FIRST and force-reinstall CUDA torch in step 3 below.
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

        # 3. Reinstall CUDA-matched torch (cu128 on x86_64, cu130 on aarch64) to swap out the
        #    CPU torch unsafe-best-match picked above. Mirrors docs/source/setup/installation/pip_installation.rst.
        result = self.run_in_uv_env(
            [
                "uv",
                "pip",
                "install",
                "--reinstall-package",
                "torch",
                "--reinstall-package",
                "torchvision",
                "torch==2.10.0",
                "torchvision==0.25.0",
                "--index-url",
                cuda_torch_index_url(),
            ],
            cwd=isaaclab_root,
            timeout=1800,
        )
        assert result.returncode == 0, f"uv pip install CUDA torch failed:\n{result.stdout}\n{result.stderr}"

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
