# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - bash tools/wheel_builder/build.sh
    - ./isaaclab.sh -u
    - uv pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
    - uv pip install <wheel>[all,isaacsim] --extra-index-url https://pypi.nvidia.com
        --index-strategy unsafe-best-match --prerelease=allow
Tests:
    - ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct-v0 --num_envs 16
        presets=newton_mjwarp --max_iterations 5 --headless
"""

from __future__ import annotations

import glob
import shutil

import pytest
from utils import UV_Mixin, run_cmd

_TRAIN_CMD = [
    "train",
    "--rl_library",
    "rsl_rl",
    "--task",
    "Isaac-Cartpole-Direct-v0",
    "--num_envs",
    "16",
    "presets=newton_mjwarp",
    "--max_iterations",
    "5",
    "--headless",
]


def _assert_training_passed(result) -> None:
    output = result.stdout + (result.stderr or "")
    assert result.returncode == 0, f"Training failed (rc={result.returncode}):\n{output}"
    assert "Traceback (most recent call last):" not in output, f"Training produced a traceback:\n{output}"
    assert "Training time:" in output, f"Training did not report completion:\n{output}"


class Test_Uv_Pip_Install_Isaaclab_All_Isaacsim_Trains_Cartpole(UV_Mixin):
    """Build the wheel, ``uv pip install <wheel>[all]``, verify cartpole training."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.docker
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(3600)
    def test_uv_pip_install_isaaclab_all_isaacsim_trains_cartpole(self, isaaclab_root):
        """Build wheel, install with ``[all]`` extras via ``uv pip``, run cartpole training."""
        try:
            build_script = isaaclab_root / "tools" / "wheel_builder" / "build.sh"
            dist_dir = isaaclab_root / "tools" / "wheel_builder" / "build" / "dist"

            # 1. Build the wheel.
            result = run_cmd(["bash", str(build_script)], cwd=isaaclab_root, timeout=600)
            assert result.returncode == 0, f"build.sh failed:\n{result.stdout}\n{result.stderr}"

            wheels = glob.glob(str(dist_dir / "isaaclab-*.whl"))
            assert len(wheels) == 1, f"Expected exactly 1 wheel in {dist_dir}, found: {wheels}"
            wheel = wheels[0]

            # 2. Create the uv env and install the wheel with [all] extras.

            self.create_uv_env(isaaclab_root)

            # pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
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

            # uv pip install "isaaclab[isaacsim]" --extra-index-url https://pypi.nvidia.com
            #   --index-strategy unsafe-best-match --prerelease=allow
            result = self.run_in_uv_env(
                    f"{wheel}[all,isaacsim]",
                    "uv",
                    "pip",
                    "install",
                    f"{wheel}[all, isaacsim]",
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
                f"uv pip install {wheel}[all, isaacsim] failed:\n{result.stdout}\n{result.stderr}"
            )

            # 3. Run cartpole training via ./isaaclab.sh train (same invocation as
            #    test_cli_install_training_in_uvenv::test_install_all_trains_cartpole).
            result = self.run_in_uv_env(
                [str(self.cli_script)] + _TRAIN_CMD,
                cwd=isaaclab_root,
                timeout=900,
            )
            _assert_training_passed(result)
        finally:
            self.destroy_uv_env()
