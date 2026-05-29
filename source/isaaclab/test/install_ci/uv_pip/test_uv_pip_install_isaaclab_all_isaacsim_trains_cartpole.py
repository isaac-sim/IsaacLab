# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (wheel supplied by runner: tools/run_install_ci.py --build-wheel or --wheel <path>)
    - ./isaaclab.sh -u
    - uv pip install <wheel>[all,isaacsim] --extra-index-url https://pypi.nvidia.com
        --index-strategy unsafe-best-match --prerelease=allow
    - uv pip install --reinstall-package torch --reinstall-package torchvision
        torch==2.10.0 torchvision==0.25.0 --index-url <cu128|cu130>
        (cu128 on x86_64, cu130 on aarch64; per docs/source/setup/installation/pip_installation.rst.
         Reinstall AFTER the wheel install: unsafe-best-match re-resolves torch from PyPI to CPU.)
    - (aarch64 only) export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
Tests:
    - ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct-v0 --num_envs 16
        presets=newton_mjwarp --max_iterations 5 --headless
"""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin, aarch64_isaacsim_env, cuda_torch_index_url

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


@pytest.mark.install_path_uv_pip
class Test_Uv_Pip_Install_Isaaclab_All_Isaacsim_Trains_Cartpole(UV_Mixin):
    """Build the wheel, ``uv pip install <wheel>[all,isaacsim]``, verify cartpole training."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.docker
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(3600)
    def test_uv_pip_install_isaaclab_all_isaacsim_trains_cartpole(self, isaaclab_root, wheel):
        """Install the runner-supplied wheel with ``[all,isaacsim]`` via ``uv pip``, run cartpole training."""
        try:
            # 1. Create the uv env and install the wheel with [all,isaacsim] extras.
            self.create_uv_env(isaaclab_root)

            # uv pip install "isaaclab[all,isaacsim]" --extra-index-url https://pypi.nvidia.com
            #   --index-strategy unsafe-best-match --prerelease=allow
            # NOTE: --index-strategy unsafe-best-match re-resolves torch from PyPI (CPU build),
            #       overriding any pre-installed CUDA torch. So install isaaclab FIRST, then
            #       force-reinstall the CUDA torch from cu128/cu130 below.
            result = self.run_in_uv_env(
                [
                    "uv",
                    "pip",
                    "install",
                    f"{wheel}[all,isaacsim]",
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
                f"uv pip install {wheel}[all,isaacsim] failed:\n{result.stdout}\n{result.stderr}"
            )

            # 2. uv pip install --reinstall-package torch --reinstall-package torchvision
            #    torch==2.10.0 torchvision==0.25.0 --index-url <cu128|cu130>
            #    cu128 on x86_64, cu130 on aarch64 (e.g. GB10 / DGX Spark with CUDA capability 12.x).
            #    --reinstall-package forces uv to swap the CPU torch installed above with the CUDA build.
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

            # 3. Run cartpole training via ./isaaclab.sh train (same invocation as
            #    test_cli_install_training_in_uvenv::test_install_all_trains_cartpole).
            result = self.run_in_uv_env(
                [str(self.cli_script)] + _TRAIN_CMD,
                cwd=isaaclab_root,
                env=aarch64_isaacsim_env(),
                timeout=900,
            )
            _assert_training_passed(result)
        finally:
            self.destroy_uv_env()
