# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (wheel supplied by runner: tools/run_install_ci.py --build-wheel or --wheel <path>)
    - ./isaaclab.sh -u
    - uv pip install <wheel>[all,isaacsim] --overrides uv_pip/uv-overrides.txt
        --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
    - uv pip install --reinstall-package torch --reinstall-package torchvision
        torch==<pinned> torchvision==<pinned> --index-url <cu128|cu130>
        (versions read from [tool.isaaclab.versions] in the root pyproject.)
        (cu128 on x86_64, cu130 on aarch64; per docs/source/setup/installation/pip_installation.rst.
         Reinstall AFTER the wheel install: unsafe-best-match re-resolves torch from PyPI to CPU.)
    - (aarch64 only) export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
Tests:
    - uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 16
        presets=newton_mjwarp --max_iterations 5; uv run isaaclab train --rl_library rsl_rl
        --task Isaac-Cartpole-Camera-Direct --num_envs 16 presets=newton_mjwarp,newton_renderer --max_iterations 2
 -> verify state training, camera rendering, and camera training work
"""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin, aarch64_isaacsim_env, cuda_torch_index_url, pinned_torch_specs


@pytest.mark.install_path_uv_pip
class Test_Uv_Pip_Install_Isaaclab_All_Isaacsim_Trains_Cartpole(UV_Mixin):
    """Build the wheel, ``uv pip install <wheel>[all,isaacsim]``, verify cartpole training."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.docker
    @pytest.mark.smoke
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(4800)
    def test_uv_pip_install_isaaclab_all_isaacsim_trains_cartpole(
        self, isaaclab_root, wheel, uv_overrides, cartpole_smoke_script
    ):
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
                    "--overrides",
                    str(uv_overrides),
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
            #    torch==<pinned> torchvision==<pinned> --index-url <cu128|cu130>
            #    (versions from [tool.isaaclab.versions]; cu128 on x86_64, cu130 on aarch64,
            #    e.g. GB10 / DGX Spark with CUDA capability 12.x).
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
                    *pinned_torch_specs(),
                    "--index-url",
                    cuda_torch_index_url(),
                ],
                cwd=isaaclab_root,
                timeout=1800,
            )
            assert result.returncode == 0, f"uv pip install CUDA torch failed:\n{result.stdout}\n{result.stderr}"

            # 3. Run the shared state and camera Cartpole smoke in the installed environment.
            result = self.run_in_uv_env(
                [str(self.python), str(cartpole_smoke_script)],
                cwd=isaaclab_root,
                env=aarch64_isaacsim_env(),
                timeout=3000,
            )
            assert result.returncode == 0, f"Cartpole smoke failed:\n{result.stdout}\n{result.stderr}"
        finally:
            self.destroy_uv_env()
