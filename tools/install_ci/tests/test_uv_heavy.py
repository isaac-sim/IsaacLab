# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heavy uv-based installation and training tests for isaaclab."""

from __future__ import annotations

import shutil

import pytest
from uv_utils import UV_Utils


class Test_UV_Heavy(UV_Utils):
    """Heavy uv-based installation and training tests."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(1200)
    def test_install_and_train_cartpole(self, isaaclab_root):
        """Run `./isaaclab.x -i assets,tasks,rl[all],physx,newton,contrib` and train Isaac-Cartpole-Direct-v0 with rsl_rl on newton."""

        try:
            self.create_uv_env(isaaclab_root)

            # Install assets, tasks, rl[all], physx, newton, contrib
            result = self.run_in_env(
                [str(self.cli_script), "-i", "assets,tasks,rl[all],physx,newton,contrib"],
                cwd=isaaclab_root,
                check=False,
            )
            assert result.returncode == 0, f"isaaclab -i failed:\n{result.stdout}\n{result.stderr}"

            # Run a short training
            result = self.run_in_env(
                [
                    str(self.cli_script),
                    "-p",
                    "scripts/reinforcement_learning/rsl_rl/train.py",
                    "--task",
                    "Isaac-Cartpole-Direct-v0",
                    "--num_envs",
                    "4096",
                    "presets=newton",
                    "--max_iterations",
                    "5",
                ],
                cwd=isaaclab_root,
                check=False,
            )
            assert result.returncode == 0, f"Training failed:\n{result.stdout}\n{result.stderr}"
        finally:
            self.destroy_uv_env()
