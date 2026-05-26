# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heavy uv-based installation and training tests for isaaclab."""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin


class Test_UV_Env_Heavy(UV_Mixin):
    """Test ./isaaclab.x -u, then run heavy training."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.bug("nvbugs_5968136")
    @pytest.mark.skip(reason="Cartpole training fails in MuJoCo stiffness conversion.")
    @pytest.mark.timeout(1200)
    def test_install_and_train_cartpole(self, isaaclab_root):
        """``./isaaclab.sh -i newton,'rl[all]'`` then train Isaac-Cartpole-Direct-v0.

        Under the new install model, the core set (assets, tasks, physx, contrib, …)
        is always installed.  Only the optional extras (newton physics library and
        RL frameworks) need to be explicitly requested.
        """

        try:
            self.create_uv_env(isaaclab_root)

            # Core set is always installed; only request optional extras.
            result = self.run_in_uv_env([str(self.cli_script), "-i", "newton,rl[all]"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i newton,rl[all] failed:\n{result.stdout}\n{result.stderr}"

            # Run a short training
            result = self.run_in_uv_env(
                [
                    str(self.cli_script),
                    "-p",
                    "scripts/reinforcement_learning/rsl_rl/train.py",
                    "--task",
                    "Isaac-Cartpole-Direct-v0",
                    "--num_envs",
                    "4096",
                    "presets=newton_mjwarp",
                    "--max_iterations",
                    "5",
                ],
                cwd=isaaclab_root,
            )
            output = result.stdout + result.stderr
            assert result.returncode == 0, f"Training failed (rc={result.returncode}):\n{output}"
            assert "Traceback (most recent call last):" not in output, (
                f"Training produced a Python traceback:\n{output}"
            )
        finally:
            self.destroy_uv_env()
