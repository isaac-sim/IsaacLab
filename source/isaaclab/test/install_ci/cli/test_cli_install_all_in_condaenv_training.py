# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - conda create -n <env> python=3.12
Tests:
    - ./isaaclab.sh -i core -> verify core submodules importable
    - ./isaaclab.sh -i newton,rl[rsl-rl] -> verify cartpole training works
"""

from __future__ import annotations

import shutil

import pytest
from utils import Conda_Mixin

# ---------------------------------------------------------------------------
# Shared training helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# conda-based tests
# ---------------------------------------------------------------------------


class Test_Cli_Install_All_In_Condaenv_Training(Conda_Mixin):
    """Installation and training smoke tests using conda environments."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("conda"):
            pytest.skip("conda is not available")

    @pytest.mark.install_path_cli
    @pytest.mark.conda
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(1200)
    def test_install_core_makes_core_submodules_importable(self, isaaclab_root):
        """conda + ``./isaaclab.sh -i core`` installs all core submodules without extras."""
        try:
            self.create_conda_env(isaaclab_root)
            result = self.run_in_conda_env(
                [str(self.cli_script), "-i", "core"],
                cwd=isaaclab_root,
                timeout=900,
            )
            assert result.returncode == 0, f"conda isaaclab -i core failed:\n{result.stdout}\n{result.stderr}"
            for pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_physx"):
                r = self.run_in_conda_env(
                    [str(self.python), "-c", f"import {pkg}; print({pkg!r}, 'ok')"],
                    cwd=isaaclab_root,
                    timeout=60,
                )
                assert r.returncode == 0, f"{pkg} not importable after conda -i core:\n{r.stdout}\n{r.stderr}"
        finally:
            self.destroy_conda_env()

    @pytest.mark.install_path_cli
    @pytest.mark.conda
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(1800)
    def test_install_newton_rl_rsl_rl_trains_cartpole(self, isaaclab_root):
        """conda + ``./isaaclab.sh -i newton,rl[rsl-rl]`` + training completes successfully."""
        try:
            self.create_conda_env(isaaclab_root)
            result = self.run_in_conda_env(
                [str(self.cli_script), "-i", "newton,rl[rsl-rl]"],
                cwd=isaaclab_root,
                timeout=1200,
            )
            assert result.returncode == 0, (
                f"conda isaaclab -i newton,rl[rsl-rl] failed:\n{result.stdout}\n{result.stderr}"
            )
            result = self.run_in_conda_env(
                [str(self.cli_script)] + _TRAIN_CMD,
                cwd=isaaclab_root,
                timeout=600,
            )
            _assert_training_passed(result)
        finally:
            self.destroy_conda_env()
