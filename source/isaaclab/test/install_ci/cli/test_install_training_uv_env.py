# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end installation and training workflow tests (uv).

Covers uv-emnv-based installation paths:
  - uv env + kitless (core-only, ``-i core``)
  - uv env + full install (``-i all``)
"""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin

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
# uv-based tests
# ---------------------------------------------------------------------------


class TestUVWorkflow(UV_Mixin):
    """Installation and training smoke tests using uv environments."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(900)
    def test_install_core_installs_core_submodules(self, isaaclab_root):
        """``./isaaclab.sh -i core`` installs all core submodules without extras."""
        try:
            self.create_uv_env(isaaclab_root)
            result = self.run_in_uv_env(
                [str(self.cli_script), "-i", "core"],
                cwd=isaaclab_root,
                timeout=600,
            )
            assert result.returncode == 0, f"isaaclab -i core failed:\n{result.stdout}\n{result.stderr}"
            output = result.stdout + result.stderr
            # All core submodules should be installed; no optional tokens should warn
            assert "WARNING" not in output or "Unknown install token" not in output, (
                f"Unexpected warnings from -i core:\n{output}"
            )
            # Verify core packages importable
            for pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_physx"):
                r = self.run_in_uv_env(
                    [str(self.python), "-c", f"import {pkg}; print({pkg!r}, 'ok')"],
                    cwd=isaaclab_root,
                    timeout=60,
                )
                assert r.returncode == 0, f"{pkg} not importable after -i core:\n{r.stdout}\n{r.stderr}"
        finally:
            self.destroy_uv_env()

    # regression for NVBug 5968136 (Cartpole training fails in MuJoCo stiffness conversion)
    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(1800)
    def test_install_all_trains_cartpole(self, isaaclab_root):
        """``./isaaclab.sh -i all`` (full install) + training completes successfully."""
        try:
            self.create_uv_env(isaaclab_root)
            result = self.run_in_uv_env(
                [str(self.cli_script), "-i", "all"],
                cwd=isaaclab_root,
                timeout=1200,
            )
            assert result.returncode == 0, f"isaaclab -i all failed:\n{result.stdout}\n{result.stderr}"
            result = self.run_in_uv_env(
                [str(self.cli_script)] + _TRAIN_CMD,
                cwd=isaaclab_root,
                timeout=600,
            )
            _assert_training_passed(result)
        finally:
            self.destroy_uv_env()
