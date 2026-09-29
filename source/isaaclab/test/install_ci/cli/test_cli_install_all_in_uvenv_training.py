# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - ./isaaclab.sh -u
Tests:
    - ./isaaclab.sh -i all
        -> verify automatic extras install without tetrahedralization dependencies
    - uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct
        --num_envs 16 presets=newton_mjwarp --max_iterations 5
        -> verify state training completes

The ``-i core`` import checks live in ``test_cli_install_in_uvenv_smoke.py``; the camera
rendering and camera training probes run once, on the wheel install path in
``uv_pip/test_uv_pip_install_isaaclab_all_trains_cartpole.py``.
"""

from __future__ import annotations

import shutil

import pytest
from misc.cartpole_training_smoke import _STATE_TRAIN_CMD, _assert_training_passed
from utils import UV_Mixin


class Test_Cli_Install_All_In_Uvenv_Training(UV_Mixin):
    """Installation and training smoke tests using uv environments."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    # regression for NVBug 5968136 (Cartpole training fails in MuJoCo stiffness conversion)
    @pytest.mark.install_path_cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(3600)
    def test_install_all_trains_cartpole(self, isaaclab_root):
        """``-i all`` installs the automatic extras and supports state training."""
        try:
            self.create_uv_env(isaaclab_root)
            result = self.run_in_uv_env(
                [str(self.cli_script), "-i", "all"],
                cwd=isaaclab_root,
                timeout=1200,
            )
            assert result.returncode == 0, f"isaaclab -i all failed:\n{result.stdout}\n{result.stderr}"
            result = self.run_in_uv_env(
                [
                    str(self.python),
                    "-c",
                    "import importlib.util; raise SystemExit(importlib.util.find_spec('pytetwild') is not None)",
                ],
                cwd=isaaclab_root,
                timeout=60,
            )
            assert result.returncode == 0, (
                f"pytetwild should not be installed by -i all:\n{result.stdout}\n{result.stderr}"
            )
            result = self.run_in_uv_env([str(self.cli_script)] + _STATE_TRAIN_CMD, cwd=isaaclab_root, timeout=600)
            _assert_training_passed(result)
        finally:
            self.destroy_uv_env()
