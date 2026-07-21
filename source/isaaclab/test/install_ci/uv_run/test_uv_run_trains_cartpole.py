# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (none: uv run creates the environment from the committed uv.lock on first invocation)
Tests:
    - uv run --frozen isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct
        --num_envs 16 presets=newton_mjwarp --max_iterations 5 -> verify training completes
        from the committed lockfile (Newton MJWarp, kitless)
"""

from __future__ import annotations

import shutil

import pytest
from misc.cartpole_training_smoke import _STATE_TRAIN_CMD, _assert_training_passed
from utils import aarch64_isaacsim_env, run_cmd


@pytest.mark.install_path_uv_run
class Test_Uv_Run_Trains_Cartpole:
    """``uv run`` from the committed lockfile trains Cartpole on Newton MJWarp.

    ``--frozen`` uses the committed ``uv.lock`` exactly as-is (no re-resolution), so
    this validates the lockfile users actually get. Lockfile freshness itself is
    checked separately and cheaply by ``misc/test_uv_lock_check_smoke.py``. The
    environment is created in a temporary directory via ``UV_PROJECT_ENVIRONMENT``
    so the repository checkout stays clean.
    """

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.docker
    @pytest.mark.smoke
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(3600)
    def test_uv_run_from_committed_lock_trains_cartpole(self, isaaclab_root, tmp_path):
        """Verify ``uv run --frozen isaaclab train`` completes from the committed lock."""
        result = run_cmd(
            ["uv", "run", "--frozen", "isaaclab", *_STATE_TRAIN_CMD],
            cwd=isaaclab_root,
            env={"UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv"), **aarch64_isaacsim_env()},
            timeout=3400,
        )
        _assert_training_passed(result)
