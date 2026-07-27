# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (none: uv run creates the environment from the committed uv.lock on first invocation)
Tests:
    - uv run --frozen --extra isaacsim isaaclab train --rl_library rsl_rl
        --task Isaac-Cartpole-Direct --num_envs 16 --max_iterations 5 physics=isaacsim_physx
        -> verify training completes from the committed lockfile (Isaac Sim PhysX via the isaacsim extra)
"""

from __future__ import annotations

import platform
import shutil

import pytest
from misc.cartpole_training_smoke import _assert_training_passed
from utils import aarch64_isaacsim_env, run_cmd

# Canonical cartpole probe, with the physics selector swapped to concrete Isaac Sim
# PhysX. The isaacsim extra is conflict-forked from the base install, so uv syncs the
# environment with the Isaac Sim wheels before running.
_ISAACSIM_PHYSX_TRAIN_CMD = [
    "train",
    "--rl_library",
    "rsl_rl",
    "--task",
    "Isaac-Cartpole-Direct",
    "--num_envs",
    "16",
    "--max_iterations",
    "5",
    "physics=isaacsim_physx",
]


@pytest.mark.install_path_uv_run
class Test_Uv_Run_Isaacsim_Trains_Cartpole:
    """``uv run --extra isaacsim`` from the committed lockfile trains Cartpole on Isaac Sim PhysX.

    ``--frozen`` uses the committed ``uv.lock`` as-is; freshness is checked cheaply by
    ``misc/test_uv_lock_check_smoke.py``. The environment goes to a temporary directory
    via ``UV_PROJECT_ENVIRONMENT`` so the repository checkout stays clean.
    """

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")
        # PhysX training currently fails at env creation on aarch64 (DGX Spark):
        # "[omni.physx.tensors.plugin] Simulation view object is invalidated and cannot
        # be used again to call updateArticulationsKinematic". The environment itself
        # installs fine from the lock; this matches the x86-only scope of the workflow
        # this test replaces. Re-enable once PhysX training works on aarch64.
        if platform.machine().lower() in ("aarch64", "arm64"):
            pytest.skip("PhysX training is not functional on aarch64 yet")

    @pytest.mark.docker
    @pytest.mark.smoke
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(4800)
    def test_uv_run_isaacsim_from_committed_lock_trains_cartpole(self, isaaclab_root, tmp_path):
        """Verify ``uv run --frozen --extra isaacsim isaaclab train`` completes on Isaac Sim PhysX."""
        result = run_cmd(
            ["uv", "run", "--frozen", "--extra", "isaacsim", "isaaclab", *_ISAACSIM_PHYSX_TRAIN_CMD],
            cwd=isaaclab_root,
            env={
                "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv"),
                "OMNI_KIT_ACCEPT_EULA": "yes",
                **aarch64_isaacsim_env(),
            },
            timeout=4500,
        )
        _assert_training_passed(result)
