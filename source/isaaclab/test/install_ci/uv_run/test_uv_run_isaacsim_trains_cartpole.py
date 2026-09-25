# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - uv sync --frozen --extra isaacsim
Tests:
    - LD_PRELOAD=$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/isaacsim/kit/kernel/plugins/libcarb.env.shim.so
        uv run --frozen --no-sync --extra isaacsim isaaclab train --rl_library rsl_rl
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

    ``--frozen`` uses the committed ``uv.lock`` as-is. The environment goes to a
    temporary directory via ``UV_PROJECT_ENVIRONMENT`` so the repository checkout
    stays clean.
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
        """Verify frozen Isaac Sim dependencies train Cartpole on Isaac Sim PhysX."""
        venv = tmp_path / "venv"
        runtime_env = {
            "UV_PROJECT_ENVIRONMENT": str(venv),
            "OMNI_KIT_ACCEPT_EULA": "yes",
            **aarch64_isaacsim_env(),
        }
        sync_result = run_cmd(
            ["uv", "sync", "--frozen", "--extra", "isaacsim"],
            cwd=isaaclab_root,
            env=runtime_env,
            timeout=4500,
        )
        assert sync_result.returncode == 0, f"uv sync failed:\n{sync_result.stdout}\n{sync_result.stderr}"

        # Kit plugins access and mutate the process environment from worker
        # threads. Preload Carbonite's environment shim before Python starts so
        # glibc getenv/setenv calls are serialized, matching the main Isaac Sim
        # Docker images and preventing intermittent startup SIGSEGVs.
        shim = venv / "lib/python3.12/site-packages/isaacsim/kit/kernel/plugins/libcarb.env.shim.so"
        assert shim.is_file(), f"Carbonite environment shim was not installed at {shim}"
        runtime_env["LD_PRELOAD"] = ":".join(filter(None, (str(shim), runtime_env.get("LD_PRELOAD"))))

        result = run_cmd(
            [
                "uv",
                "run",
                "--frozen",
                "--no-sync",
                "--extra",
                "isaacsim",
                "isaaclab",
                *_ISAACSIM_PHYSX_TRAIN_CMD,
            ],
            cwd=isaaclab_root,
            env=runtime_env,
            timeout=4500,
        )
        _assert_training_passed(result)
