# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for contributed environments that run without Isaac Sim."""

from pathlib import Path

import gymnasium as gym
import pytest

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.nist as nist

# Local imports should be imported last
from contrib_env_test_utils import contrib_environment_params, num_envs  # isort: skip
from env_test_utils import _run_environments  # isort: skip


@pytest.mark.kitless
def test_nist_registration():
    task = gym.spec("IsaacContrib-NIST-Franka")
    assert task.entry_point == "isaaclab_tasks.contrib.nist.nist_env:NISTEnv"
    assert task.kwargs["env_cfg_entry_point"] == "isaaclab_tasks.contrib.nist.nist_env_cfg:NISTEnvCfg"
    assert task.kwargs["rsl_rl_cfg_entry_point"].endswith(":NISTPPORunnerCfg")
    assert "IsaacContrib-Factory-Franka" not in gym.registry
    assert "IsaacContrib-Factory-Variant-Franka" not in gym.registry
    task_root = Path(nist.__file__).parent
    for old_file in (
        "factory_assets_cfg.py",
        "factory_presets.py",
        "factory_variant_env.py",
        "factory_variant_env_cfg.py",
        "factory_variant_scene_cfg.py",
        "variant_reset_env_cfg.py",
    ):
        assert not (task_root / old_file).exists()


@pytest.mark.parametrize("task_name", contrib_environment_params("kitless"))
def test_contrib_environments_kitless(task_name):
    _run_environments(task_name, device="cuda", num_envs=num_envs(task_name))
