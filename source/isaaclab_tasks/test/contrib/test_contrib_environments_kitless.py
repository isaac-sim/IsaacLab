# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for contributed environments that run without Isaac Sim."""

import gymnasium as gym
import pytest

import isaaclab_tasks  # noqa: F401

# Local imports should be imported last
from contrib_env_test_utils import contrib_environment_params, num_envs  # isort: skip
from env_test_utils import _run_environments  # isort: skip


@pytest.mark.kitless
def test_factory_registration():
    task = gym.spec("IsaacContrib-Factory-Franka")
    assert task.entry_point == "isaaclab_tasks.contrib.nist.factory_variant_env:FactoryVariantEnv"
    assert (
        task.kwargs["env_cfg_entry_point"] == "isaaclab_tasks.contrib.nist.factory_variant_env_cfg:FactoryVariantEnvCfg"
    )
    assert task.kwargs["rsl_rl_cfg_entry_point"].endswith(":FactoryVariantPPORunnerCfg")
    assert "IsaacContrib-Factory-Variant-Franka" not in gym.registry


@pytest.mark.parametrize("task_name", contrib_environment_params("kitless"))
def test_contrib_environments_kitless(task_name):
    _run_environments(task_name, device="cuda", num_envs=num_envs(task_name))
