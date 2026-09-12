# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Franka deformable camera benchmark task registrations."""

import gymnasium as gym
import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_ENV_CFG_MODULE = "isaaclab_tasks.benchmark.franka_deformable_camera.franka_deformable_camera_env_cfg"
_AGENT_CFG_MODULE = "isaaclab_tasks.benchmark.franka_deformable_camera.agents.rsl_rl_ppo_cfg"


@pytest.mark.parametrize(
    ("task_id", "cfg_name", "agent_name"),
    [
        ("Isaac-Lift-Cable-Franka-Camera", "FrankaCableCameraEnvCfg", "FrankaCableCameraPPORunnerCfg"),
        ("Isaac-Lift-Cloth-Franka-Camera", "FrankaClothCameraEnvCfg", "FrankaDeformableCameraPPORunnerCfg"),
        ("Isaac-Lift-Soft-Franka-Camera", "FrankaSoftCameraEnvCfg", "FrankaDeformableCameraPPORunnerCfg"),
    ],
)
def test_deformable_camera_task_is_registered_from_benchmark_package(task_id: str, cfg_name: str, agent_name: str):
    spec = gym.spec(task_id)

    assert spec.kwargs["env_cfg_entry_point"] == f"{_ENV_CFG_MODULE}:{cfg_name}"
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == f"{_AGENT_CFG_MODULE}:{agent_name}"

    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    assert type(env_cfg).__module__ == _ENV_CFG_MODULE
    assert type(agent_cfg).__module__ == _AGENT_CFG_MODULE
