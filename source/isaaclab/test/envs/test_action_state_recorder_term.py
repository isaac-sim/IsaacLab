# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks that the action-state recorder terms capture the scene state on reset.

The core scenario builds a minimal manager-based environment directly. The task-backed scenario drives the
same recorders through a registered task and the Gym wrapper; it stays in the core test tree until the Franka
task can move to the task package.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import uuid

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.test.env_cfgs import make_empty_manager_based_env_cfg
from isaaclab.test.integration_scene_cfgs import ArticulationRigidObjectSceneCfg

pytestmark = pytest.mark.integration

_TASK_NAME = "IsaacContrib-Lift-Cube-Franka"


@pytest.fixture(scope="session", autouse=True)
def setup_carb_settings():
    """Set up settings to prevent simulation getting stuck."""
    get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)


def _make_core_env(device: str, num_envs: int, recorders: ActionStateRecorderManagerCfg) -> ManagerBasedEnv:
    env_cfg = make_empty_manager_based_env_cfg(device=device, num_envs=num_envs)
    env_cfg.scene = ArticulationRigidObjectSceneCfg(num_envs=num_envs, env_spacing=2.5)
    env_cfg.recorders = recorders
    return ManagerBasedEnv(cfg=env_cfg)


def _make_task_env(device: str, num_envs: int, recorders: ActionStateRecorderManagerCfg) -> ManagerBasedEnv:
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("isaaclab_tasks")
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(_TASK_NAME, device=device, num_envs=num_envs)
    env_cfg.recorders = recorders
    return gym.make(_TASK_NAME, cfg=env_cfg).unwrapped


def _check_initial_state_recorder_term(env: ManagerBasedEnv):
    """The recorded initial state of every environment matches the current relative scene state."""
    current_state = env.scene.get_state(is_relative=True)
    for env_id in range(env.num_envs):
        recorded_state = env.recorder_manager.get_episode(env_id).get_initial_state()
        for asset_type in ("articulation", "rigid_object"):
            for asset_name, asset_state in current_state[asset_type].items():
                for state_name, runtime_state in asset_state.items():
                    torch.testing.assert_close(
                        recorded_state[asset_type][asset_name][state_name][0].to(runtime_state.device),
                        runtime_state[env_id],
                        atol=0.01,
                        rtol=0.0,
                        msg=f"State [{asset_type}][{asset_name}][{state_name}] of env {env_id} does not match",
                    )


@pytest.mark.parametrize(
    ("make_env", "device", "num_envs"),
    [
        pytest.param(_make_core_env, "cuda:0", 1, id="core-cuda-1"),
        pytest.param(_make_core_env, "cuda:0", 2, id="core-cuda-2"),
        pytest.param(_make_core_env, "cpu", 2, id="core-cpu-2"),
        pytest.param(_make_task_env, "cuda:0", 2, id="task-cuda-2"),
    ],
)
def test_action_state_recorder_terms(make_env, device, num_envs, tmp_path):
    """The initial state is recorded on a full reset and on a partial reset of the last environment."""
    sim_utils.create_new_stage()
    recorders = ActionStateRecorderManagerCfg()
    recorders.dataset_export_dir_path = str(tmp_path)
    recorders.dataset_filename = f"{uuid.uuid4()}.hdf5"
    env = make_env(device, num_envs, recorders)

    env.reset()
    _check_initial_state_recorder_term(env)

    env.reset(env_ids=torch.tensor([num_envs - 1], device=env.device))
    _check_initial_state_recorder_term(env)

    env.close()
