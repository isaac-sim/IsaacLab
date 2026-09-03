# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

pytest.importorskip("torchrl")

from torchrl.envs.utils import check_env_specs  # noqa: E402

from isaaclab_rl.torchrl import IsaacLabTorchRLWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


@pytest.fixture(scope="module")
def registered_tasks():
    registered_tasks = list()
    for task_spec in gym.registry.values():
        if "Isaac" in task_spec.id:
            cfg_entry_point = gym.spec(task_spec.id).kwargs.get("rsl_rl_cfg_entry_point")
            if cfg_entry_point is not None:
                registered_tasks.append(task_spec.id)
    registered_tasks.sort()
    registered_tasks = registered_tasks[:5]

    # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
    # test on many environments.
    get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    print(">>> All registered environments:", registered_tasks)
    return registered_tasks


def _make_env(task_name: str, num_envs: int, device: str, **cfg_overrides) -> IsaacLabTorchRLWrapper:
    sim_utils.create_new_stage()
    get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", False)
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    for key, value in cfg_overrides.items():
        setattr(env_cfg, key, value)
    env = gym.make(task_name, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    return IsaacLabTorchRLWrapper(env)


def test_random_actions(registered_tasks):
    """Roll out random actions through TorchRL's own step/reset machinery and check the signals."""
    num_envs = 64
    device = "cuda"
    for task_name in registered_tasks:
        print(f">>> Running test for environment: {task_name}")
        try:
            # terminal observations are needed for the "next" observation of done rows to be finite
            env = _make_env(task_name, num_envs, device, compute_final_obs=True)
        except Exception as e:
            pytest.fail(f"Failed to set-up the environment for task {task_name}. Error: {e}")

        check_env_specs(env)

        with torch.inference_mode():
            rollout = env.rollout(100, break_when_any_done=False)

        assert rollout.batch_size == torch.Size([num_envs, 100])
        assert not rollout.isnan().any(), f"NaN found in rollout for task {task_name}"
        assert torch.equal(rollout["next", "done"], rollout["next", "terminated"] | rollout["next", "truncated"])

        print(f">>> Closing environment: {task_name}")
        env.close()


def test_finite_horizon_reports_time_outs_as_terminal(registered_tasks):
    """Finite-horizon tasks must not surface time-outs, so value estimators do not bootstrap past them."""
    num_envs = 64
    device = "cuda"
    for task_name in registered_tasks:
        print(f">>> Running test for environment: {task_name}")
        env = _make_env(task_name, num_envs, device, is_finite_horizon=True)

        with torch.inference_mode():
            rollout = env.rollout(10, break_when_any_done=False)

        assert not rollout["next", "truncated"].any(), "Time-out signal found in finite horizon environment."

        print(f">>> Closing environment: {task_name}")
        env.close()
