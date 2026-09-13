# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

from contextlib import contextmanager

import gymnasium as gym
import pytest
import torch
from tensordict import TensorDict

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.fixture(scope="module")
def registered_tasks():
    # acquire all Isaac environments names
    registered_tasks = list()
    for task_spec in gym.registry.values():
        if "Isaac" in task_spec.id:
            cfg_entry_point = gym.spec(task_spec.id).kwargs.get("rsl_rl_cfg_entry_point")
            if cfg_entry_point is not None:
                registered_tasks.append(task_spec.id)
    # sort environments by name
    registered_tasks.sort()
    registered_tasks = registered_tasks[:5]

    # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
    # test on many environments.
    get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    # print all existing task names
    print(">>> All registered environments:", registered_tasks)
    return registered_tasks


@contextmanager
def _make_env(task_name: str, *, num_envs: int, finite_horizon: bool = False):
    """Create and close an RSL-RL environment."""
    sim_utils.create_new_stage()
    get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", False)

    env_cfg = parse_env_cfg(task_name, device="cuda", num_envs=num_envs)
    env_cfg.is_finite_horizon = finite_horizon
    env = gym.make(task_name, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        yield RslRlVecEnvWrapper(env)
    finally:
        env.close()


def test_get_observations():
    """Return the current observations from a real environment."""
    with _make_env("Isaac-Cartpole", num_envs=2) as env:
        observations, _ = env.reset()
        torch.testing.assert_close(env.get_observations()["policy"], observations["policy"])

        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        observations = env.step(actions)[0]
        torch.testing.assert_close(env.get_observations()["policy"], observations["policy"])


def test_random_actions(registered_tasks):
    """Run random actions and check environments return valid signals."""
    for task_name in registered_tasks:
        print(f">>> Running test for environment: {task_name}")
        with _make_env(task_name, num_envs=64) as env:
            obs, extras = env.reset()
            assert _check_valid_tensor(obs)
            assert _check_valid_tensor(extras)

            with torch.inference_mode():
                for _ in range(100):
                    actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                    for data in env.step(actions):
                        assert _check_valid_tensor(data), f"Invalid data: {data}"


def test_no_time_outs(registered_tasks):
    """Check that environments with finite horizon do not send time-out signals."""
    # The time-out contract belongs to the wrapper, so two environments are sufficient.
    for task_name in registered_tasks[:2]:
        print(f">>> Running test for environment: {task_name}")
        with _make_env(task_name, num_envs=64, finite_horizon=True) as env:
            _, extras = env.reset()
            assert "time_outs" not in extras, "Time-out signal found in finite horizon environment."

            with torch.inference_mode():
                for _ in range(10):
                    actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                    extras = env.step(actions)[-1]
                    assert "time_outs" not in extras, "Time-out signal found in finite horizon environment."


"""
Helper functions.
"""


@staticmethod
def _check_valid_tensor(data: torch.Tensor | dict) -> bool:
    """Checks if given data does not have corrupted values.

    Args:
        data: Data buffer.

    Returns:
        True if the data is valid.
    """
    if isinstance(data, torch.Tensor):
        return not torch.any(torch.isnan(data))
    elif isinstance(data, TensorDict):
        return not data.isnan().any()
    elif isinstance(data, dict):
        valid_tensor = True
        for value in data.values():
            if isinstance(value, dict):
                valid_tensor &= _check_valid_tensor(value)
            elif isinstance(value, torch.Tensor):
                valid_tensor &= not torch.any(torch.isnan(value))
        return valid_tensor
    else:
        raise ValueError(f"Input data of invalid type: {type(data)}.")
