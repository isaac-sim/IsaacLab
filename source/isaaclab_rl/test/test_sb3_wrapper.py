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
import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.fixture(scope="module")
def registered_tasks():
    # acquire all Isaac environments names
    registered_tasks = list()
    for task_spec in gym.registry.values():
        if "Isaac" in task_spec.id:
            cfg_entry_point = gym.spec(task_spec.id).kwargs.get("sb3_cfg_entry_point")
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


def test_random_actions(registered_tasks):
    """Run random actions and check environments return valid signals."""
    # common parameters
    num_envs = 64
    device = "cuda"
    for task_name in registered_tasks:
        # Use pytest's subtests
        print(f">>> Running test for environment: {task_name}")
        # create a new stage
        sim_utils.create_new_stage()
        # reset the rtx sensors carb setting to False
        get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", False)
        try:
            # parse configuration
            env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
            # create environment
            env = gym.make(task_name, cfg=env_cfg)
            # convert to single-agent instance if required by the RL algorithm
            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)
            # wrap environment
            env = Sb3VecEnvWrapper(env)
        except Exception as e:
            if "env" in locals() and hasattr(env, "_is_closed"):
                env.close()
            else:
                if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                    e.obj.close()
            pytest.fail(f"Failed to set-up the environment for task {task_name}. Error: {e}")

        # reset environment
        obs = env.reset()
        # check signal
        assert _check_valid_array(obs)

        # simulate environment for 100 steps
        with torch.inference_mode():
            for _ in range(100):
                # sample actions from -1 to 1
                actions = 2 * np.random.rand(env.num_envs, *env.action_space.shape) - 1
                # apply actions
                transition = env.step(actions)
                # check signals
                for data in transition:
                    assert _check_valid_array(data), f"Invalid data: {data}"

        # close the environment
        print(f">>> Closing environment: {task_name}")
        env.close()


def test_unbounded_action_space_uses_normalized_wrapper_bounds():
    """Expose normalized bounds to SB3 without modifying the underlying environment."""
    env_cfg = parse_env_cfg("Isaac-Cartpole", device="cuda", num_envs=1)
    env = gym.make("Isaac-Cartpole", cfg=env_cfg)
    try:
        assert isinstance(env.unwrapped.single_action_space, gym.spaces.Box)
        assert not env.unwrapped.single_action_space.is_bounded("both")

        wrapped_env = Sb3VecEnvWrapper(env)

        np.testing.assert_array_equal(wrapped_env.action_space.low, -1.0)
        np.testing.assert_array_equal(wrapped_env.action_space.high, 1.0)
        assert not env.unwrapped.single_action_space.is_bounded("both")
    finally:
        env.close()


def test_direct_environment_remains_unbounded():
    """Do not impose the SB3 compatibility bounds on the underlying direct environment."""
    env_cfg = parse_env_cfg("Isaac-Cartpole-Direct", device="cuda", num_envs=1)
    env = gym.make("Isaac-Cartpole-Direct", cfg=env_cfg)
    try:
        assert isinstance(env.unwrapped.single_action_space, gym.spaces.Box)
        assert not env.unwrapped.single_action_space.is_bounded("both")

        env.reset()
        env.step(torch.tensor([[2.0]], device="cuda"))

        torch.testing.assert_close(env.unwrapped.actions, torch.tensor([[200.0]], device="cuda"))
    finally:
        env.close()


def test_custom_unbounded_action_bounds():
    """Allow policies to select a different finite domain for an unbounded environment."""
    env_cfg = parse_env_cfg("Isaac-Cartpole", device="cuda", num_envs=1)
    env = gym.make("Isaac-Cartpole", cfg=env_cfg)
    try:
        wrapped_env = Sb3VecEnvWrapper(env, unbounded_action_bounds=(-2.0, 3.0))

        np.testing.assert_array_equal(wrapped_env.action_space.low, -2.0)
        np.testing.assert_array_equal(wrapped_env.action_space.high, 3.0)
    finally:
        env.close()


def test_invalid_unbounded_action_bounds_are_rejected():
    """Reject invalid compatibility bounds before constructing the SB3 wrapper."""
    env_cfg = parse_env_cfg("Isaac-Cartpole", device="cuda", num_envs=1)
    env = gym.make("Isaac-Cartpole", cfg=env_cfg)
    try:
        for bounds in ((1.0, -1.0), (0.0, 0.0), (-np.inf, 1.0), (-1.0, np.inf)):
            with pytest.raises(ValueError, match="Invalid unbounded action bounds"):
                Sb3VecEnvWrapper(env, unbounded_action_bounds=bounds)
    finally:
        env.close()


"""
Helper functions.
"""


@staticmethod
def _check_valid_array(data: np.ndarray | dict | list) -> bool:
    """Checks if given data does not have corrupted values.

    Args:
        data: Data buffer.

    Returns:
        True if the data is valid.
    """
    if isinstance(data, np.ndarray):
        return not np.any(np.isnan(data))
    elif isinstance(data, dict):
        valid_array = True
        for value in data.values():
            if isinstance(value, dict):
                valid_array &= _check_valid_array(value)
            elif isinstance(value, np.ndarray):
                valid_array &= not np.any(np.isnan(value))
        return valid_array
    elif isinstance(data, list):
        valid_array = True
        for value in data:
            valid_array &= _check_valid_array(value)
        return valid_array
    else:
        raise ValueError(f"Input data of invalid type: {type(data)}.")
