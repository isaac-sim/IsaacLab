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

from collections.abc import Iterator
from contextlib import contextmanager

import gymnasium as gym
import torch
from tensordict import TensorDict

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

_TASK_IDS = (
    "Isaac-Ant",
    "Isaac-Ant-Direct",
    "Isaac-Cartpole",
    "Isaac-Cartpole-Camera",
)


@contextmanager
def _make_env(task_id: str, *, num_envs: int, finite_horizon: bool | None = None) -> Iterator[RslRlVecEnvWrapper]:
    """Create and close an RSL-RL environment."""
    sim_utils.create_new_stage()

    env_cfg = parse_env_cfg(task_id, device="cuda", num_envs=num_envs)
    if finite_horizon is not None:
        env_cfg.is_finite_horizon = finite_horizon
    env = gym.make(task_id, cfg=env_cfg)
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


def test_random_actions():
    """Run random actions and check environments return valid signals."""
    for task_id in _TASK_IDS:
        print(f">>> Running test for environment: {task_id}")
        with _make_env(task_id, num_envs=64) as env:
            observations, extras = env.reset()
            assert _has_no_nan(observations)
            assert _has_no_nan(extras)

            with torch.inference_mode():
                for _ in range(10):
                    actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                    for data in env.step(actions):
                        assert _has_no_nan(data), f"Invalid data: {data}"


def test_no_time_outs():
    """Check that environments with finite horizon do not send time-out signals."""
    # The time-out contract belongs to the wrapper, so two environments are sufficient.
    for task_id in _TASK_IDS[:2]:
        print(f">>> Running test for environment: {task_id}")
        with _make_env(task_id, num_envs=64, finite_horizon=True) as env:
            _, extras = env.reset()
            assert "time_outs" not in extras, "Time-out signal found in finite horizon environment."

            with torch.inference_mode():
                for _ in range(10):
                    actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                    extras = env.step(actions)[-1]
                    assert "time_outs" not in extras, "Time-out signal found in finite horizon environment."


def _has_no_nan(data: torch.Tensor | TensorDict | dict[str, object]) -> bool:
    """Check that all tensors in the data structure contain no NaNs.

    Args:
        data: Tensor or nested mapping of tensors.

    Returns:
        True if none of the tensors contain NaNs.
    """
    if isinstance(data, (torch.Tensor, TensorDict)):
        return not data.isnan().any()
    if isinstance(data, dict):
        tensor_values = (value for value in data.values() if isinstance(value, (torch.Tensor, TensorDict, dict)))
        return all(_has_no_nan(value) for value in tensor_values)
    raise TypeError(f"Unsupported data type: {type(data)}.")
