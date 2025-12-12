# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test environments in pure headless mode without Omniverse dependencies.

This test script runs environments using the Newton physics backend in standalone mode,
without launching the Omniverse SimulationApp. This provides faster iteration and testing
without the overhead of the full Omniverse stack.

Usage:
    pytest source/isaaclab_tasks/test/test_environments_standalone.py -v

Note:
    - Vision-based environments (RGB, Depth, Camera) are skipped as they require Omniverse rendering.
    - Only environments compatible with the Newton physics backend can be tested.
"""

import logging

from isaaclab.app import AppLauncher

# Launch in pure headless mode without Omniverse
# Setting headless=True and enable_cameras=False triggers standalone mode
app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app  # Will be None in standalone mode


"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import pytest
from pxr import UsdUtils

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.spaces import sample_space
from isaaclab.sim.utils import create_new_stage_in_memory

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# import logger
logger = logging.getLogger(__name__)


# Keywords indicating vision/camera-based environments that require Omniverse
VISION_KEYWORDS = ["RGB", "Depth", "Vision", "Camera", "Visuomotor", "Theia"]

# Environments that require specific Omniverse features
OMNIVERSE_REQUIRED_ENVS = [
    "Isaac-AutoMate-Assembly-Direct-v0",
    "Isaac-AutoMate-Disassembly-Direct-v0",
]

# Environments with known issues in standalone mode
STANDALONE_SKIP_ENVS = [
    "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",  # Has issues in IS 5.0
]


def setup_environment():
    """Set up the test environment and return registered tasks compatible with standalone mode."""
    # disable interactive mode for wandb for automate environments
    os.environ["WANDB_DISABLED"] = "true"

    # acquire all Isaac environments names
    registered_tasks = []
    for task_spec in gym.registry.values():
        if "Isaac" not in task_spec.id:
            continue
        if task_spec.id.endswith("Play-v0"):
            continue
        if "Factory" in task_spec.id:
            continue

        # Skip vision-based environments (require Omniverse for rendering)
        if any(keyword in task_spec.id for keyword in VISION_KEYWORDS):
            continue

        # Skip environments that explicitly require Omniverse
        if task_spec.id in OMNIVERSE_REQUIRED_ENVS:
            continue

        # Skip environments with known issues in standalone mode
        if task_spec.id in STANDALONE_SKIP_ENVS:
            continue

        registered_tasks.append(task_spec.id)

    # sort environments by name
    registered_tasks.sort()

    return registered_tasks


@pytest.mark.order(2)
@pytest.mark.parametrize("num_envs, device", [(32, "cuda"), (1, "cuda")])
@pytest.mark.parametrize("task_name", setup_environment())
def test_environments_standalone(task_name, num_envs, device):
    """Test environments in standalone mode without Omniverse."""
    _run_environments(task_name, device, num_envs, num_steps=100)


def _run_environments(task_name, device, num_envs, num_steps):
    """Run all environments and check environments return valid signals."""

    # skip environments that cannot be run with 32 environments within reasonable VRAM
    if num_envs == 32 and task_name in [
        "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0",
        "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0",
        "Isaac-Stack-Cube-Instance-Randomize-Franka-v0",
    ]:
        pytest.skip(f"Environment {task_name} requires too much VRAM for 32 environments")

    print(f">>> Running test for environment: {task_name}")
    _check_random_actions(task_name, device, num_envs, num_steps=num_steps)
    print(f">>> Closing environment: {task_name}")
    print("-" * 80)


def _check_random_actions(task_name: str, device: str, num_envs: int, num_steps: int = 1000):
    """Run random actions and check environments returned signals are valid."""

    env = None
    try:
        # parse configuration
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)

        # skip test if the environment is a multi-agent task
        if hasattr(env_cfg, "possible_agents"):
            print(f"[INFO]: Skipping {task_name} as it is a multi-agent task")
            return

        # create environment
        env = gym.make(task_name, cfg=env_cfg)
    except Exception as e:
        if env is not None and hasattr(env, "_is_closed"):
            env.close()
        pytest.fail(f"Failed to set-up the environment for task {task_name}. Error: {e}")

    # disable control on stop (only if attribute exists)
    if hasattr(env.unwrapped, "sim") and hasattr(env.unwrapped.sim, "_app_control_on_stop_handle"):
        env.unwrapped.sim._app_control_on_stop_handle = None

    # override action space if set to inf for environments with unbounded action spaces
    if hasattr(env.unwrapped, "single_action_space"):
        for i in range(env.unwrapped.single_action_space.shape[0]):
            if env.unwrapped.single_action_space.low[i] == float("-inf"):
                env.unwrapped.single_action_space.low[i] = -1.0
            if env.unwrapped.single_action_space.high[i] == float("inf"):
                env.unwrapped.single_action_space.high[i] = 1.0

    try:
        # reset environment
        obs, _ = env.reset()
        # check signal
        assert _check_valid_tensor(obs), "Invalid observations on reset"

        # simulate environment for num_steps steps
        with torch.inference_mode():
            for _ in range(num_steps):
                # sample actions according to the defined space
                actions = sample_space(
                    env.unwrapped.single_action_space, device=env.unwrapped.device, batch_size=num_envs
                )
                # apply actions
                transition = env.step(actions)
                # check signals
                for data in transition[:-1]:  # exclude info
                    assert _check_valid_tensor(data), f"Invalid data: {data}"
    finally:
        # close the environment
        env.close()


def _check_valid_tensor(data: torch.Tensor | dict) -> bool:
    """Checks if given data does not have corrupted values.

    Args:
        data: Data buffer.

    Returns:
        True if the data is valid.
    """
    if isinstance(data, torch.Tensor):
        return not torch.any(torch.isnan(data))
    elif isinstance(data, (tuple, list)):
        return all(_check_valid_tensor(value) for value in data)
    elif isinstance(data, dict):
        return all(_check_valid_tensor(value) for value in data.values())
    else:
        raise ValueError(f"Input data of invalid type: {type(data)}.")
