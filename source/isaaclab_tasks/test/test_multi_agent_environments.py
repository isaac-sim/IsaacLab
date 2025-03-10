# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import torch
import unittest

import omni.usd

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.envs.utils.spaces import sample_space

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class TestEnvironments(unittest.TestCase):
    """Test cases for all registered multi-agent environments."""

    @classmethod
    def setUpClass(cls):
        # acquire all Isaac environments names
        cls.registered_tasks = list()
        for task_spec in gym.registry.values():
            if "Isaac" in task_spec.id and not task_spec.id.endswith("Play-v0"):
                cls.registered_tasks.append(task_spec.id)
        # sort environments by name
        cls.registered_tasks.sort()
        # print all existing task names
        print(">>> All registered environments:", cls.registered_tasks)

    """
    Test fixtures.
    """

    def test_multiple_instances_gpu(self):
        """Run all environments with multiple instances and check environments return valid signals."""
        # common parameters
        num_envs = 32
        device = "cuda"
        # iterate over all registered environments
        for task_name in self.registered_tasks:
            with self.subTest(task_name=task_name):
                print(f">>> Running test for environment: {task_name}")
                # check environment
                self._check_random_actions(task_name, device, num_envs, num_steps=100)
                # close the environment
                print(f">>> Closing environment: {task_name}")
                print("-" * 80)

    def test_single_instance_gpu(self):
        """Run all environments with single instance and check environments return valid signals."""
        # common parameters
        num_envs = 1
        device = "cuda"
        # iterate over all registered environments
        for task_name in self.registered_tasks:
            with self.subTest(task_name=task_name):
                print(f">>> Running test for environment: {task_name}")
                # check environment
                self._check_random_actions(task_name, device, num_envs, num_steps=100)
                # close the environment
                print(f">>> Closing environment: {task_name}")
                print("-" * 80)

    """
    Helper functions.
    """

    def _check_random_actions(self, task_name: str, device: str, num_envs: int, num_steps: int = 1000):
        """Run random actions and check environments return valid signals."""
        # create a new stage
        omni.usd.get_context().new_stage()
        try:
            # parse configuration
            env_cfg: DirectMARLEnvCfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)

            # skip test if the environment is not a multi-agent task
            if not hasattr(env_cfg, "possible_agents"):
                print(f"[INFO]: Skipping {task_name} as it is not a multi-agent task")
                return

            # create environment
            env: DirectMARLEnv = gym.make(task_name, cfg=env_cfg)
        except Exception as e:
            if "env" in locals() and hasattr(env, "_is_closed"):
                env.close()
            else:
                if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                    e.obj.close()
            self.fail(f"Failed to set-up the environment for task {task_name}. Error: {e}")

        # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
        # test on many environments.
        env.unwrapped.sim.set_setting("/physics/cooking/ujitsoCollisionCooking", False)

        # avoid shutdown of process on simulation stop
        env.unwrapped.sim._app_control_on_stop_handle = None

        # reset environment
        obs, _ = env.reset()
        # check signal
        self.assertTrue(self._check_valid_tensor(obs))
        # simulate environment for num_steps steps
        with torch.inference_mode():
            for _ in range(num_steps):
                # sample actions according to the defined space
                actions = {
                    agent: sample_space(
                        env.unwrapped.action_spaces[agent], device=env.unwrapped.device, batch_size=num_envs
                    )
                    for agent in env.unwrapped.possible_agents
                }
                # apply actions
                transition = env.step(actions)
                # check signals
                for item in transition[:-1]:  # exclude info
                    for agent, data in item.items():
                        self.assertTrue(self._check_valid_tensor(data), msg=f"Invalid data ('{agent}'): {data}")

        # close the environment
        env.close()

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
        elif isinstance(data, (tuple, list)):
            return all(TestEnvironments._check_valid_tensor(value) for value in data)
        elif isinstance(data, dict):
            return all(TestEnvironments._check_valid_tensor(value) for value in data.values())
        else:
            raise ValueError(f"Input data of invalid type: {type(data)}.")


if __name__ == "__main__":
    run_tests()
