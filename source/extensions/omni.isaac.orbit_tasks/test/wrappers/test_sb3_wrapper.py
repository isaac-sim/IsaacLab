# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


from omni.isaac.orbit.app import AppLauncher, run_tests

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
import unittest

import omni.usd

from omni.isaac.orbit.envs import RLTaskEnvCfg

import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper


class TestStableBaselines3VecEnvWrapper(unittest.TestCase):
    """Test that RSL-RL VecEnv wrapper works as expected."""

    @classmethod
    def setUpClass(cls):
        # acquire all Isaac environments names
        cls.registered_tasks = list()
        for task_spec in gym.registry.values():
            if "Isaac" in task_spec.id:
                cls.registered_tasks.append(task_spec.id)
        # sort environments by name
        cls.registered_tasks.sort()
        # only pick the first four environments to test
        cls.registered_tasks = cls.registered_tasks[:4]
        # print all existing task names
        print(">>> All registered environments:", cls.registered_tasks)

    def setUp(self) -> None:
        # common parameters
        self.num_envs = 64
        self.use_gpu = True

    def test_random_actions(self):
        """Run random actions and check environments return valid signals."""
        for task_name in self.registered_tasks:
            with self.subTest(task_name=task_name):
                print(f">>> Running test for environment: {task_name}")
                # create a new stage
                omni.usd.get_context().new_stage()
                # parse configuration
                env_cfg: RLTaskEnvCfg = parse_env_cfg(task_name, use_gpu=self.use_gpu, num_envs=self.num_envs)

                # create environment
                env = gym.make(task_name, cfg=env_cfg)
                # wrap environment
                env = Sb3VecEnvWrapper(env)

                # reset environment
                obs = env.reset()
                # check signal
                self.assertTrue(self._check_valid_array(obs))

                # simulate environment for 1000 steps
                with torch.inference_mode():
                    for _ in range(1000):
                        # sample actions from -1 to 1
                        actions = 2 * np.random.rand(env.num_envs, *env.action_space.shape) - 1
                        # apply actions
                        transition = env.step(actions)
                        # check signals
                        for data in transition:
                            self.assertTrue(self._check_valid_array(data), msg=f"Invalid data: {data}")

                # close the environment
                print(f">>> Closing environment: {task_name}")
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
                    valid_array &= TestStableBaselines3VecEnvWrapper._check_valid_array(value)
                elif isinstance(value, np.ndarray):
                    valid_array &= not np.any(np.isnan(value))
            return valid_array
        elif isinstance(data, list):
            valid_array = True
            for value in data:
                valid_array &= TestStableBaselines3VecEnvWrapper._check_valid_array(value)
            return valid_array
        else:
            raise ValueError(f"Input data of invalid type: {type(data)}.")


if __name__ == "__main__":
    run_tests()
