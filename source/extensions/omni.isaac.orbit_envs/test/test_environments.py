# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import os

from omni.isaac.kit import SimulationApp

# launch the simulator
app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
config = {"headless": True}
simulation_app = SimulationApp(config, experience=app_experience)


"""Rest everything follows."""


import gym
import gym.envs
import torch
import unittest
from typing import Dict, Union

import omni.usd

import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils.parse_cfg import parse_env_cfg


class TestEnvironments(unittest.TestCase):
    """Test cases for all registered environments."""

    @classmethod
    def tearDownClass(cls):
        """Closes simulator after running all test fixtures."""
        simulation_app.close()

    def setUp(self) -> None:
        self.num_envs = 512
        self.headless = simulation_app.config["headless"]
        # acquire all Isaac environments names
        self.registered_tasks = list()
        for task_spec in gym.envs.registry.all():
            if "Isaac" in task_spec.id:
                self.registered_tasks.append(task_spec.id)
        # sort environments by name
        self.registered_tasks.sort()
        # print all existing task names
        print(">>> All registered environments:", self.registered_tasks)

    def test_random_actions(self):
        """Run random actions and check environments return valid signals."""

        for task_name in self.registered_tasks:
            print(f">>> Running test for environment: {task_name}")
            # create a new stage
            omni.usd.get_context().new_stage()
            # parse configuration
            env_cfg = parse_env_cfg(task_name, use_gpu=True, num_envs=self.num_envs)
            # create environment
            env = gym.make(task_name, cfg=env_cfg, headless=self.headless)

            # reset environment
            obs = env.reset()
            # check signal
            self.assertTrue(self._check_valid_tensor(obs))

            # simulate environment for 1000 steps
            for _ in range(1000):
                # sample actions from -1 to 1
                actions = 2 * torch.rand((env.num_envs, env.action_space.shape[0]), device=env.device) - 1
                # apply actions
                obs, rew, dones, info = env.step(actions)
                # check signals
                self.assertTrue(self._check_valid_tensor(obs))
                self.assertTrue(self._check_valid_tensor(rew))
                self.assertTrue(self._check_valid_tensor(dones))
                self.assertTrue(self._check_valid_tensor(info))

            # close the environment
            print(f">>> Closing environment: {task_name}")
            env.close()

    """
    Helper functions.
    """

    @staticmethod
    def _check_valid_tensor(data: Union[torch.Tensor, Dict]) -> bool:
        """Checks if given data does not have corrupted values.

        Args:
            data (Union[torch.Tensor, Dict]): Data buffer.

        Returns:
            bool: True if the data is valid.
        """
        if isinstance(data, torch.Tensor):
            return not torch.any(torch.isnan(data))
        elif isinstance(data, dict):
            valid_tensor = True
            for value in data.values():
                if isinstance(value, dict):
                    return TestEnvironments._check_valid_tensor(value)
                else:
                    valid_tensor = valid_tensor and not torch.any(torch.isnan(value))
            return valid_tensor
        else:
            raise ValueError(f"Input data of invalid type: {type(data)}.")


if __name__ == "__main__":
    unittest.main()
