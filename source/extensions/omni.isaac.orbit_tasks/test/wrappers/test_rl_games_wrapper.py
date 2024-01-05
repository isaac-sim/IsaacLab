# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import os

from omni.isaac.orbit.app import AppLauncher

# launch the simulator
app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
app_launcher = AppLauncher(headless=True, experience=app_experience)
simulation_app = app_launcher.app


"""Rest everything follows."""


import gymnasium as gym
import torch
import traceback
import unittest

import carb
import omni.usd

from omni.isaac.orbit.envs import RLTaskEnvCfg

import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rl_games import RlGamesVecEnvWrapper


class TestRlGamesVecEnvWrapper(unittest.TestCase):
    """Test that RL-Games VecEnv wrapper works as expected."""

    @classmethod
    def setUpClass(cls):
        # acquire all Isaac environments names
        cls.registered_tasks = list()
        for task_spec in gym.registry.values():
            if "Isaac" in task_spec.id:
                cls.registered_tasks.append(task_spec.id)
        # sort environments by name
        cls.registered_tasks.sort()
        # only pick the first three environments to test
        cls.registered_tasks = cls.registered_tasks[:3]
        # print all existing task names
        print(">>> All registered environments:", cls.registered_tasks)

    def setUp(self) -> None:
        # common parameters
        self.num_envs = 64
        self.use_gpu = True

    def test_random_actions(self):
        """Run random actions and check environments return valid signals."""
        for task_name in self.registered_tasks:
            print(f">>> Running test for environment: {task_name}")
            # create a new stage
            omni.usd.get_context().new_stage()
            # parse configuration
            env_cfg: RLTaskEnvCfg = parse_env_cfg(task_name, use_gpu=self.use_gpu, num_envs=self.num_envs)

            # create environment
            env = gym.make(task_name, cfg=env_cfg)
            # wrap environment
            env = RlGamesVecEnvWrapper(env, "cuda:0", 100, 100)

            # reset environment
            obs = env.reset()
            # check signal
            self.assertTrue(self._check_valid_tensor(obs))

            # simulate environment for 100 steps
            with torch.inference_mode():
                for _ in range(100):
                    # sample actions from -1 to 1
                    actions = 2 * torch.rand(env.num_envs, *env.action_space.shape, device=env.device) - 1
                    # apply actions
                    transition = env.step(actions)
                    # check signals
                    for data in transition:
                        self.assertTrue(self._check_valid_tensor(data), msg=f"Invalid data: {data}")

            # close the environment
            print(f">>> Closing environment: {task_name}")
            env.close()

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
        elif isinstance(data, dict):
            valid_tensor = True
            for value in data.values():
                if isinstance(value, dict):
                    valid_tensor &= TestRlGamesVecEnvWrapper._check_valid_tensor(value)
                elif isinstance(value, torch.Tensor):
                    valid_tensor &= not torch.any(torch.isnan(value))
            return valid_tensor
        else:
            raise ValueError(f"Input data of invalid type: {type(data)}.")


if __name__ == "__main__":
    try:
        unittest.main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
