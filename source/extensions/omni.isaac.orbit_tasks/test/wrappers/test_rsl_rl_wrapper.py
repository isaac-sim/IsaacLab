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
import torch
import unittest

import omni.usd

from omni.isaac.orbit.envs import RLTaskEnvCfg

import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper


class TestRslRlVecEnvWrapper(unittest.TestCase):
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
                env = RslRlVecEnvWrapper(env)

                # reset environment
                obs, extras = env.reset()
                # check signal
                self.assertTrue(self._check_valid_tensor(obs))
                self.assertTrue(self._check_valid_tensor(extras))

                # simulate environment for 1000 steps
                with torch.inference_mode():
                    for _ in range(1000):
                        # sample actions from -1 to 1
                        actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                        # apply actions
                        transition = env.step(actions)
                        # check signals
                        for data in transition:
                            self.assertTrue(self._check_valid_tensor(data), msg=f"Invalid data: {data}")

                # close the environment
                print(f">>> Closing environment: {task_name}")
                env.close()

    def test_no_time_outs(self):
        """Check that environments with finite horizon do not send time-out signals."""
        for task_name in self.registered_tasks:
            with self.subTest(task_name=task_name):
                print(f">>> Running test for environment: {task_name}")
                # create a new stage
                omni.usd.get_context().new_stage()
                # parse configuration
                env_cfg: RLTaskEnvCfg = parse_env_cfg(task_name, use_gpu=self.use_gpu, num_envs=self.num_envs)
                # change to finite horizon
                env_cfg.is_finite_horizon = True

                # create environment
                env = gym.make(task_name, cfg=env_cfg)
                # wrap environment
                env = RslRlVecEnvWrapper(env)

                # reset environment
                _, extras = env.reset()
                # check signal
                self.assertNotIn("time_outs", extras, msg="Time-out signal found in finite horizon environment.")

                # simulate environment for 10 steps
                with torch.inference_mode():
                    for _ in range(10):
                        # sample actions from -1 to 1
                        actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                        # apply actions
                        extras = env.step(actions)[-1]
                        # check signals
                        self.assertNotIn(
                            "time_outs", extras, msg="Time-out signal found in finite horizon environment."
                        )

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
                    valid_tensor &= TestRslRlVecEnvWrapper._check_valid_tensor(value)
                elif isinstance(value, torch.Tensor):
                    valid_tensor &= not torch.any(torch.isnan(value))
            return valid_tensor
        else:
            raise ValueError(f"Input data of invalid type: {type(data)}.")


if __name__ == "__main__":
    run_tests()
