# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import torch
import unittest

import carb
import omni.usd

from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg


class TestEnvironments(unittest.TestCase):
    """Test cases for all registered environments."""

    @classmethod
    def setUpClass(cls):
        # acquire all Isaac environments names
        cls.registered_tasks = list()
        for task_spec in gym.registry.values():
            if "Isaac" in task_spec.id and not task_spec.id.endswith("Play-v0"):
                cls.registered_tasks.append(task_spec.id)
        # sort environments by name
        cls.registered_tasks.sort()

        # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
        # test on many environments.
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    """
    Test fixtures.
    """

    def test_multiple_num_envs_on_gpu(self):
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

    def test_single_env_on_gpu(self):
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

    def test_deterministic_env_creation(self):
        """Check deterministic environment creation by setting a seed."""
        # common parameters
        num_envs = 128
        # we check this only for anymal environment since it has many randomization events
        task_name = "Isaac-Velocity-Flat-Anymal-C-v0"

        # create the environment once
        for device in ["cuda", "cpu"]:
            for seed in [25, 8001]:
                with self.subTest(device=device, seed=seed):
                    # fix number of steps
                    num_steps = 600
                    # call function to create and step the environment
                    masses_1, obs_1 = self._compute_masses_and_first_obs(task_name, seed, num_envs, device, num_steps)
                    masses_2, obs_2 = self._compute_masses_and_first_obs(task_name, seed, num_envs, device, num_steps)
                    masses_3, obs_3 = self._compute_masses_and_first_obs(
                        task_name, seed * 2, num_envs, device, num_steps
                    )

                    # check everything is as expected
                    # -- masses should be the same
                    torch.testing.assert_close(masses_1, masses_2)
                    self.assertFalse(torch.allclose(masses_1, masses_3))
                    # -- observations should be the same
                    for key in obs_1.keys():
                        torch.testing.assert_close(obs_1[key], obs_2[key])
                        self.assertFalse(torch.allclose(obs_1[key], obs_3[key]))

    """
    Helper functions.
    """

    def _check_random_actions(self, task_name: str, device: str, num_envs: int, num_steps: int = 1000):
        """Run random actions and check environments returned signals are valid."""
        # create a new stage
        omni.usd.get_context().new_stage()
        # parse configuration
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
        # create environment
        env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg)

        # disable control on stop
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

        # reset environment
        obs, _ = env.reset()
        # check signal
        self.assertTrue(self._check_valid_tensor(obs))
        # simulate environment for num_steps steps
        with torch.inference_mode():
            for _ in range(num_steps):
                # sample actions from -1 to 1
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                # apply actions
                transition = env.step(actions)
                # check signals
                for data in transition:
                    self.assertTrue(self._check_valid_tensor(data), msg=f"Invalid data: {data}")

        # close the environment
        env.close()

    def _compute_masses_and_first_obs(
        self, task_name: str, seed: int, num_envs: int, device: str, num_steps: int
    ) -> tuple[torch.Tensor, dict]:
        """Run random actions and obtain first observations and simulation information."""
        # create a new stage
        omni.usd.get_context().new_stage()
        # parse configuration
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
        # set seed
        env_cfg.seed = seed

        # create environment
        env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg)

        # disable control on stop
        env.sim._app_control_on_stop_handle = None  # type: ignore

        # obtain the mass of the robot after initialization
        masses = env.scene.articulations["robot"].root_physx_view.get_masses().clone()

        # reset environment
        obs, _ = env.reset()
        # simulate environment for 10 steps
        with torch.inference_mode():
            for _ in range(num_steps):
                # sample actions from -1 to 1
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                # apply actions and get initial observation
                obs = env.step(actions)[0]

        # close the environment
        env.close()

        return masses, obs

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
                    valid_tensor &= TestEnvironments._check_valid_tensor(value)
                elif isinstance(value, torch.Tensor):
                    valid_tensor &= not torch.any(torch.isnan(value))
            return valid_tensor
        else:
            raise ValueError(f"Input data of invalid type: {type(data)}.")


if __name__ == "__main__":
    run_tests()
