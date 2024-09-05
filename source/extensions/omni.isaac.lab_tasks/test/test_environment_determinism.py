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


class TestEnvironmentDeterminism(unittest.TestCase):
    """Test cases for environment determinism."""

    @classmethod
    def setUpClass(cls):
        # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
        # test on many environments.
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    """
    Test fixtures.
    """

    def test_manipulation_env_determinism(self):
        """Check deterministic environment creation for manipulation."""
        for task_name in [
            "Isaac-Open-Drawer-Franka-v0",
            "Isaac-Lift-Cube-Franka-v0",
        ]:
            for device in ["cuda", "cpu"]:
                for seed in [25, 8001]:
                    with self.subTest(task_name=task_name, device=device, seed=seed):
                        # fix number of steps
                        num_envs = 128
                        num_steps = 600
                        # call function to create and step the environment
                        obs_1, rew_1 = self._obtain_transition_tuples(task_name, seed, num_envs, device, num_steps)
                        obs_2, rew_2 = self._obtain_transition_tuples(task_name, seed, num_envs, device, num_steps)
                        obs_3, rew_3 = self._obtain_transition_tuples(task_name, seed * 2, num_envs, device, num_steps)

                        # check everything is as expected
                        # -- rewards should be the same
                        torch.testing.assert_close(rew_1, rew_2)
                        self.assertFalse(torch.allclose(rew_1, rew_3))
                        # -- observations should be the same
                        for key in obs_1.keys():
                            torch.testing.assert_close(obs_1[key], obs_2[key])
                            self.assertFalse(torch.allclose(obs_1[key], obs_3[key]))

    def test_locomotion_env_determinism(self):
        """Check deterministic environment creation for locomotion."""
        for task_name in [
            "Isaac-Ant-v0",
            "Isaac-Velocity-Flat-Anymal-C-v0",
            "Isaac-Velocity-Rough-Anymal-C-v0",
            "Isaac-Velocity-Flat-H1-v0",
        ]:
            for device in ["cuda", "cpu"]:
                for seed in [25, 8001]:
                    with self.subTest(task_name=task_name, device=device, seed=seed):
                        # fix number of steps
                        num_envs = 128
                        num_steps = 600
                        # call function to create and step the environment
                        obs_1, rew_1 = self._obtain_transition_tuples(task_name, seed, num_envs, device, num_steps)
                        obs_2, rew_2 = self._obtain_transition_tuples(task_name, seed, num_envs, device, num_steps)
                        obs_3, rew_3 = self._obtain_transition_tuples(task_name, seed * 2, num_envs, device, num_steps)

                        # check everything is as expected
                        # -- rewards should be the same
                        torch.testing.assert_close(rew_1, rew_2)
                        self.assertFalse(torch.allclose(rew_1, rew_3))
                        # -- observations should be the same
                        for key in obs_1.keys():
                            torch.testing.assert_close(obs_1[key], obs_2[key])
                            self.assertFalse(torch.allclose(obs_1[key], obs_3[key]))

    def test_dextrous_env_determinism(self):
        """Check deterministic environment creation for dextrous manipulation."""
        for task_name in [
            "Isaac-Repose-Cube-Allegro-v0",
            "Isaac-Repose-Cube-Allegro-Direct-v0",
        ]:
            for device in ["cuda", "cpu"]:
                for seed in [25, 8001]:
                    with self.subTest(task_name=task_name, device=device, seed=seed):
                        # fix number of steps
                        num_envs = 128
                        num_steps = 600
                        # call function to create and step the environment
                        obs_1, rew_1 = self._obtain_transition_tuples(task_name, seed, num_envs, device, num_steps)
                        obs_2, rew_2 = self._obtain_transition_tuples(task_name, seed, num_envs, device, num_steps)
                        obs_3, rew_3 = self._obtain_transition_tuples(task_name, seed * 2, num_envs, device, num_steps)

                        # check everything is as expected
                        # -- rewards should be the same
                        torch.testing.assert_close(rew_1, rew_2)
                        self.assertFalse(torch.allclose(rew_1, rew_3))
                        # -- observations should be the same
                        for key in obs_1.keys():
                            torch.testing.assert_close(obs_1[key], obs_2[key])
                            self.assertFalse(torch.allclose(obs_1[key], obs_3[key]))

    """
    Helper functions.
    """

    def _obtain_transition_tuples(
        self, task_name: str, seed: int, num_envs: int, device: str, num_steps: int
    ) -> tuple[dict, torch.Tensor]:
        """Run random actions and obtain transition tuples after fixed number of steps."""
        # create a new stage
        omni.usd.get_context().new_stage()
        # parse configuration
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
        # set seed
        env_cfg.set_seed(seed)

        # create environment
        env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg)

        # disable control on stop
        env.sim._app_control_on_stop_handle = None  # type: ignore

        # reset environment
        obs, _ = env.reset()
        # simulate environment for 10 steps
        with torch.inference_mode():
            for _ in range(num_steps):
                # sample actions from -1 to 1
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                # apply actions and get initial observation
                obs, rewards = env.step(actions)[:2]

        # close the environment
        env.close()

        return obs, rewards


if __name__ == "__main__":
    run_tests()
