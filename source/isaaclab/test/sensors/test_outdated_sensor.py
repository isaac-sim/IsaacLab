# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
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
import shutil
import tempfile
import torch
import unittest

import carb
import omni.usd

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class TestFrameTransformerAfterReset(unittest.TestCase):
    """Test cases for checking FrameTransformer values after reset."""

    @classmethod
    def setUpClass(cls):
        # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
        # test on many environments.
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    def setUp(self):
        # create a temporary directory to store the test datasets
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # delete the temporary directory after the test
        shutil.rmtree(self.temp_dir)

    def test_action_state_reocrder_terms(self):
        """Check FrameTransformer values after reset."""
        for task_name in ["Isaac-Stack-Cube-Franka-IK-Rel-v0"]:
            for device in ["cuda:0", "cpu"]:
                for num_envs in [1, 2]:
                    with self.subTest(task_name=task_name, device=device):
                        omni.usd.get_context().new_stage()

                        # parse configuration
                        env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)

                        # create environment
                        env = gym.make(task_name, cfg=env_cfg)

                        # disable control on stop
                        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

                        # reset environment
                        obs = env.reset()[0]

                        # get the end effector position after the reset
                        pre_reset_eef_pos = obs["policy"]["eef_pos"].clone()
                        print(pre_reset_eef_pos)

                        # step the environment with idle actions
                        idle_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                        obs = env.step(idle_actions)[0]

                        # get the end effector position after the first step
                        post_reset_eef_pos = obs["policy"]["eef_pos"]
                        print(post_reset_eef_pos)

                        # check if the end effector position is the same after the reset and the first step
                        print(torch.all(torch.isclose(pre_reset_eef_pos, post_reset_eef_pos)))
                        self.assertTrue(torch.all(torch.isclose(pre_reset_eef_pos, post_reset_eef_pos)))

                        # close the environment
                        env.close()


if __name__ == "__main__":
    run_tests()
