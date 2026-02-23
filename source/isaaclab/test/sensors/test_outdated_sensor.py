# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
simulation_app = AppLauncher(headless=True, enable_cameras=True).app


"""Rest everything follows."""

import shutil
import tempfile

import gymnasium as gym
import pytest
import torch

import carb

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.fixture()
def temp_dir():
    """Fixture to create and clean up a temporary directory for test datasets."""
    # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
    # test on many environments.
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    # create a temporary directory to store the test datasets
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # delete the temporary directory after the test
    shutil.rmtree(temp_dir)


@pytest.mark.parametrize("task_name", ["Isaac-Stack-Cube-Franka-IK-Rel-v0"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.isaacsim_ci
def test_action_state_recorder_terms(temp_dir, task_name, device, num_envs):
    """Check FrameTransformer values after reset."""
    sim_utils.create_new_stage()

    # parse configuration
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    env_cfg.wait_for_textures = False

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
    torch.testing.assert_close(pre_reset_eef_pos, post_reset_eef_pos, atol=1e-5, rtol=1e-3)

    # close the environment
    env.close()
