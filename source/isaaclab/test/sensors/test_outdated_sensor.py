# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
if not AppLauncher.instance():
    simulation_app = AppLauncher(headless=True, enable_cameras=True).app
elif AppLauncher.instance() and AppLauncher.instance()._enable_cameras is False:
    # FIXME: workaround as AppLauncher instance can currently not be closed without terminating the test
    raise ValueError("AppLauncher instance exists but enable_cameras is False")


"""Rest everything follows."""

import gymnasium as gym
import shutil
import tempfile
import torch

import carb
import omni.usd

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import pytest

@pytest.fixture()  # scope="module"
def setup_carb_settings():
    """Fixture to set up carb settings."""
    # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
    # test on many environments.
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    return carb_settings_iface

@pytest.fixture()  # scope="function"
def temp_dir():
    """Fixture to create and clean up a temporary directory for test datasets."""
    # create a temporary directory to store the test datasets
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # delete the temporary directory after the test
    shutil.rmtree(temp_dir)

@pytest.mark.parametrize("task_name", ["Isaac-Stack-Cube-Franka-IK-Rel-v0"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 2])
def test_action_state_recorder_terms(setup_carb_settings, temp_dir, task_name, device, num_envs):
    """Check FrameTransformer values after reset."""
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
    assert torch.all(torch.isclose(pre_reset_eef_pos, post_reset_eef_pos))

    # close the environment
    env.close()
