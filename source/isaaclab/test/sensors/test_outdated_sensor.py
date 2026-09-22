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

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

pytestmark = pytest.mark.integration


@pytest.fixture()
def temp_dir():
    """Fixture to create and clean up a temporary directory for test datasets."""
    # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
    # test on many environments.
    get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    # create a temporary directory to store the test datasets
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # delete the temporary directory after the test
    shutil.rmtree(temp_dir)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_frame_transformer_observation_fresh_after_reset(temp_dir, device):
    """The end-effector observation reported by ``reset()`` matches the first idle step, i.e. is not stale."""
    task_name = "IsaacContrib-Stack-Cube-Franka-IK-Rel"
    sim_utils.create_new_stage()
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=2)
    env_cfg.wait_for_textures = False
    env = gym.make(task_name, cfg=env_cfg)
    env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

    obs = env.reset()[0]
    pre_reset_eef_pos = obs["policy"]["eef_pos"].clone()
    idle_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    obs = env.step(idle_actions)[0]
    torch.testing.assert_close(pre_reset_eef_pos, obs["policy"]["eef_pos"], atol=1e-5, rtol=1e-3)
    env.close()
