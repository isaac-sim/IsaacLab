# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import omni.usd
import pytest
from env_test_utils import setup_environment

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


@pytest.fixture(scope="function")
def setup_video_params():
    # common parameters
    num_envs = 16
    device = "cuda"
    # video parameters
    step_trigger = lambda step: step % 225 == 0  # noqa: E731
    video_length = 200
    return num_envs, device, step_trigger, video_length


@pytest.mark.parametrize("task_name", setup_environment(include_play=True))
def test_record_video(task_name, setup_video_params):
    """Run random actions agent with recording of videos."""
    num_envs, device, step_trigger, video_length = setup_video_params
    videos_dir = os.path.join(os.path.dirname(__file__), "output", "videos", "train")
    # create a new stage
    omni.usd.get_context().new_stage()

    # parse configuration
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)

    # create environment
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")

    # directory to save videos
    task_videos_dir = os.path.join(videos_dir, task_name)
    # wrap environment to record videos
    env = gym.wrappers.RecordVideo(
        env,
        task_videos_dir,
        step_trigger=step_trigger,
        video_length=video_length,
        disable_logger=True,
    )

    # reset environment
    env.reset()
    # simulate environment
    with torch.inference_mode():
        for _ in range(500):
            # compute zero actions
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            _ = env.step(actions)

    # close the simulator
    env.close()
