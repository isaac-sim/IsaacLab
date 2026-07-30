# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Local imports should be imported last
from env_test_utils import setup_environment  # isort: skip


pytestmark = pytest.mark.isaacsim_ci


@pytest.fixture(scope="function")
def setup_video_params():
    # common parameters
    num_envs = 16
    device = "cuda"
    video_length = 200
    return num_envs, device, video_length


@pytest.mark.parametrize("task_name", setup_environment(include_play=True, tier="core"))
def test_record_video(task_name, setup_video_params):
    """Run random actions agent with internal VideoRecorder capturing from the active visualizer."""
    num_envs, device, video_length = setup_video_params
    videos_dir = os.path.join(os.path.dirname(__file__), "output", "videos", "train")
    task_videos_dir = os.path.join(videos_dir, task_name)
    # create a new stage
    sim_utils.create_new_stage()

    # parse configuration
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    # attach an internal video recorder — captures from the active visualizer viewport
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="visualizer",
            output_dir=task_videos_dir,
            video_length=video_length,
            video_interval=0,
        )
    ]

    # create environment (no render_mode needed; recorder is driven internally)
    env = gym.make(task_name, cfg=env_cfg)

    # reset environment
    env.reset()
    # simulate environment
    with torch.inference_mode():
        raw_env = env.unwrapped
        device = raw_env.device
        # MARL envs expose action_space as a callable (agent → space) and expect a
        # dict of per-agent action tensors.  Single-agent envs expose a Space directly.
        is_marl = callable(env.action_space) and hasattr(raw_env, "action_spaces")
        for _ in range(500):
            if is_marl:
                actions = {
                    agent: 2 * torch.rand((raw_env.num_envs, *raw_env.action_spaces[agent].shape), device=device) - 1
                    for agent in raw_env.possible_agents
                }
            else:
                actions = 2 * torch.rand(env.action_space.shape, device=device) - 1
            _ = env.step(actions)

    # close flushes any open clip to disk
    env.close()
