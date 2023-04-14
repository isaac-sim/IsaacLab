# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import os

from omni.isaac.kit import SimulationApp

# launch the simulator
app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.render.kit"
config = {"headless": True}
simulation_app = SimulationApp(config, experience=app_experience)

"""Rest everything follows."""


import gym
import os
import torch
import unittest

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg


class TestRecordVideoWrapper(unittest.TestCase):
    """Test recording videos using the RecordVideo wrapper."""

    @classmethod
    def tearDownClass(cls):
        """Closes simulator after running all test fixtures."""
        simulation_app.close()

    def setUp(self) -> None:
        # common parameters
        self.num_envs = 64
        self.use_gpu = True
        self.headless = simulation_app.config["headless"]
        # directory to save videos
        self.videos_dir = os.path.join(os.path.dirname(__file__), "videos")
        self.step_trigger = lambda step: step % 225 == 0
        self.video_length = 200
        # acquire all Isaac environments names
        self.registered_tasks = list()
        for task_spec in gym.envs.registry.all():
            if "Isaac" in task_spec.id:
                self.registered_tasks.append(task_spec.id)
        # sort environments by name
        self.registered_tasks.sort()
        # print all existing task names
        print(">>> All registered environments:", self.registered_tasks)

    def test_record_video(self):
        """Run random actions agent with recording of videos."""
        import omni.usd

        for task_name in self.registered_tasks:
            print(f">>> Running test for environment: {task_name}")
            # create a new stage
            omni.usd.get_context().new_stage()
            # parse configuration
            env_cfg = parse_env_cfg(task_name, use_gpu=self.use_gpu, num_envs=self.num_envs)
            # create environment
            env = gym.make(task_name, cfg=env_cfg, headless=self.headless, viewport=True)

            # directory to save videos
            videos_dir = os.path.join(self.videos_dir, task_name)
            # wrap environment to record videos
            env = gym.wrappers.RecordVideo(
                env, videos_dir, step_trigger=self.step_trigger, video_length=self.video_length
            )

            # reset environment
            env.reset()
            # simulate environment
            for _ in range(500):
                # compute zero actions
                actions = 2 * torch.rand((env.num_envs, env.action_space.shape[0]), device=env.device) - 1
                # apply actions
                _, _, _, _ = env.step(actions)
                # render environment
                env.render(mode="human")
                # check if simulator is stopped
                if env.unwrapped.sim.is_stopped():
                    break

            # close the simulator
            env.close()


if __name__ == "__main__":
    unittest.main()
