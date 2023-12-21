# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import os

from omni.isaac.orbit.app import AppLauncher

# launch the simulator
app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
app_launcher = AppLauncher(headless=True, experience=app_experience)
simulation_app = app_launcher.app


"""Rest everything follows."""

import os
import torch
import unittest

from omni.isaac.orbit_tasks.utils.data_collector import RobomimicDataCollector


class TestRobomimicDataCollector(unittest.TestCase):
    """Test dataset flushing behavior of robomimic data collector."""

    def test_basic_flushing(self):
        """Adds random data into the collector and checks saving of the data."""
        # name of the environment (needed by robomimic)
        task_name = "My-Task-v0"
        # specify directory for logging experiments
        test_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(test_dir, "output", "demos")
        # name of the file to save data
        filename = "hdf_dataset.hdf5"
        # number of episodes to collect
        num_demos = 10
        # number of environments to simulate
        num_envs = 4

        # create data-collector
        collector_interface = RobomimicDataCollector(task_name, log_dir, filename, num_demos)

        # reset the collector
        collector_interface.reset()

        while not collector_interface.is_stopped():
            # generate random data to store
            # -- obs
            obs = {"joint_pos": torch.randn(num_envs, 7), "joint_vel": torch.randn(num_envs, 7)}
            # -- actions
            actions = torch.randn(num_envs, 7)
            # -- next obs
            next_obs = {"joint_pos": torch.randn(num_envs, 7), "joint_vel": torch.randn(num_envs, 7)}
            # -- rewards
            rewards = torch.randn(num_envs)
            # -- dones
            dones = torch.rand(num_envs) > 0.5

            # store signals
            # -- obs
            for key, value in obs.items():
                collector_interface.add(f"obs/{key}", value)
            # -- actions
            collector_interface.add("actions", actions)
            # -- next_obs
            for key, value in next_obs.items():
                collector_interface.add(f"next_obs/{key}", value.cpu().numpy())
            # -- rewards
            collector_interface.add("rewards", rewards)
            # -- dones
            collector_interface.add("dones", dones)

            # flush data from collector for successful environments
            # note: in this case we flush all the time
            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            collector_interface.flush(reset_env_ids)

        # close collector
        collector_interface.close()
        # TODO: Add inspection of the saved dataset as part of the test.


if __name__ == "__main__":
    unittest.main()
