# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# Can set this to False to see the GUI for debugging
HEADLESS = True

# launch omniverse app
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import unittest

import omni.usd

from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass


@configclass
class EmptySceneCfg(InteractiveSceneCfg):
    """Configuration for an empty scene."""

    pass


def get_empty_base_env_cfg(device: str = "cuda:0", num_envs: int = 1, env_spacing: float = 1.0):
    """Generate base environment config based on device"""

    @configclass
    class EmptyEnvCfg(DirectMARLEnvCfg):
        """Configuration for the empty test environment."""

        # Scene settings
        scene: EmptySceneCfg = EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        # Basic settings
        decimation = 1
        possible_agents = ["agent_0", "agent_1"]
        num_actions = {"agent_0": 1, "agent_1": 2}
        num_observations = {"agent_0": 3, "agent_1": 4}
        num_states = -1

    return EmptyEnvCfg()


class TestDirectMARLEnv(unittest.TestCase):
    """Test for direct MARL env class"""

    """
    Tests
    """

    def test_initialization(self):
        for device in ("cuda:0", "cpu"):
            with self.subTest(device=device):
                # create a new stage
                omni.usd.get_context().new_stage()
                # create environment
                env = DirectMARLEnv(cfg=get_empty_base_env_cfg(device=device))
                # check multi-agent config
                self.assertEqual(env.num_agents, 2)
                self.assertEqual(env.max_num_agents, 2)
                # check spaces
                self.assertEqual(env.state_space.shape, (7,))
                self.assertEqual(len(env.observation_spaces), 2)
                self.assertEqual(len(env.action_spaces), 2)
                # step environment to verify setup
                env.reset()
                for _ in range(2):
                    actions = {"agent_0": torch.rand((1, 1)), "agent_1": torch.rand((1, 2))}
                    obs, reward, terminated, truncate, info = env.step(actions)
                    env.state()
                # close the environment
                env.close()


if __name__ == "__main__":
    run_tests()
