# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# Can set this to False to see the GUI for debugging
HEADLESS = True

# launch omniverse app
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

"""Rest everything follows."""

import unittest

import omni.usd

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


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
        action_spaces = {"agent_0": 1, "agent_1": 2}
        observation_spaces = {"agent_0": 3, "agent_1": 4}
        state_space = -1
        episode_length_s = 100.0

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
                try:
                    # create environment
                    env = DirectMARLEnv(cfg=get_empty_base_env_cfg(device=device))
                except Exception as e:
                    if "env" in locals() and hasattr(env, "_is_closed"):
                        env.close()
                    else:
                        if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                            e.obj.close()
                    self.fail(f"Failed to set-up the DirectMARLEnv environment. Error: {e}")

                # check multi-agent config
                self.assertEqual(env.num_agents, 2)
                self.assertEqual(env.max_num_agents, 2)
                # check spaces
                self.assertEqual(env.state_space.shape, (7,))
                self.assertEqual(len(env.observation_spaces), 2)
                self.assertEqual(len(env.action_spaces), 2)
                # close the environment
                env.close()


if __name__ == "__main__":
    run_tests()
