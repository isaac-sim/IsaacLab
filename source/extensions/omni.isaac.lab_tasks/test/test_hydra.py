# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import sys

from omni.isaac.lab.app import AppLauncher, run_tests

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""


import unittest

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config


class TestHydra(unittest.TestCase):
    """Test the Hydra configuration system."""

    def test_hydra(self):
        """Test the hydra configuration system."""

        # set hardcoded command line arguments
        sys.argv = [
            sys.argv[0],
            "env.decimation=42",  # test simple env modification
            "env.events.physics_material.params.asset_cfg.joint_ids='slice(0 ,1, 2)'",  # test slice setting
            "env.scene.robot.init_state.joint_vel={.*: 4.0}",  # test regex setting
            "env.rewards.feet_air_time=null",  # test setting to none
            "agent.max_iterations=3",  # test simple agent modification
        ]

        @hydra_task_config("Isaac-Velocity-Flat-H1-v0", "rsl_rl_cfg_entry_point")
        def main(env_cfg, agent_cfg, self):
            # env
            self.assertEqual(env_cfg.decimation, 42)
            self.assertEqual(env_cfg.events.physics_material.params["asset_cfg"].joint_ids, slice(0, 1, 2))
            self.assertEqual(env_cfg.scene.robot.init_state.joint_vel, {".*": 4.0})
            self.assertIsNone(env_cfg.rewards.feet_air_time)
            # agent
            self.assertEqual(agent_cfg.max_iterations, 3)

        main(self)


if __name__ == "__main__":
    run_tests()
