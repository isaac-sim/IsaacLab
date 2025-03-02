# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import unittest
from collections import namedtuple

from isaaclab.envs.mdp import NullCommandCfg


class TestNullCommandTerm(unittest.TestCase):
    """Test cases for null command generator."""

    def setUp(self) -> None:
        self.env = namedtuple("ManagerBasedRLEnv", ["num_envs", "dt", "device"])(20, 0.1, "cpu")

    def test_str(self):
        """Test the string representation of the command manager."""
        cfg = NullCommandCfg()
        command_term = cfg.class_type(cfg, self.env)
        # print the expected string
        print()
        print(command_term)

    def test_compute(self):
        """Test the compute function. For null command generator, it does nothing."""
        cfg = NullCommandCfg()
        command_term = cfg.class_type(cfg, self.env)

        # test the reset function
        command_term.reset()
        # test the compute function
        command_term.compute(dt=self.env.dt)
        # expect error
        with self.assertRaises(RuntimeError):
            command_term.command


if __name__ == "__main__":
    run_tests()
