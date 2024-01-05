# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.kit import SimulationApp

# launch omniverse app
config = {"headless": True}
simulation_app = SimulationApp(config)

"""Rest everything follows."""

import unittest
from collections import namedtuple

from omni.isaac.orbit.envs.mdp import NullCommandCfg


class TestNullCommandTerm(unittest.TestCase):
    """Test cases for null command generator."""

    def setUp(self) -> None:
        self.env = namedtuple("RLTaskEnv", ["num_envs", "dt", "device"])(20, 0.1, "cpu")

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
    unittest.main()
