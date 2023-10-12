# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import ctypes
import traceback
import unittest

import carb
import omni.isaac.core.utils.stage as stage_utils

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.assets.config import ANYMAL_C_CFG, FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG


class TestArticulation(unittest.TestCase):
    """Test for articulation class."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Simulation time-step
        self.dt = 0.01
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=self.dt, device="cuda:0", shutdown_app_on_stop=False)
        self.sim = sim_utils.SimulationContext(sim_cfg)

    def tearDown(self):
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        # clear the stage
        self.sim.clear()
        self.sim.clear_instance()

    """
    Tests
    """

    def test_initialization_floating_base(self):
        """Test articulation initialization for a floating-base."""
        # Create articulation
        robot = Articulation(cfg=ANYMAL_C_CFG.replace(prim_path="/World/Robot"))

        # Check that boundedness of articulation is correct
        self.assertEqual(ctypes.c_long.from_address(id(robot)).value, 1)

        # Play sim
        self.sim.reset()
        # Check if robot is initialized
        self.assertTrue(robot._is_initialized)
        # Check that floating base
        self.assertFalse(robot.is_fixed_base)
        # Check buffers that exists and have correct shapes
        self.assertTrue(robot.data.root_pos_w.shape == (1, 3))
        self.assertTrue(robot.data.root_quat_w.shape == (1, 4))
        self.assertTrue(robot.data.joint_pos.shape == (1, 12))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update robot
            robot.update(self.dt)

        # Delete articulation
        del robot

    def test_initialization_fixed_base(self):
        """Test articulation initialization for fixed base."""
        # Create articulation
        robot = Articulation(cfg=FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG.replace(prim_path="/World/Robot"))

        # Check that boundedness of articulation is correct
        self.assertEqual(ctypes.c_long.from_address(id(robot)).value, 1)

        # Play sim
        self.sim.reset()
        # Check if robot is initialized
        self.assertTrue(robot._is_initialized)
        # Check that fixed base
        self.assertTrue(robot.is_fixed_base)
        # Check buffers that exists and have correct shapes
        self.assertTrue(robot.data.root_pos_w.shape == (1, 3))
        self.assertTrue(robot.data.root_quat_w.shape == (1, 4))
        self.assertTrue(robot.data.joint_pos.shape == (1, 9))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update robot
            robot.update(self.dt)

        # Delete articulation
        del robot


if __name__ == "__main__":
    try:
        unittest.main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
