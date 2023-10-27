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
import torch
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
        self.dt = 0.005
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=self.dt, device="cuda:0", shutdown_app_on_stop=False)
        self.sim = sim_utils.SimulationContext(sim_cfg)

    def tearDown(self):
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        # clear the stage
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

    def test_external_force_on_single_body(self):
        """Test application of external force on the base of the robot."""

        # Robots
        robot_cfg = ANYMAL_C_CFG
        robot_cfg.spawn.func("/World/Anymal_c/Robot_1", robot_cfg.spawn, translation=(0.0, -0.5, 0.65))
        robot_cfg.spawn.func("/World/Anymal_c/Robot_2", robot_cfg.spawn, translation=(0.0, 0.5, 0.65))
        # create handles for the robots
        robot = Articulation(robot_cfg.replace(prim_path="/World/Anymal_c/Robot.*"))

        # Play the simulator
        self.sim.reset()

        # Find bodies to apply the force
        body_ids, _ = robot.find_bodies("base")
        # Sample a large force
        external_wrench_b = torch.zeros(robot.root_view.count, len(body_ids), 6, device=self.sim.device)
        external_wrench_b[..., 1] = 1000.0

        # Now we are ready!
        for _ in range(5):
            # reset root state
            root_state = robot.data.default_root_state_w.clone()
            root_state[0, :2] = torch.tensor([0.0, -0.5], device=self.sim.device)
            root_state[1, :2] = torch.tensor([0.0, 0.5], device=self.sim.device)
            robot.write_root_state_to_sim(root_state)
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # apply force
            robot.set_external_force_and_torque(
                external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
            )
            # perform simulation
            for _ in range(100):
                # apply action to the robot
                robot.set_joint_position_target(robot.data.default_joint_pos.clone())
                robot.write_data_to_sim()
                # perform step
                self.sim.step()
                # update buffers
                robot.update(self.dt)
            # check condition that the robots have fallen down
            self.assertTrue(robot.data.root_pos_w[0, 2].item() < 0.2)
            self.assertTrue(robot.data.root_pos_w[1, 2].item() < 0.2)

    def test_external_force_on_multiple_bodies(self):
        """Test application of external force on the legs of the robot."""

        # Robots
        robot_cfg = ANYMAL_C_CFG
        robot_cfg.spawn.func("/World/Anymal_c/Robot_1", robot_cfg.spawn, translation=(0.0, -0.5, 0.65))
        robot_cfg.spawn.func("/World/Anymal_c/Robot_2", robot_cfg.spawn, translation=(0.0, 0.5, 0.65))
        # create handles for the robots
        robot = Articulation(robot_cfg.replace(prim_path="/World/Anymal_c/Robot.*"))

        # Play the simulator
        self.sim.reset()

        # Find bodies to apply the force
        body_ids, _ = robot.find_bodies(".*_SHANK")
        # Sample a large force
        external_wrench_b = torch.zeros(robot.root_view.count, len(body_ids), 6, device=self.sim.device)
        external_wrench_b[..., 1] = 100.0

        # Now we are ready!
        for _ in range(5):
            # reset root state
            root_state = robot.data.default_root_state_w.clone()
            root_state[0, :2] = torch.tensor([0.0, -0.5], device=self.sim.device)
            root_state[1, :2] = torch.tensor([0.0, 0.5], device=self.sim.device)
            robot.write_root_state_to_sim(root_state)
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # apply force
            robot.set_external_force_and_torque(
                external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
            )
            # perform simulation
            for _ in range(100):
                # apply action to the robot
                robot.set_joint_position_target(robot.data.default_joint_pos.clone())
                robot.write_data_to_sim()
                # perform step
                self.sim.step()
                # update buffers
                robot.update(self.dt)
            # check condition
            # since there is a moment applied on the robot, the robot should rotate
            self.assertTrue(robot.data.root_ang_vel_w[0, 2].item() > 0.1)
            self.assertTrue(robot.data.root_ang_vel_w[1, 2].item() > 0.1)


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
