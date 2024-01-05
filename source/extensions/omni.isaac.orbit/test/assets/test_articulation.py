# Copyright (c) 2022-2024, The ORBIT Project Developers.
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
import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets import Articulation, ArticulationCfg
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from omni.isaac.orbit_assets import ANYMAL_C_CFG, FRANKA_PANDA_CFG  # isort:skip


class TestArticulation(unittest.TestCase):
    """Test for articulation class."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Simulation time-step
        self.dt = 0.005
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=self.dt, device="cuda:0")
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

    def test_initialization_floating_base_non_root(self):
        """Test initialization for a floating-base with articulation root on a rigid body
        under the provided prim path."""
        # Create articulation
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Humanoid/humanoid_instanceable.usd"),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.34)),
            actuators={"body": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
        )
        robot = Articulation(cfg=robot_cfg)

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
        self.assertTrue(robot.data.joint_pos.shape == (1, 21))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update robot
            robot.update(self.dt)

    def test_initialization_floating_base(self):
        """Test initialization for a floating-base with articulation root on provided prim path."""
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
        """Test initialization for fixed base."""
        # Create articulation
        robot = Articulation(cfg=FRANKA_PANDA_CFG.replace(prim_path="/World/Robot"))

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
        external_wrench_b = torch.zeros(robot.num_instances, len(body_ids), 6, device=self.sim.device)
        external_wrench_b[..., 1] = 1000.0

        # Now we are ready!
        for _ in range(5):
            # reset root state
            root_state = robot.data.default_root_state.clone()
            root_state[0, :2] = torch.tensor([0.0, -0.5], device=self.sim.device)
            root_state[1, :2] = torch.tensor([0.0, 0.5], device=self.sim.device)
            robot.write_root_state_to_sim(root_state)
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # reset robot
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
        external_wrench_b = torch.zeros(robot.num_instances, len(body_ids), 6, device=self.sim.device)
        external_wrench_b[..., 1] = 100.0

        # Now we are ready!
        for _ in range(5):
            # reset root state
            root_state = robot.data.default_root_state.clone()
            root_state[0, :2] = torch.tensor([0.0, -0.5], device=self.sim.device)
            root_state[1, :2] = torch.tensor([0.0, 0.5], device=self.sim.device)
            robot.write_root_state_to_sim(root_state)
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # reset robot
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

    def test_loading_gains_from_usd(self):
        """Test that gains are loaded from USD file if actuator model has them as None."""
        # Create articulation
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Humanoid/humanoid_instanceable.usd"),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.34)),
            actuators={"body": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
        )
        robot = Articulation(cfg=robot_cfg)

        # Play sim
        self.sim.reset()

        # Expected gains
        # -- Stiffness values
        expected_stiffness = {
            ".*_waist.*": 20.0,
            ".*_upper_arm.*": 10.0,
            "pelvis": 10.0,
            ".*_lower_arm": 2.0,
            ".*_thigh:0": 10.0,
            ".*_thigh:1": 20.0,
            ".*_thigh:2": 10.0,
            ".*_shin": 5.0,
            ".*_foot.*": 2.0,
        }
        indices_list, _, values_list = string_utils.resolve_matching_names_values(expected_stiffness, robot.joint_names)
        expected_stiffness = torch.zeros(robot.num_instances, robot.num_joints, device=robot.device)
        expected_stiffness[:, indices_list] = torch.tensor(values_list, device=robot.device)
        # -- Damping values
        expected_damping = {
            ".*_waist.*": 5.0,
            ".*_upper_arm.*": 5.0,
            "pelvis": 5.0,
            ".*_lower_arm": 1.0,
            ".*_thigh:0": 5.0,
            ".*_thigh:1": 5.0,
            ".*_thigh:2": 5.0,
            ".*_shin": 0.1,
            ".*_foot.*": 1.0,
        }
        indices_list, _, values_list = string_utils.resolve_matching_names_values(expected_damping, robot.joint_names)
        expected_damping = torch.zeros_like(expected_stiffness)
        expected_damping[:, indices_list] = torch.tensor(values_list, device=robot.device)

        # Check that gains are loaded from USD file
        torch.testing.assert_close(robot.actuators["body"].stiffness, expected_stiffness)
        torch.testing.assert_close(robot.actuators["body"].damping, expected_damping)

    def test_setting_gains_from_cfg(self):
        """Test that gains are loaded from the configuration correctly.

        Note: We purposefully give one argument as int and other as float to check that it is handled correctly.
        """
        # Create articulation
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Humanoid/humanoid_instanceable.usd"),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.34)),
            actuators={"body": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10, damping=2.0)},
        )
        robot = Articulation(cfg=robot_cfg)

        # Play sim
        self.sim.reset()

        # Expected gains
        expected_stiffness = torch.full((robot.num_instances, robot.num_joints), 10.0, device=robot.device)
        expected_damping = torch.full_like(expected_stiffness, 2.0)

        # Check that gains are loaded from USD file
        torch.testing.assert_close(robot.actuators["body"].stiffness, expected_stiffness)
        torch.testing.assert_close(robot.actuators["body"].damping, expected_damping)

    def test_setting_gains_from_cfg_dict(self):
        """Test that gains are loaded from the configuration dictionary correctly.

        Note: We purposefully give one argument as int and other as float to check that it is handled correctly.
        """
        # Create articulation
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Humanoid/humanoid_instanceable.usd"),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.34)),
            actuators={"body": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness={".*": 10}, damping={".*": 2.0})},
        )
        robot = Articulation(cfg=robot_cfg)

        # Play sim
        self.sim.reset()

        # Expected gains
        expected_stiffness = torch.full((robot.num_instances, robot.num_joints), 10.0, device=robot.device)
        expected_damping = torch.full_like(expected_stiffness, 2.0)

        # Check that gains are loaded from USD file
        torch.testing.assert_close(robot.actuators["body"].stiffness, expected_stiffness)
        torch.testing.assert_close(robot.actuators["body"].damping, expected_damping)


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
