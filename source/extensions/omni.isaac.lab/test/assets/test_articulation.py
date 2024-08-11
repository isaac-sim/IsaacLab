# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

HEADLESS = True

# launch omniverse app
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

"""Rest everything follows."""

import ctypes
import torch
import unittest

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import build_simulation_context
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from omni.isaac.lab_assets import ANYMAL_C_CFG, FRANKA_PANDA_CFG, SHADOW_HAND_CFG  # isort:skip


def generate_robots_scene(num_robots: int = 1, robot_type="humanoid", stiffness=10, damping=2.0) -> Articulation:
    """Generate a scene with the provided number of robots.

    Args:
        num_robots: Number of robots to generate.
        robot_type: Type of robot to generate. Options are "humanoid", "panda", and "anymal".
        stiffness: Stiffness value for the robot's actuators. Only currently used for humanoid.
        damping: Damping value for the robot's actuators. Only currently used for humanoid.

    Returns:
        The articulation representing the robots.

    """
    # Create Top-level Xforms, one for each cube
    for i in range(num_robots):
        prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 1.5, 0.0, 0.0))

    if robot_type == "humanoid":
        robot_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Humanoid/humanoid_instanceable.usd"),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.34)),
            actuators={"body": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=stiffness, damping=damping)},
        )
    elif robot_type == "panda":
        robot_cfg = FRANKA_PANDA_CFG
    elif robot_type in ["anymal", "anymal_fixed_base"]:
        robot_cfg = ANYMAL_C_CFG

        if robot_type == "anymal_fixed_base":
            robot_cfg.spawn.articulation_props.fix_root_link = True

    elif robot_type == "shadow_hand":
        robot_cfg = SHADOW_HAND_CFG
    elif robot_type == "single_joint":
        robot_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Simple/revolute_articulation.usd"),
            actuators={
                "joint": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit=400.0,
                    velocity_limit=100.0,
                    stiffness=0.0,
                    damping=10.0,
                ),
            },
        )
    else:
        raise ValueError(f"Invalid robot type: {robot_type}, valid options are 'humanoid', 'panda', 'anymal', 'shadow_hand' and 'single_joint'.")

    # Create articulation with the default prim path
    robot = Articulation(cfg=robot_cfg.replace(prim_path="/World/Env_.*/Robot"))

    return robot


class TestArticulation(unittest.TestCase):
    """Test for articulation class."""

    """
    Tests
    """

    def test_initialization_floating_base_non_root(self):
        """Test initialization for a floating-base with articulation root on a rigid body.
        under the provided prim path."""
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        robot = generate_robots_scene(num_robots=num_robots, robot_type="humanoid")

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(robot)).value, 1)

                        # Check if robot is initialized
                        self.assertTrue(robot.is_initialized)
                        # Check that floating base
                        self.assertFalse(robot.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertEqual(robot.data.root_pos_w.shape, (num_robots, 3))
                        self.assertEqual(robot.data.root_quat_w.shape, (num_robots, 4))
                        self.assertEqual(robot.data.joint_pos.shape, (num_robots, 21))

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            robot.root_physx_view.max_dofs, robot.root_physx_view.shared_metatype.dof_count
                        )
                        # -- link related
                        self.assertEqual(
                            robot.root_physx_view.max_links, robot.root_physx_view.shared_metatype.link_count
                        )
                        # -- link names (check within articulation ordering is correct)
                        prim_path_body_names = [path.split("/")[-1] for path in robot.root_physx_view.link_paths[0]]
                        self.assertListEqual(prim_path_body_names, robot.body_names)

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update robot
                            robot.update(sim.cfg.dt)

    def test_initialization_floating_base(self):
        """Test initialization for a floating-base with articulation root on provided prim path."""
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        robot = generate_robots_scene(num_robots=num_robots, robot_type="anymal")

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(robot)).value, 1)

                        # Play sim
                        sim.reset()
                        # Check if robot is initialized
                        self.assertTrue(robot.is_initialized)
                        # Check that floating base
                        self.assertFalse(robot.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertEqual(robot.data.root_pos_w.shape, (num_robots, 3))
                        self.assertEqual(robot.data.root_quat_w.shape, (num_robots, 4))
                        self.assertEqual(robot.data.joint_pos.shape, (num_robots, 12))

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            robot.root_physx_view.max_dofs, robot.root_physx_view.shared_metatype.dof_count
                        )
                        # -- link related
                        self.assertEqual(
                            robot.root_physx_view.max_links, robot.root_physx_view.shared_metatype.link_count
                        )
                        # -- link names (check within articulation ordering is correct)
                        prim_path_body_names = [path.split("/")[-1] for path in robot.root_physx_view.link_paths[0]]
                        self.assertListEqual(prim_path_body_names, robot.body_names)

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update robot
                            robot.update(sim.cfg.dt)

    def test_initialization_fixed_base(self):
        """Test initialization for fixed base."""
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        robot = generate_robots_scene(num_robots=num_robots, robot_type="panda")

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(robot)).value, 1)

                        # Play sim
                        sim.reset()
                        # Check if robot is initialized
                        self.assertTrue(robot.is_initialized)
                        # Check that fixed base
                        self.assertTrue(robot.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertEqual(robot.data.root_pos_w.shape, (num_robots, 3))
                        self.assertEqual(robot.data.root_quat_w.shape, (num_robots, 4))
                        self.assertEqual(robot.data.joint_pos.shape, (num_robots, 9))

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            robot.root_physx_view.max_dofs, robot.root_physx_view.shared_metatype.dof_count
                        )
                        # -- link related
                        self.assertEqual(
                            robot.root_physx_view.max_links, robot.root_physx_view.shared_metatype.link_count
                        )
                        # -- link names (check within articulation ordering is correct)
                        prim_path_body_names = [path.split("/")[-1] for path in robot.root_physx_view.link_paths[0]]
                        self.assertListEqual(prim_path_body_names, robot.body_names)

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update robot
                            robot.update(sim.cfg.dt)
                            # check that the root is at the correct state
                            default_root_state = robot.data.default_root_state.clone()
                            torch.testing.assert_close(robot.data.root_state_w, default_root_state)

    def test_initialization_fixed_base_single_joint(self):
        """Test initialization for fixed base articulation with a single joint."""
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        robot = generate_robots_scene(num_robots=num_robots, robot_type="single_joint")

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(robot)).value, 1)

                        # Play sim
                        sim.reset()
                        # Check if robot is initialized
                        self.assertTrue(robot.is_initialized)
                        # Check that fixed base
                        self.assertTrue(robot.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertEqual(robot.data.root_pos_w.shape, (num_robots, 3))
                        self.assertEqual(robot.data.root_quat_w.shape, (num_robots, 4))
                        self.assertEqual(robot.data.joint_pos.shape, (num_robots, 1))

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            robot.root_physx_view.max_dofs, robot.root_physx_view.shared_metatype.dof_count
                        )
                        # -- link related
                        self.assertEqual(
                            robot.root_physx_view.max_links, robot.root_physx_view.shared_metatype.link_count
                        )
                        # -- link names (check within articulation ordering is correct)
                        prim_path_body_names = [path.split("/")[-1] for path in robot.root_physx_view.link_paths[0]]
                        self.assertListEqual(prim_path_body_names, robot.body_names)

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update robot
                            robot.update(sim.cfg.dt)

    def test_initialization_hand_with_tendons(self):
        """Test initialization for fixed base articulated hand with tendons."""
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        robot = generate_robots_scene(num_robots=num_robots, robot_type="shadow_hand")

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(robot)).value, 1)

                        # Play sim
                        sim.reset()
                        # Check if robot is initialized
                        self.assertTrue(robot.is_initialized)
                        # Check that fixed base
                        self.assertTrue(robot.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertTrue(robot.data.root_pos_w.shape == (num_robots, 3))
                        self.assertTrue(robot.data.root_quat_w.shape == (num_robots, 4))
                        self.assertTrue(robot.data.joint_pos.shape == (num_robots, 24))

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            robot.root_physx_view.max_dofs, robot.root_physx_view.shared_metatype.dof_count
                        )
                        # -- link related
                        self.assertEqual(
                            robot.root_physx_view.max_links, robot.root_physx_view.shared_metatype.link_count
                        )

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update robot
                            robot.update(sim.cfg.dt)

    def test_initialization_floating_base_made_fixed_base(self):
        """Test initialization for a floating-base articulation made fixed-base using schema properties."""
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        robot = generate_robots_scene(num_robots=num_robots, robot_type="anymal_fixed_base")

                        # Check that boundedness of articulation is correct
                        self.assertEqual(ctypes.c_long.from_address(id(robot)).value, 1)

                        # Play sim
                        sim.reset()
                        # Check if robot is initialized
                        self.assertTrue(robot.is_initialized)
                        # Check that is fixed base
                        self.assertTrue(robot.is_fixed_base)
                        # Check buffers that exists and have correct shapes
                        self.assertEqual(robot.data.root_pos_w.shape, (num_robots, 3))
                        self.assertEqual(robot.data.root_quat_w.shape, (num_robots, 4))
                        self.assertEqual(robot.data.joint_pos.shape, (num_robots, 12))

                        # Check some internal physx data for debugging
                        # -- joint related
                        self.assertEqual(
                            robot.root_physx_view.max_dofs, robot.root_physx_view.shared_metatype.dof_count
                        )
                        # -- link related
                        self.assertEqual(
                            robot.root_physx_view.max_links, robot.root_physx_view.shared_metatype.link_count
                        )
                        # -- link names (check within articulation ordering is correct)
                        prim_path_body_names = [path.split("/")[-1] for path in robot.root_physx_view.link_paths[0]]
                        self.assertListEqual(prim_path_body_names, robot.body_names)

                        # Root state should be at the default state
                        robot.write_root_state_to_sim(robot.data.default_root_state.clone())

                        # Simulate physics
                        for _ in range(10):
                            # perform rendering
                            sim.step()
                            # update robot
                            robot.update(sim.cfg.dt)

                            # check that the root is at the correct state
                            default_root_state = robot.data.default_root_state.clone()
                            torch.testing.assert_close(robot.data.root_state_w, default_root_state)

    # def test_initialization_fixed_base_made_floating_base(self):
    #     """Test initialization for fixed base made floating-base using schema properties."""
    #     # Create articulation
    #     robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot")
    #     robot_cfg.spawn.articulation_props.fix_root_link = False
    #     robot = Articulation(cfg=robot_cfg)

    #     # Check that boundedness of articulation is correct
    #     self.assertEqual(ctypes.c_long.from_address(id(robot)).value, 1)

    #     # Play sim
    #     self.sim.reset()

    #     # Check that boundedness of articulation is correct
    #     self.assertEqual(ctypes.c_long.from_address(id(robot)).value, 1)

    #     # Check if robot is initialized
    #     self.assertTrue(robot.is_initialized)
    #     # Check that fixed base
    #     self.assertFalse(robot.is_fixed_base)
    #     # Check buffers that exists and have correct shapes
    #     self.assertTrue(robot.data.root_pos_w.shape == (1, 3))
    #     self.assertTrue(robot.data.root_quat_w.shape == (1, 4))
    #     self.assertTrue(robot.data.joint_pos.shape == (1, 9))

    #     # Check some internal physx data for debugging
    #     # -- joint related
    #     self.assertEqual(robot.root_physx_view.max_dofs, robot.root_physx_view.shared_metatype.dof_count)
    #     # -- link related
    #     self.assertEqual(robot.root_physx_view.max_links, robot.root_physx_view.shared_metatype.link_count)
    #     # -- link names (check within articulation ordering is correct)
    #     prim_path_body_names = [path.split("/")[-1] for path in robot.root_physx_view.link_paths[0]]
    #     self.assertListEqual(prim_path_body_names, robot.body_names)

    #     # Simulate physics
    #     for _ in range(10):
    #         # perform rendering
    #         self.sim.step()
    #         # update robot
    #         robot.update(self.dt)
    #         # check that the root is at the correct state
    #         default_root_state = robot.data.default_root_state.clone()
    #         is_close = torch.any(torch.isclose(robot.data.root_state_w, default_root_state))
    #         self.assertFalse(is_close)

    # def test_out_of_range_default_joint_pos(self):
    #     """Test that the default joint position from configuration is out of range."""
    #     with build_simulation_context(device="cuda:0", add_ground_plane=False, auto_add_lighting=True) as sim:
    #         # Create articulation
    #         robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot")
    #         robot_cfg.init_state.joint_pos = {
    #             "panda_joint1": 10.0,
    #             "panda_joint[2, 4]": -20.0,
    #         }
    #         robot = Articulation(robot_cfg)

    #         # Check that boundedness of articulation is correct
    #         self.assertEqual(ctypes.c_long.from_address(id(robot)).value, 1)

    #         # Play sim
    #         sim.reset()
    #         # Check if robot is initialized
    #         self.assertFalse(robot._is_initialized)

    # def test_out_of_range_default_joint_vel(self):
    #     """Test that the default joint velocity from configuration is out of range."""
    #     with build_simulation_context(device="cuda:0", add_ground_plane=False, auto_add_lighting=True) as sim:
    #         # Create articulation
    #         robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot")
    #         robot_cfg.init_state.joint_vel = {
    #             "panda_joint1": 100.0,
    #             "panda_joint[2, 4]": -60.0,
    #         }
    #         robot = Articulation(robot_cfg)

    #         # Check that boundedness of articulation is correct
    #         self.assertEqual(ctypes.c_long.from_address(id(robot)).value, 1)

    #         # Play sim
    #         sim.reset()
    #         # Check if robot is initialized
    #         self.assertFalse(robot._is_initialized)

    def test_external_force_on_single_body(self):
        """Test application of external force on the base of the robot."""
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        robot = generate_robots_scene(num_robots=num_robots, robot_type="anymal")
                        # Play the simulator
                        sim.reset()

                        # Find bodies to apply the force
                        body_ids, _ = robot.find_bodies("base")
                        # Sample a large force
                        external_wrench_b = torch.zeros(robot.num_instances, len(body_ids), 6, device=sim.device)
                        external_wrench_b[..., 1] = 1000.0

                        # Now we are ready!
                        for _ in range(5):
                            # reset root state
                            root_state = robot.data.default_root_state.clone()

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
                                sim.step()
                                # update buffers
                                robot.update(sim.cfg.dt)
                            # check condition that the robots have fallen down
                            for i in range(num_robots):
                                self.assertLess(robot.data.root_pos_w[i, 2].item(), 0.2)

    def test_external_force_on_multiple_bodies(self):
        """Test application of external force on the legs of the robot."""
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        robot = generate_robots_scene(num_robots=num_robots, robot_type="anymal")

                        # Play the simulator
                        sim.reset()

                        # Find bodies to apply the force
                        body_ids, _ = robot.find_bodies(".*_SHANK")
                        # Sample a large force
                        external_wrench_b = torch.zeros(robot.num_instances, len(body_ids), 6, device=sim.device)
                        external_wrench_b[..., 1] = 100.0

                        # Now we are ready!
                        for _ in range(5):
                            # reset root state
                            robot.write_root_state_to_sim(robot.data.default_root_state.clone())
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
                                sim.step()
                                # update buffers
                                robot.update(sim.cfg.dt)
                            # check condition
                            for i in range(num_robots):
                                # since there is a moment applied on the robot, the robot should rotate
                                self.assertTrue(robot.data.root_ang_vel_w[i, 2].item() > 0.1)

    def test_loading_gains_from_usd(self):
        """Test that gains are loaded from USD file if actuator model has them as None."""
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        robot = generate_robots_scene(
                            num_robots=num_robots, robot_type="humanoid", stiffness=None, damping=None
                        )

                        # Play sim
                        sim.reset()

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
                        indices_list, _, values_list = string_utils.resolve_matching_names_values(
                            expected_stiffness, robot.joint_names
                        )
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
                        indices_list, _, values_list = string_utils.resolve_matching_names_values(
                            expected_damping, robot.joint_names
                        )
                        expected_damping = torch.zeros_like(expected_stiffness)
                        expected_damping[:, indices_list] = torch.tensor(values_list, device=robot.device)

                        # Check that gains are loaded from USD file
                        torch.testing.assert_close(robot.actuators["body"].stiffness, expected_stiffness)
                        torch.testing.assert_close(robot.actuators["body"].damping, expected_damping)

    def test_setting_gains_from_cfg(self):
        """Test that gains are loaded from the configuration correctly.

        Note: We purposefully give one argument as int and other as float to check that it is handled correctly.
        """
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        robot = generate_robots_scene(num_robots=num_robots, robot_type="humanoid")

                        # Play sim
                        sim.reset()

                        # Expected gains
                        expected_stiffness = torch.full(
                            (robot.num_instances, robot.num_joints), 10.0, device=robot.device
                        )
                        expected_damping = torch.full_like(expected_stiffness, 2.0)

                        # Check that gains are loaded from USD file
                        torch.testing.assert_close(robot.actuators["body"].stiffness, expected_stiffness)
                        torch.testing.assert_close(robot.actuators["body"].damping, expected_damping)

    def test_setting_gains_from_cfg_dict(self):
        """Test that gains are loaded from the configuration dictionary correctly.

        Note: We purposefully give one argument as int and other as float to check that it is handled correctly.
        """
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        robot = generate_robots_scene(num_robots=num_robots, robot_type="humanoid")
                        # Play sim
                        sim.reset()

                        # Expected gains
                        expected_stiffness = torch.full(
                            (robot.num_instances, robot.num_joints), 10.0, device=robot.device
                        )
                        expected_damping = torch.full_like(expected_stiffness, 2.0)

                        # Check that gains are loaded from USD file
                        torch.testing.assert_close(robot.actuators["body"].stiffness, expected_stiffness)
                        torch.testing.assert_close(robot.actuators["body"].damping, expected_damping)

    # TODO: Wrap up this test
    # def test_implicit_vs_explicit_actuator(self):
    #     """Test that verifies that actuators for implicit PD are similarly performing as explicit PD."""
    #     for num_robots in (1, 2):
    #         for device in ("cuda:0", "cpu"):
    #             with self.subTest(num_robots=num_robots, device=device):
    #                 with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
    #                     for i in range(num_robots):
    #                         prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 1.5, 0.0, 0.0))
    #                     robot_implicit_actuator_cfg = ArticulationCfg(
    #                         prim_path="/World/Env_.*/Robot_1",
    #                         spawn=sim_utils.UsdFileCfg(
    #                             usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Humanoid/humanoid_instanceable.usd"
    #                         ),
    #                         init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.34)),
    #                         actuators={
    #                             "body": ImplicitActuatorCfg(
    #                                 joint_names_expr=[".*"], stiffness=0.0, damping=0.0, velocity_limit=5.0
    #                             )
    #                         },
    #                     )
    #                     robot_explicit_actuator_cfg = ArticulationCfg(
    #                         prim_path="/World/Env_.*/Robot_2",
    #                         spawn=sim_utils.UsdFileCfg(
    #                             usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Humanoid/humanoid_instanceable.usd"
    #                         ),
    #                         init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.34)),
    #                         actuators={
    #                             "body": DCMotorCfg(
    #                                 joint_names_expr=[".*"], stiffness=0.0, damping=0.0, velocity_limit=5.0,
    #                             )
    #                         },
    #                     )
    #                     # TODO: Perform the test
    #                     robot_implicit_actuator = Articulation(cfg=robot_implicit_actuator_cfg)
    #                     robot_explicit_actuator = Articulation(cfg=robot_explicit_actuator_cfg)

    #                     # Play sim
    #                     sim.reset()

    #                                             # Now we are ready!
    #                     for _ in range(5):
    #                         # reset root state
    #                         for robot in (robot_implicit_actuator, robot_explicit_actuator):
    #                             root_state = robot.data.default_root_state.clone()
    #                             robot.write_root_state_to_sim(root_state)
    #                             # reset dof state
    #                             joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
    #                             robot.write_joint_state_to_sim(joint_pos, joint_vel)
    #                             # reset robot
    #                             robot.reset()

    #                         for robot in (robot_implicit_actuator, robot_explicit_actuator):
    #                             # apply action to the robot
    #                             robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    #                             robot.write_data_to_sim()

    #                         # perform simulation
    #                         for _ in range(100):
    #                             # perform step
    #                             sim.step()
    #                             # update buffers
    #                             robot.update(sim.cfg.dt)
    #                         # TODO: Check that current joint position is same as default joint position

    #                         assert torch.allclose(robot.data.joint_pos, robot.data.default_joint_pos)

    def test_reset(self):
        """Test that reset method works properly.

        Need to check that all actuators are reset and that forces, torques and last body velocities are reset to 0.0.

        NOTE: Currently no way to determine actuators have been reset, can leave this to actuator tests that
        implement reset method.

        """
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
                        robot = generate_robots_scene(num_robots=num_robots, robot_type="humanoid")

                        # Play the simulator
                        sim.reset()

                        # Now we are ready!
                        # reset robot
                        robot.reset()

                        # Reset should zero external forces and torques
                        self.assertFalse(robot.has_external_wrench)
                        self.assertEqual(torch.count_nonzero(robot._external_force_b), 0)
                        self.assertEqual(torch.count_nonzero(robot._external_torque_b), 0)

    def test_apply_joint_command(self):
        # TODO: Improve this test to actually ensure we get to desired joint positions
        for num_robots in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_robots=num_robots, device=device):
                    with build_simulation_context(
                        gravity_enabled=True, device=device, add_ground_plane=True, auto_add_lighting=True
                    ) as sim:
                        robot = generate_robots_scene(num_robots=num_robots, robot_type="panda")

                        # Play the simulator
                        sim.reset()

                        for _ in range(100):
                            # perform step
                            sim.step()
                            # update buffers
                            robot.update(sim.cfg.dt)

                        # reset dof state
                        joint_pos = robot.data.default_joint_pos
                        joint_pos[:, 3] = 0.0

                        # apply action to the robot
                        robot.set_joint_position_target(joint_pos)
                        robot.write_data_to_sim()

                        for _ in range(100):
                            # perform step
                            sim.step()
                            # update buffers
                            robot.update(sim.cfg.dt)

                        # Check that current joint position is not the same as default joint position, meaning
                        # the robot moved. We can't check that it reached it's desired joint position as the gains
                        # are not properly tuned
                        assert not torch.allclose(robot.data.joint_pos, joint_pos)


if __name__ == "__main__":
    run_tests()
