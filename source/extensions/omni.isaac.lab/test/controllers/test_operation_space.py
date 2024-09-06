# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import unittest

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.cloner import GridCloner

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.controllers import OperationSpaceController, OperationSpaceControllerCfg
from omni.isaac.lab.utils.math import compute_pose_error, subtract_frame_transforms, quat_rotate_inverse, combine_frame_transforms
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG

##
# Pre-defined configs
##
from omni.isaac.lab_assets import FRANKA_PANDA_CFG  # isort:skip


class TestOperationSpaceController(unittest.TestCase):
    """Test fixture for checking that differential IK controller tracks commands properly."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Wait for spawning
        stage_utils.create_new_stage()
        # Constants
        self.num_envs = 16
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=0.01)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        # TODO: Remove this once we have a better way to handle this.
        self.sim._app_control_on_stop_handle = None

        # Create a ground plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/GroundPlane", cfg)

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

        light_cfg = sim_utils.DistantLightCfg(intensity=5.0, exposure=10.0)
        light_cfg.func("/Light", light_cfg, translation=[0, 0, 1],)

        # Create interface to clone the scene
        cloner = GridCloner(spacing=2.0)
        cloner.define_base_env("/World/envs")
        self.env_prim_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        # create source prim
        prim_utils.define_prim(self.env_prim_paths[0], "Xform")
        # clone the env xform
        self.env_origins = cloner.clone(
            source_prim_path=self.env_prim_paths[0],
            prim_paths=self.env_prim_paths,
            replicate_physics=True,
        )

        self.robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        self.robot_cfg.spawn.rigid_props.disable_gravity = True
        self.robot_cfg.actuators["panda_shoulder"].stiffness = 0.0
        self.robot_cfg.actuators["panda_shoulder"].damping = 0.0
        self.robot_cfg.actuators["panda_forearm"].stiffness = 0.0
        self.robot_cfg.actuators["panda_forearm"].damping = 0.0
        self.robot_cfg.spawn.rigid_props.disable_gravity = True

        # Define goals for the arm [xyz, quat_wxyz]
        ee_goals_set = [
            [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
            [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
            [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ]
        self.ee_pose_b_des_set = torch.tensor(ee_goals_set, device=self.sim.device)

    def tearDown(self):
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        self.sim.clear()  # FIXME: This hangs the test for some reason when LIVESTREAM is not enabled.
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    """
    Test fixtures.
    """

    def test_franka_pose_abs_fixed_impedance_with_full_inertial_and_gravity_compensation(self):
        """Test absolute pose control with fixed impedance, full inertial and gravity compensation."""
        self.robot_cfg.spawn.rigid_props.disable_gravity = False
        robot = Articulation(cfg=self.robot_cfg)
        # Create OPC controller
        opc_cfg = OperationSpaceControllerCfg(command_types=["pose_abs"], impedance_mode="fixed",
                                              inertial_compensation=True,
                                              uncouple_motion_wrench=False,
                                              gravity_compensation=True,
                                              stiffness=500.0, damping_ratio=1.0)
        opc = OperationSpaceController(opc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, opc, "panda_hand", ["panda_joint.*"], self.ee_pose_b_des_set)

    def test_franka_pose_abs_fixed_impedance_with_full_inertial_compensation(self):
        """Test absolute pose control with fixed impedance and full inertial compensation."""
        robot = Articulation(cfg=self.robot_cfg)
        # Create OPC controller
        opc_cfg = OperationSpaceControllerCfg(command_types=["pose_abs"], impedance_mode="fixed",
                                              inertial_compensation=True,
                                              uncouple_motion_wrench=False,
                                              gravity_compensation=False,
                                              stiffness=500.0, damping_ratio=1.0)
        opc = OperationSpaceController(opc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, opc, "panda_hand", ["panda_joint.*"], self.ee_pose_b_des_set)

    def test_franka_pose_abs_fixed_impedance_with_decoupled_inertial_compensation(self):
        """Test absolute pose control with fixed impedance and decoupled inertial compensation."""
        robot = Articulation(cfg=self.robot_cfg)
        # Create OPC controller
        opc_cfg = OperationSpaceControllerCfg(command_types=["pose_abs"], impedance_mode="fixed",
                                              inertial_compensation=True,
                                              uncouple_motion_wrench=True,
                                              gravity_compensation=False,
                                              stiffness=1000.0, damping_ratio=1.0)
        opc = OperationSpaceController(opc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, opc, "panda_hand", ["panda_joint.*"], self.ee_pose_b_des_set)

    def test_franka_pose_abs_fixed_impedance_without_inertial_compensation(self):
        """Test absolute pose control with fixed impedance and without inertial compensation."""
        robot = Articulation(cfg=self.robot_cfg)
        # Create OPC controller
        opc_cfg = OperationSpaceControllerCfg(command_types=["pose_abs"], impedance_mode="fixed",
                                              inertial_compensation=False,
                                              gravity_compensation=False,
                                              stiffness=[400.0, 400.0, 400.0, 100.0, 100.0, 100.0],
                                              damping_ratio=[5.0, 5.0, 5.0, 0.001, 0.001, 0.001])
        opc = OperationSpaceController(opc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, opc, "panda_hand", ["panda_joint.*"], self.ee_pose_b_des_set)

    # def test_franka_position_rel(self):
    #     """Test relative position control."""

    # def test_franka_force_abs(self):
    #     """test absolute force control."""

    # def test_franka_variable_kp(self):
    #     """Tests operational space controller for franka using variable stiffness impedance."""

    # def test_franka_variable(self):
    #     """Tests operational space controller for franka using variable stiffness and damping ratio impedance."""

    # def test_franka_control_hybrid_motion_force(self):
    #     """Tests operational space control for hybrid motion and force control."""

    """
    Helper functions
    """

    def _run_op_space_controller(
        self,
        robot: Articulation,
        opc: OperationSpaceController,
        ee_frame_name: str,
        arm_joint_names: list[str],
        targets: torch.tensor,
    ):
        # Define simulation stepping
        sim_dt = self.sim.get_physics_dt()
        # Play the simulator
        self.sim.reset()

        # Obtain the frame index of the end-effector
        ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
        # Obtain joint indices
        arm_joint_ids = robot.find_joints(arm_joint_names)[0]

        # Update existing buffers
        # Note: We need to update buffers before the first step for the controller.
        robot.update(dt=sim_dt)

        # get the updated states
        jacobian, mass_matrix, gravity, ee_pose_b, ee_vel_b, root_pose_w, ee_pose_w = self._update_states(robot, ee_frame_idx, arm_joint_ids)

        # Track the given pose command
        current_goal_idx = 0  # Current goal index for the arm
        ee_pose_b_des = torch.zeros(self.num_envs, opc.action_dim, device=self.sim.device)
        ee_pose_w_des = torch.zeros(self.num_envs, opc.action_dim, device=self.sim.device)

        # Set joint efforts to zero
        zero_joint_efforts = torch.zeros(self.num_envs, robot.num_joints, device=self.sim.device)
        joint_efforts = torch.zeros(self.num_envs, len(arm_joint_ids), device=self.sim.device)

        # Now we are ready!
        for count in range(1501):
            # reset every 500 steps
            if count % 500 == 0:
                # check that we converged to the goal
                if count > 0:
                    self._check_convergence(ee_pose_b, ee_pose_b_des)
                # reset joint state
                default_joint_pos = robot.data.default_joint_pos.clone()
                default_joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                robot.set_joint_effort_target(zero_joint_efforts)  # Set zero torques in the initial step
                robot.write_data_to_sim()
                robot.reset()
                # reset target pose
                ee_pose_b_des, ee_pose_w_des, current_goal_idx = self._update_target(opc, root_pose_w, targets, current_goal_idx)
                # set the opc command (ee desired pose)
                opc.reset()
                opc.set_command(ee_pose_b_des)
            else:
                # at reset, the jacobians are not updated to the latest state
                # so we MUST skip the first step

                # get the updated states
                jacobian, mass_matrix, gravity, ee_pose_b, ee_vel_b, root_pose_w, ee_pose_w = self._update_states(robot, ee_frame_idx, arm_joint_ids)

                # compute the joint commands
                joint_efforts = opc.compute(jacobian=jacobian, ee_pose=ee_pose_b, ee_vel=ee_vel_b, mass_matrix=mass_matrix, gravity=gravity)
                robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
                robot.write_data_to_sim()

            # update marker positions
            self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            self.goal_marker.visualize(ee_pose_w_des[:, 0:3], ee_pose_w_des[:, 3:7])

            # perform step
            self.sim.step(render=False)
            # update buffers
            robot.update(sim_dt)

    def _update_states(
        self,
        robot: Articulation,
        ee_frame_idx: int,
        arm_joint_ids: list[int],
    ):
        ee_jacobi_idx = ee_frame_idx - 1
        # obtain quantities from simulation
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
        mass_matrix = robot.root_physx_view.get_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
        gravity = robot.root_physx_view.get_generalized_gravity_forces()[:, arm_joint_ids]

        # Compute current pose of the end-effector
        ee_pose_w = robot.data.body_state_w[:, ee_frame_idx, 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

        # Compute the current velocity of the end-effector
        ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]  # Extract end-effector velocity in the world frame
        root_vel_w = robot.data.root_vel_w  # Extract root velocity in the world frame
        relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
        ee_lin_vel_b = quat_rotate_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])  # From world to root frame
        ee_ang_vel_b = quat_rotate_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
        ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

        return jacobian, mass_matrix, gravity, ee_pose_b, ee_vel_b, root_pose_w, ee_pose_w

    def _update_target(
        self,
        opc: OperationSpaceController,
        root_pose_w: torch.tensor,
        targets: torch.tensor,
        current_goal_idx: int,
    ):
        # update the ee desired pose
        ee_pose_b_des = torch.zeros(self.num_envs, opc.action_dim, device=self.sim.device)
        ee_pose_b_des[:] = targets[current_goal_idx]

        # update the ee desired pose in world frame (for marker)
        ee_pos_w_des, ee_quat_w_des = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_b_des[:, 0:3], ee_pose_b_des[:, 3:7]
        )
        ee_pose_w_des = torch.cat([ee_pos_w_des, ee_quat_w_des], dim=-1)
        next_goal_idx = (current_goal_idx + 1) % len(targets)

        return ee_pose_b_des, ee_pose_w_des, next_goal_idx

    def _check_convergence(
        self,
        ee_pose_b: torch.tensor,
        ee_pose_b_des: torch.tensor,
    ):
        pos_error, rot_error = compute_pose_error(
            ee_pose_b[:, 0:3], ee_pose_b[:, 3:7], ee_pose_b_des[:, 0:3], ee_pose_b_des[:, 3:7]
        )
        pos_error_norm = torch.norm(pos_error, dim=-1)
        rot_error_norm = torch.norm(rot_error, dim=-1)
        # desired error (zer)
        des_error = torch.zeros_like(pos_error_norm)
        # check convergence
        torch.testing.assert_close(pos_error_norm, des_error, rtol=0.0, atol=1e-3)
        torch.testing.assert_close(rot_error_norm, des_error, rtol=0.0, atol=1e-3)


if __name__ == "__main__":
    run_tests()
