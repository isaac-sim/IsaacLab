# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from click import make_pass_decorator
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
from omni.isaac.lab_assets import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip


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

    def tearDown(self):
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        self.sim.clear()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    """
    Test fixtures.
    """

    def test_franka_pose_abs_fixed(self):
        """Test absolute pose control with fixed impedance."""
        robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        robot_cfg.spawn.rigid_props.disable_gravity = True
        robot = Articulation(cfg=robot_cfg)

        # Create IK controller
        opc_cfg = OperationSpaceControllerCfg(command_types=["pose_abs"], impedance_mode="fixed",
                                              inertial_compensation=True, uncouple_motion_wrench=False,
                                              stiffness=100.0, damping_ratio=1.0)
        opc = OperationSpaceController(opc_cfg, num_envs=self.num_envs, device=self.sim.device)

        # Define goals for the arm [xyz, quat_wxyz]
        ee_goals_set = [
            [0.5, -0.5, 0.7, 0.707, 0, 0.707, 0],
            [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
            [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
            [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ]
        ee_pose_b_des_set = torch.tensor(ee_goals_set, device=self.sim.device)

        self._run_op_space_controller(robot, opc, "panda_hand", ["panda_joint.*"], ee_pose_b_des_set)

    # def test_franka_position_abs(self):
    #     """Test absolute position control."""

    # def test_franka_force_abs(self):
    #     """test absolute force control."""

    # def test_franka_fixed(self):
    #     """Tests operational space controller for franka using fixed impedance."""
        
    # def test_franka_variable_kp(self):
    #     """Tests operational space controller for franka using variable stiffness impedance."""

    # def test_franka_variable(self):
    #     """Tests operational space controller for franka using variable stiffness and damping ratio impedance."""

    # def test_franka_gravity_compensation(self):
    #     """Tests operational space control with gravity compensation."""
    
    # def test_franka_inertial_compensation(self):
    #     """Tests operational space control with inertial compensation."""
    
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
        ee_jacobi_idx = ee_frame_idx - 1
        # Obtain joint indices
        arm_joint_ids = robot.find_joints(arm_joint_names)[0]
        # Update existing buffers
        # Note: We need to update buffers before the first step for the controller.
        robot.update(dt=sim_dt)

        # Track the given command
        current_goal_idx = 0
        # Current goal for the arm
        ee_pose_b_des = torch.zeros(self.num_envs, opc.action_dim, device=self.sim.device)
        ee_pose_b_des[:] = targets[current_goal_idx]
        # Compute current pose of the end-effector
        ee_pose_w = robot.data.body_state_w[:, ee_frame_idx, 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

        # update the goal marker
        ee_pos_w_des, ee_quat_w_des = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_b_des[:, 0:3], ee_pose_b_des[:, 3:7]
        )

        # Extract end-effector velocity in the world frame
        ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]  # Shape (num_robots, 6), includes linear and angular velocity
        # Extract root velocity in the world frame
        root_vel_w = robot.data.root_vel_w  # Shape (num_robots, 6), includes linear and angular velocity
        # Compute the relative velocity in the world frame
        relative_vel_w = ee_vel_w - root_vel_w
        # Transform velocities from world frame to root frame
        ee_lin_vel_b = quat_rotate_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])
        ee_ang_vel_b = quat_rotate_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
        ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)  # Shape (num_robots, 6)

        zero_joint_efforts = torch.zeros(self.num_envs, robot.num_joints, device=self.sim.device)
        joint_efforts = torch.zeros(self.num_envs, len(arm_joint_ids), device=self.sim.device)

        # Now we are ready!
        for count in range(15000):
            # reset every 250 steps
            if count % 250 == 0:
                # check that we converged to the goal
                if count > 0:
                    pos_error, rot_error = compute_pose_error(
                        ee_pos_b, ee_quat_b, ee_pose_b_des[:, 0:3], ee_pose_b_des[:, 3:7]
                    )
                    pos_error_norm = torch.norm(pos_error, dim=-1)
                    rot_error_norm = torch.norm(rot_error, dim=-1)
                    # desired error (zer)
                    des_error = torch.zeros_like(pos_error_norm)
                    # check convergence
                    # torch.testing.assert_close(pos_error_norm, des_error, rtol=0.0, atol=1e-3)
                    # torch.testing.assert_close(rot_error_norm, des_error, rtol=0.0, atol=1e-3)
                # reset joint state
                default_joint_pos = robot.data.default_joint_pos.clone()
                default_joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                robot.set_joint_effort_target(zero_joint_efforts)
                robot.write_data_to_sim()
                robot.reset()
                # reset target pose command
                ee_pose_b_des[:] = targets[current_goal_idx]
                opc.reset()
                opc.set_command(ee_pose_b_des)
                # update the goal marker
                ee_pos_w_des, ee_quat_w_des = combine_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_b_des[:, 0:3], ee_pose_b_des[:, 3:7]
                )
                # update goal for next iteration
                current_goal_idx = (current_goal_idx + 1) % len(targets)
            else:
                # at reset, the jacobians are not updated to the latest state
                # so we MUST skip the first step
                # obtain quantities from simulation
                jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
                mass_matrix = robot.root_physx_view.get_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]

                ee_pose_w = robot.data.body_state_w[:, ee_frame_idx, 0:7]
                root_pose_w = robot.data.root_state_w[:, 0:7]
                # compute frame in root frame
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

                # Extract end-effector velocity in the world frame
                ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]  # Shape (num_robots, 6), includes linear and angular velocity
                # Extract root velocity in the world frame
                root_vel_w = robot.data.root_vel_w  # Shape (num_robots, 6), includes linear and angular velocity
                # Compute the relative velocity in the world frame
                relative_vel_w = ee_vel_w - root_vel_w
                # Transform velocities from world frame to root frame
                ee_lin_vel_b = quat_rotate_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])
                ee_ang_vel_b = quat_rotate_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
                ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)  # Shape (num_robots, 6)

                # compute the joint commands
                joint_efforts = opc.compute(jacobian=jacobian, ee_pose=ee_pose_b, ee_vel=ee_vel_b, mass_matrix=mass_matrix)
                robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
                robot.write_data_to_sim()

            # update marker positions
            self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            self.goal_marker.visualize(ee_pos_w_des, ee_quat_w_des)

            # perform step
            self.sim.step(render=True)
            # update buffers
            robot.update(sim_dt)


if __name__ == "__main__":
    run_tests()
