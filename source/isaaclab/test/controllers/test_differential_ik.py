# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import unittest

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.cloner import GridCloner

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from isaaclab.utils.math import (  # isort:skip
    compute_pose_error,
    matrix_from_quat,
    quat_inv,
    random_yaw_orientation,
    subtract_frame_transforms,
)

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip


class TestDifferentialIKController(unittest.TestCase):
    """Test fixture for checking that differential IK controller tracks commands properly."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Wait for spawning
        stage_utils.create_new_stage()
        # Constants
        self.num_envs = 128
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=0.01)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        # TODO: Remove this once we have a better way to handle this.
        self.sim._app_control_on_stop_handle = None

        # Create a ground plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/GroundPlane", cfg)

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

        # Define goals for the arm
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
        self.sim.clear()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    """
    Test fixtures.
    """

    def test_franka_ik_pose_abs(self):
        """Test IK controller for Franka arm with Franka hand."""
        # Create robot instance
        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        robot = Articulation(cfg=robot_cfg)

        # Create IK controller
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.sim.device)

        # Run the controller and check that it converges to the goal
        self._run_ik_controller(robot, diff_ik_controller, "panda_hand", ["panda_joint.*"])

    def test_ur10_ik_pose_abs(self):
        """Test IK controller for UR10 arm."""
        # Create robot instance
        robot_cfg = UR10_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        robot_cfg.spawn.rigid_props.disable_gravity = True
        robot = Articulation(cfg=robot_cfg)

        # Create IK controller
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.sim.device)

        # Run the controller and check that it converges to the goal
        self._run_ik_controller(robot, diff_ik_controller, "ee_link", [".*"])

    """
    Helper functions.
    """

    def _run_ik_controller(
        self,
        robot: Articulation,
        diff_ik_controller: DifferentialIKController,
        ee_frame_name: str,
        arm_joint_names: list[str],
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
        ee_pose_b_des = torch.zeros(self.num_envs, diff_ik_controller.action_dim, device=self.sim.device)
        ee_pose_b_des[:] = self.ee_pose_b_des_set[current_goal_idx]
        # Compute current pose of the end-effector
        ee_pose_w = robot.data.body_state_w[:, ee_frame_idx, 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Now we are ready!
        for count in range(1500):
            # reset every 150 steps
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
                    torch.testing.assert_close(pos_error_norm, des_error, rtol=0.0, atol=1e-3)
                    torch.testing.assert_close(rot_error_norm, des_error, rtol=0.0, atol=1e-3)
                # reset joint state
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                # joint_pos *= sample_uniform(0.9, 1.1, joint_pos.shape, joint_pos.device)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.set_joint_position_target(joint_pos)
                robot.write_data_to_sim()
                # randomize root state yaw, ik should work regardless base rotation
                root_state = robot.data.root_state_w.clone()
                root_state[:, 3:7] = random_yaw_orientation(self.num_envs, self.sim.device)
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                robot.reset()
                # reset actions
                ee_pose_b_des[:] = self.ee_pose_b_des_set[current_goal_idx]
                joint_pos_des = joint_pos[:, arm_joint_ids].clone()
                # update goal for next iteration
                current_goal_idx = (current_goal_idx + 1) % len(self.ee_pose_b_des_set)
                # set the controller commands
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ee_pose_b_des)
            else:
                # at reset, the jacobians are not updated to the latest state
                # so we MUST skip the first step
                # obtain quantities from simulation
                jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
                ee_pose_w = robot.data.body_state_w[:, ee_frame_idx, 0:7]
                root_pose_w = robot.data.root_state_w[:, 0:7]
                base_rot = root_pose_w[:, 3:7]
                base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
                jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
                jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
                joint_pos = robot.data.joint_pos[:, arm_joint_ids]
                # compute frame in root frame
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                # compute the joint commands
                joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            # apply actions
            robot.set_joint_position_target(joint_pos_des, arm_joint_ids)
            robot.write_data_to_sim()
            # perform step
            self.sim.step(render=False)
            # update buffers
            robot.update(sim_dt)


if __name__ == "__main__":
    run_tests()
