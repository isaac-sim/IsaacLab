# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch
import warp as wp

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


@pytest.fixture
def sim():
    """Create a simulation context for testing."""
    # Wait for spawning
    stage = sim_utils.create_new_stage()
    # Constants
    num_envs = 1
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # TODO: Remove this once we have a better way to handle this.
    sim._app_control_on_stop_handle = None

    # Create a ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/GroundPlane", cfg)

    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0, stage=stage)
    cloner.define_base_env("/World/envs")
    env_prim_paths = cloner.generate_paths("/World/envs/env", num_envs)
    # create source prim
    stage.DefinePrim(env_prim_paths[0], "Xform")
    # clone the env xform
    cloner.clone(
        source_prim_path=env_prim_paths[0],
        prim_paths=env_prim_paths,
        replicate_physics=True,
    )

    # Define goals for the arm (x, y, z, qx, qy, qz, qw)
    ee_goals_set = [
        [0.5, 0.5, 0.7, 0, 0.707, 0, 0.707],
        [0.5, -0.4, 0.6, 0.707, 0, 0, 0.707],
        [0.5, 0, 0.5, 1.0, 0.0, 0.0, 0.0],
    ]
    ee_pose_b_des_set = torch.tensor(ee_goals_set, device=sim.device)

    yield sim, num_envs, ee_pose_b_des_set

    # Cleanup
    sim.stop()
    sim.clear_instance()


def test_franka_ik_pose_abs(sim):
    """Test IK controller for Franka arm with Franka hand."""
    sim_context, num_envs, ee_pose_b_des_set = sim

    # Create robot instance
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot = Articulation(cfg=robot_cfg)

    # Create IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=num_envs, device=sim_context.device)

    # Run the controller and check that it converges to the goal
    _run_ik_controller(
        robot, diff_ik_controller, "panda_hand", ["panda_joint.*"], sim_context, num_envs, ee_pose_b_des_set
    )


def test_ur10_ik_pose_abs(sim):
    """Test IK controller for UR10 arm."""
    sim_context, num_envs, ee_pose_b_des_set = sim

    # Create robot instance
    robot_cfg = UR10_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(cfg=robot_cfg)

    # Create IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=num_envs, device=sim_context.device)

    # Run the controller and check that it converges to the goal
    _run_ik_controller(robot, diff_ik_controller, "ee_link", [".*"], sim_context, num_envs, ee_pose_b_des_set)


def _run_ik_controller(
    robot: Articulation,
    diff_ik_controller: DifferentialIKController,
    ee_frame_name: str,
    arm_joint_names: list[str],
    sim: sim_utils.SimulationContext,
    num_envs: int,
    ee_pose_b_des_set: torch.Tensor,
):
    """Run the IK controller with the given parameters.

    Args:
        robot (Articulation): The robot to control.
        diff_ik_controller (DifferentialIKController): The differential IK controller.
        ee_frame_name (str): The name of the end-effector frame.
        arm_joint_names (list[str]): The names of the arm joints.
        sim (sim_utils.SimulationContext): The simulation context.
        num_envs (int): The number of environments.
        ee_pose_b_des_set (torch.Tensor): The set of desired end-effector poses.
    """
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Play the simulator
    sim.reset()

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
    ee_pose_b_des = torch.zeros(num_envs, diff_ik_controller.action_dim, device=sim.device)
    ee_pose_b_des[:] = ee_pose_b_des_set[current_goal_idx]
    # Compute current pose of the end-effector
    ee_pose_w = wp.to_torch(robot.data.body_pose_w)[:, ee_frame_idx]
    root_pose_w = wp.to_torch(robot.data.root_pose_w)
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
                pos_error_norm = torch.linalg.norm(pos_error, dim=-1)
                rot_error_norm = torch.linalg.norm(rot_error, dim=-1)
                # desired error (zer)
                des_error = torch.zeros_like(pos_error_norm)
                # check convergence
                torch.testing.assert_close(pos_error_norm, des_error, rtol=0.0, atol=1e-3)
                torch.testing.assert_close(rot_error_norm, des_error, rtol=0.0, atol=1e-3)
            # reset joint state
            joint_pos = wp.to_torch(robot.data.default_joint_pos).clone()
            joint_vel = wp.to_torch(robot.data.default_joint_vel).clone()
            # joint_pos *= sample_uniform(0.9, 1.1, joint_pos.shape, joint_pos.device)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.set_joint_position_target(joint_pos)
            robot.write_data_to_sim()
            # randomize root state yaw, ik should work regardless base rotation
            root_state = wp.to_torch(robot.data.root_state_w).clone()
            root_state[:, 3:7] = random_yaw_orientation(num_envs, sim.device)
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.reset()
            # reset actions
            ee_pose_b_des[:] = ee_pose_b_des_set[current_goal_idx]
            joint_pos_des = joint_pos[:, arm_joint_ids].clone()
            # update goal for next iteration
            current_goal_idx = (current_goal_idx + 1) % len(ee_pose_b_des_set)
            # set the controller commands
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ee_pose_b_des)
        else:
            # at reset, the jacobians are not updated to the latest state
            # so we MUST skip the first step
            # obtain quantities from simulation
            jacobian = wp.to_torch(robot.root_view.get_jacobians())[:, ee_jacobi_idx, :, arm_joint_ids]
            ee_pose_w = wp.to_torch(robot.data.body_pose_w)[:, ee_frame_idx]
            root_pose_w = wp.to_torch(robot.data.root_pose_w)
            base_rot = root_pose_w[:, 3:7]
            base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
            jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
            joint_pos = wp.to_torch(robot.data.joint_pos)[:, arm_joint_ids]
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
        sim.step(render=False)
        # update buffers
        robot.update(sim_dt)
