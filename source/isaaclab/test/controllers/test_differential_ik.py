# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from isaaclab.utils.math import (  # isort:skip
    compute_pose_error,
    matrix_from_quat,
    quat_inv,
    random_yaw_orientation,
    subtract_frame_transforms,
)

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip

pytestmark = pytest.mark.integration


@pytest.fixture
def sim():
    """Create a simulation context with one cloned environment and the desired end-effector poses."""
    stage = sim_utils.create_new_stage()
    num_envs = 1
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # TODO: Remove this once we have a better way to handle this.
    sim._app_control_on_stop_handle = None

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    env_fmt = "/World/envs/env_{}"
    env_origins, _ = cloner.grid_transforms(num_envs, spacing=2.0)
    stage.DefinePrim(env_fmt.format(0), "Xform")
    cloner.usd_replicate(stage, [env_fmt.format(0)], [env_fmt], np.arange(num_envs), positions=env_origins)

    # Define goals for the arm (x, y, z, qx, qy, qz, qw)
    ee_goals_set = [
        [0.5, 0.5, 0.7, 0, 0.707, 0, 0.707],
        [0.5, -0.4, 0.6, 0.707, 0, 0, 0.707],
        [0.5, 0, 0.5, 1.0, 0.0, 0.0, 0.0],
    ]
    ee_pose_b_des_set = torch.tensor(ee_goals_set, device=sim.device)

    yield sim, num_envs, ee_pose_b_des_set

    sim.stop()
    sim.clear_instance()


@pytest.mark.parametrize(
    ("robot_cfg", "ee_frame_name", "arm_joint_names"),
    [
        pytest.param(FRANKA_PANDA_HIGH_PD_CFG, "panda_hand", ["panda_joint.*"], id="franka"),
        pytest.param(UR10_CFG, "ee_link", [".*"], id="ur10"),
    ],
)
def test_ik_pose_abs(sim, robot_cfg, ee_frame_name, arm_joint_names):
    """The damped least-squares IK controller converges to absolute pose targets regardless of the base yaw."""
    sim_context, num_envs, ee_pose_b_des_set = sim
    robot_cfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(cfg=robot_cfg)

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=num_envs, device=sim_context.device)

    _run_ik_controller(
        robot, diff_ik_controller, ee_frame_name, arm_joint_names, sim_context, num_envs, ee_pose_b_des_set
    )


def _run_ik_controller(
    robot: Articulation,
    diff_ik_controller: DifferentialIKController,
    ee_frame_name: str,
    arm_joint_names: list[str],
    sim: sim_utils.SimulationContext,
    num_envs: int,
    ee_pose_b_des_set: torch.Tensor,
):
    """Track each desired pose for 250 steps from a re-randomized base yaw and assert convergence."""
    sim_dt = sim.get_physics_dt()
    sim.reset()

    # Obtain the frame index of the end-effector
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    ee_jacobi_idx = ee_frame_idx - 1
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]
    # buffers must be updated before the first controller step
    robot.update(dt=sim_dt)

    current_goal_idx = 0
    ee_pose_b_des = torch.zeros(num_envs, diff_ik_controller.action_dim, device=sim.device)
    ee_pos_b = ee_quat_b = None

    for count in range(1500):
        # reset every 250 steps
        if count % 250 == 0:
            if count > 0:
                pos_error, rot_error = compute_pose_error(
                    ee_pos_b, ee_quat_b, ee_pose_b_des[:, 0:3], ee_pose_b_des[:, 3:7]
                )
                des_error = torch.zeros(num_envs, device=sim.device)
                torch.testing.assert_close(torch.linalg.norm(pos_error, dim=-1), des_error, rtol=0.0, atol=1e-3)
                torch.testing.assert_close(torch.linalg.norm(rot_error, dim=-1), des_error, rtol=0.0, atol=1e-3)
            # reset joint state
            joint_pos = robot.data.default_joint_pos.torch.clone()
            joint_vel = robot.data.default_joint_vel.torch.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.set_joint_position_target(joint_pos)
            robot.write_data_to_sim()
            # randomize root state yaw, ik should work regardless base rotation
            root_state = robot.data.root_state_w.torch.clone()
            root_state[:, 3:7] = random_yaw_orientation(num_envs, sim.device)
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.reset()
            # command the next goal and hold the default joint positions for this step
            ee_pose_b_des[:] = ee_pose_b_des_set[current_goal_idx]
            joint_pos_des = joint_pos[:, arm_joint_ids].clone()
            current_goal_idx = (current_goal_idx + 1) % len(ee_pose_b_des_set)
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ee_pose_b_des)
        else:
            # at reset, the jacobians are not updated to the latest state, so the first step is skipped
            jacobian = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, arm_joint_ids]
            ee_pose_w = robot.data.body_pose_w.torch[:, ee_frame_idx]
            root_pose_w = robot.data.root_pose_w.torch
            base_rot_matrix = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
            jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
            joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, arm_joint_ids)
        robot.write_data_to_sim()
        # perform step
        sim.step(render=False)
        # update buffers
        robot.update(sim_dt)
