# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""
# Import pinocchio in the main script to force the use of the dependencies installed by IsaacLab and not the one installed by Isaac Sim
# pinocchio is required by the Pink IK controller
import sys

if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import json
import numpy as np
import os
import torch

import pytest
from pink.configuration import Configuration

from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_from_matrix, quat_inv

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.fixture(scope="module")
def test_cfg():
    """Load test configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "test_configs", "pink_ik_gr1_test_configs.json")
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_params(test_cfg):
    """Set up test parameters."""
    return {
        "position_tolerance": test_cfg["tolerances"]["position"],
        "rotation_tolerance": test_cfg["tolerances"]["rotation"],
        "pd_position_tolerance": test_cfg["tolerances"]["pd_position"],
        "check_errors": test_cfg["tolerances"]["check_errors"],
    }


def create_test_env(num_envs):
    """Create a test environment with the Pink IK controller."""
    env_name = "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0"
    device = "cuda:0"

    try:
        env_cfg = parse_env_cfg(env_name, device=device, num_envs=num_envs)
        # Modify scene config to not spawn the packing table to avoid collision with the robot
        del env_cfg.scene.packing_table
        del env_cfg.terminations.object_dropping
        del env_cfg.terminations.time_out
        return gym.make(env_name, cfg=env_cfg).unwrapped, env_cfg
    except Exception as e:
        print(f"Failed to create environment: {str(e)}")
        raise


@pytest.fixture(scope="module")
def env_and_cfg():
    """Create environment and configuration for tests."""
    env, env_cfg = create_test_env(num_envs=1)

    # Set up camera view
    env.sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 1.0])

    return env, env_cfg


@pytest.fixture
def test_setup(env_and_cfg):
    """Set up test case - runs before each test."""
    env, env_cfg = env_and_cfg

    num_joints_in_robot_hands = env_cfg.actions.pink_ik_cfg.controller.num_hand_joints

    # Get Action Term and IK controller
    action_term = env.action_manager.get_term(name="pink_ik_cfg")
    pink_controllers = action_term._ik_controllers
    articulation = action_term._asset

    # Initialize Pink Configuration for forward kinematics
    kinematics_model = Configuration(
        pink_controllers[0].robot_wrapper.model,
        pink_controllers[0].robot_wrapper.data,
        pink_controllers[0].robot_wrapper.q0,
    )
    left_target_link_name = env_cfg.actions.pink_ik_cfg.target_eef_link_names["left_wrist"]
    right_target_link_name = env_cfg.actions.pink_ik_cfg.target_eef_link_names["right_wrist"]

    return {
        "env": env,
        "env_cfg": env_cfg,
        "num_joints_in_robot_hands": num_joints_in_robot_hands,
        "action_term": action_term,
        "pink_controllers": pink_controllers,
        "articulation": articulation,
        "kinematics_model": kinematics_model,
        "left_target_link_name": left_target_link_name,
        "right_target_link_name": right_target_link_name,
    }


def test_stay_still(test_setup, test_cfg):
    """Test staying still."""
    print("Running stay still test...")
    run_movement_test(test_setup, test_cfg["tests"]["stay_still"], test_cfg)


def test_vertical_movement(test_setup, test_cfg):
    """Test vertical movement of robot hands."""
    print("Running vertical movement test...")
    run_movement_test(test_setup, test_cfg["tests"]["vertical_movement"], test_cfg)


def test_horizontal_movement(test_setup, test_cfg):
    """Test horizontal movement of robot hands."""
    print("Running horizontal movement test...")
    run_movement_test(test_setup, test_cfg["tests"]["horizontal_movement"], test_cfg)


def test_horizontal_small_movement(test_setup, test_cfg):
    """Test small horizontal movement of robot hands."""
    print("Running horizontal small movement test...")
    run_movement_test(test_setup, test_cfg["tests"]["horizontal_small_movement"], test_cfg)


def test_forward_waist_bending_movement(test_setup, test_cfg):
    """Test forward waist bending movement of robot hands."""
    print("Running forward waist bending movement test...")
    run_movement_test(test_setup, test_cfg["tests"]["forward_waist_bending_movement"], test_cfg)


def test_rotation_movements(test_setup, test_cfg):
    """Test rotation movements of robot hands."""
    print("Running rotation movements test...")
    run_movement_test(test_setup, test_cfg["tests"]["rotation_movements"], test_cfg)


def run_movement_test(test_setup, test_config, test_cfg, aux_function=None):
    """Run a movement test with the given configuration."""
    env = test_setup["env"]
    num_joints_in_robot_hands = test_setup["num_joints_in_robot_hands"]

    left_hand_poses = np.array(test_config["left_hand_pose"], dtype=np.float32)
    right_hand_poses = np.array(test_config["right_hand_pose"], dtype=np.float32)

    curr_pose_idx = 0
    test_counter = 0
    num_runs = 0

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        obs, _ = env.reset()

        while simulation_app.is_running() and not simulation_app.is_exiting():
            num_runs += 1

            # Call auxiliary function if provided
            if aux_function is not None:
                aux_function(num_runs)

            # Create actions from hand poses and joint positions
            setpoint_poses = np.concatenate([left_hand_poses[curr_pose_idx], right_hand_poses[curr_pose_idx]])
            actions = np.concatenate([setpoint_poses, np.zeros(num_joints_in_robot_hands)])
            actions = torch.tensor(actions, device=env.device, dtype=torch.float32)
            actions = actions.repeat(env.num_envs, 1)

            # Step environment
            obs, _, _, _, _ = env.step(actions)

            # Check convergence and verify errors
            if num_runs % test_config["allowed_steps_per_motion"] == 0:
                print("Computing errors...")
                errors = compute_errors(
                    test_setup, env, left_hand_poses[curr_pose_idx], right_hand_poses[curr_pose_idx]
                )
                print_debug_info(errors, test_counter)
                if test_cfg["tolerances"]["check_errors"]:
                    verify_errors(errors, test_setup, test_cfg["tolerances"])

                curr_pose_idx = (curr_pose_idx + 1) % len(left_hand_poses)
                if curr_pose_idx == 0:
                    test_counter += 1
                    if test_counter > test_config["repeat"]:
                        print("Test completed successfully")
                        break


def get_link_pose(env, link_name):
    """Get the position and orientation of a link."""
    link_index = env.scene["robot"].data.body_names.index(link_name)
    link_states = env.scene._articulations["robot"]._data.body_link_state_w
    link_pose = link_states[:, link_index, :7]
    return link_pose[:, :3], link_pose[:, 3:7]


def calculate_rotation_error(current_rot, target_rot):
    """Calculate the rotation error between current and target orientations in axis-angle format."""
    if isinstance(target_rot, torch.Tensor):
        target_rot_tensor = (
            target_rot.unsqueeze(0).expand(current_rot.shape[0], -1) if target_rot.dim() == 1 else target_rot
        )
    else:
        target_rot_tensor = torch.tensor(target_rot, device=current_rot.device)
        if target_rot_tensor.dim() == 1:
            target_rot_tensor = target_rot_tensor.unsqueeze(0).expand(current_rot.shape[0], -1)

    return axis_angle_from_quat(
        quat_from_matrix(matrix_from_quat(target_rot_tensor) * matrix_from_quat(quat_inv(current_rot)))
    )


def compute_errors(test_setup, env, left_target_pose, right_target_pose):
    """Compute all error metrics for the current state."""
    action_term = test_setup["action_term"]
    pink_controllers = test_setup["pink_controllers"]
    articulation = test_setup["articulation"]
    kinematics_model = test_setup["kinematics_model"]
    left_target_link_name = test_setup["left_target_link_name"]
    right_target_link_name = test_setup["right_target_link_name"]
    num_joints_in_robot_hands = test_setup["num_joints_in_robot_hands"]

    # Get current hand positions and orientations
    left_hand_pos, left_hand_rot = get_link_pose(env, left_target_link_name)
    right_hand_pos, right_hand_rot = get_link_pose(env, right_target_link_name)

    # Create setpoint tensors
    device = env.device
    num_envs = env.num_envs
    left_hand_pose_setpoint = torch.tensor(left_target_pose, device=device).unsqueeze(0).repeat(num_envs, 1)
    right_hand_pose_setpoint = torch.tensor(right_target_pose, device=device).unsqueeze(0).repeat(num_envs, 1)
    # compensate for the hand rotational offset
    # nominal_hand_pose_rotmat = matrix_from_quat(torch.tensor(env_cfg.actions.pink_ik_cfg.controller.hand_rotational_offset, device=env.device))
    left_hand_pose_setpoint[:, 3:7] = quat_from_matrix(matrix_from_quat(left_hand_pose_setpoint[:, 3:7]))
    right_hand_pose_setpoint[:, 3:7] = quat_from_matrix(matrix_from_quat(right_hand_pose_setpoint[:, 3:7]))

    # Calculate position and rotation errors
    left_pos_error = left_hand_pose_setpoint[:, :3] - left_hand_pos
    right_pos_error = right_hand_pose_setpoint[:, :3] - right_hand_pos
    left_rot_error = calculate_rotation_error(left_hand_rot, left_hand_pose_setpoint[:, 3:])
    right_rot_error = calculate_rotation_error(right_hand_rot, right_hand_pose_setpoint[:, 3:])

    # Calculate PD controller errors
    ik_controller = pink_controllers[0]
    pink_controlled_joint_ids = action_term._pink_controlled_joint_ids

    # Get current and target positions
    curr_joints = articulation.data.joint_pos[:, pink_controlled_joint_ids].cpu().numpy()[0]
    target_joints = action_term.processed_actions[:, :num_joints_in_robot_hands].cpu().numpy()[0]

    # Reorder joints for Pink IK
    curr_joints = np.array(curr_joints)[ik_controller.isaac_lab_to_pink_ordering]
    target_joints = np.array(target_joints)[ik_controller.isaac_lab_to_pink_ordering]

    # Run forward kinematics
    kinematics_model.update(curr_joints)
    left_curr_pos = kinematics_model.get_transform_frame_to_world(
        frame="GR1T2_fourier_hand_6dof_left_hand_pitch_link"
    ).translation
    right_curr_pos = kinematics_model.get_transform_frame_to_world(
        frame="GR1T2_fourier_hand_6dof_right_hand_pitch_link"
    ).translation

    kinematics_model.update(target_joints)
    left_target_pos = kinematics_model.get_transform_frame_to_world(
        frame="GR1T2_fourier_hand_6dof_left_hand_pitch_link"
    ).translation
    right_target_pos = kinematics_model.get_transform_frame_to_world(
        frame="GR1T2_fourier_hand_6dof_right_hand_pitch_link"
    ).translation

    # Calculate PD errors
    left_pd_error = (
        torch.tensor(left_target_pos - left_curr_pos, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(num_envs, 1)
    )
    right_pd_error = (
        torch.tensor(right_target_pos - right_curr_pos, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(num_envs, 1)
    )

    return {
        "left_pos_error": left_pos_error,
        "right_pos_error": right_pos_error,
        "left_rot_error": left_rot_error,
        "right_rot_error": right_rot_error,
        "left_pd_error": left_pd_error,
        "right_pd_error": right_pd_error,
    }


def verify_errors(errors, test_setup, tolerances):
    """Verify that all error metrics are within tolerance."""
    env = test_setup["env"]
    device = env.device
    num_envs = env.num_envs
    zero_tensor = torch.zeros(num_envs, device=device)

    for hand in ["left", "right"]:
        # Check PD controller errors
        pd_error_norm = torch.norm(errors[f"{hand}_pd_error"], dim=1)
        torch.testing.assert_close(
            pd_error_norm,
            zero_tensor,
            rtol=0.0,
            atol=tolerances["pd_position"],
            msg=(
                f"{hand.capitalize()} hand PD controller error ({pd_error_norm.item():.6f}) exceeds tolerance"
                f" ({tolerances['pd_position']:.6f})"
            ),
        )

        # Check IK position errors
        pos_error_norm = torch.norm(errors[f"{hand}_pos_error"], dim=1)
        torch.testing.assert_close(
            pos_error_norm,
            zero_tensor,
            rtol=0.0,
            atol=tolerances["position"],
            msg=(
                f"{hand.capitalize()} hand IK position error ({pos_error_norm.item():.6f}) exceeds tolerance"
                f" ({tolerances['position']:.6f})"
            ),
        )

        # Check rotation errors
        rot_error_max = torch.max(errors[f"{hand}_rot_error"])
        torch.testing.assert_close(
            rot_error_max,
            torch.zeros_like(rot_error_max),
            rtol=0.0,
            atol=tolerances["rotation"],
            msg=(
                f"{hand.capitalize()} hand IK rotation error ({rot_error_max.item():.6f}) exceeds tolerance"
                f" ({tolerances['rotation']:.6f})"
            ),
        )


def print_debug_info(errors, test_counter):
    """Print debug information about the current state."""
    print(f"\nTest iteration {test_counter + 1}:")
    for hand in ["left", "right"]:
        print(f"Measured {hand} hand position error:", errors[f"{hand}_pos_error"])
        print(f"Measured {hand} hand rotation error:", errors[f"{hand}_rot_error"])
        print(f"Measured {hand} hand PD error:", errors[f"{hand}_pd_error"])
