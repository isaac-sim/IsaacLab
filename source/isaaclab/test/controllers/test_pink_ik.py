# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import contextlib
import json
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import warp as wp
from pink.configuration import Configuration
from pink.tasks import FrameTask

import isaaclab.sim as sim_utils
from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_from_matrix, quat_inv

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def load_test_config(env_name):
    """Load test configuration based on environment type."""
    # Determine which config file to load based on environment name
    if "G1" in env_name:
        config_file = "pink_ik_g1_test_configs.json"
    elif "GR1" in env_name:
        config_file = "pink_ik_gr1_test_configs.json"
    else:
        raise ValueError(f"Unknown environment type in {env_name}. Expected G1 or GR1.")

    config_path = Path(__file__).parent / "test_ik_configs" / config_file
    with open(config_path) as f:
        return json.load(f)


def is_waist_enabled(env_cfg):
    """Check if waist joints are enabled in the environment configuration."""
    if not hasattr(env_cfg.actions, "upper_body_ik"):
        return False

    pink_controlled_joints = env_cfg.actions.upper_body_ik.pink_controlled_joint_names

    # Also check for pattern-based joint names (e.g., "waist_.*_joint")
    return any(re.match("waist", joint) for joint in pink_controlled_joints)


def create_test_env(env_name, num_envs):
    """Create a test environment with the Pink IK controller."""
    device = "cuda:0"

    sim_utils.create_new_stage()

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


@pytest.fixture(
    scope="module",
    params=[
        "Isaac-PickPlace-GR1T2-Abs-v0",
        "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
        "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0",
        "Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
    ],
)
def env_and_cfg(request):
    """Create environment and configuration for tests."""
    env_name = request.param

    # Load the appropriate test configuration based on environment type
    test_cfg = load_test_config(env_name)

    env, env_cfg = create_test_env(env_name, num_envs=1)

    # Get only the FrameTasks from variable_input_tasks
    variable_input_tasks = [
        task for task in env_cfg.actions.upper_body_ik.controller.variable_input_tasks if isinstance(task, FrameTask)
    ]
    assert len(variable_input_tasks) == 2, "Expected exactly two FrameTasks (left and right hand)."
    frames = [task.frame for task in variable_input_tasks]
    # Try to infer which is left and which is right
    left_candidates = [f for f in frames if "left" in f.lower()]
    right_candidates = [f for f in frames if "right" in f.lower()]
    assert len(left_candidates) == 1 and len(right_candidates) == 1, (
        f"Could not uniquely identify left/right frames from: {frames}"
    )
    left_eef_urdf_link_name = left_candidates[0]
    right_eef_urdf_link_name = right_candidates[0]

    # Set up camera view
    env.sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 1.0])

    # Create test parameters from test_cfg
    test_params = {
        "position": test_cfg["tolerances"]["position"],
        "rotation": test_cfg["tolerances"]["rotation"],
        "pd_position": test_cfg["tolerances"]["pd_position"],
        "check_errors": test_cfg["tolerances"]["check_errors"],
        "left_eef_urdf_link_name": left_eef_urdf_link_name,
        "right_eef_urdf_link_name": right_eef_urdf_link_name,
    }

    try:
        yield env, env_cfg, test_cfg, test_params
    finally:
        env.close()


@pytest.fixture
def test_setup(env_and_cfg):
    """Set up test case - runs before each test."""
    env, env_cfg, test_cfg, test_params = env_and_cfg

    num_joints_in_robot_hands = env_cfg.actions.upper_body_ik.controller.num_hand_joints

    # Get Action Term and IK controller
    action_term = env.action_manager.get_term(name="upper_body_ik")
    pink_controllers = action_term._ik_controllers
    articulation = action_term._asset

    # Initialize Pink Configuration for forward kinematics
    test_kinematics_model = Configuration(
        pink_controllers[0].pink_configuration.model,
        pink_controllers[0].pink_configuration.data,
        pink_controllers[0].pink_configuration.q,
    )
    left_target_link_name = env_cfg.actions.upper_body_ik.target_eef_link_names["left_wrist"]
    right_target_link_name = env_cfg.actions.upper_body_ik.target_eef_link_names["right_wrist"]

    return {
        "env": env,
        "env_cfg": env_cfg,
        "test_cfg": test_cfg,
        "test_params": test_params,
        "num_joints_in_robot_hands": num_joints_in_robot_hands,
        "action_term": action_term,
        "pink_controllers": pink_controllers,
        "articulation": articulation,
        "test_kinematics_model": test_kinematics_model,
        "left_target_link_name": left_target_link_name,
        "right_target_link_name": right_target_link_name,
        "left_eef_urdf_link_name": test_params["left_eef_urdf_link_name"],
        "right_eef_urdf_link_name": test_params["right_eef_urdf_link_name"],
    }


@pytest.mark.parametrize(
    "test_name",
    [
        "horizontal_movement",
        "horizontal_small_movement",
        "stay_still",
        "forward_waist_bending_movement",
        "vertical_movement",
        "rotation_movements",
    ],
)
def test_movement_types(test_setup, test_name):
    """Test different movement types using parametrization."""
    test_cfg = test_setup["test_cfg"]
    env_cfg = test_setup["env_cfg"]

    if test_name not in test_cfg["tests"]:
        print(f"Skipping {test_name} test for {env_cfg.__class__.__name__} environment (test not defined)...")
        pytest.skip(f"Test {test_name} not defined for {env_cfg.__class__.__name__}")
        return

    test_config = test_cfg["tests"][test_name]

    # Check if test requires waist bending and if waist is enabled
    requires_waist_bending = test_config.get("requires_waist_bending", False)
    waist_enabled = is_waist_enabled(env_cfg)

    if requires_waist_bending and not waist_enabled:
        print(
            f"Skipping {test_name} test because it requires waist bending but waist is not enabled in"
            f" {env_cfg.__class__.__name__}..."
        )
        pytest.skip(f"Test {test_name} requires waist bending but waist is not enabled")
        return

    print(f"Running {test_name} test...")
    run_movement_test(test_setup, test_config, test_cfg)


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

        # Make the first phase longer than subsequent ones
        initial_steps = test_cfg["allowed_steps_to_settle"]
        phase = "initial"
        steps_in_phase = 0

        while simulation_app.is_running() and not simulation_app.is_exiting():
            num_runs += 1
            steps_in_phase += 1

            # Call auxiliary function if provided
            if aux_function is not None:
                aux_function(num_runs)

            # Create actions from hand poses and joint positions
            setpoint_poses = np.concatenate([left_hand_poses[curr_pose_idx], right_hand_poses[curr_pose_idx]])
            actions = np.concatenate([setpoint_poses, np.zeros(num_joints_in_robot_hands)])
            actions = torch.tensor(actions, device=env.device, dtype=torch.float32)
            # Append base command for Locomanipulation environments with fixed height
            if test_setup["env_cfg"].__class__.__name__ == "LocomanipulationG1EnvCfg":
                # Use a named variable for base height for clarity and maintainability
                BASE_HEIGHT = 0.72
                base_command = torch.zeros(4, device=env.device, dtype=actions.dtype)
                base_command[3] = BASE_HEIGHT
                actions = torch.cat([actions, base_command])
            actions = actions.repeat(env.num_envs, 1)

            # Step environment
            obs, _, _, _, _ = env.step(actions)

            # Determine the step interval for error checking
            if phase == "initial":
                check_interval = initial_steps
            else:
                check_interval = test_config["allowed_steps_per_motion"]

            # Check convergence and verify errors
            if steps_in_phase % check_interval == 0:
                print("Computing errors...")
                errors = compute_errors(
                    test_setup,
                    env,
                    left_hand_poses[curr_pose_idx],
                    right_hand_poses[curr_pose_idx],
                    test_setup["left_eef_urdf_link_name"],
                    test_setup["right_eef_urdf_link_name"],
                )
                print_debug_info(errors, test_counter)
                test_params = test_setup["test_params"]
                if test_params["check_errors"]:
                    verify_errors(errors, test_setup, test_params)
                num_runs += 1

                curr_pose_idx = (curr_pose_idx + 1) % len(left_hand_poses)
                if curr_pose_idx == 0:
                    test_counter += 1
                    if test_counter > test_config["repeat"]:
                        print("Test completed successfully")
                        break
                # After the first phase, switch to normal interval
                if phase == "initial":
                    phase = "normal"
                    steps_in_phase = 0


def get_link_pose(env, link_name):
    """Get the position and orientation of a link."""
    link_index = env.scene["robot"].data.body_names.index(link_name)
    link_states = wp.to_torch(env.scene._articulations["robot"].data.body_link_state_w)
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


def compute_errors(
    test_setup, env, left_target_pose, right_target_pose, left_eef_urdf_link_name, right_eef_urdf_link_name
):
    """Compute all error metrics for the current state."""
    action_term = test_setup["action_term"]
    pink_controllers = test_setup["pink_controllers"]
    articulation = test_setup["articulation"]
    test_kinematics_model = test_setup["test_kinematics_model"]
    left_target_link_name = test_setup["left_target_link_name"]
    right_target_link_name = test_setup["right_target_link_name"]

    # Get current hand positions and orientations
    left_hand_pos, left_hand_rot = get_link_pose(env, left_target_link_name)
    right_hand_pos, right_hand_rot = get_link_pose(env, right_target_link_name)

    # Create setpoint tensors
    device = env.device
    num_envs = env.num_envs
    left_hand_pose_setpoint = torch.tensor(left_target_pose, device=device).unsqueeze(0).repeat(num_envs, 1)
    right_hand_pose_setpoint = torch.tensor(right_target_pose, device=device).unsqueeze(0).repeat(num_envs, 1)

    # Calculate position and rotation errors
    left_pos_error = left_hand_pose_setpoint[:, :3] - left_hand_pos
    right_pos_error = right_hand_pose_setpoint[:, :3] - right_hand_pos
    left_rot_error = calculate_rotation_error(left_hand_rot, left_hand_pose_setpoint[:, 3:])
    right_rot_error = calculate_rotation_error(right_hand_rot, right_hand_pose_setpoint[:, 3:])

    # Calculate PD controller errors
    ik_controller = pink_controllers[0]
    isaaclab_controlled_joint_ids = action_term._isaaclab_controlled_joint_ids

    # Get current and target positions for controlled joints only
    curr_joints = wp.to_torch(articulation.data.joint_pos)[:, isaaclab_controlled_joint_ids].cpu().numpy()[0]
    target_joints = action_term.processed_actions[:, : len(isaaclab_controlled_joint_ids)].cpu().numpy()[0]

    # Reorder joints for Pink IK (using controlled joint ordering)
    curr_joints = np.array(curr_joints)[ik_controller.isaac_lab_to_pink_controlled_ordering]
    target_joints = np.array(target_joints)[ik_controller.isaac_lab_to_pink_controlled_ordering]

    # Run forward kinematics
    test_kinematics_model.update(curr_joints)
    left_curr_pos = test_kinematics_model.get_transform_frame_to_world(frame=left_eef_urdf_link_name).translation
    right_curr_pos = test_kinematics_model.get_transform_frame_to_world(frame=right_eef_urdf_link_name).translation

    test_kinematics_model.update(target_joints)
    left_target_pos = test_kinematics_model.get_transform_frame_to_world(frame=left_eef_urdf_link_name).translation
    right_target_pos = test_kinematics_model.get_transform_frame_to_world(frame=right_eef_urdf_link_name).translation

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
        pd_error_norm = torch.linalg.norm(errors[f"{hand}_pd_error"], dim=1)
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
        pos_error_norm = torch.linalg.norm(errors[f"{hand}_pos_error"], dim=1)
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
