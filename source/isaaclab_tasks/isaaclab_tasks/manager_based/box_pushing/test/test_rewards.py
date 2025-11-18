# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch


# Fancy Gym Reward Functions (simulating based on the provided function)
def fancy_gym_tcp_box_dist_reward(box_pos, rod_tip_pos):
    return -2 * np.clip(np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100)


def fancy_gym_box_goal_pos_dist_reward(box_pos, target_pos):
    return -3.5 * np.linalg.norm(box_pos - target_pos)


def fancy_gym_box_goal_rot_dist_reward(box_quat, target_quat):
    return -rotation_distance(box_quat, target_quat) / np.pi  # Corrected without *100


def fancy_gym_energy_cost(action):
    return -0.0005 * np.sum(np.square(action))


def fancy_gym_joint_velocity_penalty(qvel, q_dot_max):
    v_coeff = 1.0
    q_dot_error = np.abs(qvel) - np.abs(q_dot_max)
    penalty = -v_coeff * np.sum(q_dot_error[q_dot_error > 0])
    return penalty


# Isaaclab Reward Functions (without compensation terms)
def isaaclab_object_ee_distance(cube_pos_w, ee_w):
    distance = torch.linalg.norm(cube_pos_w - ee_w, dim=1)
    return -2.0 * torch.clamp(distance, min=0.05, max=100).item()


def isaaclab_object_goal_position_distance(object_pos, goal_pos):
    distance = torch.linalg.norm(object_pos - goal_pos, dim=1)
    return -3.5 * distance.item()


def isaaclab_object_goal_orientation_distance(current_orientation, desired_orientation):
    orientation_distance = torch.linalg.norm(current_orientation - desired_orientation, dim=1)
    return (-1.0 / torch.pi) * orientation_distance.item()


def isaaclab_energy_cost(action):
    return -5e-4 * torch.sum(action**2).item()


def isaaclab_joint_vel_limits_bp(joint_vel, arm_dof_vel_max, soft_ratio):
    out_of_limits = torch.abs(joint_vel) - arm_dof_vel_max
    mask = out_of_limits > 0
    out_of_limits = torch.where(mask, out_of_limits, torch.tensor(0.0))
    penalty = soft_ratio * torch.sum(out_of_limits)
    return -1.0 * penalty.item()


# Rotation distance function (shared)
def rotation_distance(quat1, quat2):
    return np.linalg.norm(quat1 - quat2)


test_cases = {
    # Normal test case for Box-goal orientation distance
    "Box-goal orientation distance (Normal)": {
        "fancy_gym_func": fancy_gym_box_goal_rot_dist_reward,
        "isaaclab_func": isaaclab_object_goal_orientation_distance,
        "inputs": [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.7071, 0.7071, 0.0, 0.0]),
        ],  # 90-degree difference
        "isaac_inputs": [
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.7071, 0.7071, 0.0, 0.0]]),
        ],
    },
    # Edge case for Box-goal orientation distance (Identical orientations)
    "Box-goal orientation distance (Identical orientations)": {
        "fancy_gym_func": fancy_gym_box_goal_rot_dist_reward,
        "isaaclab_func": isaaclab_object_goal_orientation_distance,
        "inputs": [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        ],  # Identical orientations
        "isaac_inputs": [
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ],
    },
    # Edge case for Box-goal orientation distance (Opposite orientations)
    "Box-goal orientation distance (Opposite orientations)": {
        "fancy_gym_func": fancy_gym_box_goal_rot_dist_reward,
        "isaaclab_func": isaaclab_object_goal_orientation_distance,
        "inputs": [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
        ],  # Opposite orientations
        "isaac_inputs": [
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        ],
    },
    # Normal test case for Box-goal position distance
    "Box-goal position distance (Normal)": {
        "fancy_gym_func": fancy_gym_box_goal_pos_dist_reward,
        "isaaclab_func": isaaclab_object_goal_position_distance,
        "inputs": [
            np.array([5.0, 5.0, 5.0]),
            np.array([0.0, 0.0, 0.0]),
        ],  # A reasonable distance between box and target
        "isaac_inputs": [
            torch.tensor([[5.0, 5.0, 5.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
        ],
    },
    # Edge case for Box-goal position distance (Far distance)
    "Box-goal position distance (Far distance)": {
        "fancy_gym_func": fancy_gym_box_goal_pos_dist_reward,
        "isaaclab_func": isaaclab_object_goal_position_distance,
        "inputs": [
            np.array([100.0, 100.0, 100.0]),
            np.array([0.0, 0.0, 0.0]),
        ],  # Large distance
        "isaac_inputs": [
            torch.tensor([[100.0, 100.0, 100.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
        ],
    },
    # Edge case for Box-goal position distance (Zero distance)
    "Box-goal position distance (Zero distance)": {
        "fancy_gym_func": fancy_gym_box_goal_pos_dist_reward,
        "isaaclab_func": isaaclab_object_goal_position_distance,
        "inputs": [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        ],  # Zero distance
        "isaac_inputs": [
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
        ],
    },
    # Normal test case for Energy cost
    "Energy cost (Normal)": {
        "fancy_gym_func": fancy_gym_energy_cost,
        "isaaclab_func": isaaclab_energy_cost,
        "inputs": [np.array([0.5, -0.3, 0.2])],  # Small action values
        "isaac_inputs": [torch.tensor([0.5, -0.3, 0.2])],
    },
    # Edge case for Energy cost (Large action)
    "Energy cost (Large action)": {
        "fancy_gym_func": fancy_gym_energy_cost,
        "isaaclab_func": isaaclab_energy_cost,
        "inputs": [np.array([10.0, 10.0, 10.0])],  # Large action values
        "isaac_inputs": [torch.tensor([10.0, 10.0, 10.0])],
    },
    # Edge case for Energy cost (Zero action)
    "Energy cost (Zero action)": {
        "fancy_gym_func": fancy_gym_energy_cost,
        "isaaclab_func": isaaclab_energy_cost,
        "inputs": [np.array([0.0, 0.0, 0.0])],  # No movement (zero action)
        "isaac_inputs": [torch.tensor([0.0, 0.0, 0.0])],
    },
    # Normal test case for Joint velocity penalty
    "Joint velocity penalty (Normal)": {
        "fancy_gym_func": fancy_gym_joint_velocity_penalty,
        "isaaclab_func": isaaclab_joint_vel_limits_bp,
        "inputs": [
            np.array([2.0, 2.1, 1.8, 2.2, 2.5, 2.3, 2.6]),
            np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
        ],  # Some velocities slightly over limit
        "isaac_inputs": [
            torch.tensor([2.0, 2.1, 1.8, 2.2, 2.5, 2.3, 2.6]),
            torch.tensor([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
            1.0,
        ],
    },
    # Edge case for Joint velocity penalty (Exceeding velocity)
    "Joint velocity penalty (Exceeding velocity)": {
        "fancy_gym_func": fancy_gym_joint_velocity_penalty,
        "isaaclab_func": isaaclab_joint_vel_limits_bp,
        "inputs": [
            np.array([5.0, 4.0, 6.0, 5.0, 7.0, 6.0, 6.5]),
            np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
        ],
        "isaac_inputs": [
            torch.tensor([5.0, 4.0, 6.0, 5.0, 7.0, 6.0, 6.5]),
            torch.tensor([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
            1.0,
        ],
    },
    # Edge case for Joint velocity penalty (Zero velocity)
    "Joint velocity penalty (Zero velocity)": {
        "fancy_gym_func": fancy_gym_joint_velocity_penalty,
        "isaaclab_func": isaaclab_joint_vel_limits_bp,
        "inputs": [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
        ],
        "isaac_inputs": [
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
            1.0,
        ],
    },
    # Normal test case for TCP-box distance
    "TCP-box distance (Normal)": {
        "fancy_gym_func": fancy_gym_tcp_box_dist_reward,
        "isaaclab_func": isaaclab_object_ee_distance,
        "inputs": [
            np.array([2.0, 3.0, 1.0]),
            np.array([1.5, 2.5, 1.5]),
        ],  # Reasonable distance
        "isaac_inputs": [
            torch.tensor([[2.0, 3.0, 1.0]]),
            torch.tensor([[1.5, 2.5, 1.5]]),
        ],
    },
    # Edge case for TCP-box distance (Max value)
    "TCP-box distance (Max value)": {
        "fancy_gym_func": fancy_gym_tcp_box_dist_reward,
        "isaaclab_func": isaaclab_object_ee_distance,
        "inputs": [
            np.array([200.0, 200.0, 200.0]),
            np.array([0.0, 0.0, 0.0]),
        ],  # Distance > 100
        "isaac_inputs": [
            torch.tensor([[200.0, 200.0, 200.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
        ],
    },
    # Edge case for TCP-box distance (Min value)
    "TCP-box distance (Min value)": {
        "fancy_gym_func": fancy_gym_tcp_box_dist_reward,
        "isaaclab_func": isaaclab_object_ee_distance,
        "inputs": [
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
        ],  # Distance < 0.05
        "isaac_inputs": [
            torch.tensor([[1.0, 1.0, 1.0]]),
            torch.tensor([[1.0, 1.0, 1.0]]),
        ],
    },
}


# Function to compare the reward terms and print results
def run_test_cases():
    passed = 0
    failed = 0
    for case_name, case_data in test_cases.items():
        fancy_gym_func = case_data["fancy_gym_func"]
        isaaclab_func = case_data["isaaclab_func"]

        # Calculate rewards for Fancy Gym
        fancy_gym_result = fancy_gym_func(*case_data["inputs"])

        # Calculate rewards for Isaaclab
        isaaclab_result = isaaclab_func(*case_data["isaac_inputs"])

        # Print the results and comparison
        print(f"Test Case: {case_name}")
        print(f"  Fancy Gym result: {fancy_gym_result}")
        print(f"  Isaaclab result: {isaaclab_result}")

        if np.isclose(fancy_gym_result, isaaclab_result, atol=1e-6):
            print("  Result: MATCH")
            passed += 1
        else:
            print("  Result: MISMATCH")
            failed += 1
        print("-" * 40)

    # Summary
    print(f"Summary: {passed} tests passed, {failed} tests failed.")


# Run the test cases
run_test_cases()
