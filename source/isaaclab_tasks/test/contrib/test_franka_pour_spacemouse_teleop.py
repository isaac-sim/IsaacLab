# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Franka Pour joint-position SpaceMouse adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import FrankaPourEnvCfg_TELEOP


def _load_teleop_script():
    repo_root = Path(__file__).parents[4]
    script_path = repo_root / "scripts/environments/teleoperation/teleop_franka_pour_spacemouse.py"
    spec = importlib.util.spec_from_file_location("teleop_franka_pour_spacemouse", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_joint_targets_are_encoded_as_joint_position_actions():
    teleop = _load_teleop_script()
    default_joint_pos = torch.tensor([[0.0, 1.0, -1.0]])
    joint_targets = torch.tensor([[0.25, 0.5, 2.0]])
    lower_limits = torch.tensor([[-1.0, -1.0, -1.5]])
    upper_limits = torch.tensor([[1.0, 2.0, 1.5]])

    actions = teleop.joint_targets_to_actions(
        joint_targets=joint_targets,
        action_offset=default_joint_pos,
        action_scale=0.5,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )

    torch.testing.assert_close(actions, torch.tensor([[0.5, -1.0, 5.0]]))


def test_joint_targets_support_per_joint_action_scales():
    teleop = _load_teleop_script()

    actions = teleop.joint_targets_to_actions(
        joint_targets=torch.tensor([[0.5, 1.5, -0.5]]),
        action_offset=torch.zeros((1, 3)),
        action_scale=torch.tensor([[0.5, 1.5, 0.5]]),
        lower_limits=torch.full((1, 3), -2.0),
        upper_limits=torch.full((1, 3), 2.0),
    )

    torch.testing.assert_close(actions, torch.tensor([[1.0, 1.0, -1.0]]))


def test_environment_action_keeps_gripper_separate_from_seven_arm_joints():
    teleop = _load_teleop_script()
    arm_action = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    gripper_command = torch.tensor([1.0, -1.0])

    action = teleop.compose_env_action(arm_action, gripper_command)

    assert action.shape == (2, 8)
    torch.testing.assert_close(action[:, :7], arm_action)
    torch.testing.assert_close(action[:, 7], gripper_command)


def test_tcp_offset_updates_the_translational_jacobian():
    teleop = _load_teleop_script()
    jacobian = torch.zeros((1, 6, 3))
    jacobian[:, 3:, :] = torch.eye(3)
    body_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    offset_pos = torch.tensor([[0.0, 0.0, 0.1]])

    shifted = teleop.apply_tcp_offset_to_jacobian(jacobian, body_quat, offset_pos)

    expected_linear = torch.tensor([[[0.0, 0.1, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    torch.testing.assert_close(shifted[:, :3], expected_linear)
    torch.testing.assert_close(shifted[:, 3:], torch.eye(3).unsqueeze(0))


def test_tcp_offset_is_rotated_from_the_hand_frame_before_shifting_the_jacobian():
    teleop = _load_teleop_script()
    jacobian = torch.zeros((1, 6, 3))
    jacobian[:, 3:, :] = torch.eye(3)
    sin_cos_45 = 2.0**-0.5
    body_quat = torch.tensor([[0.0, 0.0, sin_cos_45, sin_cos_45]])
    offset_pos = torch.tensor([[0.1, 0.0, 0.0]])

    shifted = teleop.apply_tcp_offset_to_jacobian(jacobian, body_quat, offset_pos)

    expected_linear = torch.tensor([[[0.0, 0.0, -0.1], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    torch.testing.assert_close(shifted[:, :3], expected_linear)
    torch.testing.assert_close(shifted[:, 3:], torch.eye(3).unsqueeze(0))


def test_teleop_uses_the_registered_joint_position_task_and_kitless_launcher():
    repo_root = Path(__file__).parents[4]
    script = repo_root / "scripts/environments/teleoperation/teleop_franka_pour_spacemouse.py"
    source = script.read_text(encoding="utf-8")

    assert 'DEFAULT_TASK = "Isaac-Pour-Franka-Teleop-v0"' in source
    assert "launch_simulation" in source
    assert "AppLauncher" not in source


def test_teleop_config_finalizes_without_trajectory_controller_or_rl_distribution():
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401

    cfg = FrankaPourEnvCfg_TELEOP().finalize()
    task_spec = gym.spec("Isaac-Pour-Franka-Teleop-v0")

    assert cfg.actions.arm_action.class_type.__name__ == "CurriculumJointPositionAction"
    assert cfg.actions.gripper_action.force_open_before_phase_stage == -1
    assert cfg.actions.gripper_action.limit_to_preload is False
    assert cfg.actions.gripper_action.default_position == pytest.approx(cfg.gripper_open_pos)
    assert cfg.actions.gripper_action.neutral_position == pytest.approx(cfg.gripper_open_pos)
    assert cfg.actions.gripper_action.scale == pytest.approx(
        cfg.gripper_open_pos - cfg.actions.gripper_action.close_position
    )
    assert cfg.terminations.time_out is None
    assert "rsl_rl_cfg_entry_point" not in task_spec.kwargs
