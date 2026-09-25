# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the SO101 Keyboard task's SysID reset pose."""

import pytest
import torch

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_assets.robots.so101 import SO101_CFG


@pytest.mark.parametrize("physics", ("newton_mjwarp", "isaacsim_physx"))
def test_keyboard_robot_faces_keyboard(physics):
    """The SysID arm's native -Y working direction points toward the keyboard at +X."""
    shared_rotation = SO101_CFG.init_state.rot
    cfg = parse_env_cfg("IsaacContrib-Keyboard-SO101", device="cpu", num_envs=1, overrides=[f"physics={physics}"])

    direction = quat_apply(torch.tensor([cfg.scene.robot.init_state.rot]), torch.tensor([[0.0, -1.0, 0.0]]))

    torch.testing.assert_close(direction, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6, rtol=0)
    assert SO101_CFG.init_state.rot == shared_rotation


@pytest.mark.parametrize("physics", ("newton_mjwarp", "isaacsim_physx"))
def test_keyboard_zero_joint_seed_is_task_local(physics):
    """Reset IK starts at the original task seed without changing the shared asset."""
    shared_joint_pos = SO101_CFG.init_state.joint_pos.copy()
    cfg = parse_env_cfg("IsaacContrib-Keyboard-SO101", device="cpu", num_envs=1, overrides=[f"physics={physics}"])

    assert cfg.scene.robot.init_state.joint_pos == dict.fromkeys(
        ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"), 0.0
    )
    cfg.scene.robot.init_state.joint_pos["shoulder_pan"] = 0.5
    assert SO101_CFG.init_state.joint_pos == shared_joint_pos
