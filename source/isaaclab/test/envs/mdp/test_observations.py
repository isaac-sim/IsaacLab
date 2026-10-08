# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for geometric MDP observation contracts."""

import math
from types import SimpleNamespace

import torch

import isaaclab.envs.mdp as mdp


def test_pose_command_error_uses_robot_base_frame_and_xyzw_rotation_difference():
    """Translated and rotated roots must not change the target-minus-current error in base coordinates."""
    # Root is translated and rotated +90 degrees about Z. Current pose in that
    # root frame is (0.4, -0.2, 0.3), rotated +90 degrees about X. The target
    # differs by (0.1, 0.3, -0.05) and by a further +90 degrees about base Z.
    root_position = torch.tensor([[2.0, -1.0, 3.0]]).repeat(3, 1)
    root_orientation = torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]]).repeat(3, 1)
    body_position = torch.tensor([[[0.0, 0.0, 0.0], [2.2, -0.6, 3.3]]]).repeat(3, 1, 1)
    body_orientation = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]]]).repeat(3, 1, 1)
    command_pose = torch.tensor([[0.5, 0.1, 0.25, 0.5, 0.5, 0.5, 0.5]]).repeat(3, 1)
    # Quaternion signs describe the same rotations, including independently
    # changing the sign of the measured body orientation and desired command.
    command_pose[1, 3:] *= -1
    body_orientation[2, 1] *= -1
    data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=root_position),
        root_quat_w=SimpleNamespace(torch=root_orientation),
        body_pos_w=SimpleNamespace(torch=body_position),
        body_quat_w=SimpleNamespace(torch=body_orientation),
    )
    command = SimpleNamespace(robot=SimpleNamespace(data=data), body_idx=1, command=command_pose)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))

    error = mdp.pose_command_error(env, command_name="ee_pose")

    expected = torch.tensor([[0.1, 0.3, -0.05, 0.0, 0.0, math.pi / 2]]).repeat(3, 1)
    torch.testing.assert_close(error, expected, rtol=1e-5, atol=1e-6)
