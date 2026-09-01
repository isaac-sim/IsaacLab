# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for semantic neutral actions."""

from types import SimpleNamespace

import torch

from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction


def test_pink_neutral_actions_use_current_frame_poses() -> None:
    """Pink IK neutral actions contain valid current poses and hand joint positions."""
    action_term = object.__new__(PinkInverseKinematicsAction)
    action_term._controlled_frame_ids = [1, 0]
    action_term._hand_joint_ids = [1, 3]

    body_poses = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], [4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.0]],
            [[7.0, 8.0, 9.0, 0.0, 1.0, 0.0, 0.0], [10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0]],
        ]
    )
    joint_positions = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    env_origins = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    action_term._asset = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=body_poses),
            joint_pos=SimpleNamespace(torch=joint_positions),
        )
    )
    action_term._env = SimpleNamespace(scene=SimpleNamespace(env_origins=env_origins))

    actions = action_term.neutral_actions

    expected_poses = body_poses[:, [1, 0]].clone()
    expected_poses[..., :3] -= env_origins.unsqueeze(1)
    expected = torch.cat((expected_poses.flatten(start_dim=1), joint_positions[:, [1, 3]]), dim=-1)
    assert torch.equal(actions, expected)
    assert torch.all(torch.linalg.vector_norm(actions[:, :14].reshape(2, 2, 7)[..., 3:7], dim=-1) == 1.0)
