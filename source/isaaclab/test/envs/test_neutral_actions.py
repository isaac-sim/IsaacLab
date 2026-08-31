# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for action-term zero actions."""

from types import SimpleNamespace

import torch

from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from isaaclab.managers.action_manager import ActionManager


def test_action_manager_dispatches_none_and_records_zero_action() -> None:
    """The action manager lets terms resolve None while recording a conceptual zero action."""

    class _ActionTerm:
        action_dim = 2
        raw_actions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        received_none = False

        def process_actions(self, actions: torch.Tensor | None) -> None:
            assert actions is None
            self.received_none = True

    manager = object.__new__(ActionManager)
    term = _ActionTerm()
    manager._terms = {"term": term}
    manager._action = torch.full((2, 2), 5.0)
    manager._prev_action = torch.full((2, 2), -1.0)
    manager._resolve_terms_handle = None

    manager.process_action(None)

    assert term.received_none
    assert torch.equal(manager.action, torch.zeros(2, 2))
    assert torch.equal(manager.prev_action, torch.full((2, 2), 5.0))


def test_pink_none_actions_use_current_frame_poses() -> None:
    """Pink IK resolves None to valid current poses and hand joint positions."""
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
    action_term.cfg = SimpleNamespace(controller=SimpleNamespace(num_hand_joints=2))
    action_term._raw_actions = torch.zeros(2, 16)
    action_term._get_base_link_frame_transform = lambda: torch.eye(4).repeat(2, 1, 1)
    action_term._extract_controlled_frame_poses = lambda actions: actions[:, :14]
    action_term._transform_poses_to_base_link_frame = lambda poses: poses
    action_term._set_task_targets = lambda poses: None

    action_term.process_actions(None)

    expected_poses = body_poses[:, [1, 0]].clone()
    expected_poses[..., :3] -= env_origins.unsqueeze(1)
    expected = torch.cat((expected_poses.flatten(start_dim=1), joint_positions[:, [1, 3]]), dim=-1)
    assert torch.equal(action_term.raw_actions, expected)
    assert torch.all(
        torch.linalg.vector_norm(action_term.raw_actions[:, :14].reshape(2, 2, 7)[..., 3:7], dim=-1) == 1.0
    )
