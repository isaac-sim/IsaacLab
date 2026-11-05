# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit coverage for Pink hand-target ordering across articulation backends."""

from types import SimpleNamespace

import torch

from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from isaaclab.utils.string import resolve_matching_names


def test_pink_hand_commands_reach_the_configured_joints():
    """A non-self-inverse hand permutation must preserve physical joint targets."""

    class _Asset:
        data = SimpleNamespace(joint_names=["arm", "hand_a", "hand_b", "hand_c"])

        def __init__(self):
            self.joint_targets = torch.zeros((1, 4))

        def find_joints(self, expressions, preserve_order=False):
            return resolve_matching_names(expressions, self.data.joint_names, preserve_order=preserve_order)

        def set_joint_position_target_index(self, target, joint_ids):
            self.joint_targets[:, joint_ids] = target

    action = PinkInverseKinematicsAction.__new__(PinkInverseKinematicsAction)
    action._asset = _Asset()
    action.cfg = SimpleNamespace(
        pink_controlled_joint_names=["arm"],
        hand_joint_names=["hand_c", "hand_a", "hand_b"],
        controller=SimpleNamespace(),
        enable_gravity_compensation=False,
    )

    action._initialize_joint_info()

    action._compute_ik_solutions = lambda: torch.tensor([[9.0]])
    action._target_hand_joint_positions = torch.tensor([[30.0, 10.0, 20.0]])
    action.apply_actions()

    torch.testing.assert_close(action._asset.joint_targets, torch.tensor([[9.0, 10.0, 20.0, 30.0]]))
