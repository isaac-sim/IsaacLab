# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the pouring task's action terms."""

import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointAction

from isaaclab_tasks.contrib.franka_pour.mdp.actions import CurriculumGripperPositionAction


def test_gripper_command_is_filtered_toward_each_new_binary_target(monkeypatch):
    """The moving average blends the previous filtered target with the new binary command, written in place."""
    action = object.__new__(CurriculumGripperPositionAction)
    action._alpha = 0.5
    action._processed_actions = torch.zeros(2, 2)
    action._previous_target = action._processed_actions.clone()
    processed = action._processed_actions

    def write_binary_command(self, actions):
        # the binary term writes its open/close command into the processed actions in place
        self._processed_actions.fill_(1.0)

    monkeypatch.setattr(BinaryJointAction, "process_actions", write_binary_command)
    for expected in (0.5, 0.75, 0.875):
        action.process_actions(torch.ones(2, 1))
        torch.testing.assert_close(action._processed_actions, torch.full((2, 2), expected))
    assert action._processed_actions is processed
