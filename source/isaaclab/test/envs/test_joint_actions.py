# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, JointVelocityAction


def _make_action(action_type, default_state: torch.Tensor, joint_ids, use_default_offset: bool = True):
    action = object.__new__(action_type)
    action._raw_actions = torch.ones(default_state.shape[0], 2)
    action._offset = torch.full_like(action._raw_actions, -1.0)
    action._joint_ids = joint_ids
    action.cfg = SimpleNamespace(use_default_offset=use_default_offset)

    state_name = "default_joint_pos" if action_type is JointPositionAction else "default_joint_vel"
    action._asset = SimpleNamespace(data=SimpleNamespace(**{state_name: SimpleNamespace(torch=default_state)}))
    return action


@pytest.mark.parametrize(
    ("action_type", "default_state"),
    [
        (JointPositionAction, torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])),
        (JointVelocityAction, torch.tensor([[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0], [-6.0, -7.0, -8.0]])),
    ],
)
def test_reset_refreshes_default_offset_for_selected_environments(action_type, default_state):
    action = _make_action(action_type, default_state, torch.tensor([0, 2]))
    env_ids = torch.tensor([1])

    action.reset(env_ids)

    torch.testing.assert_close(action._offset[1], default_state[1, [0, 2]])
    torch.testing.assert_close(action._offset[[0, 2]], torch.full((2, 2), -1.0))
    torch.testing.assert_close(action._raw_actions[1], torch.zeros(2))
    torch.testing.assert_close(action._raw_actions[[0, 2]], torch.ones(2, 2))


@pytest.mark.parametrize("action_type", [JointPositionAction, JointVelocityAction])
def test_reset_refreshes_default_offset_for_all_environments(action_type):
    default_state = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    action = _make_action(action_type, default_state, slice(None))

    action.reset()

    torch.testing.assert_close(action._offset, default_state)
    torch.testing.assert_close(action._raw_actions, torch.zeros_like(action._raw_actions))


@pytest.mark.parametrize("action_type", [JointPositionAction, JointVelocityAction])
def test_reset_preserves_configured_offset(action_type):
    default_state = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    action = _make_action(action_type, default_state, slice(None), use_default_offset=False)

    action.reset([0])

    torch.testing.assert_close(action._offset, torch.full_like(action._offset, -1.0))
    torch.testing.assert_close(action._raw_actions[0], torch.zeros(2))
    torch.testing.assert_close(action._raw_actions[1], torch.ones(2))
