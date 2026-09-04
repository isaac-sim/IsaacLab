# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
import torch

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction

pytestmark = pytest.mark.unit


class _JointIds:
    def __init__(self, count: int):
        self.warp = torch.arange(count, dtype=torch.int32)

    def __len__(self) -> int:
        return len(self.warp)


class _FakeAsset:
    num_joints = 2

    def find_joints(self, *_args, **_kwargs):
        return _JointIds(2), ["finger_left", "finger_right"]


class _FakeEnv:
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self.scene = {"robot": _FakeAsset()}


def test_binary_joint_action_clip_uses_processed_joint_dimension():
    """Per-joint clipping must cover every processed gripper joint, not the 1-D binary input."""
    cfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_.*"],
        open_command_expr={"finger_left": 1.0, "finger_right": 2.0},
        close_command_expr={"finger_left": -1.0, "finger_right": -2.0},
        clip={"finger_left": (-0.5, 0.5), "finger_right": (-1.5, 1.5)},
    )

    action = BinaryJointPositionAction(cfg, _FakeEnv())
    action.process_actions(torch.ones((2, 1), dtype=torch.float32))

    expected = torch.tensor([[0.5, 1.5], [0.5, 1.5]])
    torch.testing.assert_close(action.processed_actions, expected)
