# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionToLimitsActionCfg
from isaaclab.envs.mdp.actions.joint_actions_to_limits import JointPositionToLimitsAction

pytestmark = pytest.mark.unit


class _JointIds:
    def __init__(self, count: int):
        self.torch = torch.arange(count, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.torch)


class _FakeAsset:
    num_joints = 2

    def find_joints(self, *_args, **_kwargs):
        return _JointIds(2), ["joint_left", "joint_right"]


class _FakeEnv:
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self.scene = {"robot": _FakeAsset()}


def test_joint_position_to_limits_io_descriptor_has_zero_offset():
    """The descriptor should report the term's implicit zero offset without accessing a missing attribute."""
    cfg = JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["joint_.*"],
        rescale_to_limits=False,
    )

    action = JointPositionToLimitsAction(cfg, _FakeEnv())
    descriptor = action.IO_descriptor

    assert descriptor.offset == 0.0
    assert descriptor.joint_names == ["joint_left", "joint_right"]
    assert descriptor.shape == (2,)
