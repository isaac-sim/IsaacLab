# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from isaaclab.envs.mdp.actions.actions_cfg import NonHolonomicActionCfg
from isaaclab.envs.mdp.actions.non_holonomic_actions import NonHolonomicAction

pytestmark = pytest.mark.unit


class _FakeAsset:
    def find_joints(self, name):
        mapping = {
            "base_x": ([0], ["base_x"]),
            "base_y": ([1], ["base_y"]),
            "base_yaw": ([2], ["base_yaw"]),
        }
        return mapping[name]

    def find_bodies(self, _name):
        return [0], ["base"]


class _FakeEnv:
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self.scene = {"robot": _FakeAsset()}


def test_nonholonomic_io_descriptor_without_clip():
    """Default clip=None must produce a descriptor instead of reading an uninitialized _clip attribute."""
    cfg = NonHolonomicActionCfg(
        asset_name="robot",
        body_name="base",
        x_joint_name="base_x",
        y_joint_name="base_y",
        yaw_joint_name="base_yaw",
    )

    action = NonHolonomicAction(cfg, _FakeEnv())
    descriptor = action.IO_descriptor

    assert descriptor.clip is None
    assert descriptor.shape == (2,)
    assert descriptor.x_joint_name == "base_x"
    assert descriptor.y_joint_name == "base_y"
    assert descriptor.yaw_joint_name == "base_yaw"
