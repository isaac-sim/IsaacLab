# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for MDP action terms."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.actions.actions_cfg import AbsBinaryJointPositionActionCfg, BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import AbsBinaryJointPositionAction, BinaryJointPositionAction


class _JointIds:
    """Stand-in for the proxy returned by ``find_joints(..., as_proxy=True)``."""

    def __init__(self, count: int):
        self.warp = torch.arange(count, dtype=torch.int32)

    def __len__(self) -> int:
        return len(self.warp)


def _make_env(joint_names: list[str], num_envs: int = 2) -> SimpleNamespace:
    """Create a minimal environment whose robot resolves every joint query to ``joint_names``."""
    robot = SimpleNamespace(
        num_joints=len(joint_names),
        find_joints=lambda *_args, **_kwargs: (_JointIds(len(joint_names)), list(joint_names)),
    )
    return SimpleNamespace(num_envs=num_envs, device="cpu", scene={"robot": robot})


@pytest.mark.parametrize(
    "action_cls, cfg_cls",
    [
        (BinaryJointPositionAction, BinaryJointPositionActionCfg),
        (AbsBinaryJointPositionAction, AbsBinaryJointPositionActionCfg),
    ],
)
def test_binary_joint_action_clips_each_processed_joint(action_cls, cfg_cls) -> None:
    """Per-joint clip ranges cover every processed joint, not just the 1-D binary input."""
    cfg = cfg_cls(
        asset_name="robot",
        joint_names=["finger_.*"],
        open_command_expr={"finger_left": 1.0, "finger_right": 2.0},
        close_command_expr={"finger_left": -1.0, "finger_right": -2.0},
        clip={"finger_left": (-0.5, 0.5), "finger_right": (-1.5, 1.5)},
    )
    action = action_cls(cfg, _make_env(["finger_left", "finger_right"]))

    action.process_actions(torch.ones((2, 1)))

    torch.testing.assert_close(action.processed_actions, torch.tensor([[0.5, 1.5], [0.5, 1.5]]))
