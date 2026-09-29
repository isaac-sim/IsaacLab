# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for differential inverse-kinematics action processing."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

pytestmark = pytest.mark.unit


def test_process_actions_applies_scale_then_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """DiffIK actions use the configured affine transformation before reaching the controller."""
    num_envs = 2
    controller_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    cfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        body_name="tool",
        scale=(0.5, 2.0, -1.0),
        offset=(0.1, -0.2, 0.3),
        controller=controller_cfg,
    )
    asset = SimpleNamespace(
        find_joints=lambda _names: ([0, 1, 2], ["joint1", "joint2", "joint3"]),
        find_bodies=lambda _name: ([0], ["tool"]),
        num_joints=3,
        num_base_dofs=0,
        is_fixed_base=False,
    )
    env = SimpleNamespace(scene={"robot": asset}, num_envs=num_envs, device="cpu")
    action = DifferentialInverseKinematicsAction(cfg, env)
    ee_pos = torch.zeros(num_envs, 3)
    ee_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    monkeypatch.setattr(action, "_compute_frame_pose", lambda: (ee_pos, ee_quat))

    raw_actions = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, -0.5]])
    expected = torch.tensor([[0.6, -4.2, -0.2], [-0.4, 0.3, 0.8]])

    action.process_actions(raw_actions)

    torch.testing.assert_close(action.raw_actions, raw_actions)
    torch.testing.assert_close(action.processed_actions, expected)
    torch.testing.assert_close(action._ik_controller.ee_pos_des, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
