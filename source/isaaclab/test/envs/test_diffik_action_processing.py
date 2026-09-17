# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for differential inverse-kinematics action processing."""

import pytest
import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
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
    action = object.__new__(DifferentialInverseKinematicsAction)
    action.cfg = cfg
    action._raw_actions = torch.zeros(num_envs, 3)
    action._processed_actions = torch.zeros_like(action._raw_actions)
    action._scale = torch.tensor(cfg.scale).repeat(num_envs, 1)
    action._offset = torch.tensor(cfg.offset).repeat(num_envs, 1)
    action._ik_controller = DifferentialIKController(controller_cfg, num_envs=num_envs, device="cpu")
    ee_pos = torch.zeros(num_envs, 3)
    ee_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    monkeypatch.setattr(action, "_compute_frame_pose", lambda: (ee_pos, ee_quat))

    raw_actions = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, -0.5]])
    expected = raw_actions * action._scale + action._offset

    action.process_actions(raw_actions)

    torch.testing.assert_close(action._raw_actions, raw_actions)
    torch.testing.assert_close(action._processed_actions, expected)
    torch.testing.assert_close(action._ik_controller.ee_pos_des, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
