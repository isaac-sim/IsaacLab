# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for differential inverse-kinematics action processing."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

pytestmark = pytest.mark.unit


class _ControllerStub:
    def set_command(self, command: torch.Tensor, ee_pos: torch.Tensor, ee_quat: torch.Tensor) -> None:
        self.command = command.clone()
        self.ee_pos = ee_pos
        self.ee_quat = ee_quat


class _ActionStub:
    def __init__(self) -> None:
        self._raw_actions = torch.zeros(2, 3)
        self._processed_actions = torch.zeros(2, 3)
        self._scale = torch.tensor([[0.5, 2.0, -1.0], [0.5, 2.0, -1.0]])
        self._offset = torch.tensor([[0.1, -0.2, 0.3], [0.1, -0.2, 0.3]])
        self._ik_controller = _ControllerStub()
        self.cfg = SimpleNamespace(clip=None)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(2, 3), torch.zeros(2, 4)


def test_process_actions_applies_scale_then_offset() -> None:
    """DiffIK actions use the configured affine transformation before reaching the controller."""
    action = _ActionStub()
    raw_actions = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, -0.5]])
    expected = raw_actions * action._scale + action._offset

    DifferentialInverseKinematicsAction.process_actions(action, raw_actions)

    torch.testing.assert_close(action._raw_actions, raw_actions)
    torch.testing.assert_close(action._processed_actions, expected)
    torch.testing.assert_close(action._ik_controller.command, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
