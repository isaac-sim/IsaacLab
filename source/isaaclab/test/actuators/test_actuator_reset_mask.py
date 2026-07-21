# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for mask-based actuator resets."""

from __future__ import annotations

import pytest
import torch

from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.actuators.actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from isaaclab.actuators.actuator_pd import DelayedPDActuator, IdealPDActuator, ImplicitActuator


def _fail_host_compaction(*args, **kwargs):
    raise AssertionError("mask-native reset_mask must not materialize IDs")


class _RecordingActuator(ActuatorBase):
    """Legacy model implementing only the ID-based reset."""

    def __init__(self):
        self.reset_env_ids = None

    def reset(self, env_ids):
        self.reset_env_ids = env_ids

    def compute(self, control_action, joint_pos, joint_vel):
        return control_action


def test_base_reset_mask_compacts_and_delegates():
    """The base fallback should hand legacy models compact IDs matching the mask."""
    actuator = _RecordingActuator()

    actuator.reset_mask(torch.tensor([True, False, True]))

    torch.testing.assert_close(actuator.reset_env_ids, torch.tensor([0, 2]))


def test_reset_only_override_restores_compacting_fallback():
    """A subclass overriding reset without reset_mask must not inherit a stateless no-op."""
    assert DelayedPDActuator.reset_mask is ActuatorBase.reset_mask
    assert ImplicitActuator.reset_mask is not ActuatorBase.reset_mask
    assert IdealPDActuator.reset_mask is not ActuatorBase.reset_mask

    class _StatefulImplicit(ImplicitActuator):
        def reset(self, env_ids):
            pass

    assert _StatefulImplicit.reset_mask is ActuatorBase.reset_mask


def test_stateless_models_skip_host_compaction(monkeypatch: pytest.MonkeyPatch):
    """Stateless models must not pay a host synchronization on masked resets."""
    implicit = ImplicitActuator.__new__(ImplicitActuator)

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "nonzero", _fail_host_compaction)
        implicit.reset_mask(torch.tensor([True, False]))


def test_lstm_reset_mask_zeroes_only_selected(monkeypatch: pytest.MonkeyPatch):
    """The LSTM override should zero hidden state for masked environments only."""
    actuator = ActuatorNetLSTM.__new__(ActuatorNetLSTM)
    actuator.sea_hidden_state_per_env = torch.ones(2, 4, 3, 8)
    actuator.sea_cell_state_per_env = torch.ones(2, 4, 3, 8)
    expected = torch.ones(2, 4, 3, 8)
    expected[:, [0, 2]] = 0.0

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "nonzero", _fail_host_compaction)
        actuator.reset_mask(torch.tensor([True, False, True, False]))

    torch.testing.assert_close(actuator.sea_hidden_state_per_env, expected)
    torch.testing.assert_close(actuator.sea_cell_state_per_env, expected)


def test_mlp_reset_mask_zeroes_only_selected(monkeypatch: pytest.MonkeyPatch):
    """The MLP override should zero joint history for masked environments only."""
    actuator = ActuatorNetMLP.__new__(ActuatorNetMLP)
    actuator._joint_pos_error_history = torch.ones(4, 3, 6)
    actuator._joint_vel_history = torch.ones(4, 3, 6)
    expected = torch.ones(4, 3, 6)
    expected[[1, 3]] = 0.0

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "nonzero", _fail_host_compaction)
        actuator.reset_mask(torch.tensor([False, True, False, True]))

    torch.testing.assert_close(actuator._joint_pos_error_history, expected)
    torch.testing.assert_close(actuator._joint_vel_history, expected)
