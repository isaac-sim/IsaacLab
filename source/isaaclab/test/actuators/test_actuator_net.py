# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for neural actuator models."""

from __future__ import annotations

import warnings

import pytest
import torch

from isaaclab.actuators import ActuatorNetLSTM, ActuatorNetLSTMCfg
from isaaclab.utils.types import ArticulationActions


class _DummyLSTM(torch.nn.Module):
    """Minimal LSTM network for actuator testing."""

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=4, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(4, 1)

    def forward(
        self,
        x: torch.Tensor,
        hc: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out, hc_new = self.lstm(x, hc)
        return self.fc(out[:, -1, :]), hc_new


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_actuator_net_lstm_flattens_cuda_weights(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    checkpoint = tmp_path / "dummy_lstm.pt"
    network = _DummyLSTM().eval()
    torch.jit.save(torch.jit.script(network), str(checkpoint))

    cfg = ActuatorNetLSTMCfg(
        joint_names_expr=["joint_0"],
        network_file=str(checkpoint),
        effort_limit=100.0,
        velocity_limit=100.0,
        saturation_effort=100.0,
    )
    actuator = ActuatorNetLSTM(cfg, joint_names=["joint_0"], joint_ids=[0], num_envs=1, device="cuda")
    joint_state = torch.zeros((1, 1), device="cuda")
    control_action = ArticulationActions(joint_positions=joint_state)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        actuator.compute(control_action, joint_state, joint_state)

    assert not any("not part of single contiguous chunk" in str(warning.message) for warning in caught_warnings)
