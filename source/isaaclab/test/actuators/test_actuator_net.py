# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only tests for the neural-network actuator models."""

import pytest
import torch

from isaaclab.actuators import ActuatorNetLSTMCfg
from isaaclab.utils.types import ArticulationActions

pytestmark = pytest.mark.unit


class _ConstantTorqueLSTM(torch.nn.Module):
    """LSTM-shaped actuator network that always requests the same torque."""

    def __init__(self, torque: float):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=4, num_layers=1, batch_first=True)
        self.torque = torque

    def forward(
        self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return torch.full((x.shape[0], 1), self.torque), hc


def test_lstm_actuator_clips_with_torque_speed_curve(tmp_path):
    """The DC-motor torque-speed limit uses the current joint velocity, as for the MLP actuator."""
    network_file = tmp_path / "constant_lstm.pt"
    torch.jit.script(_ConstantTorqueLSTM(torque=100.0)).save(str(network_file))
    cfg = ActuatorNetLSTMCfg(
        joint_names_expr=["joint_.*"],
        network_file=str(network_file),
        saturation_effort=120.0,
        effort_limit=80.0,
        velocity_limit=7.5,
    )
    actuator = cfg.class_type(cfg, joint_names=["joint_0", "joint_1"], joint_ids=[0, 1], num_envs=2, device="cpu")

    zeros = torch.zeros(2, 2)
    joint_vel = torch.tensor([[0.0, 3.75], [7.5, -7.5]])
    action = actuator.compute(ArticulationActions(zeros.clone(), zeros.clone(), zeros.clone()), zeros, joint_vel)

    # Upper limit: min(saturation * (1 - vel / vel_limit), effort_limit) = min(120 * (1 - vel / 7.5), 80).
    expected = torch.clamp(120.0 * (1.0 - joint_vel / 7.5), max=80.0).clamp(max=100.0)
    torch.testing.assert_close(action.joint_efforts, expected)
