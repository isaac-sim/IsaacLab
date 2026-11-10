# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resets of the neural-network actuators by environment mask, recorded in a CUDA graph."""

import pytest
import torch
import warp as wp

from isaaclab.actuators import ActuatorNetLSTMCfg, ActuatorNetMLPCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import instantiate
from isaaclab.utils.types import ArticulationActions


class _ZeroTorqueLSTM(torch.nn.Module):
    """LSTM-shaped network that requests no torque and keeps its state."""

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=4, num_layers=2, batch_first=True)

    def forward(
        self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return torch.zeros(x.shape[0], 1, device=x.device), hc


class _ZeroTorqueMLP(torch.nn.Module):
    """MLP-shaped network that requests no torque."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1, device=x.device)


def _actuator(cfg_class, network: torch.nn.Module, tmp_path, device: str, **cfg):
    network_file = tmp_path / "network.pt"
    torch.jit.script(network).save(str(network_file))
    cfg = cfg_class(
        joint_names_expr=["joint_.*"],
        network_file=str(network_file),
        saturation_effort=120.0,
        actuator_effort_limit=80.0,
        actuator_velocity_limit=7.5,
        **cfg,
    )
    return instantiate(cfg, joint_names=["joint_0", "joint_1"], joint_ids=[0, 1], num_envs=3, device=device)


def _record_masked_reset(actuator, mask: wp.array) -> wp.Graph:
    """A CUDA graph of the actuator's reset of the masked environments; the graph reads ``mask`` when it replays."""
    with wp.ScopedCapture(mask.device) as capture:
        actuator.reset(env_mask=mask)
    return capture.graph


def _assert_masked_zero(state: torch.Tensor, env_dim: int):
    """Environments 0 and 2 (masked) are zero and environment 1 keeps its value of one."""
    masked, kept = state.index_select(env_dim, torch.tensor([0, 2], device=state.device)), state.select(env_dim, 1)
    assert torch.equal(masked, torch.zeros_like(masked))
    assert torch.equal(kept, torch.ones_like(kept))


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_lstm_masked_reset_replays_from_a_cuda_graph(tmp_path, device):
    actuator = _actuator(ActuatorNetLSTMCfg, _ZeroTorqueLSTM(), tmp_path, device)
    mask = wp.array([True, False, True], dtype=wp.bool, device=device)
    graph = _record_masked_reset(actuator, mask)
    for _ in range(3):
        actuator.sea_hidden_state.fill_(1.0)
        actuator.sea_cell_state.fill_(1.0)
        wp.capture_launch(graph)
        _assert_masked_zero(actuator.sea_hidden_state_per_env, env_dim=1)
        _assert_masked_zero(actuator.sea_cell_state_per_env, env_dim=1)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_mlp_masked_reset_replays_from_a_cuda_graph_while_the_history_moves(tmp_path, device):
    """The history shifts into a spare buffer every compute; the recorded reset clears it in either buffer."""
    actuator = _actuator(
        ActuatorNetMLPCfg,
        _ZeroTorqueMLP(),
        tmp_path,
        device,
        input_idx=[0, 1, 2],
        pos_scale=1.0,
        vel_scale=1.0,
        torque_scale=1.0,
        input_order="pos_vel",
    )
    mask = wp.array([True, False, True], dtype=wp.bool, device=device)
    graph = _record_masked_reset(actuator, mask)
    zeros = torch.zeros(3, 2, device=device)
    for _ in range(3):
        actuator.compute(ArticulationActions(joint_positions=zeros), zeros, zeros)
        actuator._joint_pos_error_history.fill_(1.0)
        actuator._joint_vel_history.fill_(1.0)
        wp.capture_launch(graph)
        _assert_masked_zero(actuator._joint_pos_error_history, env_dim=0)
        _assert_masked_zero(actuator._joint_vel_history, env_dim=0)
