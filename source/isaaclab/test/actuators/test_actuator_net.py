# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resets of the neural-network actuators by environment mask, recorded in a CUDA graph."""

import pytest
import torch
import warp as wp

from isaaclab.actuators import ActuatorNetLSTMCfg, ActuatorNetMLPCfg, DCMotor
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import instantiate
from isaaclab.utils.types import ArticulationActions

pytestmark = pytest.mark.kitless


class _ZeroTorqueLSTM(torch.nn.Module):
    """LSTM-shaped network that requests no torque and keeps its state."""

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=4, num_layers=2, batch_first=True)

    def forward(
        self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return torch.zeros(x.shape[0], 1, device=x.device), hc


def _feature_torque_mlp(num_features: int) -> torch.nn.Linear:
    """A linear network that gives every packed history feature a distinct weight in the requested torque."""
    network = torch.nn.Linear(num_features, 1)
    with torch.no_grad():
        network.weight.copy_(torch.arange(1.0, num_features + 1).unsqueeze(0))
        network.bias.zero_()
    return network


def _actuator(cfg_class, network: torch.nn.Module, tmp_path, device: str, network_format: str = "pt", **cfg):
    network_file = tmp_path / f"network.{network_format}"
    if network_format == "onnx":
        # names other than the ones the actuator exports TorchScript with
        num_features = network.in_features
        torch.onnx.export(
            network,
            (torch.zeros(1, num_features),),
            str(network_file),
            input_names=["obs"],
            output_names=["act"],
            dynamic_axes={"obs": {0: "batch"}, "act": {0: "batch"}},
            dynamo=False,
        )
    else:
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
    """A recorded reset changes the next computed torque only for the environments in its current mask."""
    torch_stream = torch.cuda.Stream(device=device)
    stream = wp.stream_from_torch(torch_stream)
    with torch.cuda.stream(torch_stream), wp.ScopedStream(stream):
        actuator = _actuator(
            ActuatorNetMLPCfg,
            _feature_torque_mlp(6),
            tmp_path,
            device,
            input_idx=[0, 1, 2],
            pos_scale=1.0,
            vel_scale=1.0,
            torque_scale=1.0,
            input_order="pos_vel",
        )
        mask = torch.tensor([True, False, True], device=device)
        capture_stream = wp.Stream(device=device)
        # The reset belongs to the Warp capture even though Torch is on a different stream.
        capture_stream.wait_stream(stream)
        with wp.ScopedStream(capture_stream, sync_enter=False):
            graph = _record_masked_reset(actuator, wp.from_torch(mask))
        zeros = torch.zeros(3, 2, device=device)
        ones = torch.ones_like(zeros)
        for _ in range(3):
            # Fill the history with ones before replaying the same reset graph with a different mask.
            for _ in range(3):
                actuator.compute(ArticulationActions(joint_positions=ones), zeros, ones)
            mask.logical_not_()
            capture_stream.wait_stream(stream)
            wp.capture_launch(graph, stream=capture_stream)
            stream.wait_stream(capture_stream)
            actuator.compute(ArticulationActions(joint_positions=zeros), zeros, zeros)
            # The new sample is zero. The two older samples contribute 2 + 3 + 5 + 6 when kept.
            expected = torch.where(mask, 0.0, 16.0).unsqueeze(1).expand(-1, 2)
            torch.testing.assert_close(actuator.computed_effort, expected)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_mlp_compute_replays_from_a_warp_graph(tmp_path, device):
    """A Warp graph records the whole torque computation, network and clip included, with no Torch work in it.

    The graph is recorded on Warp's own stream, apart from Torch's, so any Torch operation in ``compute`` would run
    once at recording time instead of being replayed.
    """
    cfg = {"input_idx": [0, 1, 2], "pos_scale": 1.0, "vel_scale": 1.0, "torque_scale": 40.0, "input_order": "pos_vel"}
    eager = _actuator(ActuatorNetMLPCfg, _feature_torque_mlp(6), tmp_path, device, **cfg)
    recorded = _actuator(ActuatorNetMLPCfg, _feature_torque_mlp(6), tmp_path, device, **cfg)
    target, position, velocity = (torch.zeros(3, 2, device=device) for _ in range(3))
    torch.cuda.synchronize(device)
    with wp.ScopedCapture(device) as capture:
        recorded.compute(ArticulationActions(joint_positions=target), position, velocity)
    for _ in range(4):
        for tensor in (target, position, velocity):
            tensor.copy_(torch.randn(3, 2, device=device))
        eager.compute(ArticulationActions(joint_positions=target.clone()), position, velocity)
        torch.cuda.synchronize(device)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        torch.testing.assert_close(recorded.computed_effort, eager.computed_effort)
        torch.testing.assert_close(recorded.applied_effort, eager.applied_effort)


def test_mlp_rejects_invalid_history_before_loading_checkpoint(tmp_path):
    """The configured history error takes precedence over checkpoint I/O."""
    cfg = ActuatorNetMLPCfg(
        joint_names_expr=["joint_.*"],
        network_file=str(tmp_path / "missing-network.pt"),
        saturation_effort=120.0,
        actuator_effort_limit=80.0,
        actuator_velocity_limit=7.5,
        input_idx=[0, -2],
        pos_scale=1.0,
        vel_scale=1.0,
        torque_scale=1.0,
        input_order="pos_vel",
    )
    with pytest.raises(IndexError, match="MLP actuator history indices"):
        instantiate(cfg, joint_names=["joint_0", "joint_1"], joint_ids=[0, 1], num_envs=3, device="cpu")


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize(
    "input_idx,input_order,network_format", [([0, 1, 2], "pos_vel", "pt"), ([3, 0, -2, 3], "vel_pos", "onnx")]
)
def test_mlp_history_keeps_its_window(tmp_path, device, input_idx, input_order, network_format):
    """Computed torque follows the selected zero-padded history across wraparound and partial resets, with a
    TorchScript or an ONNX network file; the applied torque is clipped as the Torch DC motor clips it."""
    network = _feature_torque_mlp(2 * len(input_idx))
    actuator = _actuator(
        ActuatorNetMLPCfg,
        network,
        tmp_path,
        device,
        network_format,
        input_idx=input_idx,
        pos_scale=-0.3,
        vel_scale=1.25,
        torque_scale=20.0,
        input_order=input_order,
    )
    network = network.to(device)
    clipped = False
    length = max(input_idx) + 1
    position_history = torch.zeros(3, length, 2, device=device)
    velocity_history = torch.zeros_like(position_history)
    joints = torch.arange(6.0, device=device).reshape(3, 2)
    pos_scale, vel_scale = -0.3, 1.25
    for step in range(8):
        if step == 3:
            actuator.reset(env_ids=torch.tensor([0, 2], dtype=torch.int32, device=device))
            position_history[[0, 2]] = 0.0
            velocity_history[[0, 2]] = 0.0
        elif step == 5:
            actuator.reset(env_ids=slice(1, None))
            position_history[1:] = 0.0
            velocity_history[1:] = 0.0
        elif step == 6:
            pos_scale, vel_scale = 0.6, -0.75
            actuator.cfg.pos_scale, actuator.cfg.vel_scale = pos_scale, vel_scale
        position = joints * 0.1 + step * 0.05
        desired = position + joints * 0.03 + step * 0.07
        velocity = joints * 0.2 - step * 0.03
        position_history = position_history.roll(1, 1)
        velocity_history = velocity_history.roll(1, 1)
        position_history[:, 0] = desired - position
        velocity_history[:, 0] = velocity
        pos_features = torch.stack([position_history[:, index] for index in input_idx], dim=2).reshape(6, -1)
        vel_features = torch.stack([velocity_history[:, index] for index in input_idx], dim=2).reshape(6, -1)
        if input_order == "pos_vel":
            features = torch.cat([pos_features * pos_scale, vel_features * vel_scale], dim=1)
        else:
            features = torch.cat([vel_features * vel_scale, pos_features * pos_scale], dim=1)
        expected = network(features).detach().reshape(3, 2) * 20.0
        actuator.compute(ArticulationActions(joint_positions=desired), position, velocity)
        torch.testing.assert_close(actuator.computed_effort, expected)
        torch.testing.assert_close(actuator.applied_effort, DCMotor._clip_effort(actuator, expected, velocity))
        clipped |= not torch.equal(actuator.applied_effort, actuator.computed_effort)
    assert clipped
