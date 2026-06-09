# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

HEADLESS = True

# launch the simulator before importing the rest of the framework
simulation_app = AppLauncher(headless=HEADLESS).app

"""Rest of imports follows"""

import pytest
import torch

from isaaclab.actuators import ActuatorNetGRUCfg, ActuatorNetGRUResidualCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.types import ArticulationActions


@pytest.fixture
def sim(request):
    """Create simulation context with the specified device."""
    device = request.getfixturevalue("device")
    with build_simulation_context(device=device) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


"""
Helpers: scriptable GRU modules satisfying the contract ([position, position_error, velocity] -> effort).
"""


class _TinyGRUNet(torch.nn.Module):
    """GRU + linear-head module matching the actuator's TorchScript export contract.

    Mirrors the runtime GRU produced by the actuator-model exporter: a ``.gru`` submodule
    (``torch.nn.GRU``, ``batch_first``) followed by a linear head, with recurrent dropout only when
    stacking layers. ``forward(x, hidden)`` consumes ``x`` of shape (batch, 1, 3) -- the joint
    position, position error, and velocity -- and ``hidden`` of shape (num_layers, batch,
    hidden_dim), and returns ``(output, new_hidden)`` where ``output`` has shape (batch, 1,
    output_size).
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 4, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, dropout=recurrent_dropout, batch_first=True)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, new_hidden = self.gru(x, hidden)
        return self.head(out), new_hidden


def _make_network_file(tmp_path, input_dim: int = 3, hidden_dim: int = 4, num_layers: int = 1) -> str:
    """Build, script, and save a tiny GRU network, returning the saved file path."""
    torch.manual_seed(0)
    module = _TinyGRUNet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    module.eval()
    scripted = torch.jit.script(module)
    file_path = str(tmp_path / f"tiny_gru_{input_dim}_{hidden_dim}_{num_layers}.pt")
    torch.jit.save(scripted, file_path)
    return file_path


def _make_bad_network_file(tmp_path) -> str:
    """Build and save a scripted module that lacks a ``.gru`` submodule."""

    class _NoGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.linear(x), hidden

    scripted = torch.jit.script(_NoGRU().eval())
    file_path = str(tmp_path / "no_gru.pt")
    torch.jit.save(scripted, file_path)
    return file_path


def _make_runtime_gru_file(tmp_path, hidden_dim: int = 64, num_layers: int = 2) -> str:
    """Build, script, and save a production-sized multi-layer GRU (the real export architecture)."""
    torch.manual_seed(0)
    module = _TinyGRUNet(input_dim=3, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1)
    module.eval()
    scripted = torch.jit.script(module)
    file_path = str(tmp_path / f"runtime_gru_{hidden_dim}_{num_layers}.pt")
    torch.jit.save(scripted, file_path)
    return file_path


def _make_nan_network_file(tmp_path) -> str:
    """Build and save a GRU network whose head emits non-finite output (poisoned head params)."""
    torch.manual_seed(0)
    module = _TinyGRUNet(input_dim=3)
    with torch.no_grad():
        module.head.weight.fill_(float("nan"))
        module.head.bias.fill_(float("nan"))
    module.eval()
    scripted = torch.jit.script(module)
    file_path = str(tmp_path / "nan_gru.pt")
    torch.jit.save(scripted, file_path)
    return file_path


def _reference_effort(network_file, des_pos, joint_pos, joint_vel, hidden_dim=4, num_layers=1):
    """Roll the saved network forward by hand for one step (identity normalization)."""
    device = joint_pos.device
    num_envs, num_joints = joint_pos.shape
    net = torch.jit.load(network_file, map_location=device).eval()
    batch = num_envs * num_joints
    hidden = torch.zeros(num_layers, batch, hidden_dim, device=device)
    x = torch.stack([joint_pos.flatten(), (des_pos - joint_pos).flatten(), joint_vel.flatten()], dim=1).reshape(
        batch, 1, 3
    )
    with torch.inference_mode():
        out, _ = net(x, hidden)
    return out.reshape(num_envs, num_joints)


"""
Test ActuatorNetGRU (explicit, full-torque).
"""


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_actuator_net_gru_compute(sim, num_envs, num_joints, device, tmp_path):
    """ActuatorNetGRU.compute returns the network effort (matching a reference forward), nulls pos/vel."""
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = list(range(num_joints))

    network_file = _make_network_file(tmp_path)

    # large effort limit so the applied effort is the un-clipped network output
    actuator_cfg = ActuatorNetGRUCfg(joint_names_expr=joint_names, network_file=network_file, effort_limit=1.0e6)
    actuator = actuator_cfg.class_type(
        actuator_cfg, joint_names=joint_names, joint_ids=joint_ids, num_envs=num_envs, device=device
    )

    joint_pos = torch.rand(num_envs, num_joints, device=device)
    joint_vel = torch.rand(num_envs, num_joints, device=device)
    des_pos = torch.rand(num_envs, num_joints, device=device)
    control_action = ArticulationActions(
        joint_positions=des_pos,
        joint_velocities=torch.rand(num_envs, num_joints, device=device),
        joint_efforts=None,
    )

    # independent reference forward of the same network with identity normalization
    reference = _reference_effort(network_file, des_pos, joint_pos, joint_vel)

    out = actuator.compute(control_action, joint_pos, joint_vel)

    # efforts have the expected shape and positions/velocities are nulled
    assert out.joint_efforts.shape == (num_envs, num_joints)
    assert out.joint_positions is None
    assert out.joint_velocities is None
    # the returned effort matches the reference forward (catches input-assembly/order bugs)
    torch.testing.assert_close(out.joint_efforts, actuator.applied_effort)
    torch.testing.assert_close(out.joint_efforts, reference)


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_actuator_net_gru_effort_clipping(sim, num_envs, num_joints, device, tmp_path):
    """A tiny effort limit forces the applied effort to saturate at the limit."""
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = list(range(num_joints))
    effort_limit = 0.5

    network_file = _make_network_file(tmp_path)

    actuator_cfg = ActuatorNetGRUCfg(
        joint_names_expr=joint_names,
        network_file=network_file,
        effort_limit=effort_limit,
        # bias the denormalized output well above the effort limit
        output_normalization=(100.0, 1.0),
    )
    actuator = actuator_cfg.class_type(
        actuator_cfg, joint_names=joint_names, joint_ids=joint_ids, num_envs=num_envs, device=device
    )

    joint_pos = torch.rand(num_envs, num_joints, device=device)
    joint_vel = torch.rand(num_envs, num_joints, device=device)
    control_action = ArticulationActions(
        joint_positions=torch.rand(num_envs, num_joints, device=device),
        joint_velocities=torch.rand(num_envs, num_joints, device=device),
    )

    actuator.compute(control_action, joint_pos, joint_vel)
    torch.testing.assert_close(actuator.applied_effort, effort_limit * torch.ones(num_envs, num_joints, device=device))


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_actuator_net_gru_nan_output_is_sanitized(sim, num_envs, num_joints, device, tmp_path):
    """A non-finite network output is sanitized to zero effort before reaching the engine."""
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = list(range(num_joints))

    network_file = _make_nan_network_file(tmp_path)

    actuator_cfg = ActuatorNetGRUCfg(joint_names_expr=joint_names, network_file=network_file, effort_limit=5.0)
    actuator = actuator_cfg.class_type(
        actuator_cfg, joint_names=joint_names, joint_ids=joint_ids, num_envs=num_envs, device=device
    )

    joint_pos = torch.rand(num_envs, num_joints, device=device)
    joint_vel = torch.rand(num_envs, num_joints, device=device)
    control_action = ArticulationActions(
        joint_positions=torch.rand(num_envs, num_joints, device=device),
        joint_velocities=torch.rand(num_envs, num_joints, device=device),
    )

    out = actuator.compute(control_action, joint_pos, joint_vel)

    assert torch.all(torch.isfinite(out.joint_efforts))
    torch.testing.assert_close(out.joint_efforts, torch.zeros(num_envs, num_joints, device=device))


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_actuator_net_gru_output_normalization(sim, num_envs, num_joints, device, tmp_path):
    """Output denormalization scales the raw effort by std and offsets by mean."""
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = list(range(num_joints))
    out_mean, out_std = 2.0, 3.0

    network_file = _make_network_file(tmp_path)

    def _build(output_normalization):
        cfg = ActuatorNetGRUCfg(
            joint_names_expr=joint_names,
            network_file=network_file,
            effort_limit=1.0e6,
            output_normalization=output_normalization,
        )
        return cfg.class_type(cfg, joint_names=joint_names, joint_ids=joint_ids, num_envs=num_envs, device=device)

    actuator_identity = _build(None)
    actuator_scaled = _build((out_mean, out_std))

    joint_pos = torch.rand(num_envs, num_joints, device=device)
    joint_vel = torch.rand(num_envs, num_joints, device=device)
    des_pos = torch.rand(num_envs, num_joints, device=device)

    def _ca():
        return ArticulationActions(joint_positions=des_pos.clone(), joint_velocities=joint_vel.clone())

    eff_identity = actuator_identity.compute(_ca(), joint_pos, joint_vel).joint_efforts.clone()
    eff_scaled = actuator_scaled.compute(_ca(), joint_pos, joint_vel).joint_efforts.clone()

    torch.testing.assert_close(eff_scaled, eff_identity * out_std + out_mean)


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_actuator_net_gru_input_normalization(sim, num_envs, num_joints, device, tmp_path):
    """Input normalization writes ``(x - mean) / std`` for position, position error, and velocity."""
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = list(range(num_joints))
    pos_norm = (0.2, 3.0)
    pos_err_norm = (0.5, 2.0)
    vel_norm = (-1.0, 4.0)

    network_file = _make_network_file(tmp_path)

    cfg = ActuatorNetGRUCfg(
        joint_names_expr=joint_names,
        network_file=network_file,
        effort_limit=1.0e6,
        position_normalization=pos_norm,
        pos_error_normalization=pos_err_norm,
        vel_normalization=vel_norm,
    )
    actuator = cfg.class_type(cfg, joint_names=joint_names, joint_ids=joint_ids, num_envs=num_envs, device=device)

    joint_pos = torch.rand(num_envs, num_joints, device=device)
    joint_vel = torch.rand(num_envs, num_joints, device=device)
    des_pos = joint_pos + 0.3
    actuator.compute(ArticulationActions(joint_positions=des_pos, joint_velocities=joint_vel), joint_pos, joint_vel)

    pos_error = (des_pos - joint_pos).flatten()
    torch.testing.assert_close(actuator.sea_input[:, 0, 0], (joint_pos.flatten() - pos_norm[0]) / pos_norm[1])
    torch.testing.assert_close(actuator.sea_input[:, 0, 1], (pos_error - pos_err_norm[0]) / pos_err_norm[1])
    torch.testing.assert_close(actuator.sea_input[:, 0, 2], (joint_vel.flatten() - vel_norm[0]) / vel_norm[1])


@pytest.mark.parametrize("num_envs", [2])
@pytest.mark.parametrize("num_joints", [2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_actuator_net_gru_reset(sim, num_envs, num_joints, device, tmp_path):
    """reset(env_ids) zeros the GRU hidden state only for the given environments."""
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = list(range(num_joints))

    network_file = _make_network_file(tmp_path)

    cfg = ActuatorNetGRUCfg(joint_names_expr=joint_names, network_file=network_file, effort_limit=1.0e6)
    actuator = cfg.class_type(cfg, joint_names=joint_names, joint_ids=joint_ids, num_envs=num_envs, device=device)

    # advance the hidden state for all envs
    joint_pos = torch.rand(num_envs, num_joints, device=device)
    joint_vel = torch.rand(num_envs, num_joints, device=device)
    actuator.compute(
        ArticulationActions(
            joint_positions=torch.rand(num_envs, num_joints, device=device), joint_velocities=joint_vel
        ),
        joint_pos,
        joint_vel,
    )
    assert torch.any(actuator.sea_hidden_state_per_env[:, 0] != 0.0)
    assert torch.any(actuator.sea_hidden_state_per_env[:, 1] != 0.0)

    # reset env 0 only
    actuator.reset([0])
    assert torch.all(actuator.sea_hidden_state_per_env[:, 0] == 0.0)
    assert torch.any(actuator.sea_hidden_state_per_env[:, 1] != 0.0)


"""
Test ActuatorNetGRUResidual (implicit-PD + residual).
"""


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("preset_efforts", [False, True])
def test_actuator_net_gru_residual_compute(sim, num_envs, num_joints, device, preset_efforts, tmp_path):
    """ActuatorNetGRUResidual adds the residual to joint_efforts and preserves pos/vel.

    Covers both a pre-set ``joint_efforts`` (residual added on top) and ``None`` (residual becomes
    the feed-forward effort). The approximate ``computed_effort`` follows
    ``stiffness * err_pos + damping * err_vel + joint_efforts`` and positions/velocities are
    preserved on return so the engine can apply the PD term.
    """
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = list(range(num_joints))
    stiffness, damping = 40.0, 3.0

    network_file = _make_network_file(tmp_path)

    cfg = ActuatorNetGRUResidualCfg(
        joint_names_expr=joint_names,
        network_file=network_file,
        stiffness=stiffness,
        damping=damping,
        effort_limit_sim=1.0e6,
    )
    actuator = cfg.class_type(
        cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=stiffness,
        damping=damping,
    )

    joint_pos = torch.rand(num_envs, num_joints, device=device)
    joint_vel = torch.rand(num_envs, num_joints, device=device)
    des_pos = joint_pos + 0.3
    des_vel = joint_vel + 0.1
    preset = torch.rand(num_envs, num_joints, device=device) if preset_efforts else None
    control_action = ArticulationActions(
        joint_positions=des_pos, joint_velocities=des_vel, joint_efforts=preset.clone() if preset is not None else None
    )

    # independent reference residual (identity normalization, hidden starts at zero)
    residual = _reference_effort(network_file, des_pos, joint_pos, joint_vel)

    out = actuator.compute(control_action, joint_pos, joint_vel)

    # residual is added to the feed-forward effort
    expected_ff = residual if preset is None else preset + residual
    torch.testing.assert_close(out.joint_efforts, expected_ff)
    # approximate total effort follows the implicit-PD-plus-feedforward formula
    expected_computed = stiffness * (des_pos - joint_pos) + damping * (des_vel - joint_vel) + expected_ff
    torch.testing.assert_close(actuator.computed_effort, expected_computed)
    # positions/velocities are preserved so the engine can apply the PD term
    assert out.joint_positions is not None
    assert out.joint_velocities is not None


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_actuator_net_gru_residual_velocities_none(sim, num_envs, num_joints, device, tmp_path):
    """When joint_velocities is None, the velocity error falls back to ``-joint_vel``."""
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = list(range(num_joints))
    stiffness, damping = 40.0, 3.0

    network_file = _make_network_file(tmp_path)

    cfg = ActuatorNetGRUResidualCfg(
        joint_names_expr=joint_names,
        network_file=network_file,
        stiffness=stiffness,
        damping=damping,
        effort_limit_sim=1.0e6,
    )
    actuator = cfg.class_type(
        cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=stiffness,
        damping=damping,
    )

    joint_pos = torch.rand(num_envs, num_joints, device=device)
    joint_vel = torch.rand(num_envs, num_joints, device=device)
    des_pos = joint_pos + 0.3
    control_action = ArticulationActions(joint_positions=des_pos, joint_velocities=None, joint_efforts=None)

    residual = _reference_effort(network_file, des_pos, joint_pos, joint_vel)

    out = actuator.compute(control_action, joint_pos, joint_vel)

    # velocity error falls back to -joint_vel when no desired velocity is provided
    expected_computed = stiffness * (des_pos - joint_pos) + damping * (-joint_vel) + residual
    torch.testing.assert_close(actuator.computed_effort, expected_computed)
    assert out.joint_velocities is None


"""
Test initialization-time validation errors.
"""


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_actuator_net_gru_missing_gru_submodule_raises(sim, device, tmp_path):
    """A network without a ``.gru`` submodule raises ValueError at init."""
    joint_names = ["joint_0"]
    bad_file = _make_bad_network_file(tmp_path)

    cfg = ActuatorNetGRUCfg(joint_names_expr=joint_names, network_file=bad_file)
    with pytest.raises(ValueError):
        cfg.class_type(cfg, joint_names=joint_names, joint_ids=[0], num_envs=1, device=device)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_actuator_net_gru_input_dim_mismatch_raises(sim, device, tmp_path):
    """A network whose GRU does not take exactly 3 inputs raises ValueError at init."""
    joint_names = ["joint_0"]
    network_file = _make_network_file(tmp_path, input_dim=2)

    cfg = ActuatorNetGRUCfg(joint_names_expr=joint_names, network_file=network_file)
    with pytest.raises(ValueError):
        cfg.class_type(cfg, joint_names=joint_names, joint_ids=[0], num_envs=1, device=device)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_actuator_net_gru_negative_std_raises(sim, device, tmp_path):
    """A negative normalization std raises ValueError at init (rather than being floored)."""
    joint_names = ["joint_0"]
    network_file = _make_network_file(tmp_path)

    cfg = ActuatorNetGRUCfg(
        joint_names_expr=joint_names, network_file=network_file, pos_error_normalization=(0.0, -2.0)
    )
    with pytest.raises(ValueError):
        cfg.class_type(cfg, joint_names=joint_names, joint_ids=[0], num_envs=1, device=device)


"""
Test the real (production-sized) GRU export architecture.
"""


@pytest.mark.parametrize("variant", ["explicit", "residual"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_actuator_net_gru_runtime_export_architecture(sim, variant, device, tmp_path):
    """A production-sized multi-layer GRU (the real export architecture) loads and steps.

    Exercises a stacked GRU (hidden_dim=64, num_layers=2) -- matching the runtime model the
    actuator-model exporter produces -- in both the explicit and residual actuators. Verifies the
    multi-layer hidden-state buffer is allocated correctly, the effort is finite and correctly
    shaped, and the recurrent hidden state evolves across consecutive steps (the GRU memory is
    actually carried, not just zeroed).
    """
    num_envs, num_joints = 2, 3
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = list(range(num_joints))
    hidden_dim, num_layers = 64, 2

    network_file = _make_runtime_gru_file(tmp_path, hidden_dim=hidden_dim, num_layers=num_layers)

    if variant == "explicit":
        cfg = ActuatorNetGRUCfg(joint_names_expr=joint_names, network_file=network_file, effort_limit=1.0e6)
        actuator = cfg.class_type(cfg, joint_names=joint_names, joint_ids=joint_ids, num_envs=num_envs, device=device)
    else:
        stiffness, damping = 40.0, 3.0
        cfg = ActuatorNetGRUResidualCfg(
            joint_names_expr=joint_names,
            network_file=network_file,
            stiffness=stiffness,
            damping=damping,
            effort_limit_sim=1.0e6,
        )
        actuator = cfg.class_type(
            cfg,
            joint_names=joint_names,
            joint_ids=joint_ids,
            num_envs=num_envs,
            device=device,
            stiffness=stiffness,
            damping=damping,
        )

    # the recurrent buffer reflects the stacked-layer network dimensions
    assert actuator.sea_hidden_state.shape == (num_layers, num_envs * num_joints, hidden_dim)

    # frozen input across steps; rebuild the action each step (compute may consume it)
    des_pos = torch.rand(num_envs, num_joints, device=device)
    des_vel = torch.rand(num_envs, num_joints, device=device)
    joint_pos = torch.rand(num_envs, num_joints, device=device)
    joint_vel = torch.rand(num_envs, num_joints, device=device)

    def _action():
        return ArticulationActions(joint_positions=des_pos.clone(), joint_velocities=des_vel.clone())

    out = actuator.compute(_action(), joint_pos, joint_vel)
    assert out.joint_efforts.shape == (num_envs, num_joints)
    assert torch.all(torch.isfinite(out.joint_efforts))

    # after one step the hidden state has advanced away from zero, and a second identical step
    # advances it further -- confirming the GRU memory is carried across steps
    hidden_after_first = actuator.sea_hidden_state.clone()
    assert torch.any(hidden_after_first != 0.0)
    actuator.compute(_action(), joint_pos, joint_vel)
    assert not torch.allclose(hidden_after_first, actuator.sea_hidden_state)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
