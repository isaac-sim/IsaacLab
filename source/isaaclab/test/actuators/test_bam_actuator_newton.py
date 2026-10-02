# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton-native BAM actuator component.

Behavior tests load an authored USD fixture and step its Newton actuator with supplied joint
state. Authoring tests explicitly replace its actuator prims from a BamActuatorCfg.

Motor and friction outputs are checked against upstream BAM golden data. The harness
supplies the solver's external load and reads the motor torque and published friction budget.
"""

import sys
from dataclasses import MISSING, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from newton.actuators import parse_actuator_prim

from pxr import Sdf, Usd

from isaaclab.actuators import BamActuatorCfg, BamMotorCfg
from isaaclab.actuators.newton import (
    BAM_DRIVE_API,
    DriveBam,
    NewtonActuatorAdapter,
    PhysxActuatorWrapper,
    apply_bam_startup_sampling,
)
from isaaclab.sim.schemas.schemas_actuators import (
    _is_newton_native_actuator_cfg,
    author_actuator_prims,
    validate_newton_native_actuator_cfgs,
)
from isaaclab.test.utils import test_devices
from isaaclab.utils.string import to_camel_case

pytestmark = [pytest.mark.unit, pytest.mark.filterwarnings("error::DeprecationWarning")]

JOINT_NAMES = ["servo_0", "servo_1"]
"""Joints of the fixture articulation; two of them so the shared-supply sag is observable."""

DT = 1.0 / 120.0
"""Physics timestep the actuators are stepped at [s]."""

VIN = 7.4
"""Supply voltage the fixture is configured with [V]."""

KP_FW = 200.0
"""Firmware proportional gain the fixture is configured with [-]."""


def _reference_params():
    with np.load(Path(__file__).parent / "data" / "bam_xl330_m6_goldens.npz") as data:
        return SimpleNamespace(
            **{key.removeprefix("attr_"): data[key].item() for key in data.files if key.startswith("attr_")}
        )


def _make_cfg(**overrides) -> BamActuatorCfg:
    """Build the BAM config the fixture articulation is authored from."""
    params = vars(_reference_params()).copy()
    params["resistance"] = params.pop("R")
    motor = BamMotorCfg(model="m6", **{f.name: params[f.name] for f in fields(BamMotorCfg) if f.name in params})
    kwargs = {"joint_names_expr": [".*"], "motor": motor, "vin": VIN, "kp_fw": KP_FW}
    kwargs.update(overrides)
    return BamActuatorCfg(**kwargs)


def _make_stage(cfg: BamActuatorCfg | None = None) -> Usd.Stage:
    """Load a fresh fixture layer, optionally exercising configuration-to-USD authoring."""
    path = Path(__file__).parent / "data" / "bam_two_servo.usda"
    stage = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(str(path)))
    if cfg is not None:
        author_actuator_prims(stage, "/Robot", {"servo": cfg})
    return stage


class _Harness:
    """Steps one Newton actuator over test-supplied joint state.

    Wraps the flat ``sim_state`` / ``sim_control`` pair
    :meth:`~newton.actuators.Actuator.step` expects, so a test can drive the actuator with an
    arbitrary trajectory and read back the effort it asks the solver to apply.
    """

    def __init__(self, num_envs: int, device: str, **drive_overrides: float | int):
        stage = _make_stage()
        for joint_name in JOINT_NAMES:
            prim = stage.GetPrimAtPath(f"/Robot/servo_{joint_name}_actuator")
            for name, value in drive_overrides.items():
                prim.GetAttribute(f"newton:{to_camel_case(name)}").Set(value)
        self.adapter = NewtonActuatorAdapter.from_usd(
            stage=stage,
            joint_names=JOINT_NAMES,
            num_envs=num_envs,
            num_joints=len(JOINT_NAMES),
            device=device,
            articulation_prim_path="/Robot",
        )
        assert len(self.adapter.actuators) == 1, "the fixture's joints must merge into one actuator"
        self.actuator = self.adapter.actuators[0]
        self.drive: DriveBam = self.actuator.drive
        self.drive.external_torque = wp.zeros(len(self.drive.motor_torque), dtype=wp.float32, device=device)
        self.num_envs = num_envs
        self.device = device
        shape = (num_envs, len(JOINT_NAMES))
        self.state = PhysxActuatorWrapper.create(*shape, device)
        self.control = PhysxActuatorWrapper.create(*shape, device)
        self.joint_pos = wp.zeros(shape, dtype=wp.float32, device=device)
        self.joint_vel = wp.zeros(shape, dtype=wp.float32, device=device)
        self.target_pos = wp.zeros(shape, dtype=wp.float32, device=device)
        self.state.joint_q = self.joint_pos.reshape(-1)
        self.state.joint_qd = self.joint_vel.reshape(-1)
        self.control.joint_target_pos = self.target_pos.reshape(-1)
        self.control.joint_target_vel = wp.zeros(num_envs * len(JOINT_NAMES), dtype=wp.float32, device=device)
        self.control.joint_act = None
        self.adapter.finalize(self.control)

    def step(self, joint_pos: np.ndarray, joint_vel: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """Run one actuator step and return the applied effort [N.m], shape ``(num_envs, J)``."""
        self.joint_pos.assign(np.ascontiguousarray(joint_pos, dtype=np.float32))
        self.joint_vel.assign(np.ascontiguousarray(joint_vel, dtype=np.float32))
        self.target_pos.assign(np.ascontiguousarray(target_pos, dtype=np.float32))
        # The adapter's own helper kernels take the ambient Warp device, exactly as the
        # backends that scope one around the stepping loop do.
        with wp.ScopedDevice(self.device):
            self.adapter.step(self.state, self.control, DT)
        return self.control.joint_f_2d.numpy().copy()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset the actuator state of the given environments."""
        with wp.ScopedDevice(self.device):
            self.adapter.reset(env_ids)


"""
Configuration and authoring.
"""


def test_bam_cfg_is_accepted_by_newton_native_validation():
    """The BAM config must pass the gate that ``use_newton_actuators=True`` runs."""
    cfg = _make_cfg()
    assert _is_newton_native_actuator_cfg(cfg)
    validate_newton_native_actuator_cfgs({"servo": cfg})


@pytest.mark.parametrize("backend", ["physx", "ovphysx"])
def test_bam_cfg_is_rejected_on_a_host_adapter_backend(monkeypatch, backend):
    """Shared configuration validation rejects BAM before backend initialization."""
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.schemas.schemas_actuators import define_actuator_properties

    physics_cfg = PhysxCfg() if backend == "physx" else OvPhysxCfg()
    sim_cfg = SimulationCfg(physics=physics_cfg, use_newton_actuators=True)
    manager_name = physics_cfg.class_type.rsplit(":", 1)[-1]
    sim_ctx = SimpleNamespace(cfg=sim_cfg, physics_manager=SimpleNamespace(__name__=manager_name))
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim_ctx)
    # Reject the configuration even when the optional Newton backend is not installed.
    monkeypatch.setitem(sys.modules, "isaaclab_newton", None)
    monkeypatch.setitem(sys.modules, "isaaclab_newton.physics", None)
    with pytest.raises(ValueError, match="BAM requires.*Newton backend"):
        define_actuator_properties("/Robot", {"servo": _make_cfg()})


@pytest.mark.parametrize("model, flags", [("m1", (0, 0, 0)), ("m2", (1, 0, 0)), ("m5", (1, 1, 0)), ("m6", (1, 1, 1))])
def test_authored_prim_resolves_to_the_bam_drive(model, flags):
    """Authoring a BAM group must produce a parseable ``NewtonBamDriveAPI`` actuator prim."""
    cfg = _make_cfg(vin_min=6.0, min_delay=1, max_delay=3, delay_hold_prob=0.25, delay_update_period=4)
    cfg.motor.model = model
    stage = _make_stage(cfg)

    parsed = [p for prim in Usd.PrimRange(stage.GetPrimAtPath("/Robot")) if (p := parse_actuator_prim(prim))]
    assert len(parsed) == len(JOINT_NAMES)
    for entry in parsed:
        assert entry.drive_class is DriveBam
        assert entry.component_specs == [], "the BAM delay is drive-internal, not a Delay component"
        resolved = DriveBam.resolve_arguments(dict(entry.drive_kwargs))
        params = _reference_params()
        # Both deployment settings and identified constants come from the config.
        assert resolved["kp_fw"] == pytest.approx(KP_FW)
        assert resolved["vin"] == pytest.approx(VIN)
        assert resolved["vin_min"] == pytest.approx(6.0)
        assert resolved["kt"] == pytest.approx(params.kt)
        assert resolved["load_friction_external_quad"] == pytest.approx(params.load_friction_external_quad)
        assert (resolved["min_delay"], resolved["max_delay"]) == (1, 3)
        assert resolved["delay_hold_prob"] == pytest.approx(0.25)
        assert resolved["delay_update_period"] == 4
        assert (resolved["stribeck"], resolved["load_dependent"], resolved["quadratic"]) == flags

    # The drive schema token is applied on the prim, not just implied by the parse.
    # ``NewtonBamDriveAPI`` has no registered USD schema definition, so the composed
    # ``GetAppliedSchemas`` filters it out; read the authored opinion instead.
    spec = stage.GetRootLayer().GetPrimAtPath(f"/Robot/servo_{JOINT_NAMES[0]}_actuator")
    assert BAM_DRIVE_API in spec.GetInfo("apiSchemas").GetAppliedItems()


def test_configuration_replaces_existing_usd_coefficients(tmp_path):
    """A serialized asset's fit is replaced entirely, including runtime parameter values."""
    cfg = _make_cfg()
    stage = _make_stage(cfg)
    for index, name in enumerate(JOINT_NAMES):
        prim = stage.GetPrimAtPath(f"/Robot/servo_{name}_actuator")
        prim.GetAttribute("newton:kt").Set(0.3 + index * 0.1)
        prim.GetAttribute("newton:frictionScale").Set(2.0)
    path = tmp_path / "robot.usda"
    stage.Export(str(path))
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths.append(str(path))
    cfg.kp_fw = 123.0
    cfg.motor.friction_base = 0.012
    author_actuator_prims(stage, "/Robot", {"servo": cfg})
    parsed = [entry for prim in Usd.PrimRange(stage.GetPrimAtPath("/Robot")) if (entry := parse_actuator_prim(prim))]
    assert len(parsed) == len(JOINT_NAMES)
    for entry in parsed:
        resolved = DriveBam.resolve_arguments(dict(entry.drive_kwargs))
        assert resolved["kt"] == pytest.approx(_reference_params().kt)
        assert resolved["kp_fw"] == pytest.approx(123.0)
        assert resolved["friction_base"] == pytest.approx(0.012)
        assert resolved["friction_scale"] == 1.0


@pytest.mark.parametrize("field", ["motor", "kp_fw", "vin"])
def test_authoring_requires_motor_and_deployment_settings(field):
    """Existing USD values cannot fill missing configuration fields."""
    cfg = _make_cfg()
    stage = _make_stage(cfg)
    setattr(cfg, field, MISSING)
    with pytest.raises(TypeError, match=field):
        author_actuator_prims(stage, "/Robot", {"servo": cfg})


@pytest.mark.parametrize("limit, voltage_range", [(None, None), (None, (6.5, 8.2)), (0.05, (6.5, 8.2))])
def test_effort_limit_is_authored_on_the_drive_not_as_a_clamping_component(limit, voltage_range):
    """A BAM actuator prim must carry no USD-registered API schema beside the BAM token.

    Newton resolves an actuator prim's components from ``Usd.Prim.GetAppliedSchemas``, falling
    back to the raw ``apiSchemas`` metadata *only when that comes back empty*.
    ``NewtonBamDriveAPI`` has no registered schema definition, so USD drops it from the
    composed list; a registered sibling such as ``NewtonMaxEffortClampingAPI`` would make the
    composed list non-empty and the BAM drive would vanish from the parse. The effort
    limit is therefore a drive parameter, and this test is what stops it going back.
    """
    cfg = _make_cfg(actuator_effort_limit=limit, vin_range=voltage_range)
    stage = _make_stage(cfg)

    prim = stage.GetPrimAtPath(f"/Robot/servo_{JOINT_NAMES[0]}_actuator")
    parsed = parse_actuator_prim(prim)
    assert parsed is not None and parsed.drive_class is DriveBam
    assert parsed.component_specs == [], "a BAM prim must compose no clamping or delay component"
    voltage = max(voltage_range) if voltage_range is not None else cfg.vin
    expected = limit if limit is not None else voltage * cfg.motor.kt / cfg.motor.resistance
    assert DriveBam.resolve_arguments(dict(parsed.drive_kwargs))["max_effort"] == pytest.approx(expected)

    spec = stage.GetRootLayer().GetPrimAtPath(prim.GetPath())
    assert list(spec.GetInfo("apiSchemas").GetAppliedItems()) == [BAM_DRIVE_API]


def test_driven_joints_are_seeded_with_a_positive_friction():
    """Constraint allocation needs a positive seed even for a fit with zero Coulomb friction."""
    cfg = _make_cfg()
    cfg.motor.friction_base = 0.0
    stage = _make_stage(cfg)
    seeds = []
    for name in JOINT_NAMES:
        friction = stage.GetPrimAtPath(f"/Robot/{name}").GetAttribute("newton:friction")
        assert friction.IsValid() and friction.Get() > 0.0
        seeds.append(friction.Get())
        friction.Set(2.0 * friction.Get())
    # Existing joint friction must not change the initialization value.
    author_actuator_prims(stage, "/Robot", {"servo": cfg})
    for name, seed in zip(JOINT_NAMES, seeds, strict=True):
        friction = stage.GetPrimAtPath(f"/Robot/{name}").GetAttribute("newton:friction")
        assert friction.Get() == seed


"""
Kernel behaviour.
"""


@pytest.mark.parametrize("device", test_devices())
def test_drive_matches_upstream_motor_and_friction_goldens(device):
    """The USD-to-Warp path preserves upstream firmware, motor and m6 friction outputs."""
    with np.load(Path(__file__).parent / "data" / "bam_xl330_m6_goldens.npz") as data:
        goldens = {key: data[key] for key in data.files}
    samples = len(goldens["q"])
    harness = _Harness(num_envs=samples // 2, device=device)
    # The budget's prior motor load is an independent golden input, not recomputed by the port.
    state_in, state_out = harness.actuator.state(), harness.actuator.state()
    state_in.drive_state.prev_applied_torque.assign(goldens["prev_tau"].astype(np.float32))
    harness.drive.external_torque.assign(goldens["ext_tau"].astype(np.float32))
    for array, key in ((harness.joint_pos, "q"), (harness.joint_vel, "dq"), (harness.target_pos, "q_target")):
        array.assign(goldens[key].astype(np.float32).reshape(-1, 2))
    with wp.ScopedDevice(device):
        harness.actuator.step(harness.state, harness.control, state_in, state_out, dt=DT)
    np.testing.assert_allclose(harness.drive.motor_torque.numpy(), goldens["motor_torque"], rtol=1e-5, atol=1e-6)
    limit = VIN * float(goldens["attr_kt"]) / float(goldens["attr_R"])
    np.testing.assert_allclose(
        harness.control.joint_f_2d.numpy().reshape(-1),
        np.clip(goldens["motor_torque"], -limit, limit),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        harness.drive.friction_budget.numpy(), goldens["frictionloss_budget"], rtol=1e-5, atol=1e-6
    )


def test_unbound_drive_rejects_stepping(monkeypatch):
    """A BAM drive imported outside Lab's solver-binding path must raise a Python error."""
    harness = _Harness(num_envs=1, device="cpu")
    harness.drive.external_torque = None
    # Keep a missing guard from crashing the test process through a null Warp array.
    monkeypatch.setattr(wp, "launch", lambda *args, **kwargs: None)
    zeros = np.zeros((1, 2))
    with pytest.raises(RuntimeError, match="BAM.*solver binding"):
        harness.step(zeros, zeros, zeros)


@pytest.mark.parametrize("device", test_devices())
def test_solver_mode_emits_the_motor_torque_and_publishes_the_budget(device):
    """With the solver owning the friction, BAM applies the motor torque and exports the budget."""
    harness = _Harness(num_envs=1, device=device, max_effort=0.05, sag_gain=0.1)
    params = _reference_params()

    effort = harness.step(np.array([[0.3, -0.1]]), np.array([[0.5, -0.4]]), np.zeros((1, 2)))

    motor = harness.drive.motor_torque.numpy().copy()
    assert np.abs(motor).max() > 0.05, "the configured effort limit must bind"
    np.testing.assert_allclose(effort.reshape(-1), np.clip(motor, -0.05, 0.05), atol=0.0, rtol=0.0)
    budget = harness.drive.friction_budget.numpy()
    assert (budget >= params.friction_base).all(), "the published budget must keep the Coulomb floor"

    # Friction uses the previously applied effort even if the limit changes; sag uses raw torque.
    harness.drive.max_effort.fill_(0.02)
    zeros = np.zeros((1, 2))
    harness.step(zeros, zeros, zeros)
    expected_budget = (
        params.friction_base
        + params.friction_stribeck
        + np.abs(effort.reshape(-1)) * (params.load_friction_motor + params.load_friction_motor_stribeck)
    )
    np.testing.assert_allclose(harness.drive.friction_budget.numpy(), expected_budget, rtol=1e-6)
    np.testing.assert_allclose(harness.drive.effective_vin.numpy(), VIN - 0.1 * np.abs(motor).sum(), rtol=1e-6)
    harness.reset(torch.tensor([0], device=device))
    harness.step(zeros, zeros, zeros)
    np.testing.assert_allclose(
        harness.drive.friction_budget.numpy(), params.friction_base + params.friction_stribeck, rtol=1e-6
    )


@pytest.mark.parametrize("device", test_devices())
def test_friction_scale_changes_the_published_budget(device):
    """Friction scaling changes the solver budget while preserving the motor torque.

    This is the parameter an environment's domain-randomization event drives; the write goes
    through the same drive array the group-parameter API addresses.
    """
    pos, vel, target = np.array([[0.05, 0.05]]), np.array([[0.02, 0.02]]), np.zeros((1, 2))

    baseline = _Harness(num_envs=1, device=device)
    baseline_effort = baseline.step(pos, vel, target)

    scaled = _Harness(num_envs=1, device=device)
    scaled.drive.friction_scale.fill_(4.0)
    scaled_effort = scaled.step(pos, vel, target)

    np.testing.assert_allclose(
        scaled.drive.friction_budget.numpy(),
        4.0 * baseline.drive.friction_budget.numpy(),
        rtol=1e-6,
        atol=0.0,
    )
    np.testing.assert_array_equal(scaled_effort, baseline_effort)


@pytest.mark.parametrize("device", test_devices())
def test_shared_supply_sags_with_the_group_load(device):
    """Voltage and raw torque match a recording of BAM 62bd8ce's stateful compute method."""
    with np.load(Path(__file__).parent / "data" / "bam_xl330_m6_goldens.npz") as data:
        goldens = {key: data[key] for key in data.files if key.startswith("sag_")}
    shape = goldens["sag_q"].shape[1:]
    harness = _Harness(num_envs=shape[0], device=device, vin_min=float(goldens["sag_vin_min"]))
    for field, key in (("vin", "sag_vin"), ("sag_gain", "sag_gain")):
        getattr(harness.drive, field).assign(np.broadcast_to(goldens[key], shape).astype(np.float32).ravel())
    for step, (pos, vel, target) in enumerate(
        zip(goldens["sag_q"], goldens["sag_dq"], goldens["sag_target"], strict=True)
    ):
        harness.step(pos, vel, target)
        np.testing.assert_allclose(
            harness.drive.effective_vin.numpy().reshape(shape),
            np.broadcast_to(goldens["sag_effective_vin"][step], shape),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            harness.drive.motor_torque.numpy().reshape(shape),
            goldens["sag_motor_torque"][step],
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize("device", test_devices())
def test_startup_sampling_draws_one_value_per_environment(device):
    """The config's start-up ranges must reach the drive once the actuator exists.

    A USD prim is shared by every clone, so the ranges cannot be authored per environment.
    They are drawn afterwards, with one value covering all of an environment's joints.
    """
    cfg = _make_cfg(vin_range=(6.0, 8.0), vin_drop_gain_range=(0.0, 0.2))
    harness = _Harness(num_envs=8, device=device)

    apply_bam_startup_sampling(harness.drive, cfg)

    for attr, (low, high) in (("vin", cfg.vin_range), ("sag_gain", cfg.vin_drop_gain_range)):
        values = getattr(harness.drive, attr).numpy().reshape(8, len(JOINT_NAMES))
        np.testing.assert_allclose(values[:, 0], values[:, 1], atol=0.0, rtol=0.0)
        assert ((values >= low) & (values <= high)).all()
        assert len(np.unique(values[:, 0])) > 1, "every environment drew the same value"
    before_reset = {name: getattr(harness.drive, name).numpy().copy() for name in ("vin", "sag_gain")}
    harness.reset(torch.arange(8, device=device))
    for name, values in before_reset.items():
        np.testing.assert_array_equal(getattr(harness.drive, name).numpy(), values)
    # Friction remains unscaled until a task event writes it.
    np.testing.assert_allclose(harness.drive.friction_scale.numpy(), 1.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_delay_matches_recorded_mjlab_sequence(device):
    """Replay mjlab's 3--6-step lag draws through warm-up, ring wrap, and a partial reset.

    Torch and Warp use different RNGs. Inject the recorded draws into native drive state
    and hold them for each step; the independent RNG test covers sampling and episode seeds.
    """
    with np.load(Path(__file__).parent / "data" / "bam_xl330_m6_goldens.npz") as data:
        goldens = {key: data[key] for key in data.files if key.startswith("delay_")}
    shape = goldens["delay_commands"].shape[1:]
    harness = _Harness(
        num_envs=shape[0],
        device=device,
        min_delay=int(goldens["delay_min"]),
        max_delay=int(goldens["delay_max"]),
        delay_hold_prob=1.0,
    )
    state_in, state_out = harness.actuator.state(), harness.actuator.state()
    for step, command in enumerate(goldens["delay_commands"]):
        with wp.ScopedDevice(device):
            if goldens["delay_resets"][step].any():
                mask = wp.array(np.repeat(goldens["delay_resets"][step], shape[1]), dtype=wp.bool, device=device)
                state_in.reset(mask)
                state_out.reset(mask)
            state_in.drive_state.delay_lag.assign(np.repeat(goldens["delay_lags"][step], shape[1]).astype(np.int32))
            harness.target_pos.assign(command.astype(np.float32))
            harness.control.joint_f_2d.zero_()
            harness.actuator.step(harness.state, harness.control, state_in, state_out, dt=DT)
            state_in, state_out = state_out, state_in
        np.testing.assert_allclose(
            harness.control.joint_f_2d.numpy(),
            goldens["delay_motor_torque"][step],
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"step {step}",
        )


@pytest.mark.parametrize(
    "device, hold_probability, period",
    [(device, 0.0, 0) for device in test_devices()] + [("cpu", 1.0, 0), ("cpu", 0.0, 4)],
)
def test_constant_delay_replays_an_older_command(device, hold_probability, period):
    """A fixed lag of ``k`` steps must reproduce an undelayed actuator fed the ``k``-step-old command."""
    lag = 3
    delayed = _Harness(
        num_envs=2,
        device=device,
        min_delay=lag,
        max_delay=lag,
        delay_hold_prob=hold_probability,
        delay_update_period=period,
    )
    undelayed = _Harness(num_envs=2, device=device)

    commands = [np.full((2, 2), 0.01 * step) for step in range(8)]
    pos, vel = np.zeros((2, 2)), np.zeros((2, 2))
    for step, command in enumerate(commands):
        got = delayed.step(pos, vel, command)
        # The ring clamps to the oldest command it has seen, exactly like the reference buffer.
        expected = undelayed.step(pos, vel, commands[max(step - lag, 0)])
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0.0, err_msg=f"step {step}")

    delayed.reset(torch.tensor([0], device=device))
    fresh = np.full((2, 2), -0.02)
    got = delayed.step(pos, vel, fresh)
    expected_command = commands[len(commands) - lag].copy()
    expected_command[0] = fresh[0]
    expected = undelayed.step(pos, vel, expected_command)
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0.0)
    assert not np.allclose(got[0], got[1]), "the untouched environment must keep its delayed command"

    got = delayed.step(pos, vel, -fresh)
    expected_command = commands[len(commands) + 1 - lag].copy()
    expected_command[0] = fresh[0]
    expected = undelayed.step(pos, vel, expected_command)
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("hold_probability, period", [(1.0, 0), (0.0, 4)])
def test_delay_hold_and_update_period_reach_the_motor_output(hold_probability, period):
    """Hold freezes the lag; periodic refreshes remain staggered per environment."""
    harness = _Harness(
        num_envs=16,
        device="cpu",
        max_delay=3,
        delay_hold_prob=hold_probability,
        delay_update_period=period,
    )
    params = _reference_params()
    # Small position commands remain in the linear firmware regime at rest, so motor
    # torque identifies the delayed command without reading the private delay ring.
    command_step = 0.001
    torque_step = command_step * KP_FW * params.error_gain * VIN * params.kt / params.R
    zeros = np.zeros((16, 2))
    lags = []
    for step in range(24):
        efforts = harness.step(zeros, zeros, np.full_like(zeros, command_step * step))
        lags.append(step - np.rint(efforts / torque_step).astype(int))
    history = np.stack(lags)
    assert history.min() >= 0 and history.max() <= 3
    np.testing.assert_array_equal(history[:, :, 0], history[:, :, 1])
    if hold_probability == 1.0:
        np.testing.assert_array_equal(history, 0)
    else:
        phases = set()
        # Ignore warm-up: the ring initially clips lag to the available command history.
        for joint_history in history[4:].reshape(20, -1).T:
            changed = np.flatnonzero(np.diff(joint_history)) + 5
            residues = {int(index) % period for index in changed}
            assert len(residues) <= 1
            phases.update(residues)
        assert len(phases) > 1, "lag refreshes must not synchronize every driven joint"


@pytest.mark.parametrize("device", test_devices())
def test_delay_rng_changes_on_reset_and_preserves_untouched_environments(device):
    """Lag draws vary across episodes, remain reproducible, and survive CUDA graph replay."""
    harnesses = [_Harness(num_envs=2, device=device, max_delay=3) for _ in range(3)]
    zeros = np.zeros((2, 2), dtype=np.float32)
    graphs = []
    for harness in harnesses:
        harness.step(zeros, zeros, zeros)  # Compile before CUDA capture.
        harness.reset(torch.arange(2, device=device))
        if device.startswith("cuda"):
            with wp.ScopedDevice(device), wp.ScopedCapture() as capture:
                harness.adapter.step(harness.state, harness.control, DT, swap_state=False)
            graphs.append(capture.graph)
        else:
            graphs.append(None)

    def rollout():
        efforts = []
        for step in range(32):
            command = np.full_like(zeros, step * 0.001)
            step_efforts = []
            for harness, graph in zip(harnesses, graphs, strict=True):
                harness.target_pos.assign(command)
                with wp.ScopedDevice(device):
                    if graph is None:
                        harness.adapter.step(harness.state, harness.control, DT)
                    else:
                        wp.capture_launch(graph)
                step_efforts.append(harness.control.joint_f_2d.numpy().copy())
            efforts.append(step_efforts)
        return np.asarray(efforts)

    first = rollout()
    np.testing.assert_array_equal(first[..., 0], first[..., 1])
    assert not np.array_equal(first[:, 0, 0], first[:, 0, 1]), "environments must have independent lags"
    np.testing.assert_array_equal(first[:, 0], first[:, 1])
    np.testing.assert_array_equal(first[:, 0], first[:, 2])
    for harness in harnesses[:2]:
        harness.reset(torch.tensor([0], device=device))
    second = rollout()
    np.testing.assert_array_equal(second[..., 0], second[..., 1])
    # Identical reset histories give identical draws, but a new episode gets a new sequence.
    np.testing.assert_array_equal(second[:, 0], second[:, 1])
    assert not np.array_equal(first[:, 0, 0], second[:, 0, 0]), "reset replayed the same delay sequence"
    # The third harness never resets; its second environment is the continuation reference.
    np.testing.assert_array_equal(second[:, 0, 1], second[:, 2, 1])


@pytest.mark.parametrize("device", test_devices())
def test_reset_restores_the_first_step_behaviour(device):
    """Resetting an environment must clear its caches without touching the others."""
    harness = _Harness(num_envs=2, device=device)
    harness.drive.sag_gain.fill_(0.5)
    pos, vel, target = np.array([[0.2, 0.2], [0.2, 0.2]]), np.array([[1.0, 1.0], [1.0, 1.0]]), np.zeros((2, 2))

    first = harness.step(pos, vel, target)
    harness.step(pos, vel, target)
    harness.reset(torch.tensor([0], device=device))
    after = harness.step(pos, vel, target)

    np.testing.assert_allclose(after[0], first[0], atol=1e-6, rtol=0.0)
    assert np.abs(after[1] - first[1]).max() > 1e-9, "the untouched environment must keep its history"
