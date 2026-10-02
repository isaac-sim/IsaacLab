# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for BAM actuator authoring and the Newton-native BAM drive.

Drive outputs are checked against upstream BAM recordings, stepping the actuator with
test-supplied joint state and external load.
"""

import sys
from dataclasses import MISSING, fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.actuators import DriveBam, apply_bam_startup_sampling
from newton.actuators import parse_actuator_prim

from pxr import Sdf, Usd

from isaaclab.actuators import BamActuatorCfg, BamMotorCfg
from isaaclab.actuators.actuator_bam_cfg import BAM_DRIVE_API
from isaaclab.actuators.newton import NewtonActuatorAdapter, PhysxActuatorWrapper
from isaaclab.sim.schemas.schemas_actuators import author_actuator_prims
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import validate
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.string import to_camel_case

pytestmark = [pytest.mark.unit, pytest.mark.filterwarnings("error::DeprecationWarning")]

BAM_DATA_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Tests/BAM"
"""Hosted BAM fixtures: the two-servo USD and the upstream XL330/m6 recordings."""

JOINT_NAMES = ["servo_0", "servo_1"]
"""Joints of the fixture; two, so the shared-supply sag is observable."""

DT = 1.0 / 120.0
"""Physics timestep [s]."""

VIN = 7.4
"""Nominal supply voltage [V]."""

KP_FW = 200.0
"""Firmware proportional gain [-]."""


@pytest.fixture(scope="module")
def goldens() -> dict[str, np.ndarray]:
    """Upstream BAM 62bd8ce recordings and the identified fit they were made with."""
    with np.load(retrieve_file_path(f"{BAM_DATA_DIR}/bam_xl330_m6_goldens.npz")) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def servo_usd() -> str:
    """Local path of the two-servo articulation fixture."""
    return retrieve_file_path(f"{BAM_DATA_DIR}/bam_two_servo.usda")


def make_cfg(goldens: dict[str, np.ndarray], **overrides) -> BamActuatorCfg:
    """Build a BAM configuration from the recorded m6 fit."""
    params = {key.removeprefix("attr_"): goldens[key].item() for key in goldens if key.startswith("attr_")}
    params["resistance"] = params.pop("R")
    motor = BamMotorCfg(model="m6", **{f.name: params[f.name] for f in fields(BamMotorCfg) if f.name in params})
    return BamActuatorCfg(**{"joint_names_expr": [".*"], "motor": motor, "vin": VIN, "kp_fw": KP_FW, **overrides})


def make_stage(servo_usd: str, cfg: BamActuatorCfg | None = None) -> Usd.Stage:
    """Open a fresh fixture layer, optionally authoring its actuators from *cfg*."""
    stage = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(servo_usd))
    if cfg is not None:
        author_actuator_prims(stage, "/Robot", {"servo": cfg})
    return stage


def make_actuator(servo_usd: str, num_envs: int, device: str, **drive_overrides: float | int) -> SimpleNamespace:
    """Build the fixture's BAM actuator with flat joint-state buffers the test writes into."""
    stage = make_stage(servo_usd)
    for joint_name in JOINT_NAMES:
        prim = stage.GetPrimAtPath(f"/Robot/servo_{joint_name}_actuator")
        for name, value in drive_overrides.items():
            prim.GetAttribute(f"newton:{to_camel_case(name)}").Set(value)
    adapter = NewtonActuatorAdapter.from_usd(
        stage=stage,
        joint_names=JOINT_NAMES,
        num_envs=num_envs,
        num_joints=len(JOINT_NAMES),
        device=device,
        articulation_prim_path="/Robot",
    )
    (actuator,) = adapter.actuators
    # Like NewtonManager: set the BAM stride, then build the adapter that creates the states.
    actuator.drive.env_dof_stride = len(JOINT_NAMES)
    adapter = NewtonActuatorAdapter(
        actuators=[actuator], num_envs=num_envs, num_joints=len(JOINT_NAMES), dof_offset=0, device=device
    )
    actuator.drive.external_torque = wp.zeros(actuator.num_actuators, dtype=wp.float32, device=device)

    shape = (num_envs, len(JOINT_NAMES))
    state = PhysxActuatorWrapper.create(*shape, device)
    control = PhysxActuatorWrapper.create(*shape, device)
    joint_pos = wp.zeros(shape, dtype=wp.float32, device=device)
    joint_vel = wp.zeros(shape, dtype=wp.float32, device=device)
    target_pos = wp.zeros(shape, dtype=wp.float32, device=device)
    state.joint_q = joint_pos.reshape(-1)
    state.joint_qd = joint_vel.reshape(-1)
    control.joint_target_pos = target_pos.reshape(-1)
    control.joint_target_vel = wp.zeros(num_envs * len(JOINT_NAMES), dtype=wp.float32, device=device)
    control.joint_act = None
    adapter.finalize(control)
    return SimpleNamespace(
        adapter=adapter,
        actuator=actuator,
        drive=actuator.drive,
        state=state,
        control=control,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        target_pos=target_pos,
        device=device,
    )


def step(bam: SimpleNamespace, joint_pos: np.ndarray, joint_vel: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Run one actuator step and return the applied effort [N.m], shape ``(num_envs, J)``."""
    bam.joint_pos.assign(np.ascontiguousarray(joint_pos, dtype=np.float32))
    bam.joint_vel.assign(np.ascontiguousarray(joint_vel, dtype=np.float32))
    bam.target_pos.assign(np.ascontiguousarray(target_pos, dtype=np.float32))
    with wp.ScopedDevice(bam.device):
        bam.adapter.step(bam.state, bam.control, DT)
    return bam.control.joint_f_2d.numpy().copy()


def reset(bam: SimpleNamespace, env_ids: torch.Tensor) -> None:
    """Reset the actuator state of the given environments."""
    with wp.ScopedDevice(bam.device):
        bam.adapter.reset(env_ids)


"""
Configuration and authoring.
"""


def test_bam_requires_the_newton_backend(monkeypatch, goldens):
    """BAM is rejected on a PhysX simulation, even without the Newton package installed."""
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.schemas.schemas_actuators import define_actuator_properties

    physics_cfg = PhysxCfg()
    sim_cfg = SimulationCfg(physics=physics_cfg, use_newton_actuators=True)
    manager_name = physics_cfg.class_type.rsplit(":", 1)[-1]
    sim_ctx = SimpleNamespace(cfg=sim_cfg, physics_manager=SimpleNamespace(__name__=manager_name))
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim_ctx)
    monkeypatch.setitem(sys.modules, "isaaclab_newton", None)
    with pytest.raises(ValueError, match="BAM requires.*Newton backend"):
        define_actuator_properties("/Robot", {"servo": make_cfg(goldens)})


def test_motor_fit_rejects_unsupported_models_and_an_implicit_current_limit():
    """An unsupported model would silently author a different friction law."""
    motor = BamMotorCfg(
        model="m3", kt=0.36, resistance=2.8, error_gain=0.003, friction_base=0.005, friction_viscous=0.006
    )
    with pytest.raises(TypeError, match="max_current"):
        validate(motor)
    motor.max_current = 0.0
    with pytest.raises(ValueError, match="model"):
        validate(motor)
    motor.model = "m1"
    validate(motor)


@pytest.mark.parametrize("model, flags", [("m1", (0, 0, 0)), ("m2", (1, 0, 0)), ("m5", (1, 1, 0)), ("m6", (1, 1, 1))])
def test_authored_prim_resolves_to_the_bam_drive(goldens, servo_usd, model, flags):
    """Authoring a BAM group produces a ``NewtonBamDriveAPI`` prim carrying the configuration."""
    cfg = make_cfg(goldens, vin_min=6.0, min_delay=1, max_delay=3, delay_hold_prob=0.25, delay_update_period=4)
    cfg.motor.model = model
    stage = make_stage(servo_usd, cfg)

    parsed = [p for prim in Usd.PrimRange(stage.GetPrimAtPath("/Robot")) if (p := parse_actuator_prim(prim))]
    assert len(parsed) == len(JOINT_NAMES)
    for entry in parsed:
        assert entry.drive_class is DriveBam
        assert entry.component_specs == [], "the BAM delay is drive-internal, not a Delay component"
        resolved = DriveBam.resolve_arguments(dict(entry.drive_kwargs))
        assert resolved["kp_fw"] == pytest.approx(KP_FW)
        assert resolved["vin"] == pytest.approx(VIN)
        assert resolved["vin_min"] == pytest.approx(6.0)
        assert resolved["kt"] == pytest.approx(goldens["attr_kt"].item())
        assert resolved["load_friction_external_quad"] == pytest.approx(goldens["attr_load_friction_external_quad"])
        assert (resolved["min_delay"], resolved["max_delay"]) == (1, 3)
        assert resolved["delay_hold_prob"] == pytest.approx(0.25)
        assert resolved["delay_update_period"] == 4
        assert (resolved["stribeck"], resolved["load_dependent"], resolved["quadratic"]) == flags


def test_configuration_replaces_existing_usd_coefficients(tmp_path, goldens, servo_usd):
    """A serialized asset's fit is replaced entirely, including runtime parameter values."""
    cfg = make_cfg(goldens)
    stage = make_stage(servo_usd, cfg)
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
        assert resolved["kt"] == pytest.approx(goldens["attr_kt"].item())
        assert resolved["kp_fw"] == pytest.approx(123.0)
        assert resolved["friction_base"] == pytest.approx(0.012)
        assert resolved["friction_scale"] == 1.0


@pytest.mark.parametrize("field", ["motor", "kp_fw", "vin"])
def test_authoring_requires_motor_and_deployment_settings(goldens, servo_usd, field):
    """Existing USD values cannot fill missing configuration fields."""
    cfg = make_cfg(goldens)
    stage = make_stage(servo_usd, cfg)
    setattr(cfg, field, MISSING)
    with pytest.raises(TypeError, match=field):
        author_actuator_prims(stage, "/Robot", {"servo": cfg})


@pytest.mark.parametrize("limit, voltage_range", [(None, None), (None, (6.5, 8.2)), (0.05, (6.5, 8.2))])
def test_effort_limit_is_a_drive_parameter(goldens, servo_usd, limit, voltage_range):
    """The prim carries only the BAM token: a registered clamping schema would hide the drive from Newton."""
    cfg = make_cfg(goldens, actuator_effort_limit=limit, vin_range=voltage_range)
    stage = make_stage(servo_usd, cfg)

    prim = stage.GetPrimAtPath(f"/Robot/servo_{JOINT_NAMES[0]}_actuator")
    parsed = parse_actuator_prim(prim)
    assert parsed is not None and parsed.drive_class is DriveBam
    assert parsed.component_specs == []
    voltage = max(voltage_range) if voltage_range is not None else cfg.vin
    expected = limit if limit is not None else voltage * cfg.motor.kt / cfg.motor.resistance
    assert DriveBam.resolve_arguments(dict(parsed.drive_kwargs))["max_effort"] == pytest.approx(expected)
    # The unregistered token is dropped from the composed schemas, so read the authored opinion.
    spec = stage.GetRootLayer().GetPrimAtPath(prim.GetPath())
    assert list(spec.GetInfo("apiSchemas").GetAppliedItems()) == [BAM_DRIVE_API]


def test_driven_joints_are_seeded_with_a_positive_friction(goldens, servo_usd):
    """Constraint allocation needs a positive seed even for a fit with zero Coulomb friction."""
    cfg = make_cfg(goldens)
    cfg.motor.friction_base = 0.0
    stage = make_stage(servo_usd, cfg)
    seeds = []
    for name in JOINT_NAMES:
        friction = stage.GetPrimAtPath(f"/Robot/{name}").GetAttribute("newton:friction")
        assert friction.IsValid() and friction.Get() > 0.0
        seeds.append(friction.Get())
        friction.Set(2.0 * friction.Get())
    # Existing joint friction must not change the seed.
    author_actuator_prims(stage, "/Robot", {"servo": cfg})
    for name, seed in zip(JOINT_NAMES, seeds, strict=True):
        assert stage.GetPrimAtPath(f"/Robot/{name}").GetAttribute("newton:friction").Get() == seed


"""
Drive behavior.
"""


@pytest.mark.parametrize("device", test_devices())
def test_drive_matches_upstream_motor_and_friction_goldens(goldens, servo_usd, device):
    """The USD-to-Warp path preserves upstream firmware, motor and m6 friction outputs."""
    bam = make_actuator(servo_usd, num_envs=len(goldens["q"]) // 2, device=device)
    # The previous applied torque is an independent golden input, not recomputed by the port.
    state_in, state_out = bam.actuator.state(), bam.actuator.state()
    state_in.drive_state.prev_applied_torque.assign(goldens["prev_tau"].astype(np.float32))
    bam.drive.external_torque.assign(goldens["ext_tau"].astype(np.float32))
    for array, key in ((bam.joint_pos, "q"), (bam.joint_vel, "dq"), (bam.target_pos, "q_target")):
        array.assign(goldens[key].astype(np.float32).reshape(-1, 2))
    with wp.ScopedDevice(device):
        bam.actuator.step(bam.state, bam.control, state_in, state_out, dt=DT)
    np.testing.assert_allclose(bam.drive.motor_torque.numpy(), goldens["motor_torque"], rtol=1e-5, atol=1e-6)
    limit = VIN * float(goldens["attr_kt"]) / float(goldens["attr_R"])
    np.testing.assert_allclose(
        bam.control.joint_f_2d.numpy().reshape(-1),
        np.clip(goldens["motor_torque"], -limit, limit),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(bam.drive.friction_budget.numpy(), goldens["frictionloss_budget"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_backlash_feedback_adds_play_angle_but_keeps_motor_velocity(goldens, servo_usd, device):
    """The encoder sees the output angle; back-EMF still uses the rotor velocity."""
    bam = make_actuator(servo_usd, num_envs=2, device=device)
    bam.drive.has_backlash = True
    bam.drive.backlash_pos_indices = wp.array([1, 0, 3, 2], dtype=wp.uint32, device=device)
    positions = np.array([[0.01, -0.02], [0.03, -0.01]], dtype=np.float32)
    velocities = np.array([[0.3, -0.4], [-0.5, 0.2]], dtype=np.float32)
    targets = np.zeros_like(positions)
    forces = step(bam, positions, velocities, targets)
    kt, resistance = goldens["attr_kt"].item(), goldens["attr_R"].item()
    error = targets - positions - positions[:, ::-1]
    expected = kt / resistance * (VIN * KP_FW * goldens["attr_error_gain"].item() * error - kt * velocities)
    np.testing.assert_allclose(forces, expected, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("binding", ["solver", "play-hinge"])
def test_unbound_drive_rejects_stepping(monkeypatch, servo_usd, binding):
    """A drive stepped without its MJWarp or play-hinge binding raises instead of reading garbage."""
    bam = make_actuator(servo_usd, num_envs=1, device="cpu")
    if binding == "solver":
        bam.drive.external_torque = None
    else:
        bam.drive.has_backlash = True
    # Keep a missing guard from crashing the process through a null Warp array.
    monkeypatch.setattr(wp, "launch", lambda *args, **kwargs: None)
    zeros = np.zeros((1, 2))
    with pytest.raises(RuntimeError, match=f"BAM.*{binding}.*binding"):
        step(bam, zeros, zeros, zeros)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_effort_limit_and_friction_budget_feedback(goldens, servo_usd, device):
    """The effort limit binds, the budget keeps the Coulomb floor, and feedback uses the applied effort."""
    bam = make_actuator(servo_usd, num_envs=1, device=device, max_effort=0.05, sag_gain=0.1)

    effort = step(bam, np.array([[0.3, -0.1]]), np.array([[0.5, -0.4]]), np.zeros((1, 2)))

    motor = bam.drive.motor_torque.numpy().copy()
    assert np.abs(motor).max() > 0.05, "the configured effort limit must bind"
    np.testing.assert_allclose(effort.reshape(-1), np.clip(motor, -0.05, 0.05), atol=0.0, rtol=0.0)
    friction_base = goldens["attr_friction_base"].item()
    assert (bam.drive.friction_budget.numpy() >= friction_base).all()

    # Friction uses the previously applied effort even if the limit changes; sag uses raw torque.
    bam.drive.max_effort.fill_(0.02)
    zeros = np.zeros((1, 2))
    step(bam, zeros, zeros, zeros)
    rest_budget = friction_base + goldens["attr_friction_stribeck"].item()
    load_gain = goldens["attr_load_friction_motor"].item() + goldens["attr_load_friction_motor_stribeck"].item()
    np.testing.assert_allclose(
        bam.drive.friction_budget.numpy(), rest_budget + np.abs(effort.reshape(-1)) * load_gain, rtol=1e-6
    )
    np.testing.assert_allclose(bam.drive.effective_vin.numpy(), VIN - 0.1 * np.abs(motor).sum(), rtol=1e-6)
    reset(bam, torch.tensor([0], device=device))
    step(bam, zeros, zeros, zeros)
    np.testing.assert_allclose(bam.drive.friction_budget.numpy(), rest_budget, rtol=1e-6)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_friction_scale_scales_only_the_published_budget(servo_usd, device):
    """The randomized friction scale changes the solver budget, not the motor torque."""
    pos, vel, target = np.array([[0.05, 0.05]]), np.array([[0.02, 0.02]]), np.zeros((1, 2))
    baseline = make_actuator(servo_usd, num_envs=1, device=device)
    baseline_effort = step(baseline, pos, vel, target)
    scaled = make_actuator(servo_usd, num_envs=1, device=device)
    scaled.drive.friction_scale.fill_(4.0)
    scaled_effort = step(scaled, pos, vel, target)

    np.testing.assert_allclose(
        scaled.drive.friction_budget.numpy(), 4.0 * baseline.drive.friction_budget.numpy(), rtol=1e-6, atol=0.0
    )
    np.testing.assert_array_equal(scaled_effort, baseline_effort)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_shared_supply_sags_with_the_group_load(goldens, servo_usd, device):
    """Voltage and raw torque match a recording of upstream BAM's stateful compute method."""
    shape = goldens["sag_q"].shape[1:]
    bam = make_actuator(servo_usd, num_envs=shape[0], device=device, vin_min=float(goldens["sag_vin_min"]))
    for field, key in (("vin", "sag_vin"), ("sag_gain", "sag_gain")):
        getattr(bam.drive, field).assign(np.broadcast_to(goldens[key], shape).astype(np.float32).ravel())
    for index, (pos, vel, target) in enumerate(zip(goldens["sag_q"], goldens["sag_dq"], goldens["sag_target"])):
        step(bam, pos, vel, target)
        np.testing.assert_allclose(
            bam.drive.effective_vin.numpy().reshape(shape),
            np.broadcast_to(goldens["sag_effective_vin"][index], shape),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            bam.drive.motor_torque.numpy().reshape(shape), goldens["sag_motor_torque"][index], rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_startup_sampling_draws_one_value_per_environment(goldens, servo_usd, device):
    """Start-up ranges give one value per environment, held across resets."""
    cfg = make_cfg(goldens, vin_range=(6.0, 8.0), vin_drop_gain_range=(0.0, 0.2))
    bam = make_actuator(servo_usd, num_envs=8, device=device)

    apply_bam_startup_sampling(bam.drive, cfg)

    for attr, (low, high) in (("vin", cfg.vin_range), ("sag_gain", cfg.vin_drop_gain_range)):
        values = getattr(bam.drive, attr).numpy().reshape(8, len(JOINT_NAMES))
        np.testing.assert_array_equal(values[:, 0], values[:, 1])
        assert ((values >= low) & (values <= high)).all()
        assert len(np.unique(values[:, 0])) > 1, "every environment drew the same value"
    before_reset = {name: getattr(bam.drive, name).numpy().copy() for name in ("vin", "sag_gain")}
    reset(bam, torch.arange(8, device=device))
    for name, values in before_reset.items():
        np.testing.assert_array_equal(getattr(bam.drive, name).numpy(), values)
    np.testing.assert_array_equal(bam.drive.friction_scale.numpy(), 1.0)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_delay_matches_recorded_mjlab_sequence(goldens, servo_usd, device):
    """Replay mjlab's recorded 3--6-step lags through warm-up, ring wrap, and a partial reset.

    Torch and Warp RNGs differ, so the recorded lags are injected; the RNG tests below cover sampling.
    """
    shape = goldens["delay_commands"].shape[1:]
    bam = make_actuator(
        servo_usd,
        num_envs=shape[0],
        device=device,
        min_delay=int(goldens["delay_min"]),
        max_delay=int(goldens["delay_max"]),
        delay_hold_prob=1.0,
    )
    state_in, state_out = bam.actuator.state(), bam.actuator.state()
    for index, command in enumerate(goldens["delay_commands"]):
        with wp.ScopedDevice(device):
            if goldens["delay_resets"][index].any():
                mask = wp.array(np.repeat(goldens["delay_resets"][index], shape[1]), dtype=wp.bool, device=device)
                state_in.reset(mask)
                state_out.reset(mask)
            state_in.drive_state.delay_lag.assign(np.repeat(goldens["delay_lags"][index], shape[1]).astype(np.int32))
            bam.target_pos.assign(command.astype(np.float32))
            bam.control.joint_f_2d.zero_()
            bam.actuator.step(bam.state, bam.control, state_in, state_out, dt=DT)
            state_in, state_out = state_out, state_in
        np.testing.assert_allclose(
            bam.control.joint_f_2d.numpy(),
            goldens["delay_motor_torque"][index],
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"step {index}",
        )


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_constant_delay_replays_an_older_command_and_resets_per_environment(servo_usd, device):
    """A fixed lag of ``k`` steps reproduces an undelayed actuator fed the ``k``-step-old command."""
    lag = 3
    delayed = make_actuator(servo_usd, num_envs=2, device=device, min_delay=lag, max_delay=lag)
    undelayed = make_actuator(servo_usd, num_envs=2, device=device)

    commands = [np.full((2, 2), 0.01 * index) for index in range(8)]
    pos, vel = np.zeros((2, 2)), np.zeros((2, 2))
    for index, command in enumerate(commands):
        got = step(delayed, pos, vel, command)
        # Before the ring fills, the oldest command seen is used, like the reference buffer.
        expected = step(undelayed, pos, vel, commands[max(index - lag, 0)])
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0.0, err_msg=f"step {index}")

    reset(delayed, torch.tensor([0], device=device))
    fresh = np.full((2, 2), -0.02)
    got = step(delayed, pos, vel, fresh)
    expected_command = commands[len(commands) - lag].copy()
    expected_command[0] = fresh[0]
    np.testing.assert_allclose(got, step(undelayed, pos, vel, expected_command), atol=1e-6, rtol=0.0)
    assert not np.allclose(got[0], got[1]), "the untouched environment must keep its delayed command"

    got = step(delayed, pos, vel, -fresh)
    expected_command = commands[len(commands) + 1 - lag].copy()
    expected_command[0] = fresh[0]
    np.testing.assert_allclose(got, step(undelayed, pos, vel, expected_command), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("hold_probability, period", [(1.0, 0), (0.0, 4)])
def test_delay_hold_and_update_period_reach_the_motor_output(goldens, servo_usd, hold_probability, period):
    """Hold freezes the lag; periodic refreshes are staggered per environment."""
    bam = make_actuator(
        servo_usd, num_envs=16, device="cpu", max_delay=3, delay_hold_prob=hold_probability, delay_update_period=period
    )
    # Small commands stay in the linear firmware regime at rest, so the torque identifies the delayed command.
    command_step = 0.001
    kt, resistance = goldens["attr_kt"].item(), goldens["attr_R"].item()
    torque_step = command_step * KP_FW * goldens["attr_error_gain"].item() * VIN * kt / resistance
    zeros = np.zeros((16, 2))
    lags = []
    for index in range(24):
        efforts = step(bam, zeros, zeros, np.full_like(zeros, command_step * index))
        lags.append(index - np.rint(efforts / torque_step).astype(int))
    history = np.stack(lags)
    assert history.min() >= 0 and history.max() <= 3
    np.testing.assert_array_equal(history[:, :, 0], history[:, :, 1])
    if hold_probability == 1.0:
        np.testing.assert_array_equal(history, 0)
    else:
        phases = set()
        # Skip warm-up, where the ring clips the lag to the available history.
        for joint_history in history[4:].reshape(20, -1).T:
            changed = np.flatnonzero(np.diff(joint_history)) + 5
            residues = {int(index) % period for index in changed}
            assert len(residues) <= 1
            phases.update(residues)
        assert len(phases) > 1, "lag refreshes must not synchronize every environment"


@pytest.mark.parametrize("device", test_devices())
def test_delay_rng_changes_on_reset_and_preserves_untouched_environments(servo_usd, device):
    """Lag draws vary across episodes, replay deterministically, and survive CUDA graph replay."""
    actuators = [make_actuator(servo_usd, num_envs=2, device=device, max_delay=3) for _ in range(3)]
    zeros = np.zeros((2, 2), dtype=np.float32)
    graphs = []
    for bam in actuators:
        step(bam, zeros, zeros, zeros)  # Compile before CUDA capture.
        reset(bam, torch.arange(2, device=device))
        if device.startswith("cuda"):
            with wp.ScopedDevice(device), wp.ScopedCapture() as capture:
                bam.adapter.step(bam.state, bam.control, DT, swap_state=False)
            graphs.append(capture.graph)
        else:
            graphs.append(None)

    def rollout():
        efforts = []
        for index in range(32):
            command = np.full_like(zeros, index * 0.001)
            step_efforts = []
            for bam, graph in zip(actuators, graphs, strict=True):
                bam.target_pos.assign(command)
                with wp.ScopedDevice(device):
                    if graph is None:
                        bam.adapter.step(bam.state, bam.control, DT)
                    else:
                        wp.capture_launch(graph)
                step_efforts.append(bam.control.joint_f_2d.numpy().copy())
            efforts.append(step_efforts)
        return np.asarray(efforts)

    first = rollout()
    np.testing.assert_array_equal(first[..., 0], first[..., 1])
    assert not np.array_equal(first[:, 0, 0], first[:, 0, 1]), "environments must have independent lags"
    np.testing.assert_array_equal(first[:, 0], first[:, 1])
    np.testing.assert_array_equal(first[:, 0], first[:, 2])
    for bam in actuators[:2]:
        reset(bam, torch.tensor([0], device=device))
    second = rollout()
    # Identical reset histories give identical draws, but a new episode gets a new sequence.
    np.testing.assert_array_equal(second[:, 0], second[:, 1])
    assert not np.array_equal(first[:, 0, 0], second[:, 0, 0]), "reset replayed the same delay sequence"
    # The third actuator never resets; its second environment is the continuation reference.
    np.testing.assert_array_equal(second[:, 0, 1], second[:, 2, 1])


def test_delay_rng_does_not_replay_another_environments_stream(servo_usd):
    """Long episodes must not reach a neighboring environment's earlier lag sequence."""
    bam = make_actuator(servo_usd, num_envs=2, device="cpu", min_delay=3, max_delay=6)
    state_in, state_out = bam.actuator.state(), bam.actuator.state()
    # Jump ahead by the old per-environment stream offset instead of simulating thousands of steps.
    state_in.drive_state.delay_step_count.assign(np.array([7919, 7919, 0, 0], dtype=np.int32))
    efforts = []
    with wp.ScopedDevice("cpu"):
        for index in range(32):
            bam.target_pos.assign(np.full((2, 2), 0.001 * index, dtype=np.float32))
            bam.control.joint_f_2d.zero_()
            bam.actuator.step(bam.state, bam.control, state_in, state_out, dt=DT)
            state_in, state_out = state_out, state_in
            efforts.append(bam.control.joint_f_2d.numpy().copy())
    history = np.asarray(efforts)[6:]
    assert not np.array_equal(history[:, 0], history[:, 1]), "neighboring environments replayed the same stream"


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_reset_clears_supply_history_of_selected_environments(servo_usd, device):
    """Reset clears an environment's sag history without touching the others."""
    bam = make_actuator(servo_usd, num_envs=2, device=device)
    bam.drive.sag_gain.fill_(0.5)
    pos, vel, target = np.full((2, 2), 0.2), np.full((2, 2), 1.0), np.zeros((2, 2))

    first = step(bam, pos, vel, target)
    step(bam, pos, vel, target)
    reset(bam, torch.tensor([0], device=device))
    after = step(bam, pos, vel, target)

    np.testing.assert_allclose(after[0], first[0], atol=1e-6, rtol=0.0)
    assert np.abs(after[1] - first[1]).max() > 1e-9, "the untouched environment must keep its history"
