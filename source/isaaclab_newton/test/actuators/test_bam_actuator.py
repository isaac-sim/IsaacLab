# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for BAM actuator authoring and its Newton-native composition.

Newton's voltage-controlled drive, current and PWM clamp, DC motor, battery, delay and encoder, plus Isaac Lab's
command lag and gearbox friction, are checked against upstream BAM recordings, stepping the actuators with
test-supplied joint state and external load.
"""

import sys
from dataclasses import MISSING, fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.actuators.bam import BamCommandLag, BamFriction
from newton import ModelBuilder
from newton.actuators import (
    BatteryConstantResistance,
    ClampingCurrentPWM,
    DriveVoltageControlled,
    InputProcessorDelay,
    InputProcessorEncoder,
    MotorDC,
    parse_actuator_prim,
    parse_battery_prim,
)

from pxr import Sdf, Usd, UsdPhysics

from isaaclab.actuators import BamActuatorCfg, BamMotorCfg
from isaaclab.actuators.newton import NewtonActuatorAdapter, PhysxActuatorWrapper
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.schemas.schemas_actuators import author_actuator_prims, define_actuator_properties
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

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
    """Local path of the two-servo fixture: two independent servo joints, not a dynamic articulation."""
    return retrieve_file_path(f"{BAM_DATA_DIR}/bam_two_servo.usda")


def make_cfg(goldens: dict[str, np.ndarray], **overrides) -> BamActuatorCfg:
    """Build a BAM configuration from the recorded m6 fit."""
    params = {key.removeprefix("attr_"): goldens[key].item() for key in goldens if key.startswith("attr_")}
    params["resistance"] = params.pop("R")
    motor = BamMotorCfg(model="m6", **{f.name: params[f.name] for f in fields(BamMotorCfg) if f.name in params})
    return BamActuatorCfg(**{"joint_names_expr": ["servo_.*"], "motor": motor, "vin": VIN, "kp_fw": KP_FW, **overrides})


def make_stage(servo_usd: str, cfg: BamActuatorCfg | None = None) -> Usd.Stage:
    """Open a fresh fixture layer, optionally authoring its actuators from *cfg*.

    The fixture's prims carry a pre-native BAM actuator at the paths authoring reuses.
    """
    stage = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(servo_usd))
    if cfg is not None:
        author_actuator_prims(stage, "/Robot", {"servo": cfg})
    return stage


def make_actuator(servo_usd: str, cfg: BamActuatorCfg, num_envs: int, device: str) -> SimpleNamespace:
    """Build the fixture's BAM actuator on its authored supply voltage, with flat joint-state buffers to write.

    Without a Newton model the motors run on the authored ``vin``, so this covers everything but the battery.
    """
    adapter = NewtonActuatorAdapter.from_usd(
        stage=make_stage(servo_usd, cfg),
        joint_names=JOINT_NAMES,
        num_envs=num_envs,
        num_joints=len(JOINT_NAMES),
        device=device,
        articulation_prim_path="/Robot",
    )
    (actuator,) = adapter.actuators
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
        dofs=np.arange(num_envs * len(JOINT_NAMES)).reshape(shape),
        num_dofs=num_envs * len(JOINT_NAMES),
        state=state,
        control=control,
        joint_pos=joint_pos.reshape(-1),
        joint_vel=joint_vel.reshape(-1),
        target_pos=target_pos.reshape(-1),
        friction=None,
        device=device,
    )


def make_articulation_stage(backlash: bool = False) -> Usd.Stage:
    """Create a fixed-base articulation with two servo hinges, each optionally followed by a play hinge."""
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    UsdPhysics.ArticulationRootAPI.Apply(stage.DefinePrim("/Robot", "Xform"))

    def add_body(name: str) -> None:
        mass = UsdPhysics.MassAPI.Apply(stage.DefinePrim(f"/Robot/{name}", "Xform"))
        UsdPhysics.RigidBodyAPI.Apply(mass.GetPrim())
        mass.CreateMassAttr(0.1)
        mass.CreateDiagonalInertiaAttr((1.0e-3, 1.0e-3, 1.0e-3))

    def add_hinge(name: str, parent: str, child: str) -> UsdPhysics.RevoluteJoint:
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/{name}")
        joint.CreateBody0Rel().SetTargets([f"/Robot/{parent}"])
        joint.CreateBody1Rel().SetTargets([f"/Robot/{child}"])
        joint.CreateAxisAttr("Z")
        return joint

    add_body("base")
    UsdPhysics.FixedJoint.Define(stage, "/Robot/root").CreateBody1Rel().SetTargets(["/Robot/base"])
    for name in JOINT_NAMES:
        add_body(f"{name}_rotor")
        add_hinge(name, "base", f"{name}_rotor")
        if backlash:
            add_body(f"{name}_link")
            play = add_hinge(f"passive_{name}_backlash", f"{name}_rotor", f"{name}_link")
            play.CreateLowerLimitAttr(-1.0)
            play.CreateUpperLimitAttr(1.0)
    return stage


def make_model(cfg: BamActuatorCfg, num_envs: int, device: str, backlash: bool = False) -> SimpleNamespace:
    """Author *cfg* on a two-servo articulation, import it into Newton per environment and bind an adapter.

    As on the articulation path, Newton builds one battery per environment from the authored battery prim.
    """
    stage = make_articulation_stage(backlash)
    author_actuator_prims(stage, "/Robot", {"servo": cfg})
    prototype = ModelBuilder()
    prototype.add_usd(stage)
    builder = ModelBuilder()
    builder.replicate(prototype, num_envs)
    model = builder.finalize(device=device)
    (actuator,) = model.actuators
    (battery,) = model.batteries
    adapter = NewtonActuatorAdapter(
        list(model.actuators),
        num_envs,
        model.joint_dof_count // num_envs,
        dof_offset=0,
        device=device,
        batteries=model.batteries,
    )
    state = model.state()
    control = model.control()
    adapter.finalize(control)
    return SimpleNamespace(
        model=model,
        adapter=adapter,
        actuator=actuator,
        battery=battery,
        dofs=actuator.indices.numpy().reshape(num_envs, len(JOINT_NAMES)),
        num_dofs=model.joint_dof_count,
        state=state,
        control=control,
        # Revolute joints only: joint_q and joint_target_q share joint_qd's layout.
        joint_pos=state.joint_q,
        joint_vel=state.joint_qd,
        target_pos=control.joint_target_q,
        friction=None,
        device=device,
    )


def add_friction(bam: SimpleNamespace, motor: BamMotorCfg) -> BamFriction:
    """Attach gearbox friction, reset with the adapter as the articulation path does.

    ``bam.loads`` stands in for the solver's load observables and the model's joint friction, all zero.
    """
    dof_indices = wp.array(bam.dofs.ravel(), dtype=wp.uint32, device=bam.device)
    bam.friction = BamFriction(motor, dof_indices, dofs_per_env=bam.num_dofs // bam.dofs.shape[0])
    bam.adapter.register_reset_callback(bam.friction.reset)
    names = ("qfrc_bias", "qfrc_constraint", "qfrc_frictionloss", "joint_friction")
    bam.loads = SimpleNamespace(**{name: wp.zeros(bam.num_dofs, dtype=wp.float32, device=bam.device) for name in names})
    return bam.friction


def scatter(array: wp.array, dofs: np.ndarray, values: np.ndarray) -> None:
    """Write *values* into the entries *dofs* of a flat array."""
    full = array.numpy()
    full[dofs] = values
    array.assign(full)


def step(bam: SimpleNamespace, joint_pos: np.ndarray, joint_vel: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Run one physics step's actuator hooks and return the applied effort [N.m], shape ``(num_envs, J)``."""
    for array, values in ((bam.joint_pos, joint_pos), (bam.joint_vel, joint_vel), (bam.target_pos, target_pos)):
        scatter(array, bam.dofs, values)
    with wp.ScopedDevice(bam.device):
        if bam.friction is not None:
            bam.friction.compute_budget(bam.state.joint_qd, *vars(bam.loads).values())
        bam.adapter.step(bam.state, bam.control, DT)
        if bam.friction is not None:
            bam.friction.record_applied_effort(bam.control.joint_f)
    return bam.control.joint_f.numpy()[bam.dofs]


def supply_voltage(bam: SimpleNamespace) -> np.ndarray:
    """Supply voltage each environment's servos read on the next step [V], shape ``(num_envs, J)``."""
    rows = bam.actuator.actuator_indices.numpy().reshape(bam.dofs.shape)
    return bam.control.actuator_supply_voltage.numpy()[rows]


"""
Configuration and authoring.
"""


def test_bam_requires_the_newton_backend(monkeypatch, goldens):
    """BAM is rejected on a PhysX simulation, even without the Newton package installed."""
    sim_ctx = SimpleNamespace(
        cfg=SimulationCfg(use_newton_actuators=True), backend_manager=SimpleNamespace(backend_name="physx")
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim_ctx)
    monkeypatch.setitem(sys.modules, "isaaclab_newton", None)
    with pytest.raises(ValueError, match="BAM requires.*Newton backend"):
        define_actuator_properties("/Robot", {"servo": make_cfg(goldens)})


def test_authoring_rejects_incomplete_or_unsupported_settings(goldens, servo_usd):
    """Authored USD values cannot fill a missing setting, and settings the composition cannot run fail early."""
    with pytest.raises(TypeError, match="kp_fw"):
        author_actuator_prims(make_stage(servo_usd), "/Robot", {"servo": make_cfg(goldens, kp_fw=MISSING)})
    cfg = make_cfg(goldens)
    cfg.motor.model = "m3"
    with pytest.raises(ValueError, match="model"):
        author_actuator_prims(make_stage(servo_usd), "/Robot", {"servo": cfg})
    with pytest.raises(ValueError, match="BAM backlash requires a revolute sibling play hinge: /Robot/servo_0_play"):
        author_actuator_prims(
            make_stage(servo_usd), "/Robot", {"servo": make_cfg(goldens, backlash_joint_template="{}_play")}
        )
    for overrides, error in (
        # Newton's clamps act on the motor voltage, so an effort limit would silently clamp volts.
        ({"actuator_effort_limit": 0.5}, "BAM has no effort limit"),
        ({"battery_resistance": -0.1}, "battery resistance must be non-negative"),
        ({"battery_resistance_range": (0.2, 0.1)}, "battery resistance must be non-negative and ordered"),
        # The battery prim carries the scalar values, which Newton rejects on import when out of its limits.
        ({"battery_resistance": -0.1, "battery_resistance_range": (0.0, 0.1)}, "battery resistance must be non-neg"),
        # Newton's battery sags with the supply current, so a torque gain has no equivalent to convert to.
        ({"vin_drop_gain_range": (0.0, 0.2)}, "Set battery_resistance_range"),
        ({"vin_range": (6.0, 8.0), "vin_min": 6.5}, r"vin_min \(6.5\) must not exceed the lowest supply voltage"),
        ({"vin": 6.0, "vin_range": (6.5, 8.2), "vin_min": 6.2}, r"must not exceed the lowest supply voltage \(6.0\)"),
        # Without these checks a reversed lag range would drop the lag and a hold above 1 would freeze it.
        ({"min_delay": 3}, r"max_delay \(0\) must not be below min_delay \(3\)"),
        ({"delay_hold_prob": 1.5}, r"delay_hold_prob must lie in \[0, 1\]"),
        ({"max_delay": 2, "delay_update_period": -1}, "delay_update_period must not be negative"),
    ):
        with pytest.raises(ValueError, match=error):
            author_actuator_prims(make_stage(servo_usd), "/Robot", {"servo": make_cfg(goldens, **overrides)})


@pytest.mark.parametrize(
    "resistance, max_delay, backlash",
    [(0.0, 0, False), (0.05, 3, False), (0.0, 0, True), (0.05, 2, True)],
)
def test_authored_prims_compose_native_components(goldens, servo_usd, resistance, max_delay, backlash):
    """A BAM group authors Newton's BAM components in order, and one battery its servos draw from.

    Friction and lag settings are not authored: Isaac Lab runs them from the configuration.
    """
    stage = make_stage(servo_usd)
    for name in JOINT_NAMES:
        UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/passive_{name}_backlash")
    cfg = make_cfg(
        goldens,
        battery_resistance=resistance,
        vin_min=6.0,
        max_delay=max_delay,
        min_delay=min(max_delay, 1),
        backlash_joint_template="passive_{}_backlash" if backlash else None,
    )
    author_actuator_prims(stage, "/Robot", {"servo": cfg})

    battery = parse_battery_prim(stage.GetPrimAtPath("/Robot/servo_battery"))
    assert battery.battery_class is BatteryConstantResistance
    assert BatteryConstantResistance.resolve_arguments(battery.battery_kwargs) == pytest.approx(
        {"nominal_voltage": VIN, "resistance": resistance, "min_voltage": 6.0}
    )
    expected_components = [ClampingCurrentPWM, MotorDC]
    if max_delay:
        expected_components.append(InputProcessorDelay)
    if backlash:
        expected_components.append(InputProcessorEncoder)
    for name in JOINT_NAMES:
        entry = parse_actuator_prim(stage.GetPrimAtPath(f"/Robot/servo_{name}_actuator"))
        assert entry.drive_class is DriveVoltageControlled
        # kp is the firmware gain itself; the drive multiplies it by the error gain.
        assert DriveVoltageControlled.resolve_arguments(entry.drive_kwargs) == pytest.approx(
            {"kp": KP_FW, "error_gain": cfg.motor.error_gain}
        )
        assert entry.battery_path == "/Robot/servo_battery"
        assert [component for component, _ in entry.component_specs] == expected_components
        components = dict(entry.component_specs)
        assert ClampingCurrentPWM.resolve_arguments(components[ClampingCurrentPWM]) == pytest.approx(
            {"max_current": cfg.motor.max_current, "max_pwm": cfg.motor.max_pwm}
        )
        assert MotorDC.resolve_arguments({**components[MotorDC], "supply_from_battery": True}) == pytest.approx(
            {"kt": cfg.motor.kt, "resistance": cfg.motor.resistance, "vin": VIN, "supply_from_battery": 1.0}
        )
        if max_delay:
            # The buffer is sized for the longest lag; Isaac Lab writes each step's lag.
            assert InputProcessorDelay.resolve_arguments(components[InputProcessorDelay]) == {"delay_steps": max_delay}
        if backlash:
            assert InputProcessorEncoder.resolve_arguments(components[InputProcessorEncoder]) == {
                "output_joint": f"/Robot/passive_{name}_backlash"
            }


def test_configuration_replaces_existing_usd_values(tmp_path, goldens, servo_usd):
    """A serialized asset's fit, friction seed, encoder and battery are replaced, and stale properties are blocked."""
    cfg = make_cfg(goldens)
    stage = make_stage(servo_usd)
    stage.DefinePrim("/Robot/old_battery", "NewtonConstantResistanceBattery")
    seeds = []
    for index, name in enumerate(JOINT_NAMES):
        prim = stage.GetPrimAtPath(f"/Robot/servo_{name}_actuator")
        # The fixture's pre-native actuator attributes stay authored on a weaker layer.
        assert prim.GetAttribute("newton:frictionScale").HasAuthoredValue()
        friction = stage.GetPrimAtPath(f"/Robot/{name}").GetAttribute("newton:friction")
        assert friction.Get() > 0.0
        seeds.append(friction.Get())
        friction.Set(2.0 * friction.Get())
        prim.CreateAttribute("newton:kt", Sdf.ValueTypeNames.Float).Set(0.3 + index * 0.1)
        # An encoder the configuration does not set would keep reading the play hinge.
        prim.AddAppliedSchema("NewtonActuatorEncoderAPI")
        prim.CreateRelationship("newton:outputJoint").SetTargets([f"/Robot/{name}"])
        prim.CreateRelationship("newton:battery").SetTargets(["/Robot/old_battery"])
    path = tmp_path / "robot.usda"
    stage.Export(str(path))
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths.append(str(path))
    cfg.kp_fw = 123.5
    author_actuator_prims(stage, "/Robot", {"servo": cfg})

    parsed = [entry for prim in Usd.PrimRange(stage.GetPrimAtPath("/Robot")) if (entry := parse_actuator_prim(prim))]
    assert len(parsed) == len(JOINT_NAMES)
    for entry in parsed:
        assert entry.drive_class is DriveVoltageControlled
        # Without registered schemas Newton reads every authored attribute, so the stale ones must be blocked.
        assert set(entry.drive_kwargs) == {"kp", "error_gain"}
        assert entry.drive_kwargs["kp"] == pytest.approx(123.5)
        assert entry.battery_path == "/Robot/servo_battery"
        components = dict(entry.component_specs)
        assert list(components) == [ClampingCurrentPWM, MotorDC]
        assert components[MotorDC]["kt"] == pytest.approx(goldens["attr_kt"].item())
    for name, seed in zip(JOINT_NAMES, seeds, strict=True):
        assert stage.GetPrimAtPath(f"/Robot/{name}").GetAttribute("newton:friction").Get() == seed


@pytest.mark.parametrize("device", test_devices())
def test_newton_imports_one_battery_per_environment(goldens, device):
    """Newton builds the authored composition, with one battery per environment powering exactly its servos."""
    cfg = make_cfg(
        goldens, battery_resistance=0.05, vin_min=6.0, max_delay=3, backlash_joint_template="passive_{}_backlash"
    )
    bam = make_model(cfg, num_envs=3, device=device, backlash=True)
    assert isinstance(bam.actuator.drive, DriveVoltageControlled)
    assert [type(clamp) for clamp in bam.actuator.clamping] == [ClampingCurrentPWM]
    assert [type(processor) for processor in bam.actuator.input_processors] == [
        InputProcessorDelay,
        InputProcessorEncoder,
    ]
    np.testing.assert_array_equal(bam.actuator.motor.supply_from_battery.numpy(), 1.0)
    np.testing.assert_array_equal(bam.battery.world.numpy(), [0, 1, 2])
    rows = bam.actuator.actuator_indices.numpy().reshape(bam.dofs.shape)
    starts = bam.battery.actuator_start.numpy()
    for env, env_rows in enumerate(rows):
        np.testing.assert_array_equal(np.sort(bam.battery.actuators.numpy()[starts[env] : starts[env + 1]]), env_rows)
    np.testing.assert_allclose(bam.battery.resistance.numpy(), 0.05)
    np.testing.assert_allclose(bam.battery.min_voltage.numpy(), 6.0)
    np.testing.assert_allclose(supply_voltage(bam), VIN)
    # Each servo's encoder reads its play hinge, the next coordinate of its pair.
    _, encoder = bam.actuator.input_processors
    np.testing.assert_array_equal(encoder.output_joint_pos_indices.numpy(), bam.dofs.ravel() + 1)


"""
Actuator behavior.
"""


@pytest.mark.parametrize("device", test_devices())
def test_motor_and_friction_match_upstream_goldens(goldens, servo_usd, device):
    """Newton's drive, clamp and motor and Isaac Lab's friction budget reproduce upstream motor and m6 outputs.

    Nothing clamps the motor torque, so the applied and the computed effort are both the motor torque.
    """
    cfg = make_cfg(goldens)
    bam = make_actuator(servo_usd, cfg, num_envs=len(goldens["q"]) // 2, device=device)
    friction = add_friction(bam, cfg.motor)
    # The previous applied torque is an independent golden input, not recomputed by the port.
    friction.prev_applied_torque.assign(goldens["prev_tau"].astype(np.float32))
    # Spread the external load over the observables: -bias + constraint - the servo's own friction loss.
    ext_tau = goldens["ext_tau"].astype(np.float32)
    bias = np.float32(-0.25) * ext_tau
    frictionloss = np.full_like(ext_tau, 0.125)
    bam.loads.qfrc_bias.assign(bias)
    bam.loads.qfrc_frictionloss.assign(frictionloss)
    bam.loads.qfrc_constraint.assign(ext_tau + bias + frictionloss)
    effort = step(bam, *(goldens[key].reshape(-1, 2) for key in ("q", "dq", "q_target")))
    np.testing.assert_allclose(effort.ravel(), goldens["motor_torque"], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(bam.adapter.computed_effort_2d.numpy(), effort)
    np.testing.assert_allclose(friction.external_torque.numpy(), ext_tau, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(friction.friction_budget.numpy(), goldens["frictionloss_budget"], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(bam.loads.joint_friction.numpy(), friction.friction_budget.numpy())


@pytest.mark.parametrize("device", test_devices())
def test_sampled_supply_matches_the_recorded_first_step(goldens, device):
    """Per-environment nominal voltages written into the batteries reach the servos on the next step.

    The upstream sag recording runs its first step at each environment's nominal voltage, under both its
    ``vin - sag_gain * sum_j |tau_j|`` law and Newton's battery, whose later, sagged steps Newton tests.
    """
    cfg = make_cfg(goldens, vin_min=float(goldens["sag_vin_min"]))
    bam = make_model(cfg, num_envs=goldens["sag_q"].shape[1], device=device)
    # Isaac Lab writes its start-up samples the same way.
    bam.battery.nominal_voltage.assign(goldens["sag_vin"][:, 0].astype(np.float32))
    bam.battery.reset(bam.control)
    np.testing.assert_allclose(supply_voltage(bam)[:, 0], goldens["sag_effective_vin"][0, :, 0], rtol=1e-6)
    effort = step(bam, goldens["sag_q"][0], goldens["sag_dq"][0], goldens["sag_target"][0])
    np.testing.assert_allclose(effort, goldens["sag_motor_torque"][0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", test_devices())
def test_delay_matches_recorded_mjlab_sequence(goldens, servo_usd, device):
    """Replay mjlab's recorded 3--6-step lags through warm-up, ring wrap, and a partial reset.

    Torch and Warp RNGs differ, so the recorded lags are written into the delay processor; the lag tests below
    cover sampling.
    """
    shape = goldens["delay_commands"].shape[1:]
    bam = make_actuator(servo_usd, make_cfg(goldens, max_delay=int(goldens["delay_max"])), shape[0], device)
    (delay,) = bam.actuator.input_processors
    for index, command in enumerate(goldens["delay_commands"]):
        if goldens["delay_resets"][index].any():
            bam.adapter.reset(torch.from_numpy(np.flatnonzero(goldens["delay_resets"][index])).to(device))
        delay.delay_steps.assign(np.repeat(goldens["delay_lags"][index], shape[1]).astype(np.int32))
        effort = step(bam, np.zeros(shape), np.zeros(shape), command)
        np.testing.assert_allclose(
            effort, goldens["delay_motor_torque"][index], rtol=1e-5, atol=1e-6, err_msg=f"step {index}"
        )


@pytest.mark.parametrize("device", test_devices())
def test_effort_feedback_and_supply_floor(goldens, device):
    """Friction feeds back the applied torque, a reset restores only its environment, and an empty supply brakes."""
    cfg = make_cfg(goldens, battery_resistance=0.5)
    bam = make_model(cfg, num_envs=2, device=device)
    friction = add_friction(bam, cfg.motor)
    pos, vel, zeros = np.array([[0.3, -0.1]] * 2), np.array([[0.5, -0.4]] * 2), np.zeros((2, 2))
    effort = step(bam, pos, vel, zeros)
    budget = friction.friction_budget.numpy().reshape(2, 2).copy()
    np.testing.assert_array_equal(bam.adapter.computed_effort_2d.numpy()[:, :2], effort)
    assert (budget >= cfg.motor.friction_base).all()

    # The next budget reads the previously applied torque.
    step(bam, zeros, zeros, zeros)
    rest_budget = cfg.motor.friction_base + cfg.motor.friction_stribeck
    load_gain = cfg.motor.load_friction_motor + cfg.motor.load_friction_motor_stribeck
    np.testing.assert_allclose(
        friction.friction_budget.numpy().reshape(2, 2), rest_budget + np.abs(effort) * load_gain, rtol=1e-6
    )

    # Reset clears the selected environment's history; the other keeps its sagged supply.
    step(bam, pos, vel, zeros)
    sagged = supply_voltage(bam)
    assert (sagged < VIN).all(), "the servos must load their batteries"
    bam.adapter.reset(torch.tensor([0], device=device))
    np.testing.assert_allclose(supply_voltage(bam)[0], VIN, rtol=1e-6)
    np.testing.assert_array_equal(supply_voltage(bam)[1], sagged[1])
    np.testing.assert_allclose(step(bam, pos, vel, zeros)[0], effort[0], rtol=1e-6)
    np.testing.assert_allclose(friction.friction_budget.numpy().reshape(2, 2)[0], budget[0], rtol=1e-6)

    # A supply sagged to 0 V applies no voltage, so only the back-EMF brake remains.
    bam.battery.resistance.fill_(100.0)
    bam.battery.min_voltage.zero_()
    step(bam, pos, vel, zeros)
    np.testing.assert_array_equal(supply_voltage(bam), 0.0)
    np.testing.assert_allclose(step(bam, pos, vel, zeros), -(cfg.motor.kt**2) * vel / cfg.motor.resistance, rtol=1e-6)


def make_lagged_actuator(goldens, servo_usd, device, num_envs, **lag_settings) -> SimpleNamespace:
    """Build a delayed BAM actuator whose lag is drawn each step and restarted with the adapter's resets."""
    cfg = make_cfg(goldens, **lag_settings)
    bam = make_actuator(servo_usd, cfg, num_envs=num_envs, device=device)
    (delay,) = bam.actuator.input_processors
    bam.lag = BamCommandLag(
        delay,
        slots=np.arange(bam.dofs.size),
        dof_indices=bam.dofs.ravel(),
        dofs_per_env=len(JOINT_NAMES),
        min_delay=cfg.min_delay,
        max_delay=cfg.max_delay,
        hold_prob=cfg.delay_hold_prob,
        update_period=cfg.delay_update_period,
    )
    bam.adapter.register_reset_callback(bam.lag.reset)
    return bam


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_command_lags_differ_per_environment_and_episode(goldens, servo_usd, device):
    """Lags refresh on staggered per-environment phases, change after a reset, and replay through a CUDA graph."""
    num_envs, period = 16, 4
    # The second actuator never resets after start-up, so it continues every untouched environment.
    actuators = [
        make_lagged_actuator(goldens, servo_usd, device, num_envs, min_delay=1, max_delay=3, delay_update_period=period)
        for _ in range(2)
    ]
    zeros = np.zeros((num_envs, 2))
    graphs = []
    for bam in actuators:
        step(bam, zeros, zeros, zeros)  # Compile before CUDA capture.
        bam.adapter.reset(torch.arange(num_envs, device=device))
        with wp.ScopedDevice(device), wp.ScopedCapture() as capture:
            bam.lag.step()
            bam.adapter.step(bam.state, bam.control, DT, swap_state=False)
        graphs.append(capture.graph)
    # Small commands stay in the linear firmware regime at rest, so each torque identifies the delayed command.
    command_step = 0.001
    kt, resistance = goldens["attr_kt"].item(), goldens["attr_R"].item()
    torque_step = command_step * KP_FW * goldens["attr_error_gain"].item() * VIN * kt / resistance

    def rollout(steps: range) -> np.ndarray:
        """Replay a ramped command and return each step's lag, shape ``(steps, actuator, env, joint)``."""
        lags = []
        for index in steps:
            for bam, graph in zip(actuators, graphs, strict=True):
                bam.target_pos.assign(np.full(num_envs * 2, command_step * index, dtype=np.float32))
                with wp.ScopedDevice(device):
                    wp.capture_launch(graph)
            efforts = np.stack([bam.control.joint_f_2d.numpy() for bam in actuators])
            lags.append(index - np.rint(efforts / torque_step).astype(int))
        return np.stack(lags)

    first = rollout(range(24))
    # Step 0 has no history to delay; later lags are drawn or held within [min_delay, max_delay].
    np.testing.assert_array_equal(np.unique(first[1:]), [1, 2, 3])
    np.testing.assert_array_equal(first[..., 0], first[..., 1])
    np.testing.assert_array_equal(first[:, 0], first[:, 1])
    phases = set()
    # Skip warm-up, where the ring clips the lag to the available history.
    for env_lags in first[period:, 0, :, 0].T:
        changed = np.flatnonzero(np.diff(env_lags)) + period + 1
        residues = {int(index) % period for index in changed}
        assert len(residues) <= 1
        phases.update(residues)
    assert len(phases) > 1, "lag refreshes must not synchronize every environment"
    histories = {tuple(env_lags) for env_lags in first[:, 0, :, 0].T}
    assert len(histories) > period, "environments sharing a phase must draw independent lags"

    actuators[0].adapter.reset(torch.tensor([0], device=device))
    second = rollout(range(24, 48))
    assert not np.array_equal(first[:, 0, 0], second[:, 0, 0]), "reset replayed the same delay sequence"
    np.testing.assert_array_equal(second[:, 0, 1:], second[:, 1, 1:])


@pytest.mark.parametrize("device", test_devices())
def test_command_lags_over_other_dofs_draw_independently(goldens, servo_usd, device):
    """Lags of separate groups, here one per joint, draw their own sequence in each environment."""
    num_envs = 8
    bam = make_actuator(servo_usd, make_cfg(goldens, max_delay=6), num_envs=num_envs, device=device)
    (delay,) = bam.actuator.input_processors
    lags = [
        BamCommandLag(delay, bam.dofs[:, joint], bam.dofs[:, joint], len(JOINT_NAMES), min_delay=0, max_delay=6)
        for joint in range(len(JOINT_NAMES))
    ]
    draws = []
    for _ in range(4):
        with wp.ScopedDevice(device):
            for lag in lags:
                lag.step()
        draws.append([lag.lag.numpy() for lag in lags])
    draws = np.array(draws)
    assert not np.array_equal(draws[:, 0], draws[:, 1]), "groups drew the same lag sequence"
