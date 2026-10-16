# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for BAM actuator authoring and its Newton-native composition.

Newton's ``DriveBAM``, ``Battery``, delay processor and effort clamp, plus Isaac Lab's command lag and
gearbox friction, are checked against upstream BAM recordings, stepping the actuators with test-supplied
joint state and external load.
"""

import sys
from dataclasses import MISSING, fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.actuators import BamFriction
from newton.actuators import (
    Actuator,
    ClampingMaxEffort,
    DriveBAM,
    InputProcessorBacklash,
    InputProcessorRandomDelay,
    parse_actuator_prim,
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

BAM_DRIVE_KWARGS = {"kp", "kt", "resistance", "vin", "max_pwm", "max_current", "supply_from_control"}
"""Drive attributes BAM authoring sets on each actuator prim."""


@wp.kernel
def _reference_lag_kernel(seed: int, step: int, min_delay: int, max_delay: int, lags: wp.array[wp.int32]):
    """The pre-native BAM lag draw for update period 0: an episode seed per environment, a draw per step."""
    env = wp.tid()
    episode_seed = wp.int32(wp.rand_init(seed, env))
    lags[env] = wp.randi(wp.rand_init(episode_seed, step), min_delay, max_delay + 1)


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
    return BamActuatorCfg(**{"joint_names_expr": ["servo_.*"], "motor": motor, "vin": VIN, "kp_fw": KP_FW, **overrides})


def make_stage(servo_usd: str, cfg: BamActuatorCfg | None = None) -> Usd.Stage:
    """Open a fresh fixture layer, optionally authoring its actuators from *cfg*.

    The fixture's prims carry a pre-native BAM actuator at the paths authoring reuses.
    """
    stage = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(servo_usd))
    if cfg is not None:
        author_actuator_prims(stage, "/Robot", {"servo": cfg})
    return stage


def make_actuator(
    servo_usd: str,
    cfg: BamActuatorCfg,
    num_envs: int,
    device: str,
    vin: float | np.ndarray = VIN,
    sag_gain: float | np.ndarray = 0.0,
    vin_min: float = 0.0,
) -> SimpleNamespace:
    """Build the fixture's BAM actuator with flat joint-state buffers the test writes into.

    As on the articulation path, each environment's servos draw from one battery, by default at :data:`VIN`
    without sag.
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
    dofs = np.arange(num_envs * len(JOINT_NAMES)).reshape(shape)
    f32 = lambda values: wp.array(np.broadcast_to(values, num_envs).astype(np.float32), device=device)  # noqa: E731
    adapter.add_batteries(dofs, f32(vin), f32(sag_gain), min_voltage=max(vin_min, 0.0))
    return SimpleNamespace(
        adapter=adapter,
        actuator=actuator,
        dofs=dofs,
        state=state,
        control=control,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        target_pos=target_pos,
        friction=None,
        device=device,
    )


def add_friction(bam: SimpleNamespace, motor: BamMotorCfg) -> BamFriction:
    """Attach gearbox friction, reset with the adapter as the articulation path does."""
    dof_indices = wp.array(bam.dofs.ravel(), dtype=wp.uint32, device=bam.device)
    bam.friction = BamFriction(motor, dof_indices, dofs_per_env=len(JOINT_NAMES))
    bam.adapter.register_reset_callback(bam.friction.reset)
    return bam.friction


def step(bam: SimpleNamespace, joint_pos: np.ndarray, joint_vel: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Run one physics step's actuator hooks and return the applied effort [N.m], shape ``(num_envs, J)``."""
    bam.joint_pos.assign(np.ascontiguousarray(joint_pos, dtype=np.float32))
    bam.joint_vel.assign(np.ascontiguousarray(joint_vel, dtype=np.float32))
    bam.target_pos.assign(np.ascontiguousarray(target_pos, dtype=np.float32))
    with wp.ScopedDevice(bam.device):
        if bam.friction is not None:
            bam.friction.compute_budget(bam.state.joint_qd)
        bam.adapter.step(bam.state, bam.control, DT)
        if bam.friction is not None:
            bam.friction.record_applied_effort(bam.control.joint_f)
    return bam.control.joint_f_2d.numpy().copy()


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
    """Authored USD values cannot fill a missing setting, and unsupported models or lag settings fail at authoring."""
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
    # Without these checks a reversed lag range would drop the lag and a hold above 1 would freeze it.
    for overrides, error in (
        ({"min_delay": 3}, r"max_delay \(0\) must not be below min_delay \(3\)"),
        ({"delay_hold_prob": 1.5}, r"delay_hold_prob must lie in \[0, 1\]"),
        ({"max_delay": 2, "delay_update_period": -1}, "delay_update_period must not be negative"),
    ):
        with pytest.raises(ValueError, match=error):
            author_actuator_prims(make_stage(servo_usd), "/Robot", {"servo": make_cfg(goldens, **overrides)})


@pytest.mark.parametrize(
    "limit, voltage_range, max_delay, backlash",
    [
        (None, None, 0, None),
        (0.05, (6.5, 8.2), 3, None),
        (None, (6.5, 8.2), 0, "position"),
        (0.05, None, 2, "velocity"),
    ],
)
def test_authored_prim_composes_native_components(goldens, servo_usd, limit, voltage_range, max_delay, backlash):
    """A BAM group authors Newton's BAM drive, effort clamp, delay buffer and backlash processor.

    Friction, sag and lag settings are not authored: Isaac Lab runs them from the configuration.
    """
    stage = make_stage(servo_usd)
    for name in JOINT_NAMES:
        UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/passive_{name}_backlash")
    cfg = make_cfg(
        goldens,
        actuator_effort_limit=limit,
        vin_range=voltage_range,
        max_delay=max_delay,
        min_delay=min(max_delay, 1),
        backlash_joint_template=None if backlash is None else "passive_{}_backlash",
        backlash_velocity_feedback=backlash == "velocity",
    )
    author_actuator_prims(stage, "/Robot", {"servo": cfg})

    voltage = max(voltage_range) if voltage_range is not None else VIN
    expected_limit = limit if limit is not None else voltage * cfg.motor.kt / cfg.motor.resistance
    expected_components = [ClampingMaxEffort]
    if max_delay:
        expected_components.append(InputProcessorRandomDelay)
    if backlash is not None:
        expected_components.append(InputProcessorBacklash)
    for name in JOINT_NAMES:
        prim = stage.GetPrimAtPath(f"/Robot/servo_{name}_actuator")
        entry = parse_actuator_prim(prim)
        assert entry.drive_class is DriveBAM
        assert set(entry.drive_kwargs) == BAM_DRIVE_KWARGS
        resolved = DriveBAM.resolve_arguments(dict(entry.drive_kwargs))
        assert resolved["supply_from_control"] is True
        assert resolved["kp"] == pytest.approx(KP_FW * cfg.motor.error_gain)
        assert resolved["kt"] == pytest.approx(cfg.motor.kt)
        assert resolved["vin"] == pytest.approx(VIN)
        assert resolved["max_current"] == pytest.approx(cfg.motor.max_current)
        assert [component for component, _ in entry.component_specs] == expected_components
        components = {component: kwargs for component, kwargs in entry.component_specs}
        assert components[ClampingMaxEffort]["max_effort"] == pytest.approx(expected_limit)
        if max_delay:
            assert InputProcessorRandomDelay.resolve_arguments(components[InputProcessorRandomDelay]) == {
                "max_delay": max_delay,
                "min_delay": 1,
                "hold_probability": 0.0,
                "update_period": 0,
                "seed": 0,
            }
        if backlash is not None:
            resolved = InputProcessorBacklash.resolve_arguments(components[InputProcessorBacklash])
            assert resolved == {
                "backlash_joint": f"/Robot/passive_{name}_backlash",
                "velocity_feedback": backlash == "velocity",
            }


def test_configuration_replaces_existing_usd_values(tmp_path, goldens, servo_usd):
    """A serialized asset's fit, friction seed and backlash are replaced, and stale actuator properties are blocked."""
    cfg = make_cfg(goldens)
    stage = make_stage(servo_usd)
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
        # A backlash processor the configuration does not set would keep reading the play hinge.
        prim.AddAppliedSchema("NewtonActuatorBacklashAPI")
        prim.CreateRelationship("newton:backlashJoint").SetTargets([f"/Robot/{name}"])
    path = tmp_path / "robot.usda"
    stage.Export(str(path))
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths.append(str(path))
    cfg.kp_fw = 123.5
    author_actuator_prims(stage, "/Robot", {"servo": cfg})

    parsed = [entry for prim in Usd.PrimRange(stage.GetPrimAtPath("/Robot")) if (entry := parse_actuator_prim(prim))]
    assert len(parsed) == len(JOINT_NAMES)
    for entry in parsed:
        assert entry.drive_class is DriveBAM
        # Without a registered schema Newton reads every authored attribute, so the stale ones must be blocked.
        assert set(entry.drive_kwargs) == BAM_DRIVE_KWARGS
        assert entry.drive_kwargs["kt"] == pytest.approx(goldens["attr_kt"].item())
        assert entry.drive_kwargs["kp"] == pytest.approx(123.5 * goldens["attr_error_gain"].item())
        assert [component for component, _ in entry.component_specs] == [ClampingMaxEffort]
    for name, seed in zip(JOINT_NAMES, seeds, strict=True):
        assert stage.GetPrimAtPath(f"/Robot/{name}").GetAttribute("newton:friction").Get() == seed


"""
Actuator behavior.
"""


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_motor_and_friction_match_upstream_goldens(goldens, servo_usd, device):
    """The authored drive and clamp and Isaac Lab's friction budget reproduce upstream motor and m6 outputs."""
    cfg = make_cfg(goldens)
    bam = make_actuator(servo_usd, cfg, num_envs=len(goldens["q"]) // 2, device=device)
    friction = add_friction(bam, cfg.motor)
    # The previous applied torque is an independent golden input, not recomputed by the port.
    friction.prev_applied_torque.assign(goldens["prev_tau"].astype(np.float32))
    friction.external_torque.assign(goldens["ext_tau"].astype(np.float32))
    effort = step(bam, *(goldens[key].reshape(-1, 2) for key in ("q", "dq", "q_target")))
    np.testing.assert_allclose(
        bam.adapter.computed_effort_2d.numpy().ravel(), goldens["motor_torque"], rtol=1e-5, atol=1e-6
    )
    limit = VIN * cfg.motor.kt / cfg.motor.resistance
    np.testing.assert_allclose(effort.ravel(), np.clip(goldens["motor_torque"], -limit, limit), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(friction.friction_budget.numpy(), goldens["frictionloss_budget"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_battery_sag_matches_upstream_supply_recording(goldens, servo_usd, device):
    """Voltage and raw torque match a recording of upstream BAM's stateful compute method."""
    shape = goldens["sag_q"].shape[1:]
    bam = make_actuator(
        servo_usd,
        make_cfg(goldens),
        num_envs=shape[0],
        device=device,
        vin=goldens["sag_vin"][:, 0],
        sag_gain=goldens["sag_gain"][:, 0],
        vin_min=float(goldens["sag_vin_min"]),
    )
    for index, (pos, vel, target) in enumerate(zip(goldens["sag_q"], goldens["sag_dq"], goldens["sag_target"])):
        step(bam, pos, vel, target)
        np.testing.assert_allclose(
            bam.adapter.battery.voltage.numpy(), goldens["sag_effective_vin"][index, :, 0], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            bam.adapter.computed_effort_2d.numpy(), goldens["sag_motor_torque"][index], rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_delay_matches_recorded_mjlab_sequence(goldens, servo_usd, device):
    """Replay mjlab's recorded 3--6-step lags through warm-up, ring wrap, and a partial reset.

    Torch and Warp RNGs differ, so each step's lag range is pinned to the recorded lag, which every draw then
    returns; the lag tests below cover sampling.
    """
    shape = goldens["delay_commands"].shape[1:]
    bam = make_actuator(servo_usd, make_cfg(goldens, max_delay=int(goldens["delay_max"])), shape[0], device)
    (delay,) = bam.actuator.input_processors
    for index, command in enumerate(goldens["delay_commands"]):
        if goldens["delay_resets"][index].any():
            bam.adapter.reset(torch.from_numpy(np.flatnonzero(goldens["delay_resets"][index])).to(device))
        lags = np.repeat(goldens["delay_lags"][index], shape[1]).astype(np.int32)
        delay.min_delay.assign(lags)
        delay.max_delay.assign(lags)
        effort = step(bam, np.zeros(shape), np.zeros(shape), command)
        np.testing.assert_allclose(
            effort, goldens["delay_motor_torque"][index], rtol=1e-5, atol=1e-6, err_msg=f"step {index}"
        )


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("velocity_feedback", [False, True])
def test_backlash_feeds_back_the_play_angle(goldens, device, velocity_feedback):
    """The encoder sees the output angle; back-EMF uses the rotor velocity unless the play velocity is fed back."""
    kt, resistance, error_gain = (goldens[f"attr_{name}"].item() for name in ("kt", "R", "error_gain"))
    full = lambda value: wp.full(4, value, dtype=wp.float32, device=device)  # noqa: E731
    actuator = Actuator(
        indices=wp.array([0, 1, 2, 3], dtype=wp.uint32, device=device),
        drive=DriveBAM(
            kp=full(KP_FW * error_gain), kt=full(kt), resistance=full(resistance), vin=full(VIN), max_pwm=full(1.0)
        ),
        input_processors=[
            InputProcessorBacklash(
                wp.array([1, 0, 3, 2], dtype=wp.uint32, device=device), velocity_feedback=velocity_feedback
            )
        ],
    )
    positions = np.array([[0.01, -0.02], [0.03, -0.01]], dtype=np.float32)
    velocities = np.array([[0.3, -0.4], [-0.5, 0.2]], dtype=np.float32)
    control = PhysxActuatorWrapper.create(2, 2, device)
    control.joint_target_q = wp.zeros(4, dtype=wp.float32, device=device)
    control.joint_target_qd = wp.zeros(4, dtype=wp.float32, device=device)
    state = SimpleNamespace(
        joint_q=wp.array(positions.ravel(), device=device), joint_qd=wp.array(velocities.ravel(), device=device)
    )
    actuator.step(state, control, dt=DT)

    error = -positions - positions[:, ::-1]
    motor_velocity = velocities + (velocities[:, ::-1] if velocity_feedback else 0.0)
    expected = kt / resistance * (VIN * KP_FW * error_gain * error - kt * motor_velocity)
    np.testing.assert_allclose(control.joint_f_2d.numpy(), expected, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_effort_feedback_and_supply_floor(goldens, servo_usd, device):
    """Friction feeds back the clamped effort and sag the raw torque until a reset; an exhausted supply only brakes."""
    cfg = make_cfg(goldens, actuator_effort_limit=0.05)
    bam = make_actuator(servo_usd, cfg, num_envs=2, device=device, sag_gain=0.1, vin_min=-np.inf)
    friction = add_friction(bam, cfg.motor)
    pos, vel, zeros = np.array([[0.3, -0.1]] * 2), np.array([[0.5, -0.4]] * 2), np.zeros((2, 2))
    effort = step(bam, pos, vel, zeros)
    motor = bam.adapter.computed_effort_2d.numpy().copy()
    budget = friction.friction_budget.numpy().reshape(2, 2).copy()
    assert np.abs(motor).max() > 0.05, "the configured effort limit must bind"
    np.testing.assert_allclose(effort, np.clip(motor, -0.05, 0.05), atol=0.0, rtol=0.0)
    assert (budget >= cfg.motor.friction_base).all()

    # Friction uses the previously applied effort even if the limit changes; sag uses the raw torque.
    bam.actuator.clamping[0].max_effort.fill_(0.02)
    step(bam, zeros, zeros, zeros)
    rest_budget = cfg.motor.friction_base + cfg.motor.friction_stribeck
    load_gain = cfg.motor.load_friction_motor + cfg.motor.load_friction_motor_stribeck
    np.testing.assert_allclose(
        friction.friction_budget.numpy().reshape(2, 2), rest_budget + np.abs(effort) * load_gain, rtol=1e-6
    )
    np.testing.assert_allclose(bam.adapter.battery.voltage.numpy(), VIN - 0.1 * np.abs(motor).sum(axis=1), rtol=1e-6)

    # Reset clears the selected environment's history; the other keeps its sagged supply.
    step(bam, pos, vel, zeros)
    bam.adapter.reset(torch.tensor([0], device=device))
    step(bam, pos, vel, zeros)
    voltage = bam.adapter.battery.voltage.numpy()
    np.testing.assert_allclose(voltage[0], VIN, rtol=1e-6)
    assert voltage[1] < VIN, "the untouched environment must keep its history"
    np.testing.assert_allclose(bam.adapter.computed_effort_2d.numpy()[0], motor[0], rtol=1e-6)
    np.testing.assert_allclose(friction.friction_budget.numpy().reshape(2, 2)[0], budget[0], rtol=1e-6)

    # A supply sagged below zero applies no voltage, so only the back-EMF brake remains.
    bam.adapter.battery.sag_gain.fill_(100.0)
    step(bam, pos, vel, zeros)
    np.testing.assert_array_equal(bam.adapter.battery.voltage.numpy(), 0.0)
    np.testing.assert_allclose(
        bam.adapter.computed_effort_2d.numpy(), -(cfg.motor.kt**2) * vel / cfg.motor.resistance, rtol=1e-6
    )


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_command_lags_differ_per_environment_and_episode(goldens, servo_usd, device):
    """Lags refresh on staggered per-environment phases, change after a reset, and replay through a CUDA graph."""
    num_envs, period = 16, 4
    # The second actuator never resets after start-up, so it continues every untouched environment.
    cfg = make_cfg(goldens, min_delay=1, max_delay=3, delay_update_period=period)
    actuators = [make_actuator(servo_usd, cfg, num_envs, device) for _ in range(2)]
    zeros = np.zeros((num_envs, 2))
    graphs = []
    for bam in actuators:
        step(bam, zeros, zeros, zeros)  # Compile before CUDA capture.
        bam.adapter.reset(torch.arange(num_envs, device=device))
        with wp.ScopedDevice(device), wp.ScopedCapture() as capture:
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
                bam.target_pos.assign(np.full((num_envs, 2), command_step * index, dtype=np.float32))
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


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_command_lag_stream_keeps_its_seeding(goldens, servo_usd, device):
    """Each environment's lag stream is seeded from the base seed plus the reset count, as before the native port.

    Pins the draw sequence, so trajectories recorded with the pre-native BAM drive replay unchanged.
    """
    num_envs = 4
    bam = make_actuator(servo_usd, make_cfg(goldens, min_delay=3, max_delay=6), num_envs, device)
    (delay,) = bam.actuator.input_processors
    zeros = np.zeros((num_envs, len(JOINT_NAMES)))
    expected = wp.zeros(num_envs, dtype=wp.int32, device=device)
    for reset_count in range(3):
        if reset_count:
            bam.adapter.reset(None if reset_count == 1 else torch.arange(num_envs, device=device))
        for index in range(8):
            step(bam, zeros, zeros, zeros)
            with wp.ScopedDevice(device):
                wp.launch(_reference_lag_kernel, dim=num_envs, inputs=[reset_count, index, 3, 6, expected])
            np.testing.assert_array_equal(
                delay.delay_steps.numpy().reshape(num_envs, -1),
                np.repeat(expected.numpy()[:, None], len(JOINT_NAMES), axis=1),
                err_msg=f"episode {reset_count}, step {index}",
            )
