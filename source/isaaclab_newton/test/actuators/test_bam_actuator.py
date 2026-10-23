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
from isaaclab_newton.actuators import DriveBam
from newton.actuators import parse_actuator_prim

from pxr import Sdf, Usd

from isaaclab.actuators import BamActuatorCfg, BamMotorCfg
from isaaclab.actuators.actuator_bam_cfg import BAM_DRIVE_API
from isaaclab.actuators.newton import NewtonActuatorAdapter, PhysxActuatorWrapper
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.schemas.schemas_actuators import author_actuator_prims, define_actuator_properties
from isaaclab.test.utils import DeviceScope, test_devices
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
    # The adapter creates drive states, so rebuild it once the stride is set, as the articulation path does.
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
    """Authored USD values cannot fill a missing setting, and an unsupported model would author another friction law."""
    with pytest.raises(TypeError, match="kp_fw"):
        author_actuator_prims(make_stage(servo_usd), "/Robot", {"servo": make_cfg(goldens, kp_fw=MISSING)})
    cfg = make_cfg(goldens)
    cfg.motor.model = "m3"
    with pytest.raises(ValueError, match="model"):
        author_actuator_prims(make_stage(servo_usd), "/Robot", {"servo": cfg})


@pytest.mark.parametrize(
    "model, flags, limit, voltage_range",
    [
        ("m1", (0, 0, 0), None, None),
        ("m2", (1, 0, 0), None, (6.5, 8.2)),
        ("m5", (1, 1, 0), 0.05, (6.5, 8.2)),
        ("m6", (1, 1, 1), 0.05, None),
    ],
)
def test_authored_prim_resolves_to_the_bam_drive(goldens, servo_usd, model, flags, limit, voltage_range):
    """Authoring a BAM group produces a ``NewtonBamDriveAPI``-only prim carrying the configuration.

    The effort limit is a drive parameter: a registered clamping schema would hide the drive from Newton.
    """
    cfg = make_cfg(
        goldens,
        actuator_effort_limit=limit,
        vin_range=voltage_range,
        vin_min=6.0,
        min_delay=1,
        max_delay=3,
        delay_hold_prob=0.25,
        delay_update_period=4,
    )
    cfg.motor.model = model
    stage = make_stage(servo_usd, cfg)

    voltage = max(voltage_range) if voltage_range is not None else VIN
    expected_limit = limit if limit is not None else voltage * cfg.motor.kt / cfg.motor.resistance
    parsed = {
        prim.GetPath(): entry
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/Robot"))
        if (entry := parse_actuator_prim(prim))
    }
    assert len(parsed) == len(JOINT_NAMES)
    for path, entry in parsed.items():
        assert entry.drive_class is DriveBam
        assert entry.component_specs == [], "the BAM delay and clamp are drive-internal, not components"
        # The unregistered token is dropped from the composed schemas, so read the authored opinion.
        spec = stage.GetRootLayer().GetPrimAtPath(path)
        assert list(spec.GetInfo("apiSchemas").GetAppliedItems()) == [BAM_DRIVE_API]
        resolved = DriveBam.resolve_arguments(dict(entry.drive_kwargs))
        assert resolved["max_effort"] == pytest.approx(expected_limit)
        assert resolved["kp_fw"] == pytest.approx(KP_FW)
        assert resolved["vin"] == pytest.approx(VIN)
        assert resolved["vin_min"] == pytest.approx(6.0)
        assert resolved["kt"] == pytest.approx(goldens["attr_kt"].item())
        assert resolved["load_friction_external_quad"] == pytest.approx(goldens["attr_load_friction_external_quad"])
        assert (resolved["min_delay"], resolved["max_delay"]) == (1, 3)
        assert resolved["delay_hold_prob"] == pytest.approx(0.25)
        assert resolved["delay_update_period"] == 4
        assert (resolved["stribeck"], resolved["load_dependent"], resolved["quadratic"]) == flags


def test_configuration_replaces_existing_usd_values(tmp_path, goldens, servo_usd):
    """A serialized asset's fit and joint friction seed are replaced, including runtime parameter values."""
    # An int literal must not leave an int-typed attribute that truncates the replacement value.
    cfg = make_cfg(goldens, kp_fw=200)
    # Constraint allocation needs a positive seed even for a fit with zero Coulomb friction.
    cfg.motor.friction_base = 0.0
    stage = make_stage(servo_usd)
    for name in JOINT_NAMES:
        stage.RemovePrim(f"/Robot/servo_{name}_actuator")
    author_actuator_prims(stage, "/Robot", {"servo": cfg})
    seeds = []
    for index, name in enumerate(JOINT_NAMES):
        prim = stage.GetPrimAtPath(f"/Robot/servo_{name}_actuator")
        prim.GetAttribute("newton:kt").Set(0.3 + index * 0.1)
        prim.GetAttribute("newton:frictionScale").Set(2.0)
        friction = stage.GetPrimAtPath(f"/Robot/{name}").GetAttribute("newton:friction")
        assert friction.Get() > 0.0
        seeds.append(friction.Get())
        friction.Set(2.0 * friction.Get())
    path = tmp_path / "robot.usda"
    stage.Export(str(path))
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths.append(str(path))
    cfg.kp_fw = 123.5
    cfg.motor.friction_base = 0.012
    author_actuator_prims(stage, "/Robot", {"servo": cfg})

    parsed = [entry for prim in Usd.PrimRange(stage.GetPrimAtPath("/Robot")) if (entry := parse_actuator_prim(prim))]
    assert len(parsed) == len(JOINT_NAMES)
    for entry in parsed:
        resolved = DriveBam.resolve_arguments(dict(entry.drive_kwargs))
        assert resolved["kt"] == pytest.approx(goldens["attr_kt"].item())
        assert resolved["kp_fw"] == pytest.approx(123.5)
        assert resolved["friction_base"] == pytest.approx(0.012)
        assert resolved["friction_scale"] == 1.0
    for name, seed in zip(JOINT_NAMES, seeds, strict=True):
        assert stage.GetPrimAtPath(f"/Robot/{name}").GetAttribute("newton:friction").Get() == seed


"""
Drive behavior.
"""


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
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
def test_delay_matches_recorded_mjlab_sequence(goldens, servo_usd, device):
    """Replay mjlab's recorded 3--6-step lags through warm-up, ring wrap, and a partial reset.

    Torch and Warp RNGs differ, so the recorded lags are injected; the RNG test below covers sampling.
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


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_effort_feedback_and_supply_floor(goldens, servo_usd, device):
    """Friction feeds back the applied effort and sag the raw torque until a reset; an exhausted supply only brakes."""
    bam = make_actuator(servo_usd, num_envs=2, device=device, max_effort=0.05, sag_gain=0.1, vin_min=-np.inf)
    pos, vel, zeros = np.array([[0.3, -0.1]] * 2), np.array([[0.5, -0.4]] * 2), np.zeros((2, 2))
    effort = step(bam, pos, vel, zeros)
    motor = bam.drive.motor_torque.numpy().reshape(2, 2)
    budget = bam.drive.friction_budget.numpy().reshape(2, 2)
    assert np.abs(motor).max() > 0.05, "the configured effort limit must bind"
    np.testing.assert_allclose(effort, np.clip(motor, -0.05, 0.05), atol=0.0, rtol=0.0)
    friction_base = goldens["attr_friction_base"].item()
    assert (budget >= friction_base).all()

    # Friction uses the previously applied effort even if the limit changes; sag uses the raw torque.
    bam.drive.max_effort.fill_(0.02)
    step(bam, zeros, zeros, zeros)
    rest_budget = friction_base + goldens["attr_friction_stribeck"].item()
    load_gain = goldens["attr_load_friction_motor"].item() + goldens["attr_load_friction_motor_stribeck"].item()
    np.testing.assert_allclose(
        bam.drive.friction_budget.numpy().reshape(2, 2), rest_budget + np.abs(effort) * load_gain, rtol=1e-6
    )
    sagged_vin = VIN - 0.1 * np.abs(motor).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        bam.drive.effective_vin.numpy().reshape(2, 2), np.broadcast_to(sagged_vin, (2, 2)), rtol=1e-6
    )

    # Reset clears the selected environment's history; the other keeps its sagged supply.
    step(bam, pos, vel, zeros)
    bam.adapter.reset(torch.tensor([0], device=device))
    step(bam, pos, vel, zeros)
    vin = bam.drive.effective_vin.numpy().reshape(2, 2)
    np.testing.assert_allclose(vin[0], VIN, rtol=1e-6)
    assert (vin[1] < VIN).all(), "the untouched environment must keep its history"
    np.testing.assert_allclose(bam.drive.motor_torque.numpy().reshape(2, 2)[0], motor[0], rtol=1e-6)
    np.testing.assert_allclose(bam.drive.friction_budget.numpy().reshape(2, 2)[0], budget[0], rtol=1e-6)

    # A supply sagged below zero applies no voltage, so only the back-EMF brake remains.
    bam.drive.sag_gain.fill_(100.0)
    step(bam, pos, vel, zeros)
    np.testing.assert_array_equal(bam.drive.effective_vin.numpy(), 0.0)
    kt, resistance = bam.drive.kt.numpy(), bam.drive.resistance.numpy()
    np.testing.assert_allclose(bam.drive.motor_torque.numpy(), -(kt**2) * vel.ravel() / resistance, rtol=1e-6)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_delay_lags_differ_per_environment_and_episode(goldens, servo_usd, device):
    """Lags refresh on staggered per-environment phases, change after a reset, and replay through a CUDA graph."""
    num_envs, period = 16, 4
    # The second actuator never resets after start-up, so it continues every untouched environment.
    actuators = [
        make_actuator(servo_usd, num_envs=num_envs, device=device, min_delay=1, max_delay=3, delay_update_period=period)
        for _ in range(2)
    ]
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
