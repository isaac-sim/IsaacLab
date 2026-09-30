# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton-native BAM actuator component.

The suite drives the real construction path -- author ``NewtonActuator`` prims from a
:class:`~isaaclab.actuators.BamActuatorCfg`, parse them back with
:meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.from_usd`, and step the resulting
:class:`~newton.actuators.Actuator` -- so a break anywhere between the config and the Warp
kernels shows up here. No simulator is involved: the joint state is supplied by the test.

Motor and friction outputs are checked against upstream BAM golden data. The harness
supplies the solver's external load and reads the motor torque and published friction budget.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from newton.actuators import parse_actuator_prim

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.actuators import BamActuatorCfg, IdealPDActuatorCfg
from isaaclab.actuators.newton import (
    BAM_CONTROL_API,
    ControllerBam,
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

pytestmark = pytest.mark.unit

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
    kwargs = {"joint_names_expr": [".*"], "vin": VIN, "kp_fw": KP_FW}
    kwargs.update(overrides)
    return BamActuatorCfg(**kwargs)


def _make_stage(cfg: BamActuatorCfg | dict[str, BamActuatorCfg], joint_names: list[str] = JOINT_NAMES) -> Usd.Stage:
    """Author an articulation over *joint_names*, driven by the given BAM actuator group(s).

    A group covers whichever of the joints its ``joint_names_expr`` selects, so a fixture can
    carry joints no actuator drives -- which is what a play hinge is. A mapping authors several
    groups at once, under the names it is keyed by.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Robot")
    for index, name in enumerate(joint_names):
        body = UsdGeom.Xform.Define(stage, f"/World/Robot/body_{index}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/World/Robot/{name}")
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        prim = stage.DefinePrim(f"/World/Robot/asset_{name}_actuator", "NewtonActuator")
        prim.GetReferences().AddReference(str(Path(__file__).parent / "data" / "bam_xl330_m6.usda"), "/BamActuator")
        prim.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
    author_actuator_prims(stage, "/World/Robot", cfg if isinstance(cfg, dict) else {"servo": cfg})
    return stage


def _make_adapter(
    cfg: BamActuatorCfg, num_envs: int, device: str, joint_names: list[str] = JOINT_NAMES
) -> NewtonActuatorAdapter:
    """Author, parse and build the Newton actuator adapter for the fixture."""
    return NewtonActuatorAdapter.from_usd(
        stage=_make_stage(cfg, joint_names),
        joint_names=joint_names,
        num_envs=num_envs,
        num_joints=len(joint_names),
        device=device,
        articulation_prim_path="/World/Robot",
    )


class _Harness:
    """Steps one Newton actuator over test-supplied joint state.

    Wraps the flat ``sim_state`` / ``sim_control`` pair
    :meth:`~newton.actuators.Actuator.step` expects, so a test can drive the actuator with an
    arbitrary trajectory and read back the effort it asks the solver to apply.
    """

    def __init__(self, cfg: BamActuatorCfg, num_envs: int, device: str, joint_names: list[str] = JOINT_NAMES):
        self.adapter = _make_adapter(cfg, num_envs, device, joint_names)
        assert len(self.adapter.actuators) == 1, "the fixture's joints must merge into one actuator"
        self.actuator = self.adapter.actuators[0]
        self.controller: ControllerBam = self.actuator.controller
        self.controller.external_torque = wp.zeros(len(self.controller.motor_torque), dtype=wp.float32, device=device)
        self.num_envs = num_envs
        self.device = device
        self.joint_names = joint_names
        shape = (num_envs, len(joint_names))
        self.state = PhysxActuatorWrapper.create(*shape, device)
        self.control = PhysxActuatorWrapper.create(*shape, device)
        self.joint_pos = wp.zeros(shape, dtype=wp.float32, device=device)
        self.joint_vel = wp.zeros(shape, dtype=wp.float32, device=device)
        self.target_pos = wp.zeros(shape, dtype=wp.float32, device=device)
        self.state.joint_q = self.joint_pos.reshape(-1)
        self.state.joint_qd = self.joint_vel.reshape(-1)
        self.control.joint_target_pos = self.target_pos.reshape(-1)
        self.control.joint_target_vel = wp.zeros(num_envs * len(joint_names), dtype=wp.float32, device=device)
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


def test_bam_cfg_is_rejected_on_a_host_adapter_backend():
    """A backend without an in-solver actuator path must refuse the BAM config, loudly.

    The model is written in terms of solver quantities: it publishes its friction budget into
    the solver's joint dry friction and reads the external load back out of the solver's
    generalized forces. A backend that steps native actuators through the shared host adapter
    (PhysX, OVPhysX) provides neither, so the controller would silently fall back to
    a different friction model and skip its solver bindings. Failing the gate names the fix.
    """
    with pytest.raises(ValueError, match="requires the Newton backend"):
        validate_newton_native_actuator_cfgs({"servo": _make_cfg()}, host_adapter=True)

    # The restriction is BAM's alone -- every other supported config still runs there, so the
    # flag cannot be passing by rejecting the whole native path.
    validate_newton_native_actuator_cfgs({"legs": IdealPDActuatorCfg(joint_names_expr=[".*"])}, host_adapter=True)


def test_authored_prim_resolves_to_the_bam_controller():
    """Authoring a BAM group must produce a parseable ``NewtonBamControlAPI`` actuator prim."""
    cfg = _make_cfg(vin_min=6.0, min_delay=1, max_delay=3, delay_hold_prob=0.25, delay_update_period=4)
    stage = _make_stage(cfg)

    parsed = [p for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot")) if (p := parse_actuator_prim(prim))]
    assert len(parsed) == len(JOINT_NAMES)
    for entry in parsed:
        assert entry.controller_class is ControllerBam
        assert entry.component_specs == [], "the BAM delay is controller-internal, not a Delay component"
        resolved = ControllerBam.resolve_arguments(dict(entry.controller_kwargs))
        params = _reference_params()
        # Deployment settings come from the config, identified constants from the USD.
        assert resolved["kp_fw"] == pytest.approx(KP_FW)
        assert resolved["vin"] == pytest.approx(VIN)
        assert resolved["vin_min"] == pytest.approx(6.0)
        assert resolved["kt"] == pytest.approx(params.kt)
        assert resolved["load_friction_external_quad"] == pytest.approx(params.load_friction_external_quad)
        assert (resolved["min_delay"], resolved["max_delay"]) == (1, 3)
        assert resolved["delay_hold_prob"] == pytest.approx(0.25)
        assert resolved["delay_update_period"] == 4
        assert (resolved["stribeck"], resolved["load_dependent"], resolved["quadratic"]) == (1, 1, 1)

    # The controller schema token is applied on the prim, not just implied by the parse.
    # ``NewtonBamControlAPI`` has no registered USD schema definition, so the composed
    # ``GetAppliedSchemas`` filters it out; read the authored opinion instead.
    spec = stage.GetRootLayer().GetPrimAtPath(f"/World/Robot/servo_{JOINT_NAMES[0]}_actuator")
    assert BAM_CONTROL_API in spec.GetInfo("apiSchemas").GetAppliedItems()


@pytest.mark.parametrize("reauthor", [False, True])
def test_usd_coefficients_are_self_contained_and_preserved(reauthor, tmp_path):
    """USD coefficients survive serialization and config authoring without a JSON sidecar."""
    stage = _make_stage(_make_cfg(kp_fw=None, vin=None))
    for index, name in enumerate(JOINT_NAMES):
        prim = stage.GetPrimAtPath(f"/World/Robot/servo_{name}_actuator")
        assert not prim.HasAttribute("newton:paramsFile")
        prim.GetAttribute("newton:kt").Set(0.3 + index * 0.1)
    path = tmp_path / "robot.usda"
    stage.Export(str(path))
    stage = Usd.Stage.Open(str(path))
    if reauthor:
        # A previous draft's JSON pointer and another controller's attributes must not
        # survive replacement, including opinions from a referenced layer.
        layer_path = tmp_path / "legacy.usda"
        stage.GetPrimAtPath(f"/World/Robot/servo_{JOINT_NAMES[0]}_actuator").CreateAttribute(
            "newton:paramsFile", Sdf.ValueTypeNames.Asset
        ).Set(Sdf.AssetPath("missing.json"))
        stage.GetPrimAtPath(f"/World/Robot/servo_{JOINT_NAMES[0]}_actuator").CreateAttribute(
            "newton:kp", Sdf.ValueTypeNames.Float
        ).Set(10.0)
        stage.GetPrimAtPath(f"/World/Robot/servo_{JOINT_NAMES[0]}_actuator").SetMetadata(
            "apiSchemas", Sdf.TokenListOp.CreateExplicit([BAM_CONTROL_API, "NewtonPDControlAPI"])
        )
        stage.Export(str(layer_path))
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().subLayerPaths.append(str(layer_path))
        author_actuator_prims(
            stage, "/World/Robot", {"servo": _make_cfg(kp_fw=123.0, parameter_overrides={"friction_base": 0.012})}
        )
    parsed = [
        entry for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot")) if (entry := parse_actuator_prim(prim))
    ]
    assert len(parsed) == 2
    for index, entry in enumerate(parsed):
        assert entry.controller_class is ControllerBam
        resolved = ControllerBam.resolve_arguments(dict(entry.controller_kwargs))
        assert resolved["kt"] == pytest.approx(0.3 + index * 0.1)
        assert resolved["kp_fw"] == pytest.approx(123.0 if reauthor else 400.0)
        assert resolved["friction_base"] == pytest.approx(0.012 if reauthor else _reference_params().friction_base)


def test_effort_limit_is_authored_on_the_controller_not_as_a_clamping_component():
    """A BAM actuator prim must carry no USD-registered API schema beside the BAM token.

    Newton resolves an actuator prim's components from ``Usd.Prim.GetAppliedSchemas``, falling
    back to the raw ``apiSchemas`` metadata *only when that comes back empty*.
    ``NewtonBamControlAPI`` has no registered schema definition, so USD drops it from the
    composed list; a registered sibling such as ``NewtonMaxEffortClampingAPI`` would make the
    composed list non-empty and the BAM controller would vanish from the parse. The effort
    limit is therefore a controller parameter, and this test is what stops it going back.
    """
    cfg = _make_cfg(actuator_effort_limit=0.05)
    stage = _make_stage(cfg)

    prim = stage.GetPrimAtPath(f"/World/Robot/servo_{JOINT_NAMES[0]}_actuator")
    parsed = parse_actuator_prim(prim)
    assert parsed is not None and parsed.controller_class is ControllerBam
    assert parsed.component_specs == [], "a BAM prim must compose no clamping or delay component"
    assert ControllerBam.resolve_arguments(dict(parsed.controller_kwargs))["max_effort"] == pytest.approx(0.05)

    spec = stage.GetRootLayer().GetPrimAtPath(prim.GetPath())
    assert list(spec.GetInfo("apiSchemas").GetAppliedItems()) == [BAM_CONTROL_API]


def test_driven_joints_are_seeded_with_a_positive_friction():
    """MuJoCo only builds a DOF-friction row for joints whose frictionloss is positive.

    The row has to exist from the first solve -- the constraint budget is sized from the model
    as spawned -- so authoring seeds the driven joints with the budget's own floor.
    """
    stage = _make_stage(_make_cfg())
    floor = _reference_params().friction_base
    for name in JOINT_NAMES:
        friction = stage.GetPrimAtPath(f"/World/Robot/{name}").GetAttribute("newton:friction")
        assert friction.IsValid() and friction.Get() == pytest.approx(floor)


def test_authoring_preserves_a_task_authored_joint_friction():
    """A joint friction the asset already carries must win over the seed."""
    cfg = _make_cfg()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Robot")
    for index, name in enumerate(JOINT_NAMES):
        body = UsdGeom.Xform.Define(stage, f"/World/Robot/body_{index}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/World/Robot/{name}")
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        prim = stage.DefinePrim(f"/World/Robot/asset_{name}_actuator", "NewtonActuator")
        prim.GetReferences().AddReference(str(Path(__file__).parent / "data" / "bam_xl330_m6.usda"), "/BamActuator")
        prim.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
        joint.GetPrim().CreateAttribute("newton:friction", Sdf.ValueTypeNames.Float).Set(0.5)
    author_actuator_prims(stage, "/World/Robot", {"servo": cfg})
    for name in JOINT_NAMES:
        assert stage.GetPrimAtPath(f"/World/Robot/{name}").GetAttribute("newton:friction").Get() == pytest.approx(0.5)


"""
Kernel behaviour.
"""


def test_controller_rejects_stepping_without_solver_load_binding():
    """An unbound controller must fail before launching kernels with missing solver inputs."""
    harness = _Harness(_make_cfg(), num_envs=1, device="cpu")
    harness.controller.external_torque = None
    zeros = np.zeros((1, len(JOINT_NAMES)), dtype=np.float32)
    with pytest.raises(RuntimeError, match="MJWarp.*bound"):
        harness.step(zeros, zeros, zeros)


@pytest.mark.parametrize("device", test_devices())
def test_controller_matches_upstream_motor_and_friction_goldens(device):
    """The USD-to-Warp path preserves upstream firmware, motor and m6 friction outputs."""
    with np.load(Path(__file__).parent / "data" / "bam_xl330_m6_goldens.npz") as data:
        goldens = {key: data[key] for key in data.files}
    samples = len(goldens["q"])
    harness = _Harness(_make_cfg(), num_envs=samples // 2, device=device)
    # The budget's prior motor load is an independent golden input, not recomputed by the port.
    state_in, state_out = harness.actuator.state(), harness.actuator.state()
    state_in.drive_state.prev_motor_torque.assign(goldens["prev_tau"].astype(np.float32))
    harness.controller.external_torque.assign(goldens["ext_tau"].astype(np.float32))
    for array, key in ((harness.joint_pos, "q"), (harness.joint_vel, "dq"), (harness.target_pos, "q_target")):
        array.assign(goldens[key].astype(np.float32).reshape(-1, 2))
    with wp.ScopedDevice(device):
        harness.actuator.step(harness.state, harness.control, state_in, state_out, dt=DT)
    np.testing.assert_allclose(
        harness.control.joint_f_2d.numpy().reshape(-1), goldens["motor_torque"], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        harness.controller.friction_budget.numpy(), goldens["frictionloss_budget"], rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("device", test_devices())
def test_solver_mode_emits_the_motor_torque_and_publishes_the_budget(device):
    """With the solver owning the friction, BAM applies the motor torque and exports the budget."""
    harness = _Harness(_make_cfg(actuator_effort_limit=0.05), num_envs=1, device=device)
    params = _reference_params()

    effort = harness.step(np.array([[0.3, -0.1]]), np.array([[0.5, -0.4]]), np.zeros((1, 2)))

    motor = harness.controller.motor_torque.numpy()
    assert np.abs(motor).max() > 0.05, "the configured effort limit must bind"
    np.testing.assert_allclose(effort.reshape(-1), np.clip(motor, -0.05, 0.05), atol=0.0, rtol=0.0)
    budget = harness.controller.friction_budget.numpy()
    assert (budget >= params.friction_base).all(), "the published budget must keep the Coulomb floor"
    np.testing.assert_allclose(harness.controller.viscous_damping.numpy(), params.friction_viscous, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_friction_scale_changes_the_published_budget(device):
    """Friction scaling changes the solver budget while preserving the motor torque.

    This is the parameter an environment's domain-randomization event drives; the write goes
    through the same controller array the group-parameter API addresses.
    """
    pos, vel, target = np.array([[0.05, 0.05]]), np.array([[0.02, 0.02]]), np.zeros((1, 2))

    baseline = _Harness(_make_cfg(), num_envs=1, device=device)
    baseline_effort = baseline.step(pos, vel, target)

    scaled = _Harness(_make_cfg(), num_envs=1, device=device)
    scaled.controller.friction_scale.fill_(4.0)
    scaled_effort = scaled.step(pos, vel, target)

    np.testing.assert_allclose(
        scaled.controller.friction_budget.numpy(),
        4.0 * baseline.controller.friction_budget.numpy(),
        rtol=1e-6,
        atol=0.0,
    )
    np.testing.assert_array_equal(scaled_effort, baseline_effort)


@pytest.mark.parametrize("device", test_devices())
def test_shared_supply_sags_with_the_group_load(device):
    """The supply drop is driven by the whole group's load, not by each joint's own.

    ``env_dof_stride`` is what tells the controller which flat DOFs belong to one supply; the
    adapter declares it because it is the first object that knows the environment count.
    """
    harness = _Harness(_make_cfg(vin_drop_gain_range=None), num_envs=2, device=device)
    assert harness.controller.env_dof_stride == len(JOINT_NAMES)
    harness.controller.sag_gain.fill_(5.0)

    pos = np.array([[0.4, 0.4], [0.4, 0.4]])
    target = np.zeros((2, 2))
    harness.step(pos, np.zeros((2, 2)), target)
    # ``numpy()`` aliases a Warp array on the host, so the torque has to be copied out
    # before the next step overwrites it.
    motor = harness.controller.motor_torque.numpy().copy()
    harness.step(pos, np.zeros((2, 2)), target)

    expected = VIN - 5.0 * np.abs(motor.reshape(2, 2)).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(
        harness.controller.effective_vin.numpy().reshape(2, 2), np.broadcast_to(expected, (2, 2)), rtol=1e-5, atol=0.0
    )


@pytest.mark.parametrize("device", test_devices())
def test_startup_sampling_draws_one_value_per_environment(device):
    """The config's start-up ranges must reach the controller once the actuator exists.

    A USD prim is shared by every clone, so the ranges cannot be authored per environment.
    They are drawn afterwards, with one value covering all of an environment's joints.
    """
    cfg = _make_cfg(vin_range=(6.0, 8.0), friction_scale_range=(0.5, 1.5))
    harness = _Harness(cfg, num_envs=8, device=device)

    apply_bam_startup_sampling(harness.controller, cfg)

    for attr, (low, high) in (("vin", cfg.vin_range), ("friction_scale", cfg.friction_scale_range)):
        values = getattr(harness.controller, attr).numpy().reshape(8, len(JOINT_NAMES))
        np.testing.assert_allclose(values[:, 0], values[:, 1], atol=0.0, rtol=0.0)
        assert ((values >= low) & (values <= high)).all()
        assert len(np.unique(values[:, 0])) > 1, "every environment drew the same value"
    before_reset = {name: getattr(harness.controller, name).numpy().copy() for name in ("vin", "friction_scale")}
    harness.reset(torch.arange(8, device=device))
    for name, values in before_reset.items():
        np.testing.assert_array_equal(getattr(harness.controller, name).numpy(), values)
    # An unset range leaves the authored nominal in place.
    np.testing.assert_allclose(harness.controller.sag_gain.numpy(), 0.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "device, hold_probability, period",
    [(device, 0.0, 0) for device in test_devices()] + [("cpu", 1.0, 0), ("cpu", 0.0, 4)],
)
def test_constant_delay_replays_an_older_command(device, hold_probability, period):
    """A fixed lag of ``k`` steps must reproduce an undelayed actuator fed the ``k``-step-old command."""
    lag = 3
    delayed = _Harness(
        _make_cfg(min_delay=lag, max_delay=lag, delay_hold_prob=hold_probability, delay_update_period=period),
        num_envs=2,
        device=device,
    )
    undelayed = _Harness(_make_cfg(), num_envs=2, device=device)

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
    """Hold freezes the lag; periodic refreshes remain staggered per driven joint."""
    harness = _Harness(
        _make_cfg(min_delay=0, max_delay=3, delay_hold_prob=hold_probability, delay_update_period=period),
        num_envs=16,
        device="cpu",
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
def test_reset_restores_the_first_step_behaviour(device):
    """Resetting an environment must clear its caches without touching the others."""
    harness = _Harness(_make_cfg(), num_envs=2, device=device)
    harness.controller.sag_gain.fill_(0.5)
    pos, vel, target = np.array([[0.2, 0.2], [0.2, 0.2]]), np.array([[1.0, 1.0], [1.0, 1.0]]), np.zeros((2, 2))

    first = harness.step(pos, vel, target)
    harness.step(pos, vel, target)
    harness.reset(torch.tensor([0], device=device))
    after = harness.step(pos, vel, target)

    np.testing.assert_allclose(after[0], first[0], atol=1e-6, rtol=0.0)
    assert np.abs(after[1] - first[1]).max() > 1e-9, "the untouched environment must keep its history"
