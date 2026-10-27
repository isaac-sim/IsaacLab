# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PhysX-side Newton actuator adapter and its host runtime."""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import warp as wp
from newton._src.actuators.utils import load_checkpoint
from newton.actuators import (
    ActuatorParsed,
    ClampingDCMotor,
    ClampingMaxEffort,
    ClampingPositionBased,
    Delay,
    DrivePD,
    parse_actuator_prim,
)

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.actuators import (
    ActuatorBaseCfg,
    ActuatorNetLSTMCfg,
    ActuatorNetMLPCfg,
    DCMotor,
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuatorCfg,
    RemotizedPDActuatorCfg,
)
from isaaclab.actuators.newton import NewtonActuatorAdapter
from isaaclab.actuators.newton.physx_runtime import PhysxActuatorRuntime
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.schemas.schemas_actuators import (
    author_actuator_prims,
    define_actuator_properties,
    resave_checkpoint_with_metadata,
)
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.test.utils.actuator_equivalence import make_dummy_lstm_checkpoint, make_dummy_mlp_checkpoint
from isaaclab.utils import validate

_JOINT_NAMES = ["pd_a", "pd_b", "dc_a", "dc_b", "remote_a", "remote_b"]


class MisleadingImplicitActuatorDCMotor(DCMotor):
    """Explicit actuator whose class name contains ``ImplicitActuator``."""


def _write_pickle_marker(marker_path: str) -> str:
    Path(marker_path).write_text("deserialized", encoding="utf-8")
    return "deserialized"


class _PickleMarker:
    def __init__(self, marker_path: str):
        self.marker_path = marker_path

    def __reduce__(self):
        return _write_pickle_marker, (self.marker_path,)


def _make_robot_stage() -> Usd.Stage:
    """Create a robot with one revolute joint per entry of ``_JOINT_NAMES`` and no actuators."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Robot")

    bodies = [UsdGeom.Xform.Define(stage, f"/World/Robot/body_{index}") for index in range(len(_JOINT_NAMES))]
    for body in bodies:
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

    joints = [UsdPhysics.RevoluteJoint.Define(stage, f"/World/Robot/{name}") for name in _JOINT_NAMES]
    for joint, body in zip(joints, bodies, strict=True):
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
    return stage


def _make_actuator_stage() -> Usd.Stage:
    stage = _make_robot_stage()
    author_actuator_prims(
        stage,
        "/World/Robot",
        {
            "pd_a": DelayedPDActuatorCfg(
                joint_names_expr=["pd_a"], stiffness=11.0, damping=1.5, actuator_effort_limit=21.0, max_delay=2
            ),
            "pd_b": DelayedPDActuatorCfg(
                joint_names_expr=["pd_b"], stiffness=22.0, damping=2.5, actuator_effort_limit=32.0, max_delay=4
            ),
            "dc_a": DCMotorCfg(
                joint_names_expr=["dc_a"],
                stiffness=33.0,
                damping=3.5,
                actuator_effort_limit=43.0,
                actuator_velocity_limit=7.0,
                saturation_effort=53.0,
            ),
            "dc_b": DCMotorCfg(
                joint_names_expr=["dc_b"],
                stiffness=44.0,
                damping=4.5,
                actuator_effort_limit=54.0,
                actuator_velocity_limit=8.0,
                saturation_effort=64.0,
            ),
            "remote_a": RemotizedPDActuatorCfg(
                joint_names_expr=["remote_a"],
                stiffness=55.0,
                damping=5.5,
                actuator_effort_limit=65.0,
                max_delay=1,
                joint_parameter_lookup=[[-1.0, 1.0, 10.0], [1.0, 1.0, 20.0]],
            ),
            "remote_b": RemotizedPDActuatorCfg(
                joint_names_expr=["remote_b"],
                stiffness=55.0,
                damping=5.5,
                actuator_effort_limit=65.0,
                max_delay=1,
                joint_parameter_lookup=[[-1.0, 1.0, 11.0], [1.0, 1.0, 21.0]],
            ),
        },
    )
    return stage


def _make_asset_stage(tmp_path: Path, asset: Usd.Stage | None = None) -> Usd.Stage:
    """Reference *asset* (default ``_make_actuator_stage()``) as a spawned robot whose joints carry USD actuators."""
    asset_path = tmp_path / "robot.usda"
    # Keep the stage referenced: its root layer is released with it.
    asset = asset or _make_actuator_stage()
    asset.GetRootLayer().Export(str(asset_path))
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Robot").GetReferences().AddReference(str(asset_path), "/World/Robot")
    return stage


def _parsed_actuator(stage: Usd.Stage, joint_name: str) -> ActuatorParsed:
    """Parse the single active actuator that Newton builds for *joint_name*."""
    parsed = [
        entry
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot"))
        if (entry := parse_actuator_prim(prim, allow_unsafe_torch_checkpoints=True))
        and entry.target_path == f"/World/Robot/{joint_name}"
    ]
    assert len(parsed) == 1, parsed
    return parsed[0]


def test_newton_actuator_path_is_enabled_by_default():
    """Use the Newton actuator path unless the legacy path is explicitly requested."""
    assert SimulationCfg().use_newton_actuators
    assert not SimulationCfg(use_newton_actuators=False).use_newton_actuators


def test_checkpoint_metadata_rejects_pickle_without_deserializing(monkeypatch, tmp_path):
    """Reject legacy pickle checkpoints before their payload can execute."""
    checkpoint_path = tmp_path / "actuator.pt"
    marker_path = tmp_path / "pickle-executed"
    torch.save({"model": _PickleMarker(str(marker_path))}, checkpoint_path)
    monkeypatch.setattr("isaaclab.utils.assets.retrieve_file_path", lambda _: str(checkpoint_path))

    with pytest.raises(ValueError, match="expected a TorchScript archive"):
        resave_checkpoint_with_metadata(str(checkpoint_path), {})

    assert not marker_path.exists()


@pytest.mark.parametrize("kind", ["mlp", "lstm"])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_checkpoint_export_preserves_metadata_and_variable_batches(kind, device):
    """Newton can move exported Lab checkpoints to CUDA and evaluate any environment count."""
    path = make_dummy_mlp_checkpoint() if kind == "mlp" else make_dummy_lstm_checkpoint()
    converted_path = None
    try:
        original = torch.jit.load(path).eval()
        converted_path = resave_checkpoint_with_metadata(path, {"model_type": kind, "torque_scale": 3.0})
        converted, metadata = load_checkpoint(converted_path)
        original, converted = original.to(device), converted.to(device)
        assert metadata["model_type"] == kind
        assert metadata["torque_scale"] == 3.0
        if kind == "mlp":
            assert metadata["input_idx"] == [0, 1, 2]
            assert metadata["vel_scale"] == 0.5
        else:
            assert (metadata["num_layers"], metadata["hidden_size"]) == (1, 4)
        for batch_size in (1, 3, 8):
            if kind == "mlp":
                args = (torch.randn(batch_size, 6, device=device),)
            else:
                hidden = torch.randn(1, batch_size, 4, device=device)
                cell = torch.randn_like(hidden)
                args = (torch.randn(batch_size, 1, 2, device=device), (hidden, cell))
            torch.testing.assert_close(converted(*args), original(*args))
    finally:
        Path(path).unlink()
        if converted_path is not None:
            Path(converted_path).unlink()


def test_from_usd_groups_by_structure_and_preserves_per_dof_values():
    """Aggregate scalar variants while keeping incompatible shared lookup tables separate."""
    actuators = NewtonActuatorAdapter.from_usd(
        stage=_make_actuator_stage(),
        joint_names=_JOINT_NAMES,
        num_envs=2,
        num_joints=len(_JOINT_NAMES),
        device="cpu",
        articulation_prim_path="/World/Robot",
    ).actuators

    assert len(actuators) == 4

    pd = next(actuator for actuator in actuators if [type(c) for c in actuator.clamping] == [ClampingMaxEffort])
    assert type(pd.drive) is DrivePD
    np.testing.assert_array_equal(pd.indices.numpy(), [0, 1, 6, 7])
    np.testing.assert_allclose(pd.drive.kp.numpy(), [11.0, 22.0, 11.0, 22.0])
    np.testing.assert_allclose(pd.drive.kd.numpy(), [1.5, 2.5, 1.5, 2.5])
    np.testing.assert_allclose(pd.clamping[0].max_effort.numpy(), [21.0, 32.0, 21.0, 32.0])
    np.testing.assert_array_equal(pd.delay.delay_steps.numpy(), [2, 4, 2, 4])
    assert pd.delay.buf_depth == 4

    dc = next(actuator for actuator in actuators if [type(c) for c in actuator.clamping] == [ClampingDCMotor])
    assert type(dc.drive) is DrivePD
    assert dc.delay is None
    np.testing.assert_array_equal(dc.indices.numpy(), [2, 3, 8, 9])
    np.testing.assert_allclose(dc.drive.kp.numpy(), [33.0, 44.0, 33.0, 44.0])
    np.testing.assert_allclose(dc.drive.kd.numpy(), [3.5, 4.5, 3.5, 4.5])
    np.testing.assert_allclose(dc.clamping[0].saturation_effort.numpy(), [53.0, 64.0, 53.0, 64.0])
    np.testing.assert_allclose(dc.clamping[0].velocity_limit.numpy(), [7.0, 8.0, 7.0, 8.0])
    np.testing.assert_allclose(dc.clamping[0].max_motor_effort.numpy(), [43.0, 54.0, 43.0, 54.0])

    remotized = [
        actuator
        for actuator in actuators
        if any(type(clamping) is ClampingPositionBased for clamping in actuator.clamping)
    ]
    assert len(remotized) == 2
    assert {tuple(actuator.indices.numpy()) for actuator in remotized} == {(4, 10), (5, 11)}
    assert all(not any(type(clamping) is ClampingMaxEffort for clamping in actuator.clamping) for actuator in remotized)
    lookups = [next(c for c in actuator.clamping if type(c) is ClampingPositionBased) for actuator in remotized]
    assert {(tuple(c.lookup_positions.numpy()), tuple(c.lookup_efforts.numpy())) for c in lookups} == {
        ((-1.0, 1.0), (10.0, 20.0)),
        ((-1.0, 1.0), (11.0, 21.0)),
    }


def test_is_stateful_only_with_delay_state():
    """Report the adapter as stateful when an actuator keeps a delay buffer, and stateless otherwise."""
    stateless_stage = _make_robot_stage()
    author_actuator_prims(
        stateless_stage,
        "/World/Robot",
        {
            "dc": DCMotorCfg(
                joint_names_expr=[".*"],
                stiffness=1.0,
                damping=0.1,
                actuator_effort_limit=2.0,
                actuator_velocity_limit=3.0,
                saturation_effort=4.0,
            )
        },
    )
    for stage, expected in ((_make_actuator_stage(), True), (stateless_stage, False)):
        adapter = NewtonActuatorAdapter.from_usd(
            stage=stage,
            joint_names=_JOINT_NAMES,
            num_envs=2,
            num_joints=len(_JOINT_NAMES),
            device="cpu",
            articulation_prim_path="/World/Robot",
        )
        assert adapter.is_stateful is expected


def test_schema_authoring_matches_lab_effort_limit_resolution():
    """New actuator prims match the Lab path's authored fallback and partial-map limits; a zero delay authors none."""
    stage = _make_robot_stage()
    for joint_name, authored_limit in zip(("pd_a", "pd_b", "dc_a", "dc_b"), (71.0, 72.0, 73.0, 74.0), strict=True):
        joint_prim = stage.GetPrimAtPath(f"/World/Robot/{joint_name}")
        UsdPhysics.DriveAPI.Apply(joint_prim, "angular").CreateMaxForceAttr(authored_limit)
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/a")
    cfgs = {
        "mapped": DelayedPDActuatorCfg(
            joint_names_expr=["pd_.*"], stiffness=1.0, damping=0.0, actuator_effort_limit={"pd_a": 5.0}, max_delay=0
        ),
        "fallback": DelayedPDActuatorCfg(joint_names_expr=["dc_.*"], stiffness=1.0, damping=0.0, max_delay=0),
        # "mapped_pd" and joint "a" name the same new prim as "mapped" and "pd_a"; each joint still gets its own.
        "mapped_pd": IdealPDActuatorCfg(joint_names_expr=["a"], stiffness=1.0, damping=0.0, actuator_effort_limit=9.0),
    }

    author_actuator_prims(stage, "/World/Robot", cfgs)

    for joint_name, limit in {"pd_a": 5.0, "pd_b": 0.0, "dc_a": 73.0, "dc_b": 74.0, "a": 9.0}.items():
        assert _parsed_actuator(stage, joint_name).component_specs == [(ClampingMaxEffort, {"max_effort": limit})]


def test_configuration_overrides_only_the_values_it_sets(tmp_path, caplog):
    """Unset values keep the asset's actuator, and the config type keeps, swaps, or removes its components."""
    asset = _make_actuator_stage()
    remote_a = asset.GetPrimAtPath("/World/Robot/remote_a_remote_a_actuator")
    remote_a.AddAppliedSchema("NewtonMaxEffortClampingAPI")
    remote_a.CreateAttribute("newton:maxEffort", Sdf.ValueTypeNames.Float).Set(65.0)
    pd_b = asset.GetPrimAtPath("/World/Robot/pd_b_pd_b_actuator")
    pd_b.RemoveAppliedSchema("NewtonMaxEffortClampingAPI")
    pd_b.RemoveProperty("newton:maxEffort")
    # Callback-backed fields keep the asset's lookup table and trusted local checkpoints too.
    UsdPhysics.RevoluteJoint.Define(asset, "/World/Robot/remote_keep")
    layer = asset.GetRootLayer()
    Sdf.CopySpec(layer, "/World/Robot/remote_b_remote_b_actuator", layer, "/World/Robot/remote_keep_actuator")
    asset.GetPrimAtPath("/World/Robot/remote_keep_actuator").GetRelationship("newton:targets").SetTargets(
        ["/World/Robot/remote_keep"]
    )
    checkpoints = {}
    for kind, make_checkpoint in (("mlp", make_dummy_mlp_checkpoint), ("lstm", make_dummy_lstm_checkpoint)):
        checkpoint = tmp_path / f"{kind}.pt2"
        original = make_checkpoint()
        try:
            Path(resave_checkpoint_with_metadata(original, {"model_type": kind})).replace(checkpoint)
        finally:
            Path(original).unlink()
        checkpoints[kind] = str(checkpoint)
        joint = UsdPhysics.RevoluteJoint.Define(asset, f"/World/Robot/{kind}")
        prim = asset.DefinePrim(f"/World/Robot/{kind}_actuator", "NewtonActuator")
        prim.AddAppliedSchema("NewtonNeuralControlAPI")
        prim.AddAppliedSchema("NewtonDCMotorClampingAPI")
        prim.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
        prim.CreateAttribute("newton:modelPath", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(checkpoint)))
        prim.CreateAttribute("newton:saturationEffort", Sdf.ValueTypeNames.Float).Set(20.0)
    stage = _make_asset_stage(tmp_path, asset)
    # A component the asset lacks takes its joint's drive limit; a kept clamp keeps the asset's limit.
    for index, joint_name in enumerate(_JOINT_NAMES):
        joint_prim = stage.GetPrimAtPath(f"/World/Robot/{joint_name}")
        UsdPhysics.DriveAPI.Apply(joint_prim, "angular").CreateMaxForceAttr(80.0 + index)
    cfgs = {
        # Mappings that do not match pd_b keep its asset gain and give its added clamp the drive limit, not zero.
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=["pd_.*"], stiffness={"pd_a": 99.0}, damping=None, actuator_effort_limit={"pd_a": 5.0}
        ),
        # IdealPD swaps the asset's DC clamp for a max-effort clamp.
        "dc_a": IdealPDActuatorCfg(joint_names_expr=["dc_a"], stiffness=None, damping=None),
        # A DC motor without a saturation effort keeps the asset's motor curve.
        "dc_b": DCMotorCfg(joint_names_expr=["dc_b"], stiffness=1.0, damping=None),
        # A DC motor models no max-effort clamp, lookup table or delay, so all three are removed.
        "remote_a": DCMotorCfg(joint_names_expr=["remote_a"], stiffness=None, damping=None, saturation_effort=5.0),
        # A zero delay removes the asset's delay.
        "remote_b": DelayedPDActuatorCfg(joint_names_expr=["remote_b"], stiffness=None, damping=None, max_delay=0),
        "remote_keep": RemotizedPDActuatorCfg(joint_names_expr=["remote_keep"], stiffness=99.0, damping=None),
        "mlp": ActuatorNetMLPCfg(joint_names_expr=["mlp"], actuator_effort_limit=0.5),
        "lstm": ActuatorNetLSTMCfg(joint_names_expr=["lstm"], actuator_effort_limit=0.75),
    }
    for cfg in cfgs.values():
        validate(cfg)

    with caplog.at_level(logging.WARNING):
        author_actuator_prims(stage, "/World/Robot", cfgs)
    # A warning names each joint whose asset components the config type does not model.
    assert {name for name in _JOINT_NAMES if f"of joint '{name}'" in caplog.text} == {"dc_a", "remote_a", "remote_b"}
    # Authoring again overrides the same prims, so the stage does not change.
    authored = stage.GetRootLayer().ExportToString()
    author_actuator_prims(stage, "/World/Robot", cfgs)
    assert stage.GetRootLayer().ExportToString() == authored

    dc_b_curve = {"saturation_effort": 64.0, "velocity_limit": 8.0, "max_motor_effort": 54.0}
    expected = {
        "pd_a": ({"kp": 99.0, "kd": 1.5}, {ClampingMaxEffort: {"max_effort": 5.0}, Delay: {"delay_steps": 2}}),
        "pd_b": ({"kp": 22.0, "kd": 2.5}, {ClampingMaxEffort: {"max_effort": 81.0}, Delay: {"delay_steps": 4}}),
        "dc_a": ({"kp": 33.0, "kd": 3.5}, {ClampingMaxEffort: {"max_effort": 82.0}}),
        "dc_b": ({"kp": 1.0, "kd": 4.5}, {ClampingDCMotor: dc_b_curve}),
        "remote_a": ({"kp": 55.0, "kd": 5.5}, {ClampingDCMotor: {"saturation_effort": 5.0, "max_motor_effort": 84.0}}),
        "remote_b": ({"kp": 55.0, "kd": 5.5}, {ClampingMaxEffort: {"max_effort": 85.0}}),
        "remote_keep": (
            {"kp": 99.0, "kd": 5.5},
            {
                ClampingPositionBased: {"lookup_positions": [-1.0, 1.0], "lookup_efforts": [11.0, 21.0]},
                Delay: {"delay_steps": 1},
            },
        ),
        "mlp": (
            {"model_path": checkpoints["mlp"]},
            {ClampingDCMotor: {"saturation_effort": 20.0, "max_motor_effort": 0.5}},
        ),
        "lstm": (
            {"model_path": checkpoints["lstm"]},
            {ClampingDCMotor: {"saturation_effort": 20.0, "max_motor_effort": 0.75}},
        ),
    }
    for joint_name, (drive, components) in expected.items():
        actuator = _parsed_actuator(stage, joint_name)
        assert (actuator.drive_kwargs, dict(actuator.component_specs)) == (drive, components), joint_name


def test_actuators_without_a_matching_drive_are_replaced(tmp_path, caplog):
    """Another drive, several actuators on one joint, or an unknown schema fall back to a new prim with defaults."""
    asset = _make_actuator_stage()
    pid = asset.GetPrimAtPath("/World/Robot/pd_a_pd_a_actuator")
    pid.RemoveAppliedSchema("NewtonPDControlAPI")
    pid.AddAppliedSchema("NewtonPIDControlAPI")
    asset.GetPrimAtPath("/World/Robot/dc_a_dc_a_actuator").AddAppliedSchema("NewtonUnknownAPI")
    layer = asset.GetRootLayer()
    Sdf.CopySpec(layer, "/World/Robot/pd_b_pd_b_actuator", layer, "/World/Robot/pd_b_copy")
    # Newton drives only an actuator's first target, so dc_b's actuator is not a third pd_b actuator.
    asset.GetPrimAtPath("/World/Robot/dc_b_dc_b_actuator").GetRelationship("newton:targets").AddTarget(
        "/World/Robot/pd_b"
    )
    stage = _make_asset_stage(tmp_path, asset)
    cfg = IdealPDActuatorCfg(joint_names_expr=["pd_.*", "dc_a"], stiffness=7.0, damping=None)

    # The group name makes pd_a's new prim path that of the replaced asset prim.
    with caplog.at_level(logging.WARNING):
        author_actuator_prims(stage, "/World/Robot", {"pd_a": cfg})

    for joint_name in ("pd_a", "pd_b", "dc_a"):
        actuator = _parsed_actuator(stage, joint_name)
        assert actuator.drive_class is DrivePD
        assert actuator.drive_kwargs == {"kp": 7.0, "kd": 0.0}
        assert actuator.component_specs == []
    for path in ("pd_a_pd_a_actuator", "pd_b_pd_b_actuator", "pd_b_copy", "dc_a_dc_a_actuator"):
        assert not stage.GetPrimAtPath(f"/World/Robot/{path}").IsActive()
    assert all(f"joint '{name}'" in caplog.text for name in ("pd_a", "pd_b", "dc_a"))
    # Joints no config covers keep their USD actuators.
    assert _parsed_actuator(stage, "dc_b").drive_kwargs == {"kp": 44.0, "kd": 4.5}


@pytest.mark.parametrize(
    ("make_stage", "groups", "match"),
    [
        (
            _make_asset_stage,
            {
                "unsupported": ActuatorBaseCfg(
                    class_type=DCMotor, joint_names_expr=["pd_a"], stiffness=0.0, damping=0.0
                ),
                "custom_dc": DCMotorCfg(
                    class_type=f"{__name__}:MisleadingImplicitActuatorDCMotor",
                    joint_names_expr=["pd_a"],
                    stiffness=0.0,
                    damping=0.0,
                    saturation_effort=1.0,
                ),
            },
            r"'unsupported' \(ActuatorBaseCfg\), 'custom_dc' \(DCMotorCfg\).*use_newton_actuators",
        ),
        (
            _make_asset_stage,
            {
                "group": DelayedPDActuatorCfg(
                    joint_names_expr=["pd_.*"], stiffness=1.0, damping=0.0, actuator_effort_limit={"missing_joint": 5.0}
                )
            },
            "missing_joint",
        ),
        (
            lambda tmp_path: _make_robot_stage(),
            {"group": DCMotorCfg(joint_names_expr=["dc_a"], stiffness=1.0, damping=1.0)},
            "saturation_effort.*dc_a",
        ),
        (
            lambda tmp_path: _make_robot_stage(),
            {"group": RemotizedPDActuatorCfg(joint_names_expr=["pd_a"], stiffness=1.0, damping=0.0)},
            "joint_parameter_lookup.*pd_a",
        ),
        (
            lambda tmp_path: _make_robot_stage(),
            {"group": ActuatorNetMLPCfg(joint_names_expr=["pd_a"])},
            "network_file.*pd_a",
        ),
        (
            _make_asset_stage,
            {"group": ActuatorNetMLPCfg(joint_names_expr=["pd_a"], network_file="unused.pt")},
            "network_file.*input_order",
        ),
        (
            _make_asset_stage,
            {"group": ActuatorNetMLPCfg(joint_names_expr=["pd_a"], pos_scale=2.0)},
            "network_file.*metadata",
        ),
    ],
    ids=[
        "unsupported_cfgs",
        "unmatched_effort_limit",
        "new_prim_without_saturation",
        "new_prim_without_lookup",
        "new_prim_without_checkpoint",
        "incomplete_mlp_metadata",
        "metadata_without_checkpoint",
    ],
)
def test_rejected_config_raises_before_editing_the_joint(tmp_path, make_stage, groups, match):
    """A config Newton cannot author, or one that leaves a required value unset, raises before it edits the joint."""
    stage = make_stage(tmp_path)
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(ValueError, match=match):
        author_actuator_prims(stage, "/World/Robot", groups)

    assert stage.GetRootLayer().ExportToString() == before


def test_schema_authoring_accepts_supported_public_actuator_alias():
    """Accept a public import path that resolves to a supported actuator class."""
    stage = _make_robot_stage()
    cfg = DCMotorCfg(
        class_type="isaaclab.actuators:DCMotor",
        joint_names_expr=["pd_a"],
        stiffness=1.0,
        damping=0.1,
        actuator_effort_limit=2.0,
        actuator_velocity_limit=3.0,
        saturation_effort=4.0,
    )

    author_actuator_prims(stage, "/World/Robot", {"public_alias": cfg})

    assert stage.GetPrimAtPath("/World/Robot/public_alias_pd_a_actuator").IsValid()


def test_define_actuator_properties_authors_every_spawned_articulation(monkeypatch):
    """A multi-asset spawner creates one articulation per match, and each needs its own actuator prims."""
    stage = Usd.Stage.CreateInMemory()
    robot_paths = [f"/World/envs/env_{index}/Robot" for index in range(3)]
    for robot_path in robot_paths:
        UsdGeom.Xform.Define(stage, robot_path)
        body = UsdGeom.Xform.Define(stage, f"{robot_path}/body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{robot_path}/pd_a")
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
    monkeypatch.setattr(
        SimulationContext,
        "instance",
        staticmethod(lambda: SimpleNamespace(cfg=SimpleNamespace(use_newton_actuators=True))),
    )
    cfg = DelayedPDActuatorCfg(
        joint_names_expr=["pd_a"], stiffness=11.0, damping=1.5, actuator_effort_limit=21.0, max_delay=2
    )

    define_actuator_properties(r"/World/envs/env_.*/Robot", {"pd": cfg}, stage=stage)

    for robot_path in robot_paths:
        actuators = [prim for prim in stage.Traverse() if prim.GetTypeName() == "NewtonActuator"]
        assert any(str(prim.GetPath()).startswith(robot_path + "/") for prim in actuators), robot_path


# ---------------------------------------------------------------------------
# CUDA-graph handling of the host-side PhysX actuator runtime
# ---------------------------------------------------------------------------


def _runtime() -> PhysxActuatorRuntime:
    """Create a runtime for a CUDA articulation with a mocked logger."""
    return PhysxActuatorRuntime(SimpleNamespace(device="cuda:0"), logger=Mock())


def _failing_capture(fail_at: int) -> type:
    """Return a ``wp.ScopedCapture`` double whose capture number ``fail_at`` (0-based) raises on entry."""

    class _Capture:
        count = 0

        def __init__(self, *args, **kwargs) -> None:
            self.index = type(self).count
            type(self).count += 1
            self.graph = object()

        def __enter__(self):
            if self.index >= fail_at:
                raise RuntimeError("capture unavailable")
            return self

        def __exit__(self, *exc_info) -> bool:
            return False

    return _Capture


def test_graph_capture_failure_restores_adapter_state_and_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed second capture discards the graphs and restores the state swapped by the first capture."""
    state_a, state_b = object(), object()
    runtime = _runtime()
    runtime.adapter = SimpleNamespace(_states_a=state_a, _states_b=state_b)

    def _swap_adapter_state(*args, **kwargs) -> None:
        runtime.adapter._states_a, runtime.adapter._states_b = runtime.adapter._states_b, runtime.adapter._states_a

    monkeypatch.setattr(wp, "ScopedCapture", _failing_capture(fail_at=1))
    monkeypatch.setattr(runtime, "_run_native_actuator_kernels", _swap_adapter_state)

    runtime._capture_native_actuator_graphs(SimpleNamespace(), 0.01)

    assert runtime.native_actuator_graphs == ()
    assert runtime.adapter._states_a is state_a
    assert runtime.adapter._states_b is state_b
    runtime._logger.warning.assert_called_once()


def test_compute_runs_eagerly_after_graph_capture_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A graphable adapter whose capture fails still computes the current command eagerly."""
    capture = _failing_capture(fail_at=0)
    runtime = _runtime()
    runtime.adapter = SimpleNamespace(is_stateful=False, is_all_graphable=True, _states_a=object(), _states_b=object())
    eager_compute = Mock()
    monkeypatch.setattr(wp, "get_device", lambda device: SimpleNamespace(is_cuda=True, is_capturing=False))
    monkeypatch.setattr(wp, "ScopedCapture", capture)
    monkeypatch.setattr(runtime, "_run_native_actuator_kernels", eager_compute)
    collection = SimpleNamespace()

    runtime.compute(collection, 0.01)
    runtime.compute(collection, 0.01)

    # The failed capture is not retried, and every step falls back to the eager kernels.
    assert capture.count == 1
    assert runtime.native_actuator_graphs == ()
    assert eager_compute.call_count == 2
    eager_compute.assert_called_with(collection, 0.01)


def test_stateful_actuator_rejects_outer_cuda_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stateful adapters cannot safely mutate their buffers inside an outer CUDA capture."""
    runtime = _runtime()
    runtime.adapter = SimpleNamespace(is_stateful=True)
    monkeypatch.setattr(wp, "get_device", lambda device: SimpleNamespace(is_cuda=True, is_capturing=True))

    with pytest.raises(RuntimeError, match="stateful Newton actuators cannot run inside an outer CUDA graph capture"):
        runtime.compute(SimpleNamespace(), 0.01)
