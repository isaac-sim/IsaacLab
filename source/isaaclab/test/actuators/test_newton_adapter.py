# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PhysX-side Newton actuator adapter."""

import numpy as np
import pytest
from newton.actuators import ClampingDCMotor, ClampingMaxEffort, ClampingPositionBased, ControllerPD

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.actuators import ActuatorBaseCfg, DCMotor, DCMotorCfg, DelayedPDActuatorCfg, RemotizedPDActuatorCfg
from isaaclab.actuators.newton import NewtonActuatorAdapter
from isaaclab.sim.schemas.schemas_actuators import _author_actuator_prims
from isaaclab.utils.configclass import configclass

_JOINT_NAMES = ["pd_a", "pd_b", "dc_a", "dc_b", "remote_a", "remote_b"]


@configclass
class UnsupportedNewtonActuatorCfg(ActuatorBaseCfg):
    """Explicit actuator config intentionally unsupported by Newton authoring."""

    class_type: str = "unsupported:ExplicitActuator"


@configclass
class CustomDCMotorCfg(DCMotorCfg):
    """DC motor config that selects a custom Lab actuator implementation."""

    class_type: str = "unsupported:CustomDCMotor"


class MisleadingImplicitActuatorDCMotor(DCMotor):
    """Explicit actuator whose class name contains ``ImplicitActuator``."""


def _make_actuator_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Robot")

    bodies = [UsdGeom.Xform.Define(stage, f"/World/Robot/body_{index}") for index in range(len(_JOINT_NAMES))]
    for body in bodies:
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

    joints = [UsdPhysics.RevoluteJoint.Define(stage, f"/World/Robot/{name}") for name in _JOINT_NAMES]
    for joint, body in zip(joints, bodies, strict=True):
        joint.CreateBody1Rel().SetTargets([body.GetPath()])

    _author_actuator_prims(
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
    assert type(pd.controller) is ControllerPD
    np.testing.assert_array_equal(pd.indices.numpy(), [0, 1, 6, 7])
    np.testing.assert_allclose(pd.controller.kp.numpy(), [11.0, 22.0, 11.0, 22.0])
    np.testing.assert_allclose(pd.controller.kd.numpy(), [1.5, 2.5, 1.5, 2.5])
    np.testing.assert_allclose(pd.clamping[0].max_effort.numpy(), [21.0, 32.0, 21.0, 32.0])
    np.testing.assert_array_equal(pd.delay.delay_steps.numpy(), [2, 4, 2, 4])
    assert pd.delay.buf_depth == 4

    dc = next(actuator for actuator in actuators if [type(c) for c in actuator.clamping] == [ClampingDCMotor])
    assert type(dc.controller) is ControllerPD
    assert dc.delay is None
    np.testing.assert_array_equal(dc.indices.numpy(), [2, 3, 8, 9])
    np.testing.assert_allclose(dc.controller.kp.numpy(), [33.0, 44.0, 33.0, 44.0])
    np.testing.assert_allclose(dc.controller.kd.numpy(), [3.5, 4.5, 3.5, 4.5])
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
    assert {
        tuple(next(c for c in actuator.clamping if type(c) is ClampingPositionBased).lookup_efforts.numpy())
        for actuator in remotized
    } == {(10.0, 20.0), (11.0, 21.0)}


@pytest.mark.parametrize(
    ("configured_limit", "expected_limits"),
    [(None, (71.0, 72.0)), ({"pd_a": 5.0}, (5.0, 0.0))],
)
def test_schema_authoring_matches_lab_effort_limit_resolution(configured_limit, expected_limits):
    """Match authored fallback and partial-map resolution on native and Lab paths."""
    stage = _make_actuator_stage()
    for joint_name, authored_limit in zip(("pd_a", "pd_b"), (71.0, 72.0), strict=True):
        joint_prim = stage.GetPrimAtPath(f"/World/Robot/{joint_name}")
        UsdPhysics.DriveAPI.Apply(joint_prim, "angular").CreateMaxForceAttr(authored_limit)
    cfg = DelayedPDActuatorCfg(
        joint_names_expr=["pd_.*"],
        stiffness=1.0,
        damping=0.0,
        actuator_effort_limit=configured_limit,
        max_delay=0,
    )

    _author_actuator_prims(stage, "/World/Robot", {"fallback": cfg})

    for joint_name, expected_limit in zip(("pd_a", "pd_b"), expected_limits, strict=True):
        actuator_prim = stage.GetPrimAtPath(f"/World/Robot/fallback_{joint_name}_actuator")
        assert actuator_prim.GetAttribute("newton:maxEffort").Get() == pytest.approx(expected_limit)


@pytest.mark.parametrize(
    "configured_limit",
    [
        {"pd_.*": 5.0, "pd_a": 7.0},
        {"missing_joint": 5.0},
    ],
)
def test_schema_authoring_rejects_invalid_effort_limit_patterns(configured_limit):
    stage = _make_actuator_stage()
    cfg = DelayedPDActuatorCfg(
        joint_names_expr=["pd_.*"],
        stiffness=1.0,
        damping=0.0,
        actuator_effort_limit=configured_limit,
        max_delay=0,
    )

    with pytest.raises(ValueError):
        _author_actuator_prims(stage, "/World/Robot", {"invalid_limits": cfg})


@pytest.mark.parametrize(
    ("group_name", "cfg"),
    [
        ("unsupported", UnsupportedNewtonActuatorCfg(joint_names_expr=["pd_a"], stiffness=0.0, damping=0.0)),
        (
            "custom_dc",
            CustomDCMotorCfg(
                joint_names_expr=["pd_a"],
                stiffness=0.0,
                damping=0.0,
                actuator_effort_limit=1.0,
                actuator_velocity_limit=1.0,
                saturation_effort=1.0,
            ),
        ),
        (
            "misleading_name",
            CustomDCMotorCfg(
                class_type=f"{__name__}:MisleadingImplicitActuatorDCMotor",
                joint_names_expr=["pd_a"],
                stiffness=0.0,
                damping=0.0,
                actuator_effort_limit=1.0,
                actuator_velocity_limit=1.0,
                saturation_effort=1.0,
            ),
        ),
    ],
)
def test_schema_authoring_rejects_unsupported_explicit_cfg_before_removal(group_name, cfg):
    """Leave existing actuator prims intact when native authoring rejects a config."""
    stage = _make_actuator_stage()
    actuator_path = "/World/Robot/pd_a_pd_a_actuator"

    with pytest.raises(ValueError, match=rf"{group_name}.*{type(cfg).__name__}.*use_newton_actuators"):
        _author_actuator_prims(stage, "/World/Robot", {group_name: cfg})

    assert stage.GetPrimAtPath(actuator_path).IsValid()


def test_schema_authoring_accepts_supported_public_actuator_alias():
    """Accept a public import path that resolves to a supported actuator class."""
    stage = _make_actuator_stage()
    cfg = DCMotorCfg(
        class_type="isaaclab.actuators:DCMotor",
        joint_names_expr=["pd_a"],
        stiffness=1.0,
        damping=0.1,
        actuator_effort_limit=2.0,
        actuator_velocity_limit=3.0,
        saturation_effort=4.0,
    )

    _author_actuator_prims(stage, "/World/Robot", {"public_alias": cfg})

    assert stage.GetPrimAtPath("/World/Robot/public_alias_pd_a_actuator").IsValid()
