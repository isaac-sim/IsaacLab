# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PhysX-side Newton actuator adapter."""

import numpy as np
from isaaclab_newton.actuators import NewtonActuatorAdapter
from newton.actuators import ClampingDCMotor, ClampingMaxEffort, ClampingPositionBased, ControllerPD

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, RemotizedPDActuatorCfg
from isaaclab.sim.schemas.schemas_actuators import _author_actuator_prims

_JOINT_NAMES = ["pd_a", "pd_b", "dc_a", "dc_b", "remote_a", "remote_b"]


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
                joint_names_expr=["pd_a"], stiffness=11.0, damping=1.5, effort_limit=21.0, max_delay=2
            ),
            "pd_b": DelayedPDActuatorCfg(
                joint_names_expr=["pd_b"], stiffness=22.0, damping=2.5, effort_limit=32.0, max_delay=4
            ),
            "dc_a": DCMotorCfg(
                joint_names_expr=["dc_a"],
                stiffness=33.0,
                damping=3.5,
                effort_limit=43.0,
                velocity_limit=7.0,
                saturation_effort=53.0,
            ),
            "dc_b": DCMotorCfg(
                joint_names_expr=["dc_b"],
                stiffness=44.0,
                damping=4.5,
                effort_limit=54.0,
                velocity_limit=8.0,
                saturation_effort=64.0,
            ),
            "remote_a": RemotizedPDActuatorCfg(
                joint_names_expr=["remote_a"],
                stiffness=55.0,
                damping=5.5,
                effort_limit=65.0,
                max_delay=1,
                joint_parameter_lookup=[[-1.0, 1.0, 10.0], [1.0, 1.0, 20.0]],
            ),
            "remote_b": RemotizedPDActuatorCfg(
                joint_names_expr=["remote_b"],
                stiffness=55.0,
                damping=5.5,
                effort_limit=65.0,
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
    assert {
        tuple(next(c for c in actuator.clamping if type(c) is ClampingPositionBased).lookup_efforts.numpy())
        for actuator in remotized
    } == {(10.0, 20.0), (11.0, 21.0)}


def test_from_usd_ignores_an_all_zero_delay_group():
    """Treat an authored zero-step delay as an undelayed actuator."""
    stage = _make_actuator_stage()
    for name in ("pd_a_pd_a_actuator", "pd_b_pd_b_actuator"):
        actuator_prim = stage.GetPrimAtPath(f"/World/Robot/{name}")
        actuator_prim.GetAttribute("newton:delaySteps").Set(0)
        actuator_prim.GetAttribute("newton:maxDelay").Set(0)

    actuators = NewtonActuatorAdapter.from_usd(
        stage=stage,
        joint_names=_JOINT_NAMES,
        num_envs=2,
        num_joints=len(_JOINT_NAMES),
        device="cpu",
        articulation_prim_path="/World/Robot",
    ).actuators

    pd = next(actuator for actuator in actuators if [type(c) for c in actuator.clamping] == [ClampingMaxEffort])
    assert pd.delay is None
