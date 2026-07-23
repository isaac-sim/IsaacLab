# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino physical validation for joint parameters and state writers."""

import pytest
from isaaclab_newton.assets import Articulation

from isaaclab.sim import build_simulation_context
from isaaclab.test.physics.parameter_validation.fixtures import (
    ACTIVE_UPPER,
    INACTIVE_UPPER,
    JOINT_EFFECTIVE_INERTIA,
    PROBE_TARGET,
    Q_REF,
    build_single_dof,
    make_single_dof_cfg,
)
from isaaclab.test.physics.parameter_validation.oracles import (
    PROFILE_DOF_DT,
    PhysicalCase,
    assert_physical_close,
    predict_implicit_joint_step,
)


def _case(parameter_id: str, authoring: str) -> PhysicalCase:
    return PhysicalCase(
        parameter_id=parameter_id,
        backend="newton-kamino",
        authoring_path=authoring,
        profile="PROFILE-DOF",
        dt=PROFILE_DOF_DT,
        rtol=5.0e-3,
        atol=2.0e-4,
    )


def _assert_step(
    parameter_id: str,
    authoring: str,
    result: dict[str, float],
    velocity_expected: float,
    position_expected: float,
    *,
    position_initial: float = 0.0,
    velocity_initial: float = 0.0,
) -> None:
    case = _case(parameter_id, authoring)
    assert_physical_close(result["position_before"], position_initial, case)
    assert_physical_close(result["velocity_before"], velocity_initial, case)
    assert_physical_close(result["velocity_after"], velocity_expected, case)
    assert_physical_close(result["position_after"], position_expected, case)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
def test_build_yields_fixed_base_single_dof(kamino, joint_type):
    """The shared procedure imports a fixed-base one-DOF articulation with known inertia."""
    with build_simulation_context(device="cuda:0", sim_cfg=kamino.profile_dof_cfg()) as sim:
        sim._app_control_on_stop_handle = None
        build_single_dof(joint_type, usd_stiffness=100.0)
        articulation = Articulation(make_single_dof_cfg(100.0, 0.0, None))
        sim.reset()
        articulation.update(0.0)
        assert articulation.is_initialized
        assert articulation.num_joints == 1
        assert articulation.is_fixed_base
        assert float(articulation.data.mass_matrix.torch[0, 0, 0]) == pytest.approx(
            JOINT_EFFECTIVE_INERTIA[joint_type], rel=1.0e-3
        )


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("stiffness", [100.0, 300.0])
def test_drive_01_stiffness_single_step(kamino, joint_type, authoring, stiffness):
    """DRIVE-01: Authored implicit stiffness reproduces the analytical Kamino step."""
    result = kamino.run_single_dof_step(
        joint_type,
        authoring,
        stiffness=stiffness,
        damping=0.0,
        armature=0.0,
        position_target=Q_REF,
    )
    velocity, position = predict_implicit_joint_step(
        stiffness=stiffness,
        drive_damping=0.0,
        armature=0.0,
        position_target=Q_REF,
        body_inertia=result["body_inertia"],
    )
    _assert_step("DRIVE-01", authoring, result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("damping", [20.0, 60.0])
def test_drive_02_damping_single_step(kamino, joint_type, authoring, damping):
    """DRIVE-02: Authored implicit drive damping reproduces the analytical Kamino step."""
    result = kamino.run_single_dof_step(
        joint_type,
        authoring,
        stiffness=100.0,
        damping=damping,
        armature=0.0,
        position_target=Q_REF,
    )
    velocity, position = predict_implicit_joint_step(
        stiffness=100.0,
        drive_damping=damping,
        armature=0.0,
        position_target=Q_REF,
        body_inertia=result["body_inertia"],
    )
    _assert_step("DRIVE-02", authoring, result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("armature", [0.1, 0.5])
def test_joint_07_armature_single_step(kamino, joint_type, authoring, armature):
    """JOINT-07: Authored armature reproduces the analytical Kamino step."""
    result = kamino.run_single_dof_step(
        joint_type,
        authoring,
        stiffness=100.0,
        damping=0.0,
        armature=armature,
        position_target=Q_REF,
    )
    velocity, position = predict_implicit_joint_step(
        stiffness=100.0,
        drive_damping=0.0,
        armature=armature,
        position_target=Q_REF,
        body_inertia=result["body_inertia"],
    )
    _assert_step("JOINT-07", authoring, result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("passive_damping", [0.0, 3.0, 6.0])
def test_joint_08_passive_damping_usd_single_step(kamino, joint_type, passive_damping):
    """JOINT-08: USD-authored passive damping reproduces the analytical Kamino step."""
    velocity_initial = 1.0
    result = kamino.run_single_dof_step(
        joint_type,
        "usd",
        stiffness=0.0,
        damping=0.0,
        armature=0.0,
        position_target=0.0,
        velocity=velocity_initial,
        passive_damping=passive_damping,
    )
    velocity, position = predict_implicit_joint_step(
        stiffness=0.0,
        drive_damping=0.0,
        armature=0.0,
        position_target=0.0,
        body_inertia=result["body_inertia"],
        velocity=velocity_initial,
        passive_damping=passive_damping,
    )
    _assert_step(
        "JOINT-08",
        "usd",
        result,
        velocity,
        position,
        velocity_initial=velocity_initial,
    )


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("effort", [5.0, 20.0])
def test_cmd_01_feedforward_torque_implicit_single_step(kamino, joint_type, authoring, effort):
    """CMD-01: Feed-forward effort with implicit dynamics reproduces the Kamino step."""
    result = kamino.run_single_dof_step(
        joint_type,
        authoring,
        stiffness=100.0,
        damping=0.0,
        armature=0.0,
        position_target=0.0,
        effort=effort,
    )
    velocity, position = predict_implicit_joint_step(
        stiffness=100.0,
        drive_damping=0.0,
        armature=0.0,
        position_target=0.0,
        body_inertia=result["body_inertia"],
        effort=effort,
    )
    _assert_step("CMD-01", authoring, result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime", "runtime-error"])
@pytest.mark.parametrize("effort", [5.0, 20.0])
def test_cmd_01_feedforward_torque_explicit_single_step(kamino, joint_type, authoring, effort):
    """CMD-01: Feed-forward effort without joint dynamics reproduces the Kamino step."""
    if authoring == "runtime-error":
        with pytest.raises(RuntimeError, match="Changing dynamic constraint topology"):
            kamino.run_single_dof_step(
                joint_type,
                authoring,
                stiffness=0.0,
                damping=0.0,
                armature=0.0,
                position_target=0.0,
                effort=effort,
            )
        return
    result = kamino.run_single_dof_step(
        joint_type,
        authoring,
        stiffness=0.0,
        damping=0.0,
        armature=0.0,
        position_target=0.0,
        effort=effort,
    )
    velocity, position = predict_implicit_joint_step(
        stiffness=0.0,
        drive_damping=0.0,
        armature=0.0,
        position_target=0.0,
        body_inertia=result["body_inertia"],
        effort=effort,
    )
    _assert_step("CMD-01", authoring, result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("velocity_target", [2.0, 5.0])
def test_cmd_02_velocity_reference_single_step(kamino, joint_type, velocity_target):
    """CMD-02: A joint velocity target reproduces the analytical Kamino step."""
    damping = 40.0
    result = kamino.run_single_dof_step(
        joint_type,
        "cfg",
        stiffness=0.0,
        damping=damping,
        armature=0.0,
        position_target=0.0,
        velocity_target=velocity_target,
    )
    velocity, position = predict_implicit_joint_step(
        stiffness=0.0,
        drive_damping=damping,
        armature=0.0,
        position_target=0.0,
        body_inertia=result["body_inertia"],
        velocity_target=velocity_target,
    )
    _assert_step("CMD-02", "runtime", result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "runtime", "runtime-error"])
@pytest.mark.parametrize("upper", [ACTIVE_UPPER, INACTIVE_UPPER])
def test_joint_04_position_limit(kamino, joint_type, authoring, upper):
    """JOINT-04: Authored upper position limits are enforced by Kamino."""
    if authoring == "runtime-error":
        with pytest.raises(RuntimeError, match="Changing the existence of a joint limit"):
            kamino.run_position_limit_probe(joint_type, authoring, upper)
        return
    position_max, position_final = kamino.run_position_limit_probe(joint_type, authoring, upper)
    if upper < PROBE_TARGET:
        assert position_max <= upper + 0.03, "JOINT-04: active upper limit was exceeded"
        assert position_final == pytest.approx(upper, abs=0.03), (
            "JOINT-04: joint did not settle at the active upper limit"
        )
    else:
        assert position_final > 0.35, "JOINT-04: inactive limit control did not pass the low limit"


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["cfg", "runtime"])
def test_joint_02_position_state(kamino, joint_type, authoring):
    """JOINT-02: Reset-default and live joint position survive one unforced step."""
    position = 0.15
    result = kamino.run_joint_state(
        joint_type,
        authoring,
        position=position,
        velocity=0.0,
    )
    case = _case("JOINT-02", authoring)
    assert_physical_close(result["position_before"], position, case)
    assert_physical_close(result["velocity_before"], 0.0, case)
    assert_physical_close(result["position_after"], position, case)
    assert_physical_close(result["velocity_after"], 0.0, case)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["cfg", "runtime"])
def test_joint_03_velocity_state(kamino, joint_type, authoring):
    """JOINT-03: Reset-default and live joint velocity produce unforced coast motion."""
    velocity = 0.4
    result = kamino.run_joint_state(
        joint_type,
        authoring,
        position=0.0,
        velocity=velocity,
    )
    case = _case("JOINT-03", authoring)
    assert_physical_close(result["position_before"], 0.0, case)
    assert_physical_close(result["velocity_before"], velocity, case)
    assert_physical_close(result["position_after"], velocity * PROFILE_DOF_DT, case)
    assert_physical_close(result["velocity_after"], velocity, case)
