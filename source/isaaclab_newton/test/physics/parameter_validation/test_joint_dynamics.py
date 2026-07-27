# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physical validation for joint parameters and state writers."""

import pytest
import torch
from isaaclab_newton.assets import Articulation

from isaaclab.sim import build_simulation_context
from isaaclab.test.physics.parameter_validation.fixtures import (
    ACTIVE_LOWER,
    ACTIVE_UPPER,
    INACTIVE_LOWER,
    INACTIVE_UPPER,
    JOINT_EFFECTIVE_INERTIA,
    JOINT_MASS,
    PROBE_TARGET,
    Q_REF,
    build_single_dof,
    make_single_dof_cfg,
)
from isaaclab.test.physics.parameter_validation.oracles import (
    PROFILE_DOF_DT,
    PhysicalCase,
    assert_physical_close,
    predict_dry_friction_effort,
)
from isaaclab.utils.math import convert_quat, quat_apply

_PUBLIC_APIS = {
    "BODY-03": "USD MassAPI center-of-mass authoring",
    "CMD-01": "set_joint_effort_target_index",
    "CMD-02": "set_joint_velocity_target_index",
    "DRIVE-01": "write_joint_stiffness_to_sim_index",
    "DRIVE-02": "write_joint_damping_to_sim_index",
    "JOINT-01": "UsdPhysics parent and child joint frames",
    "JOINT-02": "write_joint_position_to_sim_index",
    "JOINT-03": "write_joint_velocity_to_sim_index",
    "JOINT-05": "Newton joint velocity-limit USD/config/index-writer paths",
    "JOINT-06": "Newton joint effort-limit USD/config/index-writer paths",
    "JOINT-07": "write_joint_armature_to_sim_index",
    "JOINT-08": "USD newton:damping authoring",
    "JOINT-09": "Newton joint-friction USD/config/index-writer paths",
}

_KAMINO_VELOCITY_LIMIT_REASON = "vastsoun/newton#397: Kamino does not enforce joint velocity limits"
_MJWARP_VELOCITY_LIMIT_REASON = "Accepted gap: MJWarp stores but does not physically enforce joint velocity limits"
_KAMINO_EFFORT_LIMIT_REASON = "vastsoun/newton#398: Kamino does not enforce joint effort limits"
_KAMINO_FRICTION_REASON = "vastsoun/newton#383: Kamino does not enforce Newton joint dry friction"
_VELOCITY_LIMIT_BACKENDS = [
    pytest.param(
        "kamino",
        id="kamino",
        marks=pytest.mark.xfail(strict=True, reason=_KAMINO_VELOCITY_LIMIT_REASON),
    ),
    pytest.param(
        "mjwarp",
        id="mjwarp",
        marks=pytest.mark.xfail(strict=True, reason=_MJWARP_VELOCITY_LIMIT_REASON),
    ),
]
_EFFORT_LIMIT_BACKENDS = [
    pytest.param(
        "kamino",
        id="kamino",
        marks=pytest.mark.xfail(strict=True, reason=_KAMINO_EFFORT_LIMIT_REASON),
    ),
    pytest.param("mjwarp", id="mjwarp"),
]
_FRICTION_CASES = [
    pytest.param(
        "kamino",
        "revolute",
        id="kamino",
        marks=pytest.mark.xfail(strict=True, reason=_KAMINO_FRICTION_REASON),
    ),
    pytest.param("mjwarp", "revolute", id="mjwarp-revolute"),
    pytest.param("mjwarp", "prismatic", id="mjwarp-prismatic"),
]


def _case(parameter_adapter, parameter_id: str, authoring: str) -> PhysicalCase:
    api = _PUBLIC_APIS[parameter_id]
    if authoring == "usd":
        api = "UsdPhysics joint or drive schema"
    elif authoring == "cfg":
        api = "ImplicitActuatorCfg or ArticulationCfg initialization"
    return PhysicalCase(
        parameter_id=parameter_id,
        backend=parameter_adapter.backend,
        authoring_path=authoring,
        profile="PROFILE-DOF",
        dt=PROFILE_DOF_DT,
        substeps=1,
        api=api,
        rtol=parameter_adapter.rtol,
        atol=parameter_adapter.atol,
    )


def _assert_step(
    parameter_adapter,
    parameter_id: str,
    authoring: str,
    result: dict[str, float],
    velocity_expected: float,
    position_expected: float,
    *,
    position_initial: float = 0.0,
    velocity_initial: float = 0.0,
) -> None:
    case = _case(parameter_adapter, parameter_id, authoring)
    assert_physical_close(result["position_before"], position_initial, case)
    assert_physical_close(result["velocity_before"], velocity_initial, case)
    assert_physical_close(result["velocity_after"], velocity_expected, case)
    assert_physical_close(result["position_after"], position_expected, case)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_build_yields_fixed_base_single_dof(parameter_adapter, joint_type):
    """The shared procedure imports a fixed-base one-DOF articulation with known inertia."""
    with build_simulation_context(device="cuda:0", sim_cfg=parameter_adapter.profile_dof_cfg()) as sim:
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
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_joint_01_parent_frame_rotation_controls_motion_axis(parameter_adapter, joint_type):
    """JOINT-01/FIX-JOINT-FRAME: the parent frame rotates the world-space joint motion axis."""
    half_sqrt_two = 2.0**-0.5
    orientation = (half_sqrt_two, 0.0, half_sqrt_two, 0.0)
    result = parameter_adapter.run_joint_frame_probe(
        joint_type,
        parent_frame_orientation=orientation,
    )
    local_axis = torch.tensor(
        (0.0, 0.0, 1.0) if joint_type == "revolute" else (1.0, 0.0, 0.0),
        device=result["position_before"].device,
    )
    orientation_wxyz = torch.tensor(orientation, device=local_axis.device)
    expected_axis = quat_apply(convert_quat(orientation_wxyz, to="xyzw"), local_axis)
    measured_velocity = result["angular_velocity"] if joint_type == "revolute" else result["linear_velocity"]
    measured_speed = torch.linalg.vector_norm(measured_velocity)
    case = _case(parameter_adapter, "JOINT-01", "usd")
    assert measured_speed > 1.0e-4, case.message(float(measured_speed), "speed > 1e-4")
    assert_physical_close(measured_velocity / measured_speed, expected_axis, case)


@pytest.mark.parametrize(
    ("frame", "expected_position"),
    [
        ("parent", (0.1, 0.0, 0.0)),
        ("child", (-0.1, 0.0, 0.0)),
    ],
)
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_joint_01_frame_translation_sets_link_motion_center(parameter_adapter, frame, expected_position):
    """JOINT-01/FIX-JOINT-FRAME: parent and child translations set the revolute motion center."""
    frame_position = (0.1, 0.0, 0.0)
    kwargs = {f"{frame}_frame_position": frame_position}
    result = parameter_adapter.run_joint_frame_probe("revolute", **kwargs)
    case = _case(parameter_adapter, "JOINT-01", "usd")
    expected = torch.tensor(expected_position, device=result["position_before"].device)
    assert_physical_close(result["position_before"], expected, case)
    if frame == "parent":
        assert_physical_close(result["linear_velocity"], torch.zeros_like(expected), case)
    else:
        assert abs(float(result["linear_velocity"][1])) > 1.0e-4, case.message(
            float(result["linear_velocity"][1]), "|tangential velocity| > 1e-4"
        )


@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_joint_01_child_frame_rotation_sets_link_orientation(parameter_adapter):
    """JOINT-01/FIX-JOINT-FRAME: the child frame rotates the constrained link relative to the joint."""
    half_sqrt_two = 2.0**-0.5
    child_orientation = (half_sqrt_two, 0.0, half_sqrt_two, 0.0)
    result = parameter_adapter.run_joint_frame_probe(
        "revolute",
        child_frame_orientation=child_orientation,
    )
    expected = torch.tensor(
        (0.0, -half_sqrt_two, 0.0, half_sqrt_two),
        device=result["orientation_before"].device,
    )
    case = _case(parameter_adapter, "JOINT-01", "usd")
    measured = result["orientation_before"]
    if torch.dot(measured, expected) < 0.0:
        measured = -measured
    assert_physical_close(measured, expected, case)
    assert abs(float(result["angular_velocity"][2])) > 1.0e-4, case.message(
        float(result["angular_velocity"][2]), "|angular velocity z| > 1e-4"
    )


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("stiffness", [100.0, 300.0])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_drive_01_stiffness_single_step(parameter_adapter, joint_type, authoring, stiffness):
    """DRIVE-01: Authored implicit stiffness reproduces the backend's analytical step."""
    result = parameter_adapter.run_single_dof_step(
        joint_type,
        authoring,
        stiffness=stiffness,
        damping=0.0,
        armature=0.0,
        position_target=Q_REF,
    )
    velocity, position = parameter_adapter.predict_dof_step(
        stiffness=stiffness,
        drive_damping=0.0,
        armature=0.0,
        position_target=Q_REF,
        body_inertia=result["body_inertia"],
    )
    _assert_step(parameter_adapter, "DRIVE-01", authoring, result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("damping", [20.0, 60.0])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_drive_02_damping_single_step(parameter_adapter, joint_type, authoring, damping):
    """DRIVE-02: Authored drive damping reproduces the backend's analytical step."""
    result = parameter_adapter.run_single_dof_step(
        joint_type,
        authoring,
        stiffness=100.0,
        damping=damping,
        armature=0.0,
        position_target=Q_REF,
    )
    velocity, position = parameter_adapter.predict_dof_step(
        stiffness=100.0,
        drive_damping=damping,
        armature=0.0,
        position_target=Q_REF,
        body_inertia=result["body_inertia"],
    )
    _assert_step(parameter_adapter, "DRIVE-02", authoring, result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("armature", [0.1, 0.5])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_joint_07_armature_single_step(parameter_adapter, joint_type, authoring, armature):
    """JOINT-07: Authored armature reproduces the backend's analytical step."""
    result = parameter_adapter.run_single_dof_step(
        joint_type,
        authoring,
        stiffness=100.0,
        damping=0.0,
        armature=armature,
        position_target=Q_REF,
    )
    velocity, position = parameter_adapter.predict_dof_step(
        stiffness=100.0,
        drive_damping=0.0,
        armature=armature,
        position_target=Q_REF,
        body_inertia=result["body_inertia"],
    )
    _assert_step(parameter_adapter, "JOINT-07", authoring, result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("passive_damping", [0.0, 3.0, 6.0])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_joint_08_passive_damping_usd_single_step(parameter_adapter, joint_type, passive_damping):
    """JOINT-08: USD-authored passive damping reproduces the backend's analytical step."""
    velocity_initial = 1.0
    result = parameter_adapter.run_single_dof_step(
        joint_type,
        "usd",
        stiffness=0.0,
        damping=0.0,
        armature=0.0,
        position_target=0.0,
        velocity=velocity_initial,
        passive_damping=passive_damping,
    )
    velocity, position = parameter_adapter.predict_dof_step(
        stiffness=0.0,
        drive_damping=0.0,
        armature=0.0,
        position_target=0.0,
        body_inertia=result["body_inertia"],
        velocity=velocity_initial,
        passive_damping=passive_damping,
    )
    _assert_step(
        parameter_adapter,
        "JOINT-08",
        "usd",
        result,
        velocity,
        position,
        velocity_initial=velocity_initial,
    )


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("parameter_adapter", _VELOCITY_LIMIT_BACKENDS, indirect=True)
def test_joint_05_velocity_limit_sustained_drive_and_braking(parameter_adapter, joint_type, authoring):
    """JOINT-05/FIX-LIMIT-VEL: the solver caps driven speed and brakes an over-limit state."""
    velocity_limit = 0.5
    driven = parameter_adapter.run_velocity_limit_probe(
        joint_type,
        authoring,
        velocity_limit=velocity_limit,
    )
    braking = parameter_adapter.run_velocity_limit_probe(
        joint_type,
        authoring,
        velocity_limit=velocity_limit,
        initial_velocity=2.0,
        driven=False,
    )
    case = _case(parameter_adapter, "JOINT-05", authoring)
    tolerance = 0.05
    assert driven["maximum_velocity"] <= velocity_limit + tolerance, case.message(
        driven["maximum_velocity"], f"maximum velocity <= {velocity_limit + tolerance}"
    )
    assert braking["final_velocity"] <= velocity_limit + tolerance, case.message(
        braking["final_velocity"], f"braked velocity <= {velocity_limit + tolerance}"
    )


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("parameter_adapter", _EFFORT_LIMIT_BACKENDS, indirect=True)
def test_joint_06_effort_limit_clamps_drive_response(parameter_adapter, joint_type, authoring):
    """JOINT-06/FIX-LIMIT-EFFORT: the solver clamps a drive command above its effort limit."""
    effort_limit = 5.0
    stiffness = 100.0
    result = parameter_adapter.run_single_dof_step(
        joint_type,
        authoring,
        stiffness=stiffness,
        damping=0.0,
        armature=0.0,
        position_target=Q_REF,
        effort_limit=effort_limit,
        dof_authoring="usd",
    )
    velocity, position = parameter_adapter.predict_dof_step(
        stiffness=stiffness,
        drive_damping=0.0,
        armature=0.0,
        position_target=Q_REF,
        body_inertia=result["body_inertia"],
        effort_limit=effort_limit,
    )
    _assert_step(parameter_adapter, "JOINT-06", authoring, result, velocity, position)


@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize(("parameter_adapter", "joint_type"), _FRICTION_CASES, indirect=["parameter_adapter"])
def test_joint_09_dry_friction_decelerates_unforced_motion(parameter_adapter, joint_type, authoring):
    """JOINT-09/FIX-PASSIVE: Newton dry friction applies its absolute opposing effort."""
    velocity_initial = 1.0
    friction = 1.0
    result = parameter_adapter.run_single_dof_step(
        joint_type,
        authoring,
        stiffness=0.0,
        damping=0.0,
        armature=0.0,
        position_target=0.0,
        velocity=velocity_initial,
        friction=friction,
        dof_authoring="usd",
    )
    friction_effort = predict_dry_friction_effort(velocity_initial, 0.0, friction)
    velocity, position = parameter_adapter.predict_dof_step(
        stiffness=0.0,
        drive_damping=0.0,
        armature=0.0,
        position_target=0.0,
        body_inertia=result["body_inertia"],
        effort=friction_effort,
        velocity=velocity_initial,
    )
    _assert_step(
        parameter_adapter,
        "JOINT-09",
        authoring,
        result,
        velocity,
        position,
        velocity_initial=velocity_initial,
    )


@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize(("parameter_adapter", "joint_type"), _FRICTION_CASES, indirect=["parameter_adapter"])
def test_joint_09_dry_friction_holds_at_rest_below_breakaway_effort(parameter_adapter, joint_type, authoring):
    """JOINT-09/FIX-PASSIVE: Newton dry friction cancels a sub-threshold effort at rest."""
    effort = 0.5
    friction = 1.0
    result = parameter_adapter.run_single_dof_step(
        joint_type,
        authoring,
        stiffness=0.0,
        damping=0.0,
        armature=0.0,
        position_target=0.0,
        effort=effort,
        friction=friction,
        dof_authoring="usd",
    )
    friction_effort = predict_dry_friction_effort(0.0, effort, friction)
    velocity, position = parameter_adapter.predict_dof_step(
        stiffness=0.0,
        drive_damping=0.0,
        armature=0.0,
        position_target=0.0,
        body_inertia=result["body_inertia"],
        effort=friction_effort,
    )
    _assert_step(parameter_adapter, "JOINT-09", authoring, result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("drive_authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("effort", [5.0, 20.0])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_cmd_01_feedforward_torque_implicit_single_step(parameter_adapter, joint_type, drive_authoring, effort):
    """CMD-01: Feed-forward effort with implicit dynamics reproduces the backend step."""
    result = parameter_adapter.run_single_dof_step(
        joint_type,
        drive_authoring,
        stiffness=100.0,
        damping=0.0,
        armature=0.0,
        position_target=0.0,
        effort=effort,
    )
    velocity, position = parameter_adapter.predict_dof_step(
        stiffness=100.0,
        drive_damping=0.0,
        armature=0.0,
        position_target=0.0,
        body_inertia=result["body_inertia"],
        effort=effort,
    )
    _assert_step(parameter_adapter, "CMD-01", "runtime", result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("drive_authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("effort", [5.0, 20.0])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_cmd_01_feedforward_torque_explicit_single_step(parameter_adapter, joint_type, drive_authoring, effort):
    """CMD-01: Feed-forward effort without joint dynamics reproduces the backend step."""
    result = parameter_adapter.run_single_dof_step(
        joint_type,
        drive_authoring,
        stiffness=0.0,
        damping=0.0,
        armature=0.0,
        position_target=0.0,
        effort=effort,
    )
    velocity, position = parameter_adapter.predict_dof_step(
        stiffness=0.0,
        drive_damping=0.0,
        armature=0.0,
        position_target=0.0,
        body_inertia=result["body_inertia"],
        effort=effort,
    )
    _assert_step(parameter_adapter, "CMD-01", "runtime", result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("effort", [5.0, 20.0])
def test_cmd_01_feedforward_torque_explicit_runtime_topology_error(kamino, joint_type, effort):
    """CMD-01: Kamino rejects adding dynamic-constraint topology at runtime."""
    with pytest.raises(RuntimeError, match="Changing dynamic constraint topology"):
        kamino.run_single_dof_step(
            joint_type,
            "runtime-error",
            stiffness=0.0,
            damping=0.0,
            armature=0.0,
            position_target=0.0,
            effort=effort,
        )


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("velocity_target", [2.0, 5.0])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_cmd_02_velocity_reference_single_step(parameter_adapter, joint_type, velocity_target):
    """CMD-02: A joint velocity target reproduces the analytical backend step."""
    damping = 40.0
    result = parameter_adapter.run_single_dof_step(
        joint_type,
        "cfg",
        stiffness=0.0,
        damping=damping,
        armature=0.0,
        position_target=0.0,
        velocity_target=velocity_target,
    )
    velocity, position = parameter_adapter.predict_dof_step(
        stiffness=0.0,
        drive_damping=damping,
        armature=0.0,
        position_target=0.0,
        body_inertia=result["body_inertia"],
        velocity_target=velocity_target,
    )
    # The cfg drive seed establishes VELOCITY mode; the target itself is a runtime command.
    _assert_step(parameter_adapter, "CMD-02", "runtime", result, velocity, position)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize(
    ("limit", "active_bound", "inactive_bound"),
    [
        ("lower", ACTIVE_LOWER, INACTIVE_LOWER),
        ("upper", ACTIVE_UPPER, INACTIVE_UPPER),
    ],
)
@pytest.mark.parametrize("bound_kind", ["active", "inactive"])
def test_joint_04_position_limit_runtime_topology_error(
    kamino, joint_type, limit, active_bound, inactive_bound, bound_kind
):
    """JOINT-04: Kamino rejects changing the existence of a joint limit at runtime."""
    bound = active_bound if bound_kind == "active" else inactive_bound
    with pytest.raises(RuntimeError, match="Changing the existence of a joint limit"):
        kamino.run_position_limit_probe(joint_type, "runtime-error", limit, bound)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "runtime"])
@pytest.mark.parametrize(
    ("limit", "active_bound", "inactive_bound"),
    [
        ("lower", ACTIVE_LOWER, INACTIVE_LOWER),
        ("upper", ACTIVE_UPPER, INACTIVE_UPPER),
    ],
)
@pytest.mark.parametrize("bound_kind", ["active", "inactive"])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_joint_04_position_limit(
    parameter_adapter, joint_type, authoring, limit, active_bound, inactive_bound, bound_kind
):
    """JOINT-04: Authored lower and upper position limits are enforced by the backend."""
    bound = active_bound if bound_kind == "active" else inactive_bound
    position_extreme, position_final = parameter_adapter.run_position_limit_probe(joint_type, authoring, limit, bound)
    is_active = abs(bound) < PROBE_TARGET
    if is_active:
        if limit == "upper":
            assert position_extreme <= bound + 0.03, "JOINT-04: active upper limit was exceeded"
        else:
            assert position_extreme >= bound - 0.03, "JOINT-04: active lower limit was exceeded"
        assert position_final == pytest.approx(bound, abs=0.03), (
            f"JOINT-04: joint did not settle at the active {limit} limit"
        )
    else:
        if limit == "upper":
            assert position_final > 0.35, "JOINT-04: inactive upper-limit control did not pass the low limit"
        else:
            assert position_final < -0.35, "JOINT-04: inactive lower-limit control did not pass the low limit"


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["cfg", "runtime"])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_joint_02_position_state(parameter_adapter, joint_type, authoring):
    """JOINT-02: Reset-default and live joint position survive one unforced step."""
    position = 0.15
    result = parameter_adapter.run_joint_state(
        joint_type,
        authoring,
        position=position,
        velocity=0.0,
    )
    case = _case(parameter_adapter, "JOINT-02", authoring)
    assert_physical_close(result["position_before"], position, case)
    assert_physical_close(result["velocity_before"], 0.0, case)
    assert_physical_close(result["position_after"], position, case)
    assert_physical_close(result["velocity_after"], 0.0, case)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["cfg", "runtime"])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_joint_03_velocity_state(parameter_adapter, joint_type, authoring):
    """JOINT-03: Reset-default and live joint velocity produce unforced coast motion."""
    velocity = 0.4
    result = parameter_adapter.run_joint_state(
        joint_type,
        authoring,
        position=0.0,
        velocity=velocity,
    )
    case = _case(parameter_adapter, "JOINT-03", authoring)
    assert_physical_close(result["position_before"], 0.0, case)
    assert_physical_close(result["velocity_before"], velocity, case)
    assert_physical_close(result["position_after"], velocity * PROFILE_DOF_DT, case)
    assert_physical_close(result["velocity_after"], velocity, case)


@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_body_03_center_of_mass_gravity_moment(parameter_adapter):
    """BODY-03: An authored COM offset produces the expected fixed-pivot gravity moment."""
    offset = (0.1, 0.0, 0.0)
    velocity, body_inertia = parameter_adapter.run_com_gravity_probe(offset)
    torque_z = offset[0] * JOINT_MASS["revolute"] * -9.81
    expected_velocity = torque_z * PROFILE_DOF_DT / body_inertia
    case = _case(parameter_adapter, "BODY-03", "usd")
    assert_physical_close(velocity, expected_velocity, case)

    control_velocity, _ = parameter_adapter.run_com_gravity_probe((0.0, 0.0, 0.0))
    assert_physical_close(control_velocity, 0.0, case)
