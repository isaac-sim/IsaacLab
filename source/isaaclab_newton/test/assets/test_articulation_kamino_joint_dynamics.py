# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino joint-property dynamics integration tests.

These tests verify that joint properties (stiffness, drive damping, passive damping, position limit, armature) authored
through the different supported paths -- the USD source model, the Python actuator config, and the
runtime ``write_joint_*_to_sim`` API -- all end up producing the *correct Kamino dynamics*. They also
verify the per-step joint commands that feed the same dynamics: the feed-forward joint torque
(:meth:`set_joint_effort_target_index`) and the joint velocity reference
(:meth:`set_joint_velocity_target_index`).
"""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math

import pytest
import torch
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg

from pxr import Gf, Sdf, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.sim.schemas import UsdPhysicsDriveCfg, apply_drive

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Kamino solver tests require a CUDA device")

# Kamino tests run on GPU.
DEVICE = "cuda:0"
# Single physics step; small enough to keep the reconstruction accurate, large enough to be well above float noise.
DT = 1.0 / 120.0
# Position-target offset used by the single-step probes [rad for revolute, m for prismatic].
Q_REF = 0.2
# Base drive stiffness authored in USD so Kamino allocates the per-DOF dynamic constraint.
BASE_KE = 100.0
BASE_KD = 10.0
# Per-joint moving-link mass [kg] and diagonal inertia [kg m^2]; COM sits on the joint axis so the
# joint-space inertia equals the mass (prismatic) or the on-axis principal moment (revolute).
_MASS = {"revolute": 1.0, "prismatic": 2.0}
_INERTIA = (0.1, 0.1, 0.05)
_I_BODY_ANALYTIC = {"revolute": _INERTIA[2], "prismatic": _MASS["prismatic"]}
_RAD2DEG = 180.0 / math.pi

# Tolerances for the single-step reconstruction (float32 solve + P-ADMM residual tolerance of 1e-6).
_RTOL = 5e-3
_ATOL = 2e-4


def _sim_cfg(*, alpha: float = 0.0, beta: float = 0.0) -> SimulationCfg:
    """Kamino simulation config pinned for exact single-step reconstruction.

    ``integrator="euler"`` and ``num_substeps=1`` make one ``sim.step()`` a single semi-implicit
    Euler solver step. ``constraints_alpha=0`` disables bilateral Baumgarte so a step from a clean
    (on-manifold) reset is not perturbed; the limit test overrides it.
    """
    return SimulationCfg(
        dt=DT,
        gravity=(0.0, 0.0, 0.0),
        device=DEVICE,
        physics=NewtonCfg(
            solver_cfg=KaminoSolverCfg(integrator="euler", constraints_alpha=alpha, constraints_beta=beta),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )


def _build_single_dof(
    joint_type: str,
    *,
    usd_ke: float,
    usd_kd: float = 0.0,
    usd_armature: float = 0.0,
    usd_passive_damping: float = 0.0,
    usd_lower: float | None = None,
    usd_upper: float | None = None,
) -> None:
    """Author a fixed-base single-DOF articulation on the current stage under ``/World/Env_0/Robot``.

    The structure mirrors ``revolute_articulation.usd``: a ``base`` rigid body fixed to the world by a
    ``FixedJoint`` and a moving ``link`` connected to the base by a revolute/prismatic DOF joint. The
    moving link carries an explicit mass and diagonal inertia with its COM on the joint axis.
    """
    root = "/World/Env_0/Robot"
    base = f"{root}/base"
    link = f"{root}/link"
    stage = sim_utils.get_current_stage()

    sim_utils.create_prim("/World/Env_0", "Xform")
    sim_utils.create_prim(root, "Xform")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root))

    # base body, fixed to the world
    sim_utils.create_prim(base, "Sphere", attributes={"radius": 0.05})
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(base))
    base_mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(base))
    base_mass.CreateMassAttr().Set(1.0)
    base_mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1e-4, 1e-4, 1e-4))
    UsdPhysics.FixedJoint.Define(stage, f"{base}/FixedJoint").CreateBody0Rel().SetTargets([base])

    # moving link with a known mass and inertia (COM on the joint axis)
    sim_utils.create_prim(link, "Cube", attributes={"size": 0.2})
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(link))
    link_mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(link))
    link_mass.CreateMassAttr().Set(_MASS[joint_type])
    link_mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*_INERTIA))
    link_mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    dof_joint = f"{link}/joint"
    if joint_type == "revolute":
        joint = UsdPhysics.RevoluteJoint.Define(stage, dof_joint)
        axis = "Z"
        limit_conv = _RAD2DEG  # USD authors angular limits in degrees
        damping_conv = 1.0 / _RAD2DEG  # USD authors angular damping per degree
    else:
        joint = UsdPhysics.PrismaticJoint.Define(stage, dof_joint)
        axis = "X"
        limit_conv = 1.0
        damping_conv = 1.0
    joint.CreateBody0Rel().SetTargets([base])
    joint.CreateBody1Rel().SetTargets([link])
    joint.CreateAxisAttr().Set(axis)
    if usd_lower is not None and usd_upper is not None:
        joint.CreateLowerLimitAttr().Set(float(usd_lower * limit_conv))
        joint.CreateUpperLimitAttr().Set(float(usd_upper * limit_conv))
    if usd_armature > 0.0:
        joint.GetPrim().CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float).Set(float(usd_armature))
    joint.GetPrim().CreateAttribute("newton:damping", Sdf.ValueTypeNames.Float).Set(
        float(usd_passive_damping * damping_conv)
    )

    # apply_drive handles the rad->deg conversion of angular stiffness/damping; Newton converts back on import.
    apply_drive(
        UsdPhysicsDriveCfg(drive_type="force", stiffness=float(usd_ke), damping=float(usd_kd), max_force=1.0e9),
        dof_joint,
        stage,
    )


def _make_cfg(stiffness: float | None, damping: float | None, armature: float | None) -> ArticulationCfg:
    """Articulation config over the pre-built prims. ``None`` fields load from the USD joint prim."""
    return ArticulationCfg(
        prim_path="/World/Env_.*/Robot",
        spawn=None,
        actuators={
            "joint": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=stiffness, damping=damping, armature=armature
            )
        },
    )


def _predict_single_step(
    *,
    ke: float,
    kd: float,
    a: float,
    q_ref: float,
    i_body: float,
    tau: float = 0.0,
    dq_ref: float = 0.0,
    q: float = 0.0,
    dq: float = 0.0,
    passive_damping: float = 0.0,
    dt: float = DT,
):
    """Analytic Kamino ``euler`` single step for a fixed-base 1-DOF joint.

    The implicit joint dynamics add armature, drive damping, passive damping, and stiffness to the
    effective joint inertia. The resulting update is::

        m_eff = i_body + a + dt * (kd + passive_damping) + dt**2 * ke
        dq_plus = ((i_body + a) * dq + dt * (tau + ke * (q_ref - q) + kd * dq_ref)) / m_eff
        q_plus = q + dt * dq_plus

    Passive damping is therefore dynamically equivalent to drive damping with a zero velocity
    reference, while remaining a distinct authored model property.

    Returns the predicted ``(dq_plus, q_plus)``.
    """
    m_eff = i_body + a + dt * (kd + passive_damping) + dt * dt * ke
    drive_effort = tau + ke * (q_ref - q) + kd * dq_ref
    dq_plus = ((i_body + a) * dq + dt * drive_effort) / m_eff
    return dq_plus, q + dt * dq_plus


def _authoring_spec(authoring: str, *, ke: float, kd: float, a: float):
    """Map an authoring path to (USD gains, actuator-cfg gains, runtime gains)."""
    # We need a non-zero USD gain to trigger the right actuation type in cfg and runtime mode.
    # TODO: Change once https://github.com/isaac-sim/IsaacLab/issues/6649 is resolved.
    usd_ke = BASE_KE if ke > 0.0 else 0.0
    usd_kd = BASE_KD if kd > 0.0 else 0.0

    if authoring == "usd":
        return dict(usd=(ke, kd, a), cfg=(None, None, None), runtime=None)
    if authoring == "cfg":
        return dict(usd=(usd_ke, usd_kd, 0.0), cfg=(ke, kd, a), runtime=None)
    if authoring == "runtime":
        # Kamino only allocates the constraint if USD or cfg authoring triggers it.
        # A USD or cfg value must be non-zero to trigger the constraint allocation.
        usd_a = a / 10.0
        return dict(usd=(usd_ke, usd_kd, usd_a), cfg=(None, None, None), runtime=(ke, kd, a))
    if authoring == "runtime-error":
        # Set up for an error case where the runtime write tries to change the dynamic constraint topology.
        if ke == 0.0 and kd == 0.0 and a == 0.0:
            return dict(usd=(BASE_KE, 0.0, 0.0), cfg=(None, None, None), runtime=(0.0, 0.0, 0.0))
        else:
            return dict(usd=(0.0, 0.0, 0.0), cfg=(None, None, None), runtime=(ke, kd, a))
    raise ValueError(f"unknown authoring path: {authoring}")


def _single_step(
    joint_type: str,
    authoring: str,
    *,
    ke: float,
    kd: float,
    a: float,
    q_ref: float = Q_REF,
    dq_ref: float = 0.0,
    tau: float = 0.0,
    q_initial: float = 0.0,
    dq_initial: float = 0.0,
    passive_damping: float = 0.0,
) -> dict:
    """Build the articulation, author properties via ``authoring``, and take one Kamino step.

    The per-step command inputs -- position target ``q_ref``, velocity reference ``dq_ref`` and
    feed-forward torque ``tau`` -- are applied through the standard actuator command API.

    Returns a dict with the read-back effective inertia and the pre/post joint state.
    """
    spec = _authoring_spec(authoring, ke=ke, kd=kd, a=a)
    usd_ke, usd_kd, usd_a = spec["usd"]
    cfg_ke, cfg_kd, cfg_a = spec["cfg"]

    with build_simulation_context(device=DEVICE, sim_cfg=_sim_cfg()) as sim:
        sim._app_control_on_stop_handle = None

        # This sequence replicate the IsaacLab asset authoring and simulation pipeline.
        # 1. Read the USD asset. Here, we build the articulation procedurally.
        _build_single_dof(
            joint_type,
            usd_ke=usd_ke,
            usd_kd=usd_kd,
            usd_armature=usd_a,
            usd_passive_damping=passive_damping,
        )
        # 2. Author the gains via the Python actuator config.
        art = Articulation(_make_cfg(cfg_ke, cfg_kd, cfg_a))
        # 3. Reset the simulation. This call sets up the Kamino solver and model.
        sim.reset()
        art.update(0.0)

        i_body = float(art.data.mass_matrix.torch[0, 0, 0])

        if spec["runtime"] is not None:
            r_ke, r_kd, r_a = spec["runtime"]
            art.write_joint_stiffness_to_sim_index(stiffness=r_ke)
            art.write_joint_damping_to_sim_index(damping=r_kd)
            art.write_joint_armature_to_sim_index(armature=r_a)

        art.write_joint_position_to_sim_index(position=torch.full((1, 1), q_initial, device=DEVICE))
        art.write_joint_velocity_to_sim_index(velocity=torch.full((1, 1), dq_initial, device=DEVICE))
        art.set_joint_position_target_index(target=torch.full((1, 1), q_ref, device=DEVICE))
        art.set_joint_velocity_target_index(target=torch.full((1, 1), dq_ref, device=DEVICE))
        art.set_joint_effort_target_index(target=torch.full((1, 1), tau, device=DEVICE))
        art.update(0.0)
        q_minus = float(art.data.joint_pos.torch[0, 0])
        dq_minus = float(art.data.joint_vel.torch[0, 0])

        art.write_data_to_sim()
        sim.step()
        art.update(DT)
        return {
            "i_body": i_body,
            "q_minus": q_minus,
            "dq_minus": dq_minus,
            "q_plus": float(art.data.joint_pos.torch[0, 0]),
            "dq_plus": float(art.data.joint_vel.torch[0, 0]),
        }


def _assert_step_matches(
    parameter_id: str,
    result: dict,
    dq_expected: float,
    q_expected: float,
    *,
    q_initial: float = 0.0,
    dq_initial: float = 0.0,
) -> None:
    """Assert one-step state and dynamics with parameter traceability."""
    assert result["q_minus"] == pytest.approx(q_initial, rel=_RTOL, abs=_ATOL), (
        f"{parameter_id}: articulation did not start at the requested joint position"
    )
    assert result["dq_minus"] == pytest.approx(dq_initial, rel=_RTOL, abs=_ATOL), (
        f"{parameter_id}: articulation did not start at the requested joint velocity"
    )
    assert result["dq_plus"] == pytest.approx(dq_expected, rel=_RTOL, abs=_ATOL), (
        f"{parameter_id}: joint velocity does not match the analytic Kamino step"
    )
    assert result["q_plus"] == pytest.approx(q_expected, rel=_RTOL, abs=_ATOL), (
        f"{parameter_id}: joint position does not match the analytic Kamino step"
    )


# ---------------------------------------------------------------------------
# Model construction / effective-inertia sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
def test_build_yields_fixed_base_single_dof(joint_type):
    """The procedural asset imports as a fixed-base 1-DOF articulation with the expected inertia."""
    with build_simulation_context(device=DEVICE, sim_cfg=_sim_cfg()) as sim:
        sim._app_control_on_stop_handle = None
        _build_single_dof(joint_type, usd_ke=BASE_KE)
        art = Articulation(_make_cfg(BASE_KE, 0.0, None))
        sim.reset()
        art.update(0.0)

        assert art.is_initialized
        assert art.num_joints == 1
        assert art.is_fixed_base

        i_body = float(art.data.mass_matrix.torch[0, 0, 0])
        assert i_body == pytest.approx(_I_BODY_ANALYTIC[joint_type], rel=1e-3)


# ---------------------------------------------------------------------------
# Stiffness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("ke", [100.0, 300.0])
def test_drive_01_stiffness_single_step(joint_type, authoring, ke):
    """DRIVE-01: Authored implicit stiffness reproduces the analytic single Kamino step."""
    r = _single_step(joint_type, authoring, ke=ke, kd=0.0, a=0.0)

    dq_pred, q_pred = _predict_single_step(ke=ke, kd=0.0, a=0.0, q_ref=Q_REF, i_body=r["i_body"])
    _assert_step_matches("DRIVE-01", r, dq_pred, q_pred)


# ---------------------------------------------------------------------------
# Damping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("kd", [20.0, 60.0])
def test_drive_02_damping_single_step(joint_type, authoring, kd):
    """DRIVE-02: Authored implicit drive damping reproduces the analytic single Kamino step.

    This is the drive damping gain, not the passive damping classified as ``JOINT-08``. It enters the
    implicit ``m_j`` denominator, so from rest with a fixed stiffness it measurably reduces the
    first-step velocity.
    """
    r = _single_step(joint_type, authoring, ke=BASE_KE, kd=kd, a=0.0)

    dq_pred, q_pred = _predict_single_step(ke=BASE_KE, kd=kd, a=0.0, q_ref=Q_REF, i_body=r["i_body"])
    _assert_step_matches("DRIVE-02", r, dq_pred, q_pred)


# ---------------------------------------------------------------------------
# Armature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("armature", [0.1, 0.5])
def test_joint_07_armature_single_step(joint_type, authoring, armature):
    """JOINT-07: Authored armature reproduces the analytic single Kamino step.

    Armature adds to the effective joint inertia ``m_j`` (not the body ``mass_matrix``), reducing the
    first-step velocity for a fixed stiffness.
    """
    r = _single_step(joint_type, authoring, ke=BASE_KE, kd=0.0, a=armature)

    dq_pred, q_pred = _predict_single_step(ke=BASE_KE, kd=0.0, a=armature, q_ref=Q_REF, i_body=r["i_body"])
    _assert_step_matches("JOINT-07", r, dq_pred, q_pred)


# ---------------------------------------------------------------------------
# Passive damping (USD authoring path only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("passive_damping", [0.0, 3.0, 6.0])
def test_joint_08_passive_damping_usd_single_step(joint_type, passive_damping):
    """JOINT-08: USD-authored passive damping reproduces the analytic single Kamino step.

    This is Newton's passive ``joint_damping`` property, not the implicit drive damping classified as
    ``DRIVE-02``. It is dynamically equivalent to drive damping with a zero velocity reference. The
    damping value is scaled by the known effective joint inertia so both joint types exercise the
    same nominal decay rate.

    Expand to other authoring paths once https://github.com/isaac-sim/IsaacLab/issues/6517 is resolved.
    """
    dq_initial = 1.0

    r = _single_step(
        joint_type,
        "usd",
        ke=0.0,
        kd=0.0,
        a=0.0,
        q_ref=0.0,
        dq_initial=dq_initial,
        passive_damping=passive_damping,
    )
    dq_pred, q_pred = _predict_single_step(
        ke=0.0,
        kd=0.0,
        a=0.0,
        q_ref=0.0,
        i_body=r["i_body"],
        dq=dq_initial,
        passive_damping=passive_damping,
    )
    _assert_step_matches(
        "JOINT-08",
        r,
        dq_pred,
        q_pred,
        dq_initial=dq_initial,
    )


# ---------------------------------------------------------------------------
# Feed-forward torque
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("tau", [5.0, 20.0])
def test_cmd_01_feedforward_torque_implicit_single_step(joint_type, authoring, tau):
    """CMD-01: Feed-forward effort with implicit PD reproduces the analytic single Kamino step.

    A non-zero implicit stiffness is present so Kamino allocates the joint's dynamic constraint.
    """
    r = _single_step(joint_type, authoring, ke=BASE_KE, kd=0.0, a=0.0, q_ref=0.0, tau=tau)

    dq_pred, q_pred = _predict_single_step(ke=BASE_KE, kd=0.0, a=0.0, q_ref=0.0, i_body=r["i_body"], tau=tau)
    _assert_step_matches("CMD-01", r, dq_pred, q_pred)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime", "runtime-error"])
@pytest.mark.parametrize("tau", [5.0, 20.0])
def test_cmd_01_feedforward_torque_explicit_single_step(joint_type, authoring, tau):
    """CMD-01: Feed-forward effort without joint dynamics reproduces the analytic single Kamino step.

    Runtime authoring starts from an implicit-stiffness configuration and then clears its stiffness.
    Kamino must reject that dynamic-constraint topology change instead of silently rebuilding.
    """
    if authoring == "runtime-error":
        with pytest.raises(RuntimeError, match="Changing dynamic constraint topology"):
            _single_step(joint_type, authoring, ke=0.0, kd=0.0, a=0.0, q_ref=0.0, tau=tau)
        return

    r = _single_step(joint_type, authoring, ke=0.0, kd=0.0, a=0.0, q_ref=0.0, tau=tau)

    dq_pred, q_pred = _predict_single_step(ke=0.0, kd=0.0, a=0.0, q_ref=0.0, i_body=r["i_body"], tau=tau)
    _assert_step_matches("CMD-01", r, dq_pred, q_pred)


# ---------------------------------------------------------------------------
# Joint velocity reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("dq_ref", [2.0, 5.0])
def test_cmd_02_velocity_reference_single_step(joint_type, dq_ref):
    """CMD-02: A joint velocity target reproduces the analytic single Kamino step.

    Zero stiffness and non-zero damping make Newton import the drive in velocity mode. With the position
    target at the rest configuration (``q_ref=0``), the velocity reference enters the joint torque as
    ``kd*dq_ref``. It is delivered through :meth:`set_joint_velocity_target_index`.
    """
    kd = 40.0
    r = _single_step(joint_type, "cfg", ke=0.0, kd=kd, a=0.0, q_ref=0.0, dq_ref=dq_ref)

    dq_pred, q_pred = _predict_single_step(ke=0.0, kd=kd, a=0.0, q_ref=0.0, i_body=r["i_body"], dq_ref=dq_ref)
    _assert_step_matches("CMD-02", r, dq_pred, q_pred)


# ---------------------------------------------------------------------------
# Position limits
# ---------------------------------------------------------------------------

_PROBE_TARGET = 0.45
_LIMIT_LOWER = -1.0
_ACTIVE_UPPER = 0.3
_INACTIVE_UPPER = 2.0


def _position_limit_authoring_spec(authoring: str, upper: float) -> dict:
    """Map an authoring path to USD limits and runtime limits."""
    if authoring == "usd":
        return dict(usd_lower=_LIMIT_LOWER, usd_upper=upper, runtime_limit=None)
    if authoring == "runtime":
        # USD carries the opposite probe limit; the runtime write selects the limit under test.
        usd_upper = _INACTIVE_UPPER if upper < _PROBE_TARGET else _ACTIVE_UPPER
        return dict(usd_lower=_LIMIT_LOWER, usd_upper=usd_upper, runtime_limit=(_LIMIT_LOWER, upper))
    if authoring == "runtime-error":
        return dict(usd_lower=None, usd_upper=None, runtime_limit=(_LIMIT_LOWER, upper))
    raise ValueError(f"unknown authoring path: {authoring}")


def _assert_position_limit_behavior(upper: float, q_max: float, q_final: float) -> None:
    """Assert active/inactive upper-limit trajectories for FIX-LIMIT-POS."""
    if upper < _PROBE_TARGET:
        assert q_max <= upper + 0.03, "JOINT-04: active upper limit was exceeded"
        assert q_final == pytest.approx(upper, abs=0.03), "JOINT-04: joint did not settle at the active upper limit"
    else:
        assert q_final > 0.35, "JOINT-04: inactive-limit control did not pass the low-limit position"


def _run_position_limit_probe(joint_type: str, authoring: str, upper: float) -> tuple[float, float]:
    """Drive toward ``_PROBE_TARGET`` and return ``(q_max, q_final)``."""
    spec = _position_limit_authoring_spec(authoring, upper)
    with build_simulation_context(device=DEVICE, sim_cfg=_sim_cfg(alpha=0.01, beta=0.01)) as sim:
        sim._app_control_on_stop_handle = None
        _build_single_dof(
            joint_type,
            usd_ke=30.0,
            usd_lower=spec["usd_lower"],
            usd_upper=spec["usd_upper"],
        )
        art = Articulation(_make_cfg(30.0, 60.0, None))
        sim.reset()

        if spec["runtime_limit"] is not None:
            lower, runtime_upper = spec["runtime_limit"]
            art.write_joint_position_limit_to_sim_index(
                limits=torch.tensor([[[lower, runtime_upper]]], device=DEVICE)
            )

        art.set_joint_position_target_index(target=torch.full((1, 1), _PROBE_TARGET, device=DEVICE))
        q_max = -1.0e9
        for _ in range(600):
            art.write_data_to_sim()
            sim.step()
            art.update(DT)
            q_max = max(q_max, float(art.data.joint_pos.torch[0, 0]))
        return q_max, float(art.data.joint_pos.torch[0, 0])


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "runtime", "runtime-error"])
@pytest.mark.parametrize("upper", [_ACTIVE_UPPER, _INACTIVE_UPPER])
def test_joint_04_position_limit(joint_type, authoring, upper):
    """JOINT-04: Authored upper position limits are enforced by Kamino.

    Driving the joint toward a target beyond the limit clamps it at the limit (``upper=0.3``); with a
    far limit (``upper=2.0``) the same drive carries the joint well past ``0.3`` toward the target,
    confirming the clamp above was due to the limit and not the drive. A gentle, damped drive keeps
    the soft-constraint penetration small.

    USD and runtime :meth:`~isaaclab.assets.Articulation.write_joint_position_limit_to_sim_index` paths
    are covered. Runtime writes that introduce limits on a previously unlimited joint must raise instead
    of silently changing Kamino joint-limit topology.
    """
    if authoring == "runtime-error":
        with pytest.raises(RuntimeError, match="Changing the existence of a joint limit"):
            _run_position_limit_probe(joint_type, authoring, upper)
        return

    q_max, q_final = _run_position_limit_probe(joint_type, authoring, upper)
    _assert_position_limit_behavior(upper, q_max, q_final)
