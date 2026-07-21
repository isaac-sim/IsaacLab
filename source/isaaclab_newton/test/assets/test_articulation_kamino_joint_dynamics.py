# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino joint-property dynamics integration tests.

These tests verify that joint properties (stiffness, damping, position limit, armature) authored
through the different supported paths -- the USD source model, the Python actuator config, and the
runtime ``write_joint_*_to_sim`` API -- all end up producing the *correct Kamino dynamics*. They also
verify the per-step joint commands that feed the same dynamics: the feed-forward joint torque
(:meth:`set_joint_effort_target_index`) and the joint velocity reference
(:meth:`set_joint_velocity_target_index`).

Rather than compare against long-term/period responses, we reconstruct a *single* Kamino
integration step analytically and compare it to the observed joint state after one ``sim.step()``.
This is possible because we build the articulation procedurally, so the effective joint-space
inertia is known (and read back from :attr:`data.mass_matrix`), and because the Kamino ``"euler"``
integrator applies the implicit joint PD as a per-DOF dynamic constraint whose converged solution
is closed-form for a fixed-base 1-DOF system on the constraint manifold (see :func:`_predict_from_rest`).

Test system: a procedurally built, fixed-base, single-DOF articulation (a base link fixed to the
world by a ``FixedJoint`` plus one moving link connected by a revolute or prismatic joint), mirroring
the structure of ``SimpleArticulation/revolute_articulation.usd``. Both joint types are tested
separately. Gravity is disabled and there are no collisions, so the joint reduces to a scalar ODE.

Notes on Kamino behavior exercised here:
- Kamino allocates the per-DOF implicit-dynamics constraint at *model-build* time whenever a joint
  has a non-zero stiffness/damping/armature imported from USD. A joint built with zero gains is not
  given a constraint, and later runtime writes to its gains are ignored. All tests therefore author a
  non-zero drive stiffness in USD so the constraint exists; the value under test is then supplied by
  the selected authoring path.
- Kamino honors USD-authored joint position limits but does not honor runtime limit writes, so the
  limit test exercises the USD authoring path only.
- Velocity/effort limits and joint friction are intentionally out of scope (Kamino does not enforce
  the velocity/effort limits in its step, and maps ``joint_friction`` to viscous damping).
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
    else:
        joint = UsdPhysics.PrismaticJoint.Define(stage, dof_joint)
        axis = "X"
        limit_conv = 1.0
    joint.CreateBody0Rel().SetTargets([base])
    joint.CreateBody1Rel().SetTargets([link])
    joint.CreateAxisAttr().Set(axis)
    if usd_lower is not None and usd_upper is not None:
        joint.CreateLowerLimitAttr().Set(float(usd_lower * limit_conv))
        joint.CreateUpperLimitAttr().Set(float(usd_upper * limit_conv))
    if usd_armature > 0.0:
        joint.GetPrim().CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float).Set(float(usd_armature))

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


def _predict_from_rest(
    *,
    ke: float,
    kd: float,
    a: float,
    q_ref: float,
    i_body: float,
    tau: float = 0.0,
    dq_ref: float = 0.0,
    dt: float = DT,
):
    """Analytic Kamino ``euler`` single step for a fixed-base 1-DOF joint starting from rest.

    The implicit joint dynamics use the effective joint inertia ``m_j = a + dt*kd + dt^2*ke`` and a
    bias velocity built from the total commanded joint torque -- the feed-forward torque ``tau``, the
    stiffness term ``ke*q_ref`` and the velocity-reference term ``kd*dq_ref``::

        dq_b = dt * (tau + ke * q_ref + kd * dq_ref) / m_j
        dq_plus = dq_b - dq_b * i_body / (m_j + i_body)
        q_plus = dt * dq_plus

    (Friction is out of scope and starting state is rest, so ``dq_minus = q_minus = 0``.)

    Returns the predicted ``(dq_plus, q_plus)``.
    """
    m_j = a + dt * kd + dt * dt * ke
    if m_j > 0.0:
        dq_b = dt * (tau + ke * q_ref + kd * dq_ref) / m_j
        dq_plus = dq_b - dq_b * i_body / (m_j + i_body)
    else:
        dq_plus = dt * tau / i_body
    return dq_plus, dt * dq_plus


def _authoring_spec(authoring: str, *, ke: float, kd: float, a: float):
    """Map an authoring path to (USD gains, actuator-cfg gains, runtime gains)."""
    if authoring == "usd":
        return dict(usd=(ke, kd, a), cfg=(None, None, None), runtime=None)
    if authoring == "cfg":
        return dict(usd=(0.0, 0.0, 0.0), cfg=(ke, kd, a), runtime=None)
    if authoring == "runtime":
        # Kamino only allocates the constraint if USD or cfg authoring triggers it.
        # We write a non-zero cfg stiffness so that the constraint is allocated.
        return dict(usd=(0.0, 0.0, 0.0), cfg=(BASE_KE, 0.0, 0.0), runtime=(ke, kd, a))
    raise ValueError(f"unknown authoring path: {authoring}")


def _single_step_from_rest(
    joint_type: str,
    authoring: str,
    *,
    ke: float,
    kd: float,
    a: float,
    q_ref: float = Q_REF,
    dq_ref: float = 0.0,
    tau: float = 0.0,
) -> dict:
    """Build the articulation, author gains via ``authoring``, and take one Kamino step from rest.

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
        _build_single_dof(joint_type, usd_ke=usd_ke, usd_kd=usd_kd, usd_armature=usd_a)
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
def test_stiffness_single_step(joint_type, authoring, ke):
    """Authored stiffness (USD / cfg / runtime) reproduces the analytic single Kamino step."""
    r = _single_step_from_rest(joint_type, authoring, ke=ke, kd=0.0, a=0.0)
    assert abs(r["q_minus"]) < 1e-5 and abs(r["dq_minus"]) < 1e-5, "articulation should start from rest"

    dq_pred, q_pred = _predict_from_rest(ke=ke, kd=0.0, a=0.0, q_ref=Q_REF, i_body=r["i_body"])
    assert r["dq_plus"] == pytest.approx(dq_pred, rel=_RTOL, abs=_ATOL)
    assert r["q_plus"] == pytest.approx(q_pred, rel=_RTOL, abs=_ATOL)


# ---------------------------------------------------------------------------
# Damping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("kd", [20.0, 60.0])
def test_damping_single_step(joint_type, authoring, kd):
    """Authored damping (USD / cfg / runtime) reproduces the analytic single Kamino step.

    Damping enters the implicit ``m_j`` denominator, so from rest with a fixed stiffness it measurably
    reduces the first-step velocity.
    """
    r = _single_step_from_rest(joint_type, authoring, ke=BASE_KE, kd=kd, a=0.0)
    assert abs(r["q_minus"]) < 1e-5 and abs(r["dq_minus"]) < 1e-5, "articulation should start from rest"

    dq_pred, q_pred = _predict_from_rest(ke=BASE_KE, kd=kd, a=0.0, q_ref=Q_REF, i_body=r["i_body"])
    assert r["dq_plus"] == pytest.approx(dq_pred, rel=_RTOL, abs=_ATOL)
    assert r["q_plus"] == pytest.approx(q_pred, rel=_RTOL, abs=_ATOL)


# ---------------------------------------------------------------------------
# Armature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("armature", [0.1, 0.5])
def test_armature_single_step(joint_type, authoring, armature):
    """Authored armature (USD / cfg / runtime) reproduces the analytic single Kamino step.

    Armature adds to the effective joint inertia ``m_j`` (not the body ``mass_matrix``), reducing the
    first-step velocity for a fixed stiffness.
    """
    r = _single_step_from_rest(joint_type, authoring, ke=BASE_KE, kd=0.0, a=armature)
    assert abs(r["q_minus"]) < 1e-5 and abs(r["dq_minus"]) < 1e-5, "articulation should start from rest"

    dq_pred, q_pred = _predict_from_rest(ke=BASE_KE, kd=0.0, a=armature, q_ref=Q_REF, i_body=r["i_body"])
    assert r["dq_plus"] == pytest.approx(dq_pred, rel=_RTOL, abs=_ATOL)
    assert r["q_plus"] == pytest.approx(q_pred, rel=_RTOL, abs=_ATOL)


# ---------------------------------------------------------------------------
# Feed-forward torque
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("tau", [5.0, 20.0])
def test_feedforward_torque_implicit_single_step(joint_type, authoring, tau):
    """A commanded feed-forward joint torque alongside implicit PD control reproduces the analytic single Kamino step.

    A non-zero implicit stiffness is present so Kamino allocates the joint's dynamic constraint.
    """
    r = _single_step_from_rest(joint_type, authoring, ke=BASE_KE, kd=0.0, a=0.0, q_ref=0.0, tau=tau)
    assert abs(r["q_minus"]) < 1e-5 and abs(r["dq_minus"]) < 1e-5, "articulation should start from rest"

    dq_pred, q_pred = _predict_from_rest(ke=BASE_KE, kd=0.0, a=0.0, q_ref=0.0, i_body=r["i_body"], tau=tau)
    assert r["dq_plus"] == pytest.approx(dq_pred, rel=_RTOL, abs=_ATOL)
    assert r["q_plus"] == pytest.approx(q_pred, rel=_RTOL, abs=_ATOL)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("tau", [5.0, 20.0])
def test_feedforward_torque_explicit_single_step(joint_type, authoring, tau):
    """A commanded feed-forward joint torque without any joint dynamicsreproduces the analytic single Kamino step."""
    r = _single_step_from_rest(joint_type, authoring, ke=0.0, kd=0.0, a=0.0, q_ref=0.0, tau=tau)
    assert abs(r["q_minus"]) < 1e-5 and abs(r["dq_minus"]) < 1e-5, "articulation should start from rest"

    dq_pred, q_pred = _predict_from_rest(ke=0.0, kd=0.0, a=0.0, q_ref=0.0, i_body=r["i_body"], tau=tau)
    assert r["dq_plus"] == pytest.approx(dq_pred, rel=_RTOL, abs=_ATOL)
    assert r["q_plus"] == pytest.approx(q_pred, rel=_RTOL, abs=_ATOL)


# ---------------------------------------------------------------------------
# Joint velocity reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("dq_ref", [2.0, 5.0])
def test_velocity_reference_single_step(joint_type, dq_ref):
    """A commanded joint velocity reference reproduces the analytic single Kamino step.

    With the position target at the rest configuration (``q_ref=0``) and a non-zero damping gain, the
    velocity reference enters the joint torque as ``kd*dq_ref``. It is delivered through
    :meth:`set_joint_velocity_target_index`; the effect vanishes at ``kd=0``, so a non-zero damping is
    required for the reference to matter.
    """
    kd = 40.0
    r = _single_step_from_rest(joint_type, "cfg", ke=BASE_KE, kd=kd, a=0.0, q_ref=0.0, dq_ref=dq_ref)
    assert abs(r["q_minus"]) < 1e-5 and abs(r["dq_minus"]) < 1e-5, "articulation should start from rest"

    dq_pred, q_pred = _predict_from_rest(ke=BASE_KE, kd=kd, a=0.0, q_ref=0.0, i_body=r["i_body"], dq_ref=dq_ref)
    assert r["dq_plus"] == pytest.approx(dq_pred, rel=_RTOL, abs=_ATOL)
    assert r["q_plus"] == pytest.approx(q_pred, rel=_RTOL, abs=_ATOL)


# ---------------------------------------------------------------------------
# Position limits (USD authoring path only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("upper", [0.3, 2.0])
def test_position_limit_usd(joint_type, upper):
    """A USD-authored upper limit is enforced by Kamino.

    Driving the joint toward a target beyond the limit clamps it at the limit (``upper=0.3``); with a
    far limit (``upper=2.0``) the same drive carries the joint well past ``0.3`` toward the target,
    confirming the clamp above was due to the limit and not the drive. A gentle, damped drive keeps
    the soft-constraint penetration small.

    TODO: Kamino honors USD-authored position limits but not runtime limit writes,
    so the limit test uses the USD path only. We should fix Kamino and update this test.
    """
    target = 0.45
    with build_simulation_context(device=DEVICE, sim_cfg=_sim_cfg(alpha=0.01, beta=0.01)) as sim:
        sim._app_control_on_stop_handle = None
        _build_single_dof(joint_type, usd_ke=30.0, usd_lower=-1.0, usd_upper=upper)
        art = Articulation(_make_cfg(30.0, 60.0, None))
        sim.reset()

        art.set_joint_position_target_index(target=torch.full((1, 1), target, device=DEVICE))
        q_max = -1.0e9
        for _ in range(600):
            art.write_data_to_sim()
            sim.step()
            art.update(DT)
            q_max = max(q_max, float(art.data.joint_pos.torch[0, 0]))
        q_final = float(art.data.joint_pos.torch[0, 0])

        if upper < target:
            # active limit: joint clamps at the limit and never overshoots it appreciably
            assert q_max <= upper + 0.03
            assert q_final == pytest.approx(upper, abs=0.03)
        else:
            # inactive limit: joint moves well past where the low limit would have clamped it
            assert q_final > 0.35
