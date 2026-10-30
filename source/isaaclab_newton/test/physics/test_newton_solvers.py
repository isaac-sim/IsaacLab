# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton solver configurations and the Isaac Lab solvers they select."""

from __future__ import annotations

import functools
import logging
from types import SimpleNamespace

import isaaclab_newton.physics.mjwarp_solver as mjwarp_solver_module
import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.cloner import newton_physics_replicate
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    KaminoDVICfg,
    KaminoDVISolverCfg,
    KaminoDynamicsCfg,
    KaminoPADMMCfg,
    KaminoPADMMSolverCfg,
    MJWarpSolver,
    MJWarpSolverCfg,
    MPMSolver,
    MPMSolverCfg,
    NewtonCfg,
    NewtonManager,
    VBDSolver,
    VBDSolverCfg,
    XPBDSolverCfg,
)
from isaaclab_newton.physics.mpm_solver import _make_solver_config, implicit_mpm_solvers, mpm_supports_graph_capture
from isaaclab_newton.physics.newton_builder import register_builder_attributes
from isaaclab_newton.physics.newton_solver import solver_kwargs
from newton import ModelBuilder
from newton.solvers import SolverImplicitMPM, SolverMuJoCo, SolverXPBD

from isaaclab.sim import SimulationContext


def _model_stub(state_attributes=()) -> SimpleNamespace:
    """Model stand-in carrying what solver construction reads before the Newton solver exists."""
    return SimpleNamespace(
        get_requested_state_attributes=lambda: list(state_attributes), articulation_start=None, articulation_end=None
    )


def test_solver_kwargs_include_newton_deterministic_mode():
    """Solver construction receives the mode configured on the outer Newton config."""
    kwargs = solver_kwargs(SolverXPBD, XPBDSolverCfg(), wp.DeterministicMode.GPU_TO_GPU)

    assert kwargs["deterministic"] == wp.DeterministicMode.GPU_TO_GPU


@pytest.mark.parametrize(
    ("solver_cfg", "active", "inactive"),
    [
        (MJWarpSolverCfg(), "mujoco:condim", ("kamino:max_solver_iterations", "mpm:young_modulus")),
        (KaminoPADMMSolverCfg(), "kamino:max_solver_iterations", ("mujoco:condim", "mpm:young_modulus")),
        (MPMSolverCfg(), "mpm:young_modulus", ("mujoco:condim", "kamino:max_solver_iterations")),
    ],
)
def test_solver_registers_only_its_builder_attributes(solver_cfg, active, inactive):
    """A solver declares its own builder schema once and no inactive solver schema."""
    builder = ModelBuilder()

    register_builder_attributes(builder, solver_cfg)
    register_builder_attributes(builder, solver_cfg)

    assert builder.has_custom_attribute(active)
    assert all(not builder.has_custom_attribute(name) for name in inactive)
    if isinstance(solver_cfg, MJWarpSolverCfg):
        assert {"mujoco:actuator", "mujoco:tendon"} <= set(builder.custom_frequencies)


# ----- USD import --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("solver_cfg", "expected_damping"),
    [pytest.param(MJWarpSolverCfg(), 0.23, id="mjwarp"), pytest.param(FeatherstoneSolverCfg(), 0.0, id="featherstone")],
)
def test_replicated_joint_properties_follow_schema_precedence(monkeypatch, solver_cfg, expected_damping):
    """Newton attributes win over PhysX, PhysX over MuJoCo, and only MJWarp imports the MuJoCo fallback."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    root_path = "/Sources/robot"
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    root = UsdGeom.Cube.Define(stage, root_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Cube.Define(stage, f"{root_path}/child").GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root_path}/child/joint")
    joint.CreateAxisAttr().Set("Z")
    joint.CreateBody0Rel().SetTargets([root_path])
    joint.CreateBody1Rel().SetTargets([f"{root_path}/child"])
    for name, value in (("mjc:frictionloss", 0.11), ("mjc:damping", 0.23), ("mjc:armature", 0.12)):
        joint.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Double, True).Set(value)
    joint.GetPrim().CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float, True).Set(0.21)
    joint.GetPrim().CreateAttribute("newton:friction", Sdf.ValueTypeNames.Double, True).Set(0.31)

    physics_cfg = NewtonCfg(solver_cfg=solver_cfg, load_visual_shapes=False)
    sim = object.__new__(SimulationContext)
    sim.cfg = SimpleNamespace(physics=physics_cfg, physics_prim_path="/physicsScene", device="cpu")
    sim.backend_manager, sim._backend_registry = NewtonManager(physics_cfg), []
    sim.backend_manager._sim, sim.backend_manager._device = sim, "cpu"
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim)

    builder, _ = newton_physics_replicate(
        stage=stage,
        sources=(root_path,),
        destinations=("/World/envs/env_{}/robot",),
        env_ids=np.array([0], dtype=np.int64),
        mapping=np.ones((1, 1), dtype=np.bool_),
    )
    model = builder.finalize(device="cpu")

    assert model.joint_friction.numpy()[-1] == pytest.approx(0.31)
    assert model.joint_damping.numpy()[-1] == pytest.approx(expected_damping)
    assert model.joint_armature.numpy()[-1] == pytest.approx(0.21)


# ----- MuJoCo Warp -------------------------------------------------------------------------


@pytest.fixture
def recorded_mujoco_kwargs(monkeypatch) -> list[dict]:
    """Replace Newton's MuJoCo solver with one recording its constructor arguments."""
    constructed = []
    record = functools.wraps(SolverMuJoCo.__init__)(lambda self, model, **kwargs: constructed.append(kwargs))
    monkeypatch.setattr(MJWarpSolver, "solver_class", type("RecordingSolver", (), {"__init__": record}))
    return constructed


def test_mjwarp_determinism_disables_sensors_unless_a_sensor_needs_them(recorded_mujoco_kwargs):
    """A guarantee builds MJWarp without its sensor stage, and refuses sensors that need that stage."""
    solver_cfg = MJWarpSolverCfg()

    MJWarpSolver.build(solver_cfg, _model_stub(), wp.DeterministicMode.NOT_GUARANTEED)
    MJWarpSolver.build(solver_cfg, _model_stub(), wp.DeterministicMode.RUN_TO_RUN)
    MJWarpSolver.build(solver_cfg, _model_stub({"body_qdd"}), wp.DeterministicMode.NOT_GUARANTEED)
    for attr in ("body_qdd", "body_parent_f"):
        with pytest.raises(ValueError, match="does not support deterministic physics"):
            MJWarpSolver.build(solver_cfg, _model_stub({attr}), wp.DeterministicMode.RUN_TO_RUN)

    assert [kwargs["disable_sensors"] for kwargs in recorded_mujoco_kwargs] == [False, True, False]
    assert solver_cfg.disable_sensors is False


def test_mujoco_cpu_reports_an_explicit_mode_instead_of_applying_it(caplog):
    """MuJoCo on the CPU is reproducible on its own, so a requested mode is reported and not applied."""
    with caplog.at_level(logging.INFO, logger=mjwarp_solver_module.__name__):
        mode = MJWarpSolver.applied_deterministic_mode(
            MJWarpSolverCfg(use_mujoco_cpu=True), wp.DeterministicMode.RUN_TO_RUN
        )

    assert mode == wp.DeterministicMode.NOT_GUARANTEED
    assert any("already reproducible" in record.getMessage() for record in caplog.records)


# ----- Kamino ------------------------------------------------------------------------------

_KAMINO_PADMM_FIELD_VALUES = [
    ("max_iterations", 13),
    ("primal_tolerance", 1.0e-5),
    ("dual_tolerance", 1.0e-5),
    ("compl_tolerance", 1.0e-5),
    ("restart_tolerance", 0.5),
    ("rho_0", 0.5),
    ("rho_min", 1.0e-4),
    ("a_0", 0.5),
    ("alpha", 11.0),
    ("tau", 1.6),
    ("eta", 1.0e-4),
    ("penalty_update_freq", 2),
    ("penalty_update_method", "balanced"),
    ("linear_solver_tolerance", 1.0e-3),
    ("linear_solver_tolerance_ratio", 0.1),
    ("use_acceleration", False),
    ("use_graph_conditionals", False),
    ("warmstart_mode", "none"),
    ("contact_warmstart_method", "geom_pair_net_force"),
]

_KAMINO_DVI_FIELD_VALUES = [
    ("tolerance", 1.0e-4),
    ("regularization", 1.0e-5),
    ("omega", 1.5),
    ("max_alternating_iterations", 15),
    ("inequality_sweeps_per_iteration", 2),
    ("bilateral_solve_interval", 2),
    ("bilateral_solver_type", "LLTBRCM"),
    ("bilateral_solver_kwargs", {"block_size": 32}),
    ("warmstart_mode", "internal"),
    ("contact_warmstart_method", "geom_pair_net_force"),
]

_KAMINO_DYNAMICS_FIELD_VALUES = [
    ("preconditioning", False),
    ("linear_solver_type", "LLTBRCM"),
    ("linear_solver_kwargs", {"maxiter": 9}),
]


def _assert_forwarded(config, field_values: list[tuple[str, object]]) -> None:
    for field_name, value in field_values:
        assert hasattr(config, field_name), f"{field_name!r} disappeared from {type(config).__name__}."
        assert getattr(config, field_name) == value


def test_kamino_padmm_cfg_forwards_its_fields_and_not_isaac_lab_metadata():
    """Every tunable P-ADMM and dynamics field reaches Newton's config; Isaac Lab-only fields do not."""
    solver_cfg = KaminoPADMMSolverCfg(
        sparse_jacobian=True,
        sparse_dynamics=True,
        dynamics=KaminoDynamicsCfg(**dict(_KAMINO_DYNAMICS_FIELD_VALUES[1:])),
        dynamics_solver_cfg=KaminoPADMMCfg(**dict(_KAMINO_PADMM_FIELD_VALUES)),
        solver_type="isaaclab_metadata_should_not_forward",
        max_contacts_per_world=32,
    )

    newton_cfg = solver_cfg.to_solver_config()

    assert newton_cfg.dynamics_solver == "padmm"
    _assert_forwarded(newton_cfg.padmm, _KAMINO_PADMM_FIELD_VALUES)
    _assert_forwarded(newton_cfg.dynamics, _KAMINO_DYNAMICS_FIELD_VALUES[1:])
    assert not any(hasattr(newton_cfg, name) for name in ("class_type", "solver_type", "max_contacts_per_world"))


def test_kamino_dvi_cfg_forwards_its_fields():
    """Every tunable DVI field, and disabled preconditioning, reaches Newton's config."""
    solver_cfg = KaminoDVISolverCfg(
        dynamics=KaminoDynamicsCfg(preconditioning=False),
        dynamics_solver_cfg=KaminoDVICfg(**dict(_KAMINO_DVI_FIELD_VALUES)),
    )

    newton_cfg = solver_cfg.to_solver_config()

    assert newton_cfg.dynamics_solver == "dvi"
    _assert_forwarded(newton_cfg.dvi, _KAMINO_DVI_FIELD_VALUES)
    _assert_forwarded(newton_cfg.dynamics, _KAMINO_DYNAMICS_FIELD_VALUES[:1])


@pytest.mark.parametrize(
    ("solver_cfg", "match"),
    [
        pytest.param(
            KaminoPADMMSolverCfg(dynamics_solver_cfg=KaminoPADMMCfg(penalty_update_method="balanced")),
            "sparse_dynamics=True",
            id="padmm_adaptive_penalties_dense",
        ),
        pytest.param(
            KaminoDVISolverCfg(dynamics=KaminoDynamicsCfg(preconditioning=True)), "preconditioning", id="dvi_precond"
        ),
    ],
)
def test_kamino_rejects_incompatible_settings(solver_cfg, match):
    with pytest.raises(ValueError, match=match):
        solver_cfg.to_solver_config()


# ----- Implicit MPM ------------------------------------------------------------------------

_MPM_FIELD_VALUES = [
    ("max_iterations", 13),
    ("tolerance", 5.0e-5),
    ("solver", "gauss-seidel"),
    ("warmstart_mode", "particles"),
    ("collider_velocity_mode", "backward"),
    ("voxel_size", 0.0375),
    ("grid_type", "dense"),
    ("grid_padding", 4),
    ("max_active_cell_count", 1024),
    ("max_leaf_node_count", 512),
    ("max_lower_node_count", 128),
    ("max_upper_node_count", 32),
    ("separate_worlds", True),
    ("transfer_scheme", "pic"),
    ("integration_scheme", "gimp"),
    ("critical_fraction", 0.25),
    ("air_drag", 0.5),
    ("collider_normal_from_sdf_gradient", True),
    ("collider_basis", "Q1"),
    ("strain_basis", "P1d"),
    ("velocity_basis", "B2"),
]

_MPM_PARTICLE_ATTRIBUTES = {
    "mpm:viscosity": 50.0,
    "mpm:friction": 0.0,
    "mpm:tensile_yield_ratio": 1.0,
    "mpm:yield_pressure": 1.0e15,
    "mpm:yield_stress": 0.0,
    "mpm:young_modulus": 1.0e15,
    "mpm:damping": 0.0,
}


def test_mpm_cfg_forwards_every_solver_field_and_not_isaac_lab_metadata():
    """Every tunable MPM field reaches ``SolverImplicitMPM.Config``; Isaac Lab-only fields do not."""
    newton_cfg = _make_solver_config(MPMSolverCfg(**dict(_MPM_FIELD_VALUES)))

    _assert_forwarded(newton_cfg, _MPM_FIELD_VALUES)
    assert not any(hasattr(newton_cfg, name) for name in ("class_type", "solver_type", "project_outside_colliders"))


@pytest.mark.parametrize(
    "mode, expected, deprecated",
    [("instantaneous", "forward", True), ("finite_difference", "backward", True), ("forward", "forward", False)],
)
def test_mpm_cfg_translates_deprecated_collider_velocity_modes(mode, expected, deprecated, recwarn):
    """Deprecated collider velocity modes warn and map to Newton's values; canonical modes pass through silently."""
    newton_cfg = _make_solver_config(MPMSolverCfg(collider_velocity_mode=mode))

    assert newton_cfg.collider_velocity_mode == expected
    deprecations = [str(w.message) for w in recwarn if issubclass(w.category, DeprecationWarning)]
    assert any(f"use {expected!r}" in message for message in deprecations) is deprecated


def test_mpm_prepare_builder_makes_kinematic_bodies_massless():
    """Kinematic bodies become massless so MPM treats them as kinematic colliders; dynamic bodies keep their mass."""
    builder = ModelBuilder()
    kinematic = builder.add_body(mass=0.35, inertia=wp.mat33(1.0), is_kinematic=True)
    dynamic = builder.add_body(mass=1.2, inertia=wp.mat33(2.0))

    MPMSolver.prepare_builder(MPMSolverCfg(), builder)

    assert builder.body_mass[kinematic] == builder.body_inv_mass[kinematic] == 0.0
    assert np.allclose(np.array(builder.body_inertia[kinematic]), 0.0)
    assert np.allclose(np.array(builder.body_inv_inertia[kinematic]), 0.0)
    assert builder.body_mass[dynamic] == pytest.approx(1.2)
    assert np.allclose(np.array(builder.body_inertia[dynamic]), 2.0)


@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_mpm_prepare_builder_converts_convex_meshes_before_solver_construction():
    """Convex meshes become triangle meshes before implicit MPM consumes the model."""
    builder = ModelBuilder()
    register_builder_attributes(builder, MPMSolverCfg())
    mesh = newton.Mesh(vertices=[(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)], indices=[0, 1, 2])
    shape = builder.add_shape_mesh(builder.add_body(), mesh=mesh)
    builder.shape_type[shape] = newton.GeoType.CONVEX_MESH
    builder.add_particles(
        pos=[(0.0, 0.0, 0.1)],
        vel=[(0.0, 0.0, 0.0)],
        mass=[0.01],
        radius=[0.02],
        custom_attributes=_MPM_PARTICLE_ATTRIBUTES,
    )

    MPMSolver.prepare_builder(MPMSolverCfg(), builder)
    solver_cfg = MPMSolverCfg(max_iterations=2, voxel_size=0.05)
    solver = MPMSolver.build(solver_cfg, builder.finalize(device="cuda:0"), wp.DeterministicMode.NOT_GUARANTEED)

    assert builder.shape_type[shape] == newton.GeoType.MESH
    assert isinstance(solver, SolverImplicitMPM)


@pytest.mark.parametrize("project_outside", [True, False])
def test_mpm_step_projects_particles_out_of_colliders_only_when_configured(project_outside):
    calls = []
    solver = SimpleNamespace(
        step=lambda *args: calls.append("step"), project_outside=lambda *args: calls.append("project")
    )

    MPMSolver.step(MPMSolverCfg(project_outside_colliders=project_outside), solver, None, None, None, None, 0.01)

    assert calls == (["step", "project"] if project_outside else ["step"])


def test_mpm_keeps_solver_history_on_automatic_resets():
    """Automatic world resets must not clear shared-grid MPM history; tasks reset it explicitly."""

    def reject_reset(*args, **kwargs):
        raise AssertionError("automatic resets must not clear implicit-MPM history")

    MPMSolver.reset(MPMSolverCfg(), SimpleNamespace(reset=reject_reset), None, None)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({"grid_type": "fixed"}, True, id="bounded_fixed"),
        pytest.param({"grid_type": "fixed", "max_active_cell_count": -1}, False, id="unbounded_fixed"),
        pytest.param({}, True, id="bounded_sparse"),
        pytest.param({"max_active_cell_count": -1}, False, id="unbounded_sparse"),
        pytest.param({"grid_type": "dense"}, False, id="dense"),
        pytest.param({"grid_padding": 1}, False, id="padded_sparse"),
        pytest.param({"velocity_basis": "P0"}, False, id="velocity_basis"),
        pytest.param({"strain_basis": "GIMP"}, False, id="strain_basis"),
        pytest.param({"collider_basis": "GIMP"}, False, id="collider_basis"),
    ],
)
def test_mpm_capture_supports_only_static_topology(overrides, expected):
    """Only fixed and capacity-bounded rebuildable sparse grids support CUDA graph capture."""
    values = {
        "grid_type": "sparse",
        "max_active_cell_count": 1024,
        "grid_padding": 0,
        "velocity_basis": "Q1",
        "strain_basis": "P0",
        "collider_basis": "S2",
    }

    assert mpm_supports_graph_capture(SimpleNamespace(**(values | overrides))) is expected


def test_mpm_checks_sparse_grid_status_only_after_a_graph_replay():
    calls = []
    solver = object.__new__(SolverImplicitMPM)
    solver.check_sparse_grid_rebuild_status = lambda: calls.append("check")

    MPMSolver.check_step(MPMSolverCfg(), solver, False, False)
    MPMSolver.check_step(MPMSolverCfg(), solver, True, False)

    assert calls == ["check"]


def test_implicit_mpm_solvers_finds_mpm_entries_of_a_coupled_solver():
    mpm_solver = object.__new__(SolverImplicitMPM)
    coupled = SimpleNamespace(
        entry_names=lambda: ("media", "rigid"), solver=lambda name: mpm_solver if name == "media" else object()
    )

    assert implicit_mpm_solvers(coupled) == (mpm_solver,)
    assert implicit_mpm_solvers(mpm_solver) == (mpm_solver,)
    assert implicit_mpm_solvers(None) == ()


# ----- VBD ---------------------------------------------------------------------------------


@pytest.mark.parametrize("particle_count", [0, 1])
def test_vbd_rebuilds_its_particle_bvh_before_each_physics_step(particle_count):
    rebuilt = []
    solver = SimpleNamespace(model=SimpleNamespace(particle_count=particle_count), rebuild_bvh=rebuilt.append)

    VBDSolver.pre_step(VBDSolverCfg(), solver, "state")

    assert rebuilt == ["state"] * particle_count
