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
from isaaclab_newton.physics.mpm_solver import implicit_mpm_solvers, mpm_solver_config
from isaaclab_newton.physics.newton_builder import register_builder_attributes
from isaaclab_newton.physics.newton_solver import solver_kwargs
from isaaclab_newton.test.fixtures.mock_newton import solver_driver
from newton import ModelBuilder
from newton.solvers import SolverImplicitMPM, SolverMuJoCo, SolverXPBD

from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.kitless


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
    joints = []
    for child in ("child_a", "child_b"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Cube.Define(stage, f"{root_path}/{child}").GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root_path}/{child}/joint")
        joint.CreateAxisAttr().Set("Z")
        joint.CreateBody0Rel().SetTargets([root_path])
        joint.CreateBody1Rel().SetTargets([f"{root_path}/{child}"])
        for name, value in (("mjc:frictionloss", 0.11), ("mjc:damping", 0.23), ("mjc:armature", 0.12)):
            joint.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Double, True).Set(value)
        joint.GetPrim().CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float, True).Set(0.21)
        joints.append(joint.GetPrim())
    # Joint a sets every schema; joint b leaves Newton unset so PhysX competes with MuJoCo.
    joints[0].CreateAttribute("newton:friction", Sdf.ValueTypeNames.Double, True).Set(0.31)
    joints[0].CreateAttribute("newton:armature", Sdf.ValueTypeNames.Double, True).Set(0.41)

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

    joint_qd_start = model.joint_qd_start.numpy()
    joint_a, joint_b = (
        joint_qd_start[model.joint_label.index(f"/World/envs/env_0/robot/{child}/joint")]
        for child in ("child_a", "child_b")
    )
    assert model.joint_friction.numpy()[joint_a] == pytest.approx(0.31)
    assert model.joint_damping.numpy()[joint_a] == pytest.approx(expected_damping)
    assert model.joint_armature.numpy()[joint_a] == pytest.approx(0.41)
    assert model.joint_armature.numpy()[joint_b] == pytest.approx(0.21)


# ----- MuJoCo Warp -------------------------------------------------------------------------


@pytest.mark.parametrize("device", test_devices())
def test_mjwarp_authored_reset_imports_only_selected_worlds(device):
    """Authored state reaches MuJoCo for selected worlds, including during CUDA graph replay."""
    builder = ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for _ in range(2):
        builder.begin_world()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        builder.add_articulation([joint])
        builder.end_world()
    model = builder.finalize(device=device)
    cfg = MJWarpSolverCfg(use_mujoco_cpu=device == "cpu", update_data_interval=2)
    solver = SolverMuJoCo(model, use_mujoco_cpu=cfg.use_mujoco_cpu, separate_worlds=True, update_data_interval=2)
    state = model.state()
    world_mask = wp.zeros(3, dtype=wp.bool, device=device)
    data = solver.mj_data if cfg.use_mujoco_cpu else solver.mjw_data
    # Zero-copy views give native NumPy and Warp buffers the same fill/read interface.
    qpos = wp.array(data.qpos, copy=False, device=device)
    qvel = wp.array(data.qvel, copy=False, device=device)
    qpos.fill_(1.25)
    qvel.fill_(-0.75)
    reset = functools.partial(solver_driver(MJWarpSolver, cfg, solver).reset, state, world_mask)
    reset()  # Warm the reset kernels with an empty mask before capture.
    if not cfg.use_mujoco_cpu:
        with wp.ScopedCapture(device=device) as capture:
            reset()
        reset = functools.partial(wp.capture_launch, capture.graph)

    count = 1 if cfg.use_mujoco_cpu else 2  # Native MuJoCo exposes only template world 0.
    for world, expected_q, expected_qd in ((1, [1.25, 0.25], [-0.75, 0.5]), (0, [-0.25, 0.25], [-0.5, 0.5])):
        state.joint_q.fill_(expected_q[world])
        state.joint_qd.fill_(expected_qd[world])
        world_mask.zero_()
        world_mask[world : world + 1].fill_(True)
        reset()
        np.testing.assert_allclose(qpos.numpy().ravel(), expected_q[:count])
        np.testing.assert_allclose(qvel.numpy().ravel(), expected_qd[:count])


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

    MJWarpSolver(solver_cfg, _model_stub(), wp.DeterministicMode.NOT_GUARANTEED)
    MJWarpSolver(solver_cfg, _model_stub(), wp.DeterministicMode.RUN_TO_RUN)
    MJWarpSolver(solver_cfg, _model_stub({"body_qdd"}), wp.DeterministicMode.NOT_GUARANTEED)
    for attr in ("body_qdd", "body_parent_f"):
        with pytest.raises(ValueError, match="does not support deterministic physics"):
            MJWarpSolver(solver_cfg, _model_stub({attr}), wp.DeterministicMode.RUN_TO_RUN)

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
    newton_cfg = MPMSolverCfg(**dict(_MPM_FIELD_VALUES)).to_solver_config()

    _assert_forwarded(newton_cfg, _MPM_FIELD_VALUES)
    assert not any(hasattr(newton_cfg, name) for name in ("class_type", "solver_type", "project_outside_colliders"))


@pytest.mark.parametrize(
    "mode, expected, deprecated",
    [("instantaneous", "forward", True), ("finite_difference", "backward", True), ("forward", "forward", False)],
)
def test_mpm_cfg_translates_deprecated_collider_velocity_modes(mode, expected, deprecated, recwarn):
    """Deprecated collider velocity modes warn and map to Newton's values; canonical modes pass through silently."""
    newton_cfg = MPMSolverCfg(collider_velocity_mode=mode).to_solver_config()

    assert newton_cfg.collider_velocity_mode == expected
    deprecations = [str(w.message) for w in recwarn if issubclass(w.category, DeprecationWarning)]
    assert any(f"use {expected!r}" in message for message in deprecations) is deprecated


def test_mpm_kinematic_colliders_preserve_mass_for_other_solvers():
    """MPM uses zero effective collider mass without breaking a shared rigid model."""
    import newton

    builder = newton.ModelBuilder()
    register_builder_attributes(builder, MPMSolverCfg())
    kinematic_body = builder.add_body(
        mass=0.35,
        inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
        is_kinematic=True,
        label="kinematic_collider",
    )
    dynamic_body = builder.add_body(
        mass=1.2,
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02),
        is_kinematic=False,
        label="dynamic_body",
    )
    mesh = newton.Mesh(
        vertices=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)],
        indices=[0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
    )
    builder.add_shape_mesh(kinematic_body, mesh=mesh, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    builder.add_particle(pos=(0.2, 0.2, 0.2), vel=(0.0, 0.0, 0.0), mass=0.01, radius=0.02)

    MPMSolver.prepare_builder(MPMSolverCfg(), builder)
    model = builder.finalize(device="cpu")
    # Explicit zero physical mass makes MuJoCo reject this free body. Kinematic
    # collider treatment must instead remain local to the MPM solver.
    rigid_solver = SolverMuJoCo(model, use_mujoco_cpu=True)
    mpm_solver = MPMSolver(
        MPMSolverCfg(voxel_size=0.1, grid_type="dense", grid_padding=1),
        model,
        wp.DeterministicMode.NOT_GUARANTEED,
    ).native

    assert rigid_solver.mj_model.nbody == 3
    assert model.body_mass.numpy()[kinematic_body] == pytest.approx(0.35)
    assert model.body_mass.numpy()[dynamic_body] == pytest.approx(1.2)
    np.testing.assert_allclose(model.body_inertia.numpy()[kinematic_body], np.eye(3) * 0.01)
    effective_mass = mpm_solver._mpm_model.collider_body_mass.numpy()
    assert effective_mass[kinematic_body] == 0.0
    assert effective_mass[dynamic_body] == pytest.approx(1.2)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_mpm_mesh_colliders_support_b3_capture_and_replay(device):
    """Imported mesh colliders and bounded B3 grids step through the native captured solver."""
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
    solver_cfg = MPMSolverCfg(
        max_iterations=2,
        voxel_size=0.05,
        max_active_cell_count=256,
        velocity_basis="B3",
        solver="jacobi",
        warmstart_mode="none",
    )
    model = builder.finalize(device=device)
    driver = MPMSolver(solver_cfg, model, wp.DeterministicMode.NOT_GUARANTEED)
    state = model.state()
    assert builder.shape_type[shape] == newton.GeoType.MESH
    assert driver.supports_capture()
    driver.step(state, state, None, None, 0.001)
    before = state.particle_q.numpy().copy()
    with wp.ScopedCapture(device=device) as capture:
        driver.step(state, state, None, None, 0.001)
    wp.capture_launch(capture.graph)
    driver.check_step(captured=True, debug=False)
    after = state.particle_q.numpy()
    assert np.isfinite(after).all()
    assert not np.array_equal(before, after)


def test_mpm_solver_build_preserves_python_precision_and_active_usd():
    """Native construction must not mutate the scene or round grid values through USD floats."""
    cfg = MPMSolverCfg(voxel_size=0.051234567890123, grid_type="fixed", grid_padding=1, max_iterations=1)
    with build_simulation_context(sim_cfg=SimulationCfg(device="cpu", physics=NewtonCfg(solver_cfg=cfg))) as sim:
        builder = ModelBuilder()
        register_builder_attributes(builder, cfg)
        builder.add_particle(pos=(0.0, 0.0, 0.1), vel=(0.0, 0.0, 0.0), mass=0.01, radius=0.02)
        model = builder.finalize(device="cpu")
        original_layer = sim.stage.GetRootLayer().ExportToString()

        solver = MPMSolver(cfg, model, wp.DeterministicMode.NOT_GUARANTEED).native

        assert solver.voxel_size == cfg.voxel_size
        assert sim.stage.GetRootLayer().ExportToString() == original_layer
        with pytest.deprecated_call(match="mpm_solver_config"):
            legacy = mpm_solver_config(cfg)
        assert legacy.voxel_size == cfg.voxel_size
        assert sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path).GetAttribute(
            "newton:mpm:voxelSize"
        ).Get() == pytest.approx(cfg.voxel_size)


@pytest.mark.parametrize("project_outside", [True, False])
def test_mpm_step_projects_particles_out_of_colliders_only_when_configured(project_outside):
    calls = []
    solver = SimpleNamespace(
        step=lambda *args: calls.append("step"), project_outside=lambda *args: calls.append("project")
    )

    solver_driver(MPMSolver, MPMSolverCfg(project_outside_colliders=project_outside), solver).step(
        None, None, None, None, 0.01, observables=None
    )

    assert calls == (["step", "project"] if project_outside else ["step"])


def test_mpm_keeps_solver_history_on_automatic_resets():
    """Automatic world resets must not clear shared-grid MPM history; tasks reset it explicitly."""

    def reject_reset(*args, **kwargs):
        raise AssertionError("automatic resets must not clear implicit-MPM history")

    solver_driver(MPMSolver, MPMSolverCfg(), SimpleNamespace(reset=reject_reset)).reset(None, None)


def test_mpm_checks_sparse_grid_status_only_after_a_graph_replay():
    calls = []
    solver = object.__new__(SolverImplicitMPM)
    solver.check_sparse_grid_rebuild_status = lambda: calls.append("check")

    driver = solver_driver(MPMSolver, MPMSolverCfg(), solver)
    driver.check_step(False, False)
    driver.check_step(True, False)

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

    solver_driver(VBDSolver, VBDSolverCfg(), solver).pre_step("state")

    assert rebuilt == ["state"] * particle_count
