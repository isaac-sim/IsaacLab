# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the :class:`NewtonManager` shell, the solver manager hooks, and the Newton step graph.

Covers:

* Solver configurations describe their Newton solver, and :attr:`NewtonCfg.class_type` selects
  :class:`NewtonManager` independently.
* Solver validation, such as rejecting ``MJWarp + use_mujoco_contacts=True + collision_cfg``.
* Fixed-root pose writes refresh MuJoCo's solver-owned root transform.
* End-to-end: spinning up a simulation with each solver binds the correct
  solver with the right capabilities, and one step graph runs the whole step
  in the same order whether it replays from CUDA graphs or eagerly.
"""

from __future__ import annotations

import functools
import logging
from functools import partial as P
from types import SimpleNamespace

import isaaclab_newton.physics.mjwarp_solver as mjwarp_solver_module
import isaaclab_newton.physics.newton_manager as newton_manager_module
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.articulation import articulation as articulation_module
from isaaclab_newton.assets.rigid_object import rigid_object as rigid_object_module
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
    NewtonBuilderCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonShapeCfg,
    NewtonSolver,
    NewtonSolverCfg,
    VBDSolverCfg,
    XPBDSolverCfg,
)
from isaaclab_newton.physics.mpm_solver import (
    _make_solver_config,
    implicit_mpm_solvers,
    mpm_supports_graph_capture,
)
from isaaclab_newton.physics.newton_backend_cfg import NewtonBackendCfg
from isaaclab_newton.physics.newton_builder import register_builder_attributes, usd_schema_resolvers
from isaaclab_newton.physics.newton_solver import solver_kwargs
from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer
from newton import JointTargetMode, JointType, ModelBuilder, ModelFlags
from newton.actuators import DrivePID
from newton.selection import ArticulationView
from newton.solvers import SolverFeatherstone, SolverImplicitMPM, SolverKamino, SolverMuJoCo, SolverVBD, SolverXPBD

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.physics import PhysicsEvent
from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context
from isaaclab.test.utils import test_devices

resolve_deterministic_mode = NewtonManager._resolve_deterministic_mode


# ---------------------------------------------------------------------------
# Lightweight (no sim) parametrisation
# ---------------------------------------------------------------------------

# (solver_cfg_factory, expected_solver_cls, single_state, needs_collision_pipeline)
SOLVER_MATRIX = [
    (P(MJWarpSolverCfg, use_mujoco_contacts=True), SolverMuJoCo, True, False),
    (P(MJWarpSolverCfg, use_mujoco_contacts=False), SolverMuJoCo, True, True),
    (XPBDSolverCfg, SolverXPBD, False, True),
    (VBDSolverCfg, SolverVBD, False, True),
    (FeatherstoneSolverCfg, SolverFeatherstone, False, True),
    (P(KaminoPADMMSolverCfg, use_collision_detector=True), SolverKamino, False, False),
    (P(KaminoPADMMSolverCfg, use_collision_detector=False), SolverKamino, False, True),
    (P(MPMSolverCfg, max_iterations=2, voxel_size=0.05), SolverImplicitMPM, True, False),
]
SOLVER_IDS = ["mj_internal", "mj_pipeline", "xpbd", "vbd", "featherstone", "kamino_internal", "kamino_pipeline", "mpm"]


# ---------------------------------------------------------------------------
# Solver and manager selection (no SimulationContext required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver_cfg_factory, solver_cls, single_state, _pipeline", SOLVER_MATRIX, ids=SOLVER_IDS)
def test_solver_cfg_describes_its_solver(solver_cfg_factory, solver_cls, single_state, _pipeline):
    """A solver configuration selects the solver that builds its Newton solver; the manager stays one class."""
    cfg = NewtonCfg(solver_cfg=solver_cfg_factory())
    assert cfg.class_type.__name__ == "NewtonManager"
    assert cfg.solver_cfg.class_type.solver_classes(cfg.solver_cfg) == (solver_cls,)
    assert cfg.solver_cfg.class_type.single_state is single_state


@pytest.mark.parametrize(
    "num_substeps, collision_decimation, should_warn",
    [
        (8, 0, False),  # Default: feature disabled, no warning.
        (8, 7, False),  # Valid edge: one mid-loop re-collide at i=6.
        (8, 8, True),  # Equal to num_substeps: gate never fires.
        (8, 16, True),  # Above num_substeps: pins the ``>=`` rather than ``==`` boundary.
    ],
)
def test_newton_cfg_collision_decimation_warning(num_substeps, collision_decimation, should_warn, caplog):
    """``NewtonCfg.__post_init__`` warns when ``collision_decimation >= num_substeps``."""
    import logging

    with caplog.at_level(logging.WARNING, logger="isaaclab_newton.physics.newton_manager_cfg"):
        cfg = NewtonCfg(num_substeps=num_substeps, collision_decimation=collision_decimation)
    assert any("collision_decimation" in rec.getMessage() for rec in caplog.records) is should_warn
    assert cfg.collision_decimation == collision_decimation


def test_solver_kwargs_include_newton_deterministic_mode() -> None:
    """Solver construction should receive the mode configured on the outer Newton config."""
    kwargs = solver_kwargs(SolverXPBD, XPBDSolverCfg(), wp.DeterministicMode.GPU_TO_GPU)

    assert kwargs["deterministic"] == wp.DeterministicMode.GPU_TO_GPU


def _model_stub(state_attributes=()) -> SimpleNamespace:
    """Model stand-in carrying what solver construction reads before the Newton solver exists."""
    return SimpleNamespace(
        get_requested_state_attributes=lambda: list(state_attributes), articulation_start=None, articulation_end=None
    )


@pytest.mark.parametrize(
    "solver_cfg",
    [pytest.param(KaminoPADMMSolverCfg(), id="kamino_padmm"), pytest.param(MPMSolverCfg(), id="implicit_mpm")],
)
def test_deterministic_mode_rejects_unsupported_solver_cfg(solver_cfg) -> None:
    """Unsupported solvers should not silently ignore a determinism guarantee."""
    manager = NewtonManager(NewtonCfg())
    manager._cfg = NewtonCfg(solver_cfg=solver_cfg, deterministic_mode="gpu_to_gpu")
    manager.backend = SimpleNamespace(model=_model_stub())
    with pytest.raises(ValueError, match="not supported"):
        manager.initialize_solver()


@pytest.mark.parametrize("collision_cfg", [None, NewtonCollisionPipelineCfg()], ids=["default", "configured"])
def test_deterministic_collision_pipeline_matches_expanded_contact_capacity(collision_cfg) -> None:
    """Deterministic sorting buffers should grow with the solver contact buffer."""
    pipeline_calls: list[dict] = []

    class FakeCollisionPipeline:
        def __init__(self, _model, **kwargs):
            pipeline_calls.append(kwargs)
            self._rigid_contact_max = kwargs.get("rigid_contact_max") or 1

        def contacts(self):
            return SimpleNamespace(rigid_contact_max=self._rigid_contact_max)

    prepared = []

    class RecordingSolver(NewtonSolver):
        @classmethod
        def create_collision_pipeline(cls, cfg, solver, **kwargs):
            return FakeCollisionPipeline(solver.model, **kwargs)

        @classmethod
        def prepare_contacts(cls, cfg, solver, contacts, pipeline):
            prepared.append(contacts)

    solver = SimpleNamespace(model=SimpleNamespace(), get_max_contact_count=lambda: 2)
    backend = SimpleNamespace(model=solver.model, solver=solver, collision_pipeline=None)
    manager = NewtonManager(NewtonCfg())
    solver_cfg = NewtonSolverCfg(class_type=RecordingSolver)
    manager._cfg, manager._device = NewtonCfg(collision_cfg=collision_cfg, solver_cfg=solver_cfg), "cpu"
    manager.backend = backend
    manager._deterministic_mode = wp.DeterministicMode.GPU_TO_GPU

    manager._initialize_contacts()

    assert len(pipeline_calls) == 2
    assert pipeline_calls[1] == {**pipeline_calls[0], "rigid_contact_max": 2}
    assert pipeline_calls[0]["deterministic"] is True
    assert backend.contacts.rigid_contact_max == 2
    assert prepared == [backend.contacts]


@pytest.mark.parametrize("cloth", [False, True])
@pytest.mark.parametrize("device", test_devices())
def test_queries_share_native_bvhs_and_read_only_through_sdp(monkeypatch, cloth, device):
    """Two consumers reuse one refit and see fresh rigid and deformable publications."""
    sim = object.__new__(SimulationContext)
    sim._backend_registry = []
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim)
    sim.backend_manager = NewtonManager(NewtonCfg())
    sim.backend_manager._sim, sim.backend_manager._device = sim, device
    cfg = NewtonBackendCfg(physics_cfg=object(), device=device)
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg.physics_cfg))
    builder.add_shape_sphere(builder.add_body(label="/Object"))
    for position in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        builder.add_particle(pos=wp.vec3(position), vel=wp.vec3(), mass=1.0)
    if cloth:
        builder.add_triangle(0, 1, 2)
    backend = sim.get_or_create_backend(cfg)
    backend.geometry_offsets = {"/Cloth": 0}
    transforms, points = SceneDataFormat.Transform(), SceneDataFormat.Points()
    transforms.transforms = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
    points.points = wp.array(np.tile([1.0, 2.0, 3.0], (3, 1)), dtype=wp.vec3, device=device)
    publication = SimpleNamespace(
        transforms=transforms,
        transform_paths=["/Object"],
        transforms_timestamp=0,
        geometry_timestamp=0,
        get_transforms=lambda output_format: transforms,
        get_geometry_batches=lambda output_format: [(points, {"/Cloth": (0, 3)})],
    )
    provider = SceneDataProvider(publication)
    renderer = object.__new__(NewtonWarpRenderer)
    renderer.backend = backend
    renderer._scene_data_provider = provider
    renderer._transform_mapping = provider.create_mapping(list(backend.model.body_label))
    # Triangle rendering performs host work and must stay eager even when capture is enabled.
    renderer.cfg = SimpleNamespace(use_cuda_graph=cloth)
    refits, observed = [], []
    refit = backend.model.bvh_refit_shapes
    monkeypatch.setattr(backend.model, "bvh_refit_shapes", lambda state: (refits.append(state), refit(state)))

    def read(_data):
        assert backend.state_0.body_q is transforms.transforms
        np.testing.assert_array_equal(backend.state_0.particle_q.numpy(), points.points.numpy())
        observed.append(backend.state_0.body_q.numpy().copy())

    monkeypatch.setattr(renderer, "_launch_render", read)
    for _ in range(2):
        render_data = SimpleNamespace(graph=None, ppisp_pipeline=None)
        renderer.render(render_data)
        assert render_data.graph is None
    assert len(refits) == 1
    transforms.transforms = wp.array(
        [wp.transform(wp.vec3(4.0, 5.0, 6.0), wp.quat_identity())], dtype=wp.transform, device=device
    )
    points.points.assign(np.tile(np.asarray([7.0, 8.0, 9.0], dtype=np.float32), (3, 1)))
    publication.transforms_timestamp += 1
    publication.geometry_timestamp += 1
    renderer.render(render_data)
    assert len(refits) == 2
    np.testing.assert_array_equal(observed[-1][0, :3], [4.0, 5.0, 6.0])
    for expected, attribute in enumerate(("geometry_timestamp", "transforms_timestamp"), start=3):
        setattr(publication, attribute, getattr(publication, attribute) + 1)
        renderer.render(render_data)
        renderer.render(render_data)
        assert len(refits) == expected
    backend.close()


def test_newton_shape_cfg_defaults_match_newton_shape_config():
    """``NewtonShapeCfg`` contact defaults mirror Newton's, so applying them never clobbers shape materials."""
    import newton

    upstream = newton.ModelBuilder().default_shape_cfg
    shape_cfg = NewtonShapeCfg()
    assert shape_cfg.ke == upstream.ke
    assert shape_cfg.kd == upstream.kd
    assert shape_cfg.mu == upstream.mu


@pytest.mark.parametrize(
    "mode, expected, deprecated",
    [
        ("instantaneous", "forward", True),
        ("finite_difference", "backward", True),
        ("forward", "forward", False),
        ("backward", "backward", False),
    ],
)
def test_mpm_solver_cfg_translates_deprecated_collider_velocity_modes(mode, expected, deprecated, recwarn):
    """Deprecated collider velocity modes warn and map to Newton's values; canonical modes pass through silently."""
    newton_cfg = _make_solver_config(MPMSolverCfg(collider_velocity_mode=mode))

    assert newton_cfg.collider_velocity_mode == expected
    deprecations = [str(w.message) for w in recwarn if issubclass(w.category, DeprecationWarning)]
    if deprecated:
        assert any(f"use {expected!r}" in message for message in deprecations)
    else:
        assert not deprecations


# A non-default value of every solver-tunable MPMSolverCfg field.
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


def test_mpm_solver_cfg_forwards_every_solver_field():
    """Every tunable MPM cfg field round-trips into ``SolverImplicitMPM.Config``, and Isaac Lab fields do not."""
    newton_cfg = _make_solver_config(MPMSolverCfg(**dict(_MPM_FIELD_VALUES)))
    assert not any(hasattr(newton_cfg, name) for name in ("class_type", "solver_type", "project_outside_colliders"))
    for field_name, value in _MPM_FIELD_VALUES:
        assert hasattr(newton_cfg, field_name), (
            f"{field_name!r} disappeared from SolverImplicitMPM.Config — MPMSolverCfg needs to drop or rename it."
        )
        assert getattr(newton_cfg, field_name) == value


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


def test_kamino_solver_cfg_forwards_padmm_fields():
    """Every tunable P-ADMM cfg field round-trips into ``PADMMSolverConfig``."""
    # Adaptive penalty updates require the sparse solver path.
    solver_cfg = KaminoPADMMSolverCfg(
        sparse_jacobian=True,
        sparse_dynamics=True,
        dynamics_solver_cfg=KaminoPADMMCfg(**dict(_KAMINO_PADMM_FIELD_VALUES)),
    )
    newton_cfg = solver_cfg.to_solver_config()
    for field_name, value in _KAMINO_PADMM_FIELD_VALUES:
        assert hasattr(newton_cfg.padmm, field_name), (
            f"{field_name!r} disappeared from PADMMSolverConfig — KaminoPADMMCfg needs to drop or rename it."
        )
        assert getattr(newton_cfg.padmm, field_name) == value


def test_kamino_padmm_rejects_adaptive_penalties_with_dense_dynamics():
    """Kamino's adaptive P-ADMM penalties require the sparse solver path."""
    solver_cfg = KaminoPADMMSolverCfg(dynamics_solver_cfg=KaminoPADMMCfg(penalty_update_method="balanced"))
    with pytest.raises(ValueError, match="sparse_dynamics=True"):
        solver_cfg.to_solver_config()


def test_kamino_solver_cfg_forwards_dvi_fields():
    """Every tunable DVI cfg field round-trips into ``DVISolverConfig``."""
    solver_cfg = KaminoDVISolverCfg(
        dynamics=KaminoDynamicsCfg(preconditioning=False),
        dynamics_solver_cfg=KaminoDVICfg(**dict(_KAMINO_DVI_FIELD_VALUES)),
    )
    newton_cfg = solver_cfg.to_solver_config()
    for field_name, value in _KAMINO_DVI_FIELD_VALUES:
        assert hasattr(newton_cfg.dvi, field_name), (
            f"{field_name!r} disappeared from DVISolverConfig — KaminoDVICfg needs to drop or rename it."
        )
        assert getattr(newton_cfg.dvi, field_name) == value


@pytest.mark.parametrize("field_name, value", _KAMINO_DYNAMICS_FIELD_VALUES)
def test_kamino_solver_cfg_forwards_dynamics_fields(field_name, value):
    """Every tunable dynamics cfg field round-trips into ``ConstrainedDynamicsConfig``."""
    solver_type = KaminoDVISolverCfg if field_name == "preconditioning" and value is False else KaminoPADMMSolverCfg
    solver_cfg = solver_type(dynamics=KaminoDynamicsCfg(**{field_name: value}))
    newton_cfg = solver_cfg.to_solver_config()
    assert hasattr(newton_cfg.dynamics, field_name), (
        f"{field_name!r} disappeared from ConstrainedDynamicsConfig — KaminoDynamicsCfg needs updating."
    )
    assert getattr(newton_cfg.dynamics, field_name) == value


def test_kamino_to_solver_config_metadata_excluded():
    """Isaac Lab metadata and manager-only fields do not leak into Newton config."""
    solver_cfg = KaminoPADMMSolverCfg(
        solver_type="isaaclab_metadata_should_not_forward",
        max_contacts_per_world=32,
    )
    newton_cfg = solver_cfg.to_solver_config()
    assert not hasattr(newton_cfg, "class_type")
    assert not hasattr(newton_cfg, "solver_type")
    assert not hasattr(newton_cfg, "max_contacts_per_world")


def test_kamino_concrete_solver_configs_select_their_backends():
    """Concrete solver config types select their corresponding Newton backends."""
    padmm_cfg = KaminoPADMMSolverCfg(sparse_jacobian=True)
    dvi_cfg = KaminoDVISolverCfg()
    assert padmm_cfg.to_solver_config().dynamics_solver == "padmm"
    assert dvi_cfg.to_solver_config().dynamics_solver == "dvi"


def test_kamino_dvi_rejects_preconditioning():
    """DVI preserves Newton's preconditioning compatibility check."""
    solver_cfg = KaminoDVISolverCfg(dynamics=KaminoDynamicsCfg(preconditioning=True))
    with pytest.raises(ValueError, match="preconditioning"):
        solver_cfg.to_solver_config()


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

    assert builder.has_custom_attribute(active)
    assert all(not builder.has_custom_attribute(name) for name in inactive)
    if isinstance(solver_cfg, MJWarpSolverCfg):
        assert "mujoco:actuator" in builder.custom_frequencies
        assert "mujoco:tendon" in builder.custom_frequencies

    # A second registration on the same builder is a no-op.
    register_builder_attributes(builder, solver_cfg)
    assert builder.has_custom_attribute(active)


def test_mpm_prepare_builder_makes_kinematic_bodies_massless():
    """Kinematic bodies must be massless so MPM treats them as kinematic colliders."""
    import newton

    builder = newton.ModelBuilder()
    kinematic_body = builder.add_body(
        mass=0.35,
        inertia=wp.mat33(1.0),
        is_kinematic=True,
        label="kinematic_collider",
    )
    dynamic_body = builder.add_body(
        mass=1.2,
        inertia=wp.mat33(2.0),
        is_kinematic=False,
        label="dynamic_body",
    )

    MPMSolver.prepare_builder(MPMSolverCfg(), builder)

    assert builder.body_flags[kinematic_body] & int(newton.BodyFlags.KINEMATIC)
    assert builder.body_mass[kinematic_body] == 0.0
    assert builder.body_inv_mass[kinematic_body] == 0.0
    assert np.allclose(np.array(builder.body_inertia[kinematic_body]), 0.0)
    assert np.allclose(np.array(builder.body_inv_inertia[kinematic_body]), 0.0)

    assert builder.body_mass[dynamic_body] == pytest.approx(1.2)
    assert builder.body_inv_mass[dynamic_body] == pytest.approx(1.0 / 1.2)
    assert np.allclose(np.array(builder.body_inertia[dynamic_body]), 2.0)


_MPM_PARTICLE_ATTRIBUTES = {
    "mpm:viscosity": 50.0,
    "mpm:friction": 0.0,
    "mpm:tensile_yield_ratio": 1.0,
    "mpm:yield_pressure": 1.0e15,
    "mpm:yield_stress": 0.0,
    "mpm:young_modulus": 1.0e15,
    "mpm:damping": 0.0,
}


@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_mpm_prepare_builder_converts_convex_mesh_before_solver_construction():
    """Convex meshes must become triangle meshes before implicit MPM consumes the model."""
    import newton

    builder = newton.ModelBuilder()
    register_builder_attributes(builder, MPMSolverCfg())
    body = builder.add_body(label="convex_mesh_collider")
    mesh = newton.Mesh(
        vertices=[(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
    )
    shape = builder.add_shape_mesh(body, mesh=mesh)
    builder.shape_type[shape] = newton.GeoType.CONVEX_MESH
    builder.add_particles(
        pos=[(0.0, 0.0, 0.1)],
        vel=[(0.0, 0.0, 0.0)],
        mass=[0.01],
        radius=[0.02],
        custom_attributes=_MPM_PARTICLE_ATTRIBUTES,
    )

    MPMSolver.prepare_builder(MPMSolverCfg(), builder)
    model = builder.finalize(device="cuda:0")
    solver_cfg = MPMSolverCfg(max_iterations=2, voxel_size=0.05)
    solver = MPMSolver.build(solver_cfg, model, wp.DeterministicMode.NOT_GUARANTEED)

    assert builder.shape_type[shape] == newton.GeoType.MESH
    assert isinstance(solver, SolverImplicitMPM)


def _revolute_joint_stage(root_path: str):
    """A stage with one revolute joint carrying MuJoCo joint properties and a PhysX armature that overrides them."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Cube.Define(stage, root_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    child_path = f"{root_path}/child"
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Cube.Define(stage, child_path).GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{child_path}/joint")
    joint.CreateAxisAttr().Set("Z")
    joint.CreateBody0Rel().SetTargets([root_path])
    joint.CreateBody1Rel().SetTargets([child_path])
    joint_prim = joint.GetPrim()
    for name, value in (("mjc:frictionloss", 0.11), ("mjc:damping", 0.23), ("mjc:armature", 0.12)):
        joint_prim.CreateAttribute(name, Sdf.ValueTypeNames.Double, True).Set(value)
    joint_prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float, True).Set(0.21)
    return stage, joint_prim


@pytest.mark.parametrize(
    ("solver_cfg", "expected_friction", "expected_damping"),
    [
        pytest.param(MJWarpSolverCfg(), 0.11, 0.23, id="mjwarp"),
        pytest.param(FeatherstoneSolverCfg(), 0.0, 0.0, id="featherstone"),
    ],
)
def test_production_imports_scope_mujoco_joint_properties(monkeypatch, solver_cfg, expected_friction, expected_damping):
    """Only MJWarp imports MuJoCo joint properties through clone-plan construction."""

    from pxr import UsdGeom, UsdPhysics

    root_path, physics_prim_path = "/Sources/robot", "/physicsScene"
    stage, _ = _revolute_joint_stage(root_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, physics_prim_path)

    physics_cfg = NewtonCfg(solver_cfg=solver_cfg, load_visual_shapes=False)
    sim = object.__new__(SimulationContext)
    sim.cfg = SimpleNamespace(physics=physics_cfg, physics_prim_path=physics_prim_path, device="cpu")
    sim.backend_manager, sim._backend_registry = NewtonManager(NewtonCfg()), []
    manager = sim.backend_manager
    manager._sim, manager._cfg, manager._device = sim, physics_cfg, "cpu"
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim)

    builder, _ = newton_physics_replicate(
        stage=stage,
        sources=(root_path,),
        destinations=("/World/envs/env_{}/robot",),
        env_ids=np.array([0], dtype=np.int64),
        mapping=np.ones((1, 1), dtype=np.bool_),
    )

    model = builder.finalize(device="cpu")

    assert model.joint_friction.numpy()[-1] == pytest.approx(expected_friction)
    assert model.joint_damping.numpy()[-1] == pytest.approx(expected_damping)
    assert model.joint_armature.numpy()[-1] == pytest.approx(0.21)


@pytest.mark.parametrize(
    ("solver_cfg", "imports_mujoco"),
    [
        pytest.param(MJWarpSolverCfg(), True, id="mjwarp"),
        pytest.param(FeatherstoneSolverCfg(), False, id="featherstone"),
    ],
)
def test_schema_resolver_policy_and_precedence(solver_cfg, imports_mujoco):
    """Resolver precedence and MuJoCo fallback selection follow active solver needs."""
    from pxr import Sdf

    stage, joint_prim = _revolute_joint_stage("/World/robot")
    joint_prim.CreateAttribute("newton:friction", Sdf.ValueTypeNames.Double, True).Set(0.31)
    joint_prim.CreateAttribute("newton:armature", Sdf.ValueTypeNames.Double, True).Set(0.41)

    schema_resolvers = usd_schema_resolvers(solver_cfg)
    builder = ModelBuilder()
    register_builder_attributes(builder, solver_cfg)
    builder.add_usd(stage, schema_resolvers=schema_resolvers)
    model = builder.finalize(device="cpu")

    # Newton values win over PhysX and MuJoCo; the MuJoCo fallback fills only unset fields.
    assert model.joint_friction.numpy()[-1] == pytest.approx(0.31)
    assert model.joint_damping.numpy()[-1] == pytest.approx(0.23 if imports_mujoco else 0.0)
    assert model.joint_armature.numpy()[-1] == pytest.approx(0.41)


@pytest.mark.parametrize("project_outside", [True, False])
def test_mpm_project_outside_colliders_gates_projection(project_outside):
    """``project_outside_colliders`` controls whether ``project_outside`` runs per substep."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05, project_outside_colliders=project_outside),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        builder.add_particles(
            pos=[(0.0, 0.0, 0.10), (0.05, 0.0, 0.10), (0.0, 0.05, 0.10)],
            vel=[(0.0, 0.0, 0.0)] * 3,
            mass=[0.01] * 3,
            radius=[0.02] * 3,
            custom_attributes=_MPM_PARTICLE_ATTRIBUTES,
        )
        sim.reset()
        solver = sim.backend_manager.backend.solver
        assert isinstance(solver, SolverImplicitMPM)

        calls = []
        original_project = solver.project_outside
        solver.project_outside = lambda *args, **kwargs: (calls.append(1), original_project(*args, **kwargs))
        sim.step(render=False)
        assert bool(calls) is project_outside


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
def test_mpm_cuda_graph_capture_supports_static_topology(overrides, expected):
    """Only fixed and capacity-bounded rebuildable sparse grids support outer capture."""
    values = {
        "grid_type": "sparse",
        "max_active_cell_count": 1024,
        "grid_padding": 0,
        "velocity_basis": "Q1",
        "strain_basis": "P0",
        "collider_basis": "S2",
    }
    solver = SimpleNamespace(**(values | overrides))

    assert mpm_supports_graph_capture(solver) is expected


def test_mpm_status_check_runs_only_after_graph_capture():
    """Sparse-grid asynchronous failures are queried only after graph replay."""
    calls = []
    solver = object.__new__(SolverImplicitMPM)
    solver.check_sparse_grid_rebuild_status = lambda: calls.append("check")

    MPMSolver.check_step(MPMSolverCfg(), solver, False, False)
    MPMSolver.check_step(MPMSolverCfg(), solver, True, False)

    assert calls == ["check"]


def test_nested_mpm_solver_discovery_selects_implicit_mpm_entries():
    """A coupled solver exposes only its implicit-MPM entries for status checks and capture decisions."""
    mpm_solver = object.__new__(SolverImplicitMPM)

    class CoupledSolver:
        def entry_names(self):
            return ("media", "rigid")

        def solver(self, name):
            return mpm_solver if name == "media" else object()

    assert implicit_mpm_solvers(CoupledSolver()) == (mpm_solver,)
    assert implicit_mpm_solvers(mpm_solver) == (mpm_solver,)
    assert implicit_mpm_solvers(None) == ()


@pytest.mark.parametrize("supported", [True, False])
def test_step_captures_only_when_the_solver_supports_it(monkeypatch, supported):
    """The first step runs eagerly; later steps replay a graph only when the solver can be captured."""
    events = []
    manager = _recording_manager(events, single_state=True, num_substeps=1, device="cuda:0", captures=supported)
    replays = []
    monkeypatch.setattr(manager, "_capture_graph", lambda segment: segment)
    monkeypatch.setattr(wp, "capture_launch", lambda segment: (replays.append(segment), segment()))
    manager._invalidate_graph()

    for _ in range(2):
        manager.step()

    assert (manager._graph is not None) is supported
    assert len(replays) == int(supported)
    # The capturing step ran eagerly first, so both steps advanced physics exactly once.
    assert events.count(("step", "state_0", "state_0")) == 2


# ---------------------------------------------------------------------------
# Manager state-refresh boundaries (no SimulationContext required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asset_class", [articulation_module.Articulation, rigid_object_module.RigidObject])
@pytest.mark.parametrize(
    "writer, selector",
    [
        ("write_root_link_pose_to_sim_index", "env_ids"),
        ("write_root_link_pose_to_sim_mask", "env_mask"),
        ("write_root_com_pose_to_sim_index", "env_ids"),
        ("write_root_com_pose_to_sim_mask", "env_mask"),
    ],
)
def test_fixed_root_pose_write_updates_solver(asset_class, writer, selector):
    """A model-backed root write immediately refreshes MuJoCo's root transform."""
    builder = ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.begin_world()
    initial_pose = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    root = builder.add_link(xform=initial_pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
    root_joint = builder.add_joint_fixed(-1, root, parent_xform=initial_pose)
    builder.add_articulation([root_joint], label="robot")
    builder.end_world()
    model = builder.finalize(device="cpu")
    view = ArticulationView(model, "robot", verbose=False)
    body_com = view.get_attribute("body_com", model)[:, 0]
    data = SimpleNamespace(
        root_link_pose_w=view.get_root_transforms(model)[:, 0],
        root_com_pose_w=wp.empty(1, dtype=wp.transform, device="cpu"),
        _sim_bind_body_com_pos_b=body_com,
        body_com_pos_b=body_com,
    )
    asset = SimpleNamespace(
        root_view=view,
        data=data,
        device="cpu",
        _ALL_ENV_MASK=wp.array([True], dtype=wp.bool, device="cpu"),
        _resolve_env_ids=lambda ids: ids,
        _resolve_mask=lambda mask, default: default if mask is None else mask,
        assert_shape_and_dtype=lambda *args: None,
        assert_shape_and_dtype_mask=lambda *args: None,
    )
    solver = SolverMuJoCo(model)
    asset._backend_manager = SimpleNamespace(backend=SimpleNamespace(solver=solver))
    target = wp.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], dtype=wp.transform, device="cpu")
    selection = wp.array([0], dtype=wp.int32, device="cpu") if selector == "env_ids" else asset._ALL_ENV_MASK

    getattr(asset_class, writer)(asset, root_pose=target, skip_forward=True, **{selector: selection})

    np.testing.assert_allclose(solver.mjw_data.mocap_pos.numpy()[0, 0], target.numpy()[0, :3])


def _backend_with_masks(world_mask: wp.array, fk_mask: wp.array, solver: object) -> SimpleNamespace:
    """Backend stand-in holding authored-state masks and a solver, without a model."""
    return SimpleNamespace(
        state_0=object(),
        world_mask=world_mask,
        fk_mask=fk_mask,
        kinematics_dirty=True,
        transforms_may_change_on_graph_replay=False,
        solver=solver,
    )


def test_forward_consumes_existing_reset_masks(monkeypatch):
    """A fresh state read consumes pending FK once; raw array binding leaves it pending."""
    world_mask = wp.array([False, True], dtype=wp.bool, device="cpu")
    fk_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    observed: list[tuple[list[bool], list[bool]]] = []
    solver_resets: list[list[bool]] = []

    class RecordingSolver(NewtonSolver):
        @classmethod
        def reset(cls, cfg, solver, state, worlds):
            solver_resets.append(worlds.numpy().tolist())

        @classmethod
        def eval_fk(cls, cfg, solver, state, worlds, articulations):
            observed.append((worlds.numpy().tolist(), articulations.numpy().tolist()))

    owner = NewtonManager(NewtonCfg(solver_cfg=NewtonSolverCfg(class_type=RecordingSolver)))
    owner._device = "cpu"
    owner.backend = _backend_with_masks(world_mask, fk_mask, object())

    scene_backend = newton_manager_module.NewtonSceneDataBackend(owner)
    raw_state = owner.backend.state_0
    with wp.ScopedCapture(device="cpu"):
        assert scene_backend.state is raw_state
    assert observed == []
    assert solver_resets == []
    assert scene_backend.state is owner.backend.state_0
    assert scene_backend.state is owner.backend.state_0

    assert observed == [([False, True], [True, False])]
    assert solver_resets == [[False, True]]
    assert world_mask.numpy().tolist() == [False, False]
    assert fk_mask.numpy().tolist() == [False, False]


def test_forward_keeps_implicit_mpm_history_on_automatic_resets(monkeypatch):
    """Authored-state boundaries must not reset shared-grid MPM history; tasks reset it explicitly."""
    world_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    fk_mask = wp.array([], dtype=wp.bool, device="cpu")

    class _RejectingSolver:
        def reset(self, state, world_mask=None, flags=0):
            raise AssertionError("automatic resets must not clear implicit-MPM history")

    class NoFKSolver(MPMSolver):
        @classmethod
        def eval_fk(cls, *args):
            pass

    manager = NewtonManager(NewtonCfg(solver_cfg=MPMSolverCfg(class_type=NoFKSolver)))
    manager.backend = _backend_with_masks(world_mask, fk_mask, _RejectingSolver())

    manager.forward()

    assert world_mask.numpy().tolist() == [False, False]


# ---------------------------------------------------------------------------
# Manager lifecycle contracts
# ---------------------------------------------------------------------------


def test_articulation_target_modes_are_resolved_once_for_replicas(monkeypatch):
    """Resolve target modes once, then copy them to the replicated articulations."""
    builder = SimpleNamespace(
        articulation_label=["/World/Env_0/Robot", "/World/Env_1/Robot"],
        articulation_start=[0, 1],
        articulation_end=[1, 2],
        joint_type=[JointType.REVOLUTE, JointType.REVOLUTE],
        joint_qd_start=[0, 1],
        joint_label=["/World/Env_0/Robot/joint", "/World/Env_1/Robot/joint"],
        joint_target_mode=[int(JointTargetMode.NONE)] * 2,
        joint_target_ke=[0.0] * 2,
        joint_target_kd=[0.0] * 2,
    )
    cfg = SimpleNamespace(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="",
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=0.0)},
    )
    original = articulation_module.resolve_matching_names
    actuator_resolutions = 0

    def count_actuator_resolutions(name_keys, names, *args, **kwargs):
        nonlocal actuator_resolutions
        if names == ["joint"]:
            actuator_resolutions += 1
        return original(name_keys, names, *args, **kwargs)

    monkeypatch.setattr(articulation_module, "resolve_matching_names", count_actuator_resolutions)
    sim = SimpleNamespace(backend_manager=SimpleNamespace(builder=builder))
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim)
    articulation_module.Articulation._configure_joint_target_modes(SimpleNamespace(cfg=cfg), None)

    assert builder.joint_target_mode == [int(JointTargetMode.POSITION)] * 2
    assert actuator_resolutions == 1


def test_reset_binds_consumers_before_constructing_the_solver(monkeypatch):
    """Each hard reset finalizes a new backend, lets consumers bind it, then constructs the solver."""
    events: list[str] = []
    allocations = []
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cpu",
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:

        def on_physics_ready(_):
            events.append("ready")
            backend = sim.get_or_create_backend(NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device))
            assert backend is sim.backend_manager.backend
            # Consumers author the model before solvers that copy it at construction exist.
            assert backend.solver is None
            allocations.append(backend)
            different = NewtonCfg(
                solver_cfg=MJWarpSolverCfg(), load_visual_shapes=not sim.cfg.physics.load_visual_shapes
            )
            assert sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=different)) is not builder
            sim.get_scene_data_provider().get_transforms(SceneDataFormat.Transform())

        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        # Picking registers its forces once the solver exists, before any capture.
        monkeypatch.setattr(sim, "_prepare_newton_visualizer_for_capture", lambda: events.append("prepare"))
        sim.backend_manager.register_callback(on_physics_ready, PhysicsEvent.PHYSICS_READY, wrap_weak_ref=False)

        sim.reset()
        assert isinstance(sim.backend_manager.backend.solver, SolverMuJoCo)
        sim.reset()

    assert events == ["ready", "prepare"] * 2
    assert len(allocations) == 2 and allocations[0] is not allocations[1]


# ---------------------------------------------------------------------------
# End-to-end: build each solver via SimulationContext
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_solver_cls, expected_use_single_state, expected_needs_collision_pipeline",
    SOLVER_MATRIX,
    ids=SOLVER_IDS,
)
def test_reset_binds_solver_and_capabilities(
    solver_cfg_factory,
    expected_solver_cls,
    expected_use_single_state,
    expected_needs_collision_pipeline,
):
    """``reset`` binds the configured solver and its capabilities on the backend, and steps.

    The builder gets the minimal content each solver needs: particles for MPM, cloth for VBD, and one jointed body
    (plus geometry for Kamino's internal detector) otherwise.
    """
    solver_cfg = solver_cfg_factory()
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=solver_cfg, use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        assert isinstance(sim.backend_manager, NewtonManager)

        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        if expected_solver_cls is SolverImplicitMPM:
            assert builder.has_custom_attribute("mpm:young_modulus")
            builder.add_particle_grid(
                pos=wp.vec3(-0.05, -0.05, 0.10),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=2,
                dim_y=2,
                dim_z=2,
                cell_x=0.05,
                cell_y=0.05,
                cell_z=0.05,
                mass=0.01,
                jitter=0.0,
                radius_mean=0.02,
            )
        elif expected_solver_cls is SolverVBD:
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, 0.1),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0),
                vertices=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.1, 0.0, 0.0), wp.vec3(0.0, 0.1, 0.0)],
                indices=[0, 1, 2],
                density=1.0,
                particle_radius=0.01,
            )
        else:
            body = builder.add_body(mass=1.0)
            builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
            if isinstance(solver_cfg, (KaminoPADMMSolverCfg, KaminoDVISolverCfg)) and solver_cfg.use_collision_detector:
                builder.add_shape_sphere(body=body, radius=0.05)
                builder.add_ground_plane()

        sim.reset()

        solver, solver_cls = sim.backend_manager.backend.solver, solver_cfg.class_type
        assert isinstance(solver, expected_solver_cls)
        assert solver_cls.single_state is expected_use_single_state
        assert solver_cls.uses_collision_pipeline(solver_cfg, solver) is expected_needs_collision_pipeline
        assert solver_cls.applies_body_forces(solver_cfg, solver) is (expected_solver_cls is not SolverImplicitMPM)
        # only Featherstone ignores inertial property changes after construction, and warns about it
        ignores_inertia = ModelFlags.BODY_INERTIAL_PROPERTIES in solver_cls.ignored_model_changes
        assert ignores_inertia is (expected_solver_cls is SolverFeatherstone)

        if expected_solver_cls not in (SolverKamino, SolverImplicitMPM):
            assert sim.backend_manager.backend.contacts is not None

        sim.step(render=False)


def test_step_combines_pending_model_changes(monkeypatch, caplog):
    """Queued property changes reach the solver together, once before the next step."""
    sim_cfg = SimulationCfg(
        device="cuda:0",
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), use_cuda_graph=False),
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        body = builder.add_link()
        builder.add_shape_sphere(body, radius=0.05)
        joint = builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        builder.add_articulation([joint])
        sim.reset()
        backend = sim.backend_manager.backend
        solver = backend.solver
        notify = solver.notify_model_changed
        calls = []

        def record_notification(flags):
            calls.append(flags)
            notify(flags)

        monkeypatch.setattr(solver, "notify_model_changed", record_notification)
        ignored = ModelFlags.BODY_INERTIAL_PROPERTIES
        monkeypatch.setattr(MJWarpSolver, "ignored_model_changes", {ignored: "Test ignored model change"})
        flags = (ignored, ModelFlags.JOINT_DOF_PROPERTIES, ignored)
        for flag in flags:
            sim.backend_manager.add_model_change(flag)
        assert calls == []

        with caplog.at_level(logging.WARNING, logger="isaaclab_newton.physics.newton_manager"):
            sim.step(render=False)
            sim.step(render=False)
            assert calls == [ignored | ModelFlags.JOINT_DOF_PROPERTIES]
            sim.backend_manager.add_model_change(ignored)
            sim.step(render=False)

        assert calls == [ignored | ModelFlags.JOINT_DOF_PROPERTIES, ignored]
        assert sum(record.message == "Test ignored model change" for record in caplog.records) == 1


def test_mjwarp_internal_contacts_with_collision_cfg_raises():
    """A ``collision_cfg`` is rejected on ``sim.reset()`` when the solver does not use Newton's collision pipeline."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True),
            collision_cfg=NewtonCollisionPipelineCfg(),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))

        with pytest.raises(ValueError, match="collision_cfg cannot be set"):
            sim.reset()


@pytest.mark.parametrize(
    "num_substeps, collision_decimation, expected_mid_loop_collides",
    [
        (8, 0, 0),  # Feature disabled.
        (8, 2, 3),  # Re-collide after substeps 2, 4, 6 (skip last).
        (8, 7, 1),  # Re-collide after substep 7 only.
    ],
)
def test_collision_decimation_invokes_mid_loop_collide(num_substeps, collision_decimation, expected_mid_loop_collides):
    """The step graph re-invokes ``collide`` at the expected substeps.

    Wraps the backend's collision pipeline ``collide`` with a counter and
    runs one physics tick. The collide-call count is ``1`` (top-of-tick) plus
    one per matching mid-loop substep, excluding the last substep.

    The scene has a free-joint sphere falling onto a ground plane so the
    broadphase actually generates pairs — guards against a future change
    that skips ``collide()`` when there are no collidable shapes.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),
            num_substeps=num_substeps,
            collision_decimation=collision_decimation,
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        body = builder.add_body(mass=1.0)
        builder.add_joint_free(child=body)
        builder.add_shape_sphere(body=body, radius=0.05)
        builder.add_ground_plane()
        # Lift the sphere to 0.5 m above the plane so the scene is non-degenerate.
        # joint_q for a free joint is [tx, ty, tz, qx, qy, qz, qw].
        builder.joint_q[-7:] = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        sim.reset()

        # Wrap collide() with a counter — must run after sim.reset() so the
        # pipeline is allocated and before the first step builds the step graph,
        # and use_cuda_graph=False so the wrapped callable isn't bypassed by a captured graph.
        calls = {"n": 0}
        pipeline = sim.backend_manager.backend.collision_pipeline
        original_collide = pipeline.collide

        def counting_collide(state, contacts):
            calls["n"] += 1
            return original_collide(state, contacts)

        pipeline.collide = counting_collide
        try:
            sim.step(render=False)
        finally:
            pipeline.collide = original_collide

        # Expect: 1 (top-of-tick) + expected_mid_loop_collides.
        assert calls["n"] == 1 + expected_mid_loop_collides


class _State:
    """Named state stand-in that records force clearing."""

    body_count = particle_count = 0

    def __init__(self, name, events):
        self.name, self._events = name, events

    def clear_forces(self):
        self._events.append(("clear", self.name))

    def assign(self, other):
        self._events.append(("assign", self.name, other.name))


def _recording_manager(events, *, single_state, num_substeps, device="cpu", captures=True):
    """Build a manager over a backend stand-in whose solver hooks and states record every operation of a step."""

    class RecordingSolver(NewtonSolver):
        @classmethod
        def supports_capture(cls, cfg, solver):
            return captures

        @classmethod
        def uses_collision_pipeline(cls, cfg, solver):
            return False

        @classmethod
        def step(cls, cfg, solver, state_in, state_out, *_args):
            events.append(("step", state_in.name, state_out.name))

        @classmethod
        def reset(cls, cfg, solver, state, worlds):
            events.append("reset")

        @classmethod
        def eval_fk(cls, cfg, solver, state, worlds, articulations):
            events.append("fk")

    RecordingSolver.single_state = single_state
    manager = NewtonManager(NewtonCfg())
    solver_cfg = NewtonSolverCfg(class_type=RecordingSolver)
    manager._cfg, manager._device = NewtonCfg(num_substeps=num_substeps, solver_cfg=solver_cfg), device
    manager._sim = SimpleNamespace(is_playing=lambda: True, has_gui=False, has_offscreen_render=False)
    manager.backend = SimpleNamespace(
        solver=object(),
        state_0=_State("state_0", events),
        state_1=_State("state_1", events),
        control=object(),
        collision_pipeline=None,
        contacts=None,
        dt=0.002,
        num_substeps=num_substeps,
        solver_dt=0.002 / num_substeps,
        actuators=None,
        contact_sensors={},
        imu_sensors=[],
        world_mask=wp.zeros(2, dtype=wp.bool, device=device),
        fk_mask=wp.zeros(1, dtype=wp.bool, device=device),
        kinematics_dirty=False,
    )
    return manager


@pytest.mark.parametrize("use_single_state", [True, False], ids=["single_state", "double_state"])
def test_state_force_callback_runs_before_every_solver_substep(use_single_state):
    """Viewer forces are applied to each current input state before solver stepping."""
    events = []
    manager = _recording_manager(events, single_state=use_single_state, num_substeps=2)
    manager.register_state_force_callback(lambda state: events.append(("force", state.name)))

    manager._run_solver_substeps(contacts=None)

    if use_single_state:
        assert events == [
            ("force", "state_0"),
            ("step", "state_0", "state_0"),
            ("clear", "state_0"),
            ("force", "state_0"),
            ("step", "state_0", "state_0"),
            ("clear", "state_0"),
        ]
    else:
        assert events == [
            ("force", "state_0"),
            ("step", "state_0", "state_1"),
            ("clear", "state_1"),
            ("force", "state_1"),
            ("step", "state_1", "state_0"),
            ("clear", "state_0"),
        ]


def test_decimation_loop_orders_callbacks_and_ends_each_physics_step_in_state_0():
    """Each physics step runs pre-actuator callbacks before substeps and ends in the canonical state; post-step runs
    once."""
    events = []
    manager = _recording_manager(events, single_state=False, num_substeps=1)
    manager.register_pre_actuator_callback(lambda: events.append(("pre",)))
    manager.register_post_step_callback(lambda: events.append(("post",)))
    manager._decimation = 2

    manager._simulate_full()

    step = [("pre",), ("step", "state_0", "state_1"), ("assign", "state_0", "state_1"), ("clear", "state_0")]
    assert events == step * 2 + [("post",)]


@wp.kernel
def _count_steps(counter: wp.array(dtype=wp.int32)):
    counter[0] += 1


@pytest.mark.parametrize(
    ("solver_cfg", "rtx_capture", "num_substeps", "decimation"),
    [
        (MJWarpSolverCfg(use_mujoco_contacts=True), True, 1, 1),
        (KaminoPADMMSolverCfg(), False, 1, 1),
        (MJWarpSolverCfg(use_mujoco_contacts=True), False, 2, 3),
        (KaminoPADMMSolverCfg(), False, 2, 3),
    ],
    ids=["mjwarp_rtx", "kamino_standard", "mjwarp_substeps_decimation", "kamino_substeps_decimation"],
)
def test_graph_capture_preserves_first_step_and_recapture(
    monkeypatch, solver_cfg, rtx_capture, num_substeps, decimation
):
    """The first step runs eagerly, later steps replay; staged forces hold for every substep and callbacks run once."""
    sim_cfg = SimulationCfg(
        dt=0.005,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=solver_cfg, num_substeps=num_substeps, use_cuda_graph=True),
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        builder.add_body(mass=1.0)
        builder.joint_q[-7:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        sim.reset()
        if rtx_capture:
            monkeypatch.setattr(newton_manager_module, "has_kit", lambda: True)
            monkeypatch.setattr(sim, "_has_offscreen_render", True)
        sim.backend_manager.activate_newton_actuator_path()
        counter = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        sim.backend_manager.register_post_step_callback(lambda: wp.launch(_count_steps, 1, inputs=[counter]))
        wrench = wp.array([[0.0, 0.0, 9.81, 0.0, 0.0, 0.0]], dtype=wp.spatial_vector, device="cuda:0")
        for step in range(3):
            if step != 1:
                sim.backend_manager.set_decimation(decimation)
            sim.backend_manager.backend.state_0.body_f.assign(wrench)
            sim.step(render=False)
            assert sim.backend_manager._graph is not None
            np.testing.assert_allclose(sim.backend_manager.backend.state_0.joint_qd.numpy(), 0.0, atol=1e-6)
            assert counter.numpy()[0] == step + 1


def test_stateful_actuator_graph_matches_eager_across_decimation_changes(monkeypatch):
    """PID and delay history survive repeated replay, recapture, and odd/even loop lengths."""
    trajectories = []
    for use_graph in (False, True):
        sim_cfg = SimulationCfg(
            dt=0.005,
            device="cuda:0",
            gravity=(0.0, 0.0, 0.0),
            physics=NewtonCfg(
                solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True), num_substeps=1, use_cuda_graph=use_graph
            ),
        )
        with build_simulation_context(sim_cfg=sim_cfg) as sim:
            builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
            builder.begin_world()
            body = builder.add_link(mass=1.0)
            joint = builder.add_joint_prismatic(parent=-1, child=body, axis=(1.0, 0.0, 0.0))
            builder.add_articulation([joint])
            builder.add_actuator(DrivePID, index=0, kp=1.0, kd=0.0, ki=0.5, delay_steps=2)
            builder.end_world()
            sim.reset()
            monkeypatch.setattr(newton_manager_module, "has_kit", lambda: True)
            monkeypatch.setattr(sim, "_has_offscreen_render", True)
            sim.backend_manager.activate_newton_actuator_path()
            samples = []
            for decimation in (1, 2, 3):
                sim.backend_manager.set_decimation(decimation)
                for target in (1.0, -2.0, 3.0, 0.0):
                    sim.backend_manager.backend.control.joint_target_q.fill_(target)
                    sim.step(render=False)
                    samples.append(
                        np.concatenate(
                            (
                                sim.backend_manager.backend.state_0.joint_q.numpy(),
                                sim.backend_manager.backend.state_0.joint_qd.numpy(),
                                sim.backend_manager.backend.control.joint_f.numpy(),
                            )
                        )
                    )
            trajectories.append(np.asarray(samples))
    np.testing.assert_allclose(trajectories[1], trajectories[0], atol=1e-6, rtol=1e-5)


def _build_collision_scene(sim, num_boxes=8):
    """Add ``num_boxes`` free-falling boxes over a ground plane."""
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
    for _ in range(num_boxes):
        body = builder.add_body(mass=1.0)
        builder.add_joint_free(child=body)
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()


# Model device arrays ``CollisionPipeline.collide()`` reads off its cached model.
_COLLIDE_MODEL_ARRAYS = (
    "shape_transform",
    "shape_body",
    "shape_type",
    "shape_scale",
    "shape_collision_radius",
    "shape_source_ptr",
    "shape_margin",
    "shape_gap",
    "shape_collision_aabb_lower",
    "shape_collision_aabb_upper",
)


def _free_model_collide_arrays_and_churn(model, device):
    """Free the arrays ``collide()`` reads off ``model``, then churn the allocator so stale reads see garbage."""
    import gc

    for attr in _COLLIDE_MODEL_ARRAYS:
        arr = getattr(model, attr, None)
        if isinstance(arr, wp.array) and arr.device.is_cuda:
            setattr(model, attr, None)
    gc.collect()
    wp.synchronize_device(device)
    _churn = [wp.zeros(1 << 16, dtype=wp.float32, device=device) for _ in range(128)]  # noqa: F841
    wp.synchronize_device(device)


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_hard_reset_then_step_runs(use_cuda_graph):
    """A step after a hard reset never touches the old model's freed collision buffers (CUDA 700 before the fix)."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),
            num_substeps=2,
            use_cuda_graph=use_cuda_graph,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        _build_collision_scene(sim)

        sim.reset()
        old_backend = sim.backend_manager.backend
        assert old_backend.collision_pipeline is not None
        old_model = old_backend.collision_pipeline.model
        sim.step(render=False)

        sim.reset()
        assert sim.backend_manager.backend is not old_backend
        assert old_backend.model is old_backend.state_0 is old_backend.state_1 is old_backend.control is None
        assert sum(isinstance(cfg, NewtonBackendCfg) for cfg, _ in sim._backend_registry) == 1

        _free_model_collide_arrays_and_churn(old_model, "cuda:0")

        # A hard device sync surfaces any deferred illegal access as an exception.
        sim.step(render=False)
        wp.synchronize_device("cuda:0")


@pytest.mark.parametrize(
    "solver_cfg, deterministic, expected",
    [
        pytest.param(XPBDSolverCfg(), False, wp.DeterministicMode.NOT_GUARANTEED, id="no_request"),
        pytest.param(XPBDSolverCfg(), True, wp.DeterministicMode.RUN_TO_RUN, id="request_xpbd"),
        pytest.param(MJWarpSolverCfg(), True, wp.DeterministicMode.RUN_TO_RUN, id="request_mjwarp_gpu"),
        # MuJoCo on the CPU is already reproducible; Warp's mode does not reach it.
        pytest.param(
            MJWarpSolverCfg(use_mujoco_cpu=True), True, wp.DeterministicMode.NOT_GUARANTEED, id="request_mujoco_cpu"
        ),
    ],
)
def test_apply_deterministic_request_translates_the_generic_flag(solver_cfg, deterministic, expected) -> None:
    """The backend owns translation of :attr:`PhysicsCfg.deterministic` into Newton settings."""
    cfg = NewtonCfg(solver_cfg=solver_cfg, deterministic=deterministic)

    assert resolve_deterministic_mode(cfg) == expected


@pytest.mark.parametrize("mode", ["run_to_run", "gpu_to_gpu"])
def test_explicit_mode_on_mujoco_cpu_is_reported_not_enforced(mode, caplog) -> None:
    """MuJoCo-C is reproducible on its own, so an explicit mode is reported rather than enforced."""
    cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_cpu=True), deterministic_mode=mode)

    with caplog.at_level(logging.INFO, logger=mjwarp_solver_module.__name__):
        assert resolve_deterministic_mode(cfg) == wp.DeterministicMode.NOT_GUARANTEED

    assert any("already reproducible" in r.getMessage() for r in caplog.records)


@pytest.fixture
def recorded_mujoco_kwargs(monkeypatch) -> list[dict]:
    """Replace Newton's MuJoCo solver with one recording its constructor arguments."""
    constructed = []
    record = functools.wraps(SolverMuJoCo.__init__)(lambda self, model, **kwargs: constructed.append(kwargs))
    monkeypatch.setattr(MJWarpSolver, "solver_class", type("RecordingSolver", (), {"__init__": record}))
    return constructed


def test_determinism_guarantee_disables_mjwarp_sensors(recorded_mujoco_kwargs) -> None:
    """MuJoCo Warp's sensor stage cannot honor a guarantee, so the solver is built without it."""
    solver_cfg = MJWarpSolverCfg()

    MJWarpSolver.build(solver_cfg, _model_stub(), wp.DeterministicMode.NOT_GUARANTEED)
    MJWarpSolver.build(solver_cfg, _model_stub(), wp.DeterministicMode.RUN_TO_RUN)

    assert [kwargs["disable_sensors"] for kwargs in recorded_mujoco_kwargs] == [False, True]
    assert solver_cfg.disable_sensors is False


def test_apply_deterministic_request_keeps_an_explicit_mode() -> None:
    """An explicitly requested mode is the more specific instruction and wins."""
    cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(), deterministic=True, deterministic_mode="gpu_to_gpu")

    assert resolve_deterministic_mode(cfg) == wp.DeterministicMode.GPU_TO_GPU


@pytest.mark.parametrize("attr", ["body_qdd", "body_parent_f"])
def test_deterministic_mode_rejects_sensors_that_need_the_sensor_stage(attr, recorded_mujoco_kwargs) -> None:
    """Disabling MJWarp sensors starves IMU/PVA/joint-wrench, so the request is refused."""
    with pytest.raises(ValueError, match="does not support deterministic physics"):
        MJWarpSolver.build(MJWarpSolverCfg(), _model_stub({attr}), wp.DeterministicMode.RUN_TO_RUN)
    assert recorded_mujoco_kwargs == []


def test_deterministic_mode_allows_those_sensors_without_a_guarantee(recorded_mujoco_kwargs) -> None:
    """The sensors are only incompatible with the guarantee, not with MJWarp itself."""
    MJWarpSolver.build(MJWarpSolverCfg(), _model_stub({"body_qdd"}), wp.DeterministicMode.NOT_GUARANTEED)
    assert len(recorded_mujoco_kwargs) == 1
