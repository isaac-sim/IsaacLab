# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the per-solver :class:`NewtonManager` abstraction.

Covers:

* :attr:`NewtonSolverCfg.class_type` resolves to the matching manager subclass.
* :meth:`NewtonCfg.__post_init__` propagates ``solver_cfg.class_type`` onto
  :attr:`NewtonCfg.class_type` so that ``SimulationContext`` picks the right
  manager.
* Each leaf manager subclasses :class:`NewtonManager` and implements
  :meth:`_build_solver` (with the abstract base raising ``NotImplementedError``).
* The cross-config validation in :meth:`NewtonMJWarpManager._build_solver`
  rejects the ``MJWarp + use_mujoco_contacts=True + collision_cfg`` combination.
* Manager name dispatch (used by :class:`InteractiveScene` and the various
  factory dispatchers) still starts with ``"newton"``.
* End-to-end: spinning up a simulation with each solver builds the correct
  solver, sets the right ``_use_single_state`` / ``_needs_collision_pipeline``
  flags, and lands canonical state on :class:`NewtonManager` so that external
  ``NewtonManager._foo`` reads keep working.
"""

from __future__ import annotations

from types import SimpleNamespace

import isaaclab_newton.physics.newton_manager as newton_manager_module
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    KaminoDVICfg,
    KaminoDVISolverCfg,
    KaminoDynamicsCfg,
    KaminoPADMMCfg,
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonFeatherstoneManager,
    NewtonKaminoManager,
    NewtonManager,
    NewtonMJWarpManager,
    NewtonMPMManager,
    NewtonShapeCfg,
    NewtonSolverCfg,
    NewtonVBDManager,
    NewtonXPBDManager,
    VBDSolverCfg,
    XPBDSolverCfg,
)
from isaaclab_newton.physics.mpm_manager import _make_solver_config
from newton.solvers import SolverFeatherstone, SolverImplicitMPM, SolverKamino, SolverMuJoCo, SolverVBD, SolverXPBD

from isaaclab.physics import PhysicsManager
from isaaclab.sim import SimulationCfg, build_simulation_context

# ---------------------------------------------------------------------------
# Lightweight (no sim) parametrisation
# ---------------------------------------------------------------------------

# (solver_cfg_factory, expected_manager, expected_solver_cls,
#  expected_use_single_state, expected_needs_collision_pipeline)
SOLVER_MATRIX = [
    pytest.param(
        lambda: MJWarpSolverCfg(use_mujoco_contacts=True),
        NewtonMJWarpManager,
        SolverMuJoCo,
        True,
        False,
        id="mjwarp_internal_contacts",
    ),
    pytest.param(
        lambda: MJWarpSolverCfg(use_mujoco_contacts=False),
        NewtonMJWarpManager,
        SolverMuJoCo,
        True,
        True,
        id="mjwarp_newton_pipeline",
    ),
    pytest.param(
        lambda: XPBDSolverCfg(),
        NewtonXPBDManager,
        SolverXPBD,
        False,
        True,
        id="xpbd",
    ),
    pytest.param(
        lambda: VBDSolverCfg(),
        NewtonVBDManager,
        SolverVBD,
        False,
        True,
        id="vbd",
    ),
    pytest.param(
        lambda: FeatherstoneSolverCfg(),
        NewtonFeatherstoneManager,
        SolverFeatherstone,
        False,
        True,
        id="featherstone",
    ),
    pytest.param(
        lambda: KaminoPADMMSolverCfg(use_collision_detector=True),
        NewtonKaminoManager,
        SolverKamino,
        False,
        False,
        id="kamino_internal_contacts",
    ),
    pytest.param(
        lambda: KaminoPADMMSolverCfg(use_collision_detector=False),
        NewtonKaminoManager,
        SolverKamino,
        False,
        True,
        id="kamino_newton_pipeline",
    ),
    pytest.param(
        lambda: MPMSolverCfg(max_iterations=2, voxel_size=0.05),
        NewtonMPMManager,
        SolverImplicitMPM,
        True,
        False,
        id="implicit_mpm",
    ),
]

RIGID_BODY_FORCE_INPUT_SUPPORT = {
    NewtonMJWarpManager: True,
    NewtonVBDManager: True,
    NewtonXPBDManager: True,
    NewtonFeatherstoneManager: True,
    NewtonKaminoManager: True,
    NewtonMPMManager: False,
}


# ---------------------------------------------------------------------------
# class_type wiring (no SimulationContext required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline",
    SOLVER_MATRIX,
)
def test_solver_cfg_class_type_resolves_to_subclass(
    solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline
):
    """Each ``*SolverCfg.class_type`` resolves to its matching manager subclass."""
    solver_cfg = solver_cfg_factory()
    # ``class_type`` is a lazy ``"module:Class"`` reference; calling its
    # ``_resolve()`` returns the actual class. ``__name__`` works without
    # forcing import (LazyType caches metadata) and is sufficient identity.
    assert solver_cfg.class_type.__name__ == expected_manager.__name__


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline",
    SOLVER_MATRIX,
)
def test_newton_cfg_post_init_propagates_class_type(
    solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline
):
    """``NewtonCfg.__post_init__`` lifts ``solver_cfg.class_type`` onto ``NewtonCfg.class_type``."""
    cfg = NewtonCfg(solver_cfg=solver_cfg_factory())
    assert cfg.class_type.__name__ == expected_manager.__name__


@pytest.mark.parametrize(
    "num_substeps, collision_decimation, should_warn",
    [
        (8, 0, False),  # Default: feature disabled, no warning.
        (8, 1, False),  # Valid: re-collide every substep.
        (8, 2, False),  # Valid: re-collide every 2 substeps.
        (8, 7, False),  # Valid edge: one mid-loop re-collide at i=6.
        (8, 8, True),  # Equal to num_substeps: gate never fires.
        (8, 16, True),  # Larger than num_substeps: gate never fires.
    ],
)
def test_newton_cfg_collision_decimation_warning(num_substeps, collision_decimation, should_warn, caplog):
    """``NewtonCfg.__post_init__`` warns when ``collision_decimation >= num_substeps``."""
    import logging

    with caplog.at_level(logging.WARNING, logger="isaaclab_newton.physics.newton_manager_cfg"):
        cfg = NewtonCfg(num_substeps=num_substeps, collision_decimation=collision_decimation)
    warned = any("collision_decimation" in rec.getMessage() for rec in caplog.records)
    assert warned is should_warn
    # Cfg field round-trips regardless of warning.
    assert cfg.collision_decimation == collision_decimation


def test_solver_kwargs_include_newton_deterministic_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solver construction should receive the mode configured on the outer Newton config."""
    monkeypatch.setattr(NewtonManager, "_deterministic_mode", wp.DeterministicMode.GPU_TO_GPU)

    kwargs = NewtonManager._filter_solver_kwargs(SolverXPBD, XPBDSolverCfg())

    assert kwargs["deterministic"] == wp.DeterministicMode.GPU_TO_GPU


@pytest.mark.parametrize(
    "solver_cfg",
    [
        pytest.param(KaminoPADMMSolverCfg(), id="kamino_padmm"),
        pytest.param(MPMSolverCfg(), id="implicit_mpm"),
        pytest.param(MJWarpSolverCfg(use_mujoco_cpu=True), id="mujoco_cpu"),
        pytest.param(MJWarpSolverCfg(), id="mujoco_warp_sensors"),
    ],
)
def test_deterministic_mode_rejects_unsupported_solver_cfg(solver_cfg) -> None:
    """Unsupported solvers should not silently ignore a determinism guarantee."""
    with pytest.raises(ValueError, match="not supported"):
        NewtonManager._validate_deterministic_solver_cfg(solver_cfg, wp.DeterministicMode.GPU_TO_GPU)


@pytest.mark.parametrize(
    "solver_cfg_cls, solver_cfg_kwargs",
    [
        pytest.param(FeatherstoneSolverCfg, {}, id="featherstone"),
        pytest.param(MJWarpSolverCfg, {"disable_sensors": True}, id="mujoco_warp"),
        pytest.param(XPBDSolverCfg, {}, id="xpbd"),
    ],
)
def test_deterministic_mode_accepts_supported_solver_cfg_subclasses(solver_cfg_cls, solver_cfg_kwargs) -> None:
    """Custom subclasses of supported solver configs should retain deterministic support."""

    class CustomSolverCfg(solver_cfg_cls):
        pass

    NewtonManager._validate_deterministic_solver_cfg(
        CustomSolverCfg(**solver_cfg_kwargs), wp.DeterministicMode.GPU_TO_GPU
    )


def test_deterministic_collision_pipeline_matches_expanded_contact_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deterministic sorting buffers should grow with the solver contact buffer."""
    pipeline_calls: list[dict] = []

    class FakeCollisionPipeline:
        def __init__(self, _model, **kwargs):
            pipeline_calls.append(kwargs)
            self._rigid_contact_max = kwargs.get("rigid_contact_max", 1)

        def contacts(self):
            return SimpleNamespace(rigid_contact_max=self._rigid_contact_max)

    solver = SimpleNamespace(get_max_contact_count=lambda: 2)
    monkeypatch.setattr(newton_manager_module, "CollisionPipeline", FakeCollisionPipeline)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", True)
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", None)
    monkeypatch.setattr(NewtonManager, "_collision_cfg", None)
    monkeypatch.setattr(NewtonManager, "_contacts", None)
    monkeypatch.setattr(NewtonManager, "_solver", solver)
    monkeypatch.setattr(NewtonManager, "_model", SimpleNamespace())
    monkeypatch.setattr(NewtonManager, "_deterministic_mode", wp.DeterministicMode.GPU_TO_GPU)

    NewtonManager._initialize_contacts()

    assert pipeline_calls == [
        {"broad_phase": "explicit", "deterministic": True},
        {"broad_phase": "explicit", "deterministic": True, "rigid_contact_max": 2},
    ]
    assert NewtonManager._contacts.rigid_contact_max == 2


def test_refit_sensor_bvh_rejects_missing_sensor_state(monkeypatch):
    """BVH refitting raises when a particle BVH exists without an initialized sensor state."""
    model = SimpleNamespace(shape_count=0, particle_count=1, bvh_particles=object())
    monkeypatch.setattr(NewtonManager, "_model", model, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_state", None, raising=False)

    with pytest.raises(RuntimeError, match="requires an initialized sensor state"):
        NewtonManager._refit_sensor_bvh()


def test_sensor_task_builds_and_refits_bvhs_before_rendering(monkeypatch):
    """Shape and particle BVHs are built and refit before a render task runs."""

    state = object()
    status = {"shape_refit": False, "particle_refit": False, "rendered": False}

    class FakeModel:
        shape_count = 1
        particle_count = 1
        bvh_shapes = None
        bvh_particles = None

        def bvh_build_shapes(self, current_state):
            assert current_state is state
            self.bvh_shapes = object()

        def bvh_build_particles(self, current_state):
            assert current_state is state
            self.bvh_particles = object()

        def bvh_refit_shapes(self, current_state):
            assert current_state is state
            status["shape_refit"] = True

        def bvh_refit_particles(self, current_state):
            assert current_state is state
            status["particle_refit"] = True

    model = FakeModel()

    def render():
        assert model.bvh_shapes is not None
        assert model.bvh_particles is not None
        assert status["shape_refit"]
        assert status["particle_refit"]
        status["rendered"] = True

    monkeypatch.setattr(NewtonManager, "get_model", classmethod(lambda cls: model))
    monkeypatch.setattr(NewtonManager, "get_state_0", classmethod(lambda cls: state))
    monkeypatch.setattr(NewtonManager, "_model", model, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_tasks", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_state", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_state_dirty", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_graph", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_flags", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_flags_host", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_graph_capture_failed", False, raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=False), raising=False)

    NewtonManager._register_sensor_task("render", render)
    NewtonManager._update_sensor_tasks("render")

    assert status["rendered"]


def test_newton_shape_cfg_defaults_match_newton_shape_config():
    """``NewtonShapeCfg`` contact defaults mirror Newton's ``ShapeConfig``.

    Guards the invariant that keeps ``checked_apply`` a no-op for envs that do
    not override ``ke``/``kd``/``mu``: if Newton's upstream defaults drift, this
    fails instead of silently clobbering every Newton scene's shape materials.
    """
    import newton

    upstream = newton.ModelBuilder().default_shape_cfg
    shape_cfg = NewtonShapeCfg()
    assert shape_cfg.ke == upstream.ke
    assert shape_cfg.kd == upstream.kd
    assert shape_cfg.mu == upstream.mu


def test_mpm_solver_cfg_maps_only_newton_solver_fields():
    """MPM config forwarding ignores Isaac Lab metadata fields explicitly."""

    solver_cfg = MPMSolverCfg(
        max_iterations=7,
        voxel_size=0.04,
        solver_type="isaaclab_metadata_should_not_forward",
    )

    newton_cfg = _make_solver_config(solver_cfg)

    assert newton_cfg.max_iterations == 7
    assert newton_cfg.voxel_size == 0.04
    assert not hasattr(newton_cfg, "class_type")
    assert not hasattr(newton_cfg, "solver_type")
    # Manager-level stepping option must not leak into the Newton solver config.
    assert not hasattr(newton_cfg, "project_outside_colliders")


@pytest.mark.parametrize(
    "deprecated_value, replacement",
    [
        ("instantaneous", "forward"),
        ("finite_difference", "backward"),
    ],
)
def test_mpm_solver_cfg_translates_deprecated_collider_velocity_modes(deprecated_value, replacement):
    """Deprecated collider velocity modes warn and map to Newton's current values."""
    with pytest.warns(DeprecationWarning, match=f"use {replacement!r}"):
        newton_cfg = _make_solver_config(MPMSolverCfg(collider_velocity_mode=deprecated_value))

    assert newton_cfg.collider_velocity_mode == replacement


@pytest.mark.parametrize("mode", ["forward", "backward"])
def test_mpm_solver_cfg_preserves_canonical_collider_velocity_modes(mode, recwarn):
    """Canonical collider velocity modes pass through without deprecation warnings."""
    newton_cfg = _make_solver_config(MPMSolverCfg(collider_velocity_mode=mode))

    assert newton_cfg.collider_velocity_mode == mode
    assert not [warning for warning in recwarn if issubclass(warning.category, DeprecationWarning)]


# Tuples of ``(field_name, non_default_value)`` covering every solver-tunable
# field on :class:`MPMSolverCfg`. Each entry exercises the implementation-side
# SolverImplicitMPM.Config construction so a Newton field rename or accidental
# drop is caught here instead of silently producing wrong-physics runs.
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


@pytest.mark.parametrize("field_name, value", _MPM_FIELD_VALUES)
def test_mpm_solver_cfg_forwards_every_solver_field(field_name, value):
    """Every tunable MPM cfg field round-trips into ``SolverImplicitMPM.Config``.

    Guards against MPM manager construction dropping or mis-naming a field if
    Newton's config surface changes.
    """
    solver_cfg = MPMSolverCfg(**{field_name: value})
    newton_cfg = _make_solver_config(solver_cfg)
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


@pytest.mark.parametrize("field_name, value", _KAMINO_PADMM_FIELD_VALUES)
def test_kamino_solver_cfg_forwards_padmm_fields(field_name, value):
    """Every tunable P-ADMM cfg field round-trips into ``PADMMSolverConfig``."""
    solver_cfg = KaminoPADMMSolverCfg(dynamics_solver_cfg=KaminoPADMMCfg(**{field_name: value}))
    newton_cfg = solver_cfg.to_solver_config()
    assert hasattr(newton_cfg.padmm, field_name), (
        f"{field_name!r} disappeared from PADMMSolverConfig — KaminoPADMMCfg needs to drop or rename it."
    )
    assert getattr(newton_cfg.padmm, field_name) == value


@pytest.mark.parametrize("field_name, value", _KAMINO_DVI_FIELD_VALUES)
def test_kamino_solver_cfg_forwards_dvi_fields(field_name, value):
    """Every tunable DVI cfg field round-trips into ``DVISolverConfig``."""
    solver_cfg = KaminoDVISolverCfg(
        dynamics=KaminoDynamicsCfg(preconditioning=False),
        dynamics_solver_cfg=KaminoDVICfg(**{field_name: value}),
    )
    newton_cfg = solver_cfg.to_solver_config()
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


def test_mpm_register_builder_attributes_is_idempotent():
    """The MPM custom-attribute hook is a no-op when attributes are already registered."""
    import newton

    builder = newton.ModelBuilder()
    assert not builder.has_custom_attribute("mpm:young_modulus")

    NewtonMPMManager._register_builder_attributes(builder)
    assert builder.has_custom_attribute("mpm:young_modulus")

    # Second call must be a no-op (no exceptions, attribute still present).
    NewtonMPMManager._register_builder_attributes(builder)
    assert builder.has_custom_attribute("mpm:young_modulus")


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

    NewtonMPMManager._prepare_builder_for_finalize(builder)

    assert builder.body_flags[kinematic_body] & int(newton.BodyFlags.KINEMATIC)
    assert builder.body_mass[kinematic_body] == 0.0
    assert builder.body_inv_mass[kinematic_body] == 0.0
    assert np.allclose(np.array(builder.body_inertia[kinematic_body]), 0.0)
    assert np.allclose(np.array(builder.body_inv_inertia[kinematic_body]), 0.0)

    assert builder.body_mass[dynamic_body] == pytest.approx(1.2)
    assert builder.body_inv_mass[dynamic_body] == pytest.approx(1.0 / 1.2)
    assert np.allclose(np.array(builder.body_inertia[dynamic_body]), 2.0)


@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_mpm_prepare_builder_converts_convex_mesh_before_solver_construction():
    """Convex meshes must become triangle meshes before implicit MPM consumes the model."""
    import newton

    builder = newton.ModelBuilder()
    NewtonMPMManager._register_builder_attributes(builder)
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
        custom_attributes={
            "mpm:viscosity": 50.0,
            "mpm:friction": 0.0,
            "mpm:tensile_yield_ratio": 1.0,
            "mpm:yield_pressure": 1.0e15,
            "mpm:yield_stress": 0.0,
            "mpm:young_modulus": 1.0e15,
            "mpm:damping": 0.0,
        },
    )

    NewtonMPMManager._prepare_builder_for_finalize(builder)
    model = builder.finalize(device="cuda:0")
    solver = NewtonMPMManager._create_solver(model, MPMSolverCfg(max_iterations=2, voxel_size=0.05))

    assert builder.shape_type[shape] == newton.GeoType.MESH
    assert isinstance(solver, SolverImplicitMPM)


def test_active_manager_create_builder_registers_mpm_attributes():
    """The active MPM manager registers solver-specific builder attributes."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()

    assert builder.has_custom_attribute("mpm:young_modulus")


def test_mpm_end_to_end_with_particle_custom_attributes():
    """End-to-end MPM step using ``add_particles(custom_attributes=...)`` — the production path."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()
        # MPM custom attrs must exist on the builder before particles use them.
        assert builder.has_custom_attribute("mpm:young_modulus")

        positions = [(0.0, 0.0, 0.10), (0.05, 0.0, 0.10), (0.0, 0.05, 0.10)]
        builder.add_particles(
            pos=positions,
            vel=[(0.0, 0.0, 0.0)] * len(positions),
            mass=[0.01] * len(positions),
            radius=[0.02] * len(positions),
            custom_attributes={
                "mpm:viscosity": 50.0,
                "mpm:friction": 0.0,
                "mpm:tensile_yield_ratio": 1.0,
                "mpm:yield_pressure": 1.0e15,
                "mpm:yield_stress": 0.0,
                "mpm:young_modulus": 1.0e15,
                "mpm:damping": 0.0,
            },
        )
        NewtonManager.set_builder(builder)

        sim.reset()
        assert isinstance(NewtonManager._solver, SolverImplicitMPM)
        sim.step(render=False)


@pytest.mark.parametrize("project_outside", [True, False])
def test_mpm_project_outside_colliders_gates_projection(project_outside):
    """``project_outside_colliders`` controls whether ``project_outside`` runs per substep.

    Wraps the solver's ``project_outside`` with a counter after ``sim.reset()``
    (``use_cuda_graph=False`` keeps the Python callable on the step path) and
    runs one tick. The call count is positive only when the flag is set.
    """
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
        builder = sim.physics_manager.create_builder()
        builder.add_particles(
            pos=[(0.0, 0.0, 0.10), (0.05, 0.0, 0.10), (0.0, 0.05, 0.10)],
            vel=[(0.0, 0.0, 0.0)] * 3,
            mass=[0.01] * 3,
            radius=[0.02] * 3,
            custom_attributes={
                "mpm:viscosity": 50.0,
                "mpm:friction": 0.0,
                "mpm:tensile_yield_ratio": 1.0,
                "mpm:yield_pressure": 1.0e15,
                "mpm:yield_stress": 0.0,
                "mpm:young_modulus": 1.0e15,
                "mpm:damping": 0.0,
            },
        )
        NewtonManager.set_builder(builder)
        sim.reset()

        calls = {"n": 0}
        original_project = NewtonManager._solver.project_outside

        def counting_project(*args, **kwargs):
            calls["n"] += 1
            return original_project(*args, **kwargs)

        NewtonManager._solver.project_outside = counting_project
        try:
            sim.step(render=False)
        finally:
            NewtonManager._solver.project_outside = original_project

        if project_outside:
            assert calls["n"] >= 1
        else:
            assert calls["n"] == 0


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({"grid_type": "fixed"}, True, id="fixed"),
        pytest.param({}, True, id="bounded_sparse"),
        pytest.param({"max_active_cell_count": -1}, False, id="unbounded_sparse"),
        pytest.param({"grid_type": "dense"}, False, id="dense"),
        pytest.param({"grid_padding": 1}, False, id="padded_sparse"),
        pytest.param({"velocity_basis": "P0"}, False, id="velocity_basis"),
        pytest.param({"strain_basis": "GIMP"}, False, id="strain_basis"),
        pytest.param({"collider_basis": "GIMP"}, False, id="collider_basis"),
    ],
)
def test_mpm_cuda_graph_capture_supports_static_topology(monkeypatch, overrides, expected):
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
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)

    assert NewtonMPMManager._supports_cuda_graph_capture() is expected


def test_mpm_status_check_runs_only_after_graph_capture(monkeypatch):
    """Sparse-grid asynchronous failures are queried only after graph replay."""
    calls = []
    solver = SimpleNamespace(check_sparse_grid_rebuild_status=lambda: calls.append("check"))
    monkeypatch.setattr(NewtonMPMManager, "_implicit_mpm_solvers", classmethod(lambda cls: (solver,)))
    monkeypatch.setattr(NewtonManager, "_graph", None)

    NewtonMPMManager._check_solver_status()
    monkeypatch.setattr(NewtonManager, "_graph", object())
    NewtonMPMManager._check_solver_status()

    assert calls == ["check"]


def test_nested_mpm_solver_discovery_is_cached(monkeypatch):
    """A coupled solver's immutable entry table is traversed only once per solver instance."""
    mpm_solver = object.__new__(SolverImplicitMPM)

    class CoupledSolver:
        calls = 0

        def entry_names(self):
            self.calls += 1
            return ("media",)

        def solver(self, _name):
            return mpm_solver

    root = CoupledSolver()
    monkeypatch.setattr(NewtonManager, "_solver", root)
    monkeypatch.setattr(NewtonMPMManager, "_implicit_mpm_solver_root", None)
    monkeypatch.setattr(NewtonMPMManager, "_implicit_mpm_solver_cache", ())

    assert NewtonMPMManager._implicit_mpm_solvers() == (mpm_solver,)
    assert NewtonMPMManager._implicit_mpm_solvers() == (mpm_solver,)
    assert root.calls == 1


def test_mpm_supported_cuda_graph_capture_defers_until_initial_reset(monkeypatch):
    """A bounded sparse grid must not capture before reset-authored topology exists."""
    solver = SimpleNamespace(
        grid_type="sparse",
        max_active_cell_count=1024,
        grid_padding=0,
        velocity_basis="Q1",
        strain_basis="P0",
        collider_basis="S2",
    )
    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", False, raising=False)

    class UnexpectedCapture:
        def __init__(self, *args, **kwargs):
            pytest.fail("MPM capture started before the initial environment reset.")

    monkeypatch.setattr(wp, "ScopedCapture", UnexpectedCapture)

    NewtonMPMManager._capture_or_defer_graph()

    assert NewtonManager._graph is None
    assert NewtonManager._graph_capture_pending is True


def test_mpm_unsupported_cuda_graph_capture_uses_eager_execution(monkeypatch):
    """An unbounded sparse grid should retain the eager-execution fallback."""
    solver = SimpleNamespace(
        grid_type="sparse",
        max_active_cell_count=-1,
        grid_padding=0,
        velocity_basis="Q1",
        strain_basis="P0",
        collider_basis="S2",
    )
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(solver_cfg=MPMSolverCfg(grid_type="sparse"), use_cuda_graph=True),
        raising=False,
    )
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True, raising=False)

    NewtonMPMManager._capture_or_defer_graph()

    assert NewtonManager._graph is None
    assert NewtonManager._graph_capture_pending is False


def test_cuda_graph_capture_uses_simulation_device(monkeypatch):
    """CUDA graph capture should use the simulation device instead of Warp's default device."""

    captured_devices = []
    captured_graph = object()

    class FakeScopedCapture:
        def __init__(self, device=None):
            captured_devices.append(device)
            self.graph = captured_graph

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:1", raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_is_all_graphable", classmethod(lambda cls: False))
    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(lambda cls: None))
    monkeypatch.setattr(wp, "ScopedCapture", FakeScopedCapture)

    NewtonManager._capture_or_defer_graph()

    assert captured_devices == ["cuda:1"]
    assert NewtonManager._graph is captured_graph


# ---------------------------------------------------------------------------
# Manager state-refresh boundaries (no SimulationContext required)
# ---------------------------------------------------------------------------


def test_forward_consumes_existing_reset_masks(monkeypatch):
    """The existing device masks are the complete input to masked FK and the solver reset hook."""
    world_mask = wp.array([False, True], dtype=wp.bool, device="cpu")
    fk_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    observed: list[tuple[list[bool], list[bool]]] = []
    solver_resets: list[list[bool]] = []

    def record_fk(worlds, articulations):
        observed.append((worlds.numpy().tolist(), articulations.numpy().tolist()))

    class _RecordingSolver:
        def reset(self, state, world_mask=None, flags=0):
            solver_resets.append(world_mask.numpy().tolist())

    monkeypatch.setattr(NewtonManager, "_world_reset_mask", world_mask, raising=False)
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", fk_mask, raising=False)
    monkeypatch.setattr(NewtonManager, "_eval_fk", record_fk, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", _RecordingSolver(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_reset_solver_internals_delegate",
        NewtonManager._reset_solver_internals,
        raising=False,
    )

    NewtonManager.forward()

    assert observed == [([False, True], [True, False])]
    assert solver_resets == [[False, True]]
    assert world_mask.numpy().tolist() == [False, False]
    assert fk_mask.numpy().tolist() == [False, False]


def test_forward_dispatches_active_mpm_reset_hook_through_base_manager(monkeypatch):
    """Base-class state reads must use the active MPM manager's reset behavior."""
    world_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    fk_mask = wp.array([], dtype=wp.bool, device="cpu")

    class _RejectingSolver:
        def reset(self, state, world_mask=None, flags=0):
            raise AssertionError("the base reset hook must not run for implicit MPM")

    monkeypatch.setattr(NewtonManager, "_world_reset_mask", world_mask, raising=False)
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", fk_mask, raising=False)
    monkeypatch.setattr(NewtonManager, "_eval_fk", lambda worlds, articulations: None, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", _RejectingSolver(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_reset_solver_internals_delegate",
        NewtonMPMManager._reset_solver_internals,
        raising=False,
    )

    NewtonManager.forward()

    assert world_mask.numpy().tolist() == [False, False]


# ---------------------------------------------------------------------------
# Manager class hierarchy and factory contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manager",
    [
        NewtonMJWarpManager,
        NewtonXPBDManager,
        NewtonVBDManager,
        NewtonFeatherstoneManager,
        NewtonKaminoManager,
        NewtonMPMManager,
    ],
)
def test_subclass_of_newton_manager(manager):
    """All concrete managers inherit from :class:`NewtonManager`."""
    assert issubclass(manager, NewtonManager)
    # Subclasses must override the abstract factory.
    assert manager._build_solver is not NewtonManager._build_solver
    assert manager._create_solver is not NewtonManager._create_solver


def test_clear_resets_rigid_body_force_capability(monkeypatch):
    """Teardown clears the canonical solver capability without subclass shadowing."""
    monkeypatch.setattr(NewtonManager, "_supports_rigid_body_force_input", True)

    NewtonManager.clear()

    assert NewtonManager._supports_rigid_body_force_input is False
    for manager in (
        NewtonMJWarpManager,
        NewtonXPBDManager,
        NewtonVBDManager,
        NewtonFeatherstoneManager,
        NewtonKaminoManager,
        NewtonMPMManager,
    ):
        assert manager._supports_rigid_body_force_input is False


def test_initialize_solver_prepares_picking_before_graph_capture(monkeypatch):
    """Viewer force callbacks are registered after capability publication and before capture."""
    events: list[str] = []
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        NewtonManager.set_builder(builder)
        monkeypatch.setattr(sim, "_prepare_newton_visualizer_for_capture", lambda: events.append("prepare"))
        monkeypatch.setattr(
            NewtonMJWarpManager,
            "_capture_or_defer_graph",
            classmethod(lambda cls: events.append("capture")),
        )

        sim.reset()

    assert events == ["prepare", "capture"]


def test_abstract_build_solver_raises():
    """Calling :meth:`_build_solver` on the abstract base raises."""
    with pytest.raises(NotImplementedError):
        NewtonManager._build_solver(model=None, solver_cfg=NewtonSolverCfg())


def test_abstract_create_solver_raises():
    """Calling :meth:`_create_solver` on the base manager raises."""
    with pytest.raises(NotImplementedError):
        NewtonManager._create_solver(model=None, solver_cfg=NewtonSolverCfg())


@pytest.mark.parametrize(
    "manager",
    [
        NewtonMJWarpManager,
        NewtonXPBDManager,
        NewtonVBDManager,
        NewtonFeatherstoneManager,
        NewtonKaminoManager,
        NewtonMPMManager,
    ],
)
def test_manager_name_starts_with_newton(manager):
    """The ``"newton"`` prefix is required by :class:`InteractiveScene` and the
    various backend factories that dispatch on ``physics_manager.__name__.lower()``.
    """
    assert manager.__name__.lower().startswith("newton")


# ---------------------------------------------------------------------------
# End-to-end: build each solver via SimulationContext
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, expected_solver_cls,"
    " expected_use_single_state, expected_needs_collision_pipeline",
    SOLVER_MATRIX,
)
def test_initialize_solver_populates_canonical_state(
    solver_cfg_factory,
    expected_manager,
    expected_solver_cls,
    expected_use_single_state,
    expected_needs_collision_pipeline,
):
    """End-to-end: ``SimulationContext`` resolves the right manager subclass and
    ``initialize_solver`` lands the right solver + flags on :class:`NewtonManager`.

    External code reads :class:`NewtonManager` attributes directly (``_solver``,
    ``_use_single_state``, ``_needs_collision_pipeline``).  Even though dispatch
    runs through a leaf subclass (e.g. :class:`NewtonMJWarpManager`), shared
    state is assigned through the explicit base class so that those reads keep
    working regardless of which leaf is active.  This test is the regression
    guard for that contract.

    The builder is pre-populated directly (instead of relying on a USD stage)
    with either a minimal particle grid for MPM or a one-body / one-joint scene
    for rigid/articulation solvers:

    1. :class:`SolverImplicitMPM` requires particles and MPM custom attributes
       registered on the builder before particle creation.
    2. :class:`SolverMuJoCo` requires at least one joint to convert the model
       to MJCF; a ground-plane-only scene fails MJCF conversion.
    3. Kamino's internal collision detector requires collidable geometry to
       construct its collision pipeline.
    4. Pre-populating ``NewtonManager._builder`` causes
       :meth:`NewtonManager.start_simulation` to skip
       :meth:`instantiate_builder_from_stage`, so the test does not depend on
       USD asset packages.
    """
    solver_cfg = solver_cfg_factory()
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=solver_cfg, use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        # Resolved manager class matches the expected leaf.
        resolved_manager = sim.physics_manager
        # ``physics_manager`` is a LazyType proxy — compare by ``__name__`` to
        # avoid forcing identity-by-id checks against the unresolved proxy.
        assert resolved_manager.__name__ == expected_manager.__name__
        assert resolved_manager.__name__.lower().startswith("newton")

        builder = resolved_manager.create_builder()
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
            # Pre-populate the builder with a minimal scene so MJCF conversion has
            # something to work with.
            body = builder.add_body(mass=1.0)
            builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
            if isinstance(solver_cfg, (KaminoPADMMSolverCfg, KaminoDVISolverCfg)) and solver_cfg.use_collision_detector:
                builder.add_shape_sphere(body=body, radius=0.05)
                builder.add_ground_plane()
        NewtonManager.set_builder(builder)

        # Force resolution and bring up the solver.
        expected_supports_force_input = RIGID_BODY_FORCE_INPUT_SUPPORT[expected_manager]
        NewtonManager._supports_rigid_body_force_input = not expected_supports_force_input
        sim.reset()

        # Canonical state lives on the base class.
        assert NewtonManager._solver is not None
        assert isinstance(NewtonManager._solver, expected_solver_cls)
        assert NewtonManager._use_single_state is expected_use_single_state
        assert NewtonManager._needs_collision_pipeline is expected_needs_collision_pipeline
        assert NewtonManager._supports_rigid_body_force_input is expected_supports_force_input
        assert NewtonManager._reset_solver_internals_delegate.__self__ is expected_manager
        assert (
            NewtonManager._reset_solver_internals_delegate.__func__ is expected_manager._reset_solver_internals.__func__
        )

        # ``_contacts`` is allocated whichever way contacts are handled
        # (MuJoCo internal buffer or Newton pipeline output).
        # Kamino with internal contacts and MPM do not currently set NewtonManager._contacts.
        if expected_solver_cls not in (SolverKamino, SolverImplicitMPM):
            assert NewtonManager._contacts is not None

        # One step should not raise — proves the dispatch wiring lines up
        # end-to-end.  (We do not assert physics; that's covered by the
        # asset/sensor test suites.)
        sim.step(render=False)


def test_mjwarp_internal_contacts_with_collision_cfg_raises():
    """Combining ``use_mujoco_contacts=True`` with a ``collision_cfg`` is rejected.

    The check lives in :meth:`NewtonMJWarpManager._build_solver` because it
    needs both the solver cfg subtype and the parent :class:`NewtonCfg`, so it
    fires during :meth:`NewtonManager.initialize_solver` (i.e. on
    ``sim.reset()``) rather than at cfg construction time.
    """
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
        builder = sim.physics_manager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        NewtonManager.set_builder(builder)

        with pytest.raises(ValueError, match="collision_cfg cannot be set"):
            sim.reset()


@pytest.mark.parametrize(
    "num_substeps, collision_decimation, expected_mid_loop_collides",
    [
        (8, 0, 0),  # Feature disabled.
        (8, 2, 3),  # Re-collide after substeps 2, 4, 6 (skip last).
        (8, 4, 1),  # Re-collide after substep 4 only.
        (8, 7, 1),  # Re-collide after substep 7 only.
        (8, 8, 0),  # Gated off (>= num_substeps).
    ],
)
def test_collision_decimation_invokes_mid_loop_collide(num_substeps, collision_decimation, expected_mid_loop_collides):
    """``_run_solver_substeps`` re-invokes ``collide`` at the expected substeps.

    Wraps :attr:`NewtonManager._collision_pipeline.collide` with a counter and
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
        builder = sim.physics_manager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_free(child=body)
        builder.add_shape_sphere(body=body, radius=0.05)
        builder.add_ground_plane()
        # Lift the sphere to 0.5 m above the plane so the scene is non-degenerate.
        # joint_q for a free joint is [tx, ty, tz, qx, qy, qz, qw].
        builder.joint_q[-7:] = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        NewtonManager.set_builder(builder)
        sim.reset()

        # Wrap collide() with a counter — must run after sim.reset() so the
        # pipeline is allocated, and use_cuda_graph=False so the wrapped
        # Python callable isn't bypassed by a captured graph.
        calls = {"n": 0}
        original_collide = NewtonManager._collision_pipeline.collide

        def counting_collide(state, contacts):
            calls["n"] += 1
            return original_collide(state, contacts)

        NewtonManager._collision_pipeline.collide = counting_collide
        try:
            sim.step(render=False)
        finally:
            NewtonManager._collision_pipeline.collide = original_collide

        # Expect: 1 (top-of-tick) + expected_mid_loop_collides.
        assert calls["n"] == 1 + expected_mid_loop_collides


@pytest.mark.parametrize("use_single_state", [True, False], ids=["single_state", "double_state"])
def test_state_force_callback_runs_before_every_solver_substep(monkeypatch, use_single_state):
    """Viewer forces are applied to each current input state before solver stepping."""
    events = []

    class _State:
        def __init__(self, name):
            self.name = name

        def clear_forces(self):
            pass

    state_0 = _State("state_0")
    state_1 = _State("state_1")

    monkeypatch.setattr(NewtonManager, "_state_0", state_0)
    monkeypatch.setattr(NewtonManager, "_state_1", state_1)
    monkeypatch.setattr(NewtonManager, "_control", object())
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.001)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 2)
    monkeypatch.setattr(NewtonManager, "_collision_decimation", 0)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False)
    monkeypatch.setattr(NewtonManager, "_use_single_state", use_single_state)
    monkeypatch.setattr(
        NewtonManager,
        "_state_force_callbacks",
        [lambda state: events.append(("force", state.name))],
    )
    monkeypatch.setattr(
        NewtonManager,
        "_step_solver",
        staticmethod(lambda state_in, state_out, *_args: events.append(("step", state_in.name, state_out.name))),
    )

    NewtonManager._run_solver_substeps(contacts=None)

    if use_single_state:
        assert events == [
            ("force", "state_0"),
            ("step", "state_0", "state_0"),
            ("force", "state_0"),
            ("step", "state_0", "state_0"),
        ]
    else:
        assert events == [
            ("force", "state_0"),
            ("step", "state_0", "state_1"),
            ("force", "state_1"),
            ("step", "state_1", "state_0"),
        ]


# ---------------------------------------------------------------------------
# Regression: an env reset written through the data layer must land in the
# manager's canonical _state_0 after an odd number of steps when CUDA graphs
# are disabled (the use_cuda_graph state-swap gating bug).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [1, 3])
def test_reset_lands_in_state_0_after_odd_kamino_steps_without_cuda_graph(num_steps):
    """An env reset written through the data-layer binding lands in ``_state_0``.

    Kamino is double-buffered (``_use_single_state=False``), so each substep
    ping-pongs ``_state_0`` / ``_state_1``. With a single substep the loop must
    copy the result back into ``_state_0`` instead of swapping, otherwise after
    an *odd* number of steps the canonical ``_state_0`` ends up on the other
    buffer. This copy-on-last was previously gated on ``use_cuda_graph``, so with
    CUDA graphs disabled ``_state_0`` flipped buffers and env-reset writes landed
    in the stale buffer.

    :class:`~isaaclab_newton.assets.ArticulationData` binds its joint-state write
    target to ``_state_0.joint_q`` once at setup (``_sim_bind_joint_pos``) and
    never re-binds on env resets, so a flipped ``_state_0`` makes reset writes
    miss the live state. This test reproduces that contract without a full USD
    articulation: it caches the same ``_state_0.joint_q`` binding, steps Kamino an
    odd number of times, writes a sentinel through the cached binding (mimicking
    the reset write), and asserts the manager's ``_state_0`` observes it.

    Without the fix the swap-on-last flips ``_state_0`` for odd ``num_steps`` and
    the sentinel lands in ``_state_1`` instead, so the final assertion fails.
    """
    sentinel = 1.2345
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=KaminoPADMMSolverCfg(),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = NewtonManager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        NewtonManager.set_builder(builder)
        sim.reset()

        # Kamino keeps separate input/output states; the bug only exists there.
        assert NewtonManager._use_single_state is False
        # The data layer binds its joint-state write target to _state_0 at setup.
        reset_target = NewtonManager._state_0.joint_q
        assert reset_target.shape[0] > 0  # guard against a vacuous assertion

        for _ in range(num_steps):
            sim.step(render=False)

        # An env reset writes joint state through the (still bound) target.
        reset_target.fill_(sentinel)

        # The reset must be visible in the manager's canonical _state_0; if the
        # buffer flipped it landed in _state_1 instead.
        canonical_joint_q = NewtonManager._state_0.joint_q.numpy()
        assert np.allclose(canonical_joint_q, sentinel), (
            f"reset write did not land in _state_0 after {num_steps} steps: {canonical_joint_q}"
        )


def _build_collision_scene(sim, num_boxes=8):
    """Add ``num_boxes`` free-falling boxes over a ground plane.

    Uses ``MJWarpSolverCfg(use_mujoco_contacts=False)`` so the Newton collision
    pipeline / contacts are allocated on ``sim.reset()``.
    """
    builder = sim.physics_manager.create_builder()
    for _ in range(num_boxes):
        body = builder.add_body(mass=1.0)
        builder.add_joint_free(child=body)
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    NewtonManager.set_builder(builder)


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
    """Free the arrays ``collide()`` reads off ``model``, then churn the allocator.

    Reusing the freed blocks mimics the GPU memory pressure a real workload
    applies between resets, so a stale pipeline still pointing at ``model``
    would read overwritten memory on its next ``collide()``.
    """
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
    """A step after a second (hard) ``sim.reset()`` runs without a CUDA error.

    Drives reset -> step -> hard reset, frees the old model's collide arrays and
    churns the allocator to mimic GPU memory pressure, then steps and syncs.
    Without the fix the stale pipeline reads the freed buffers and faults
    (CUDA 700). Run with CUDA graphs off and on.
    """
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
        assert NewtonManager._needs_collision_pipeline is True
        old_model = NewtonManager._collision_pipeline.model
        sim.step(render=False)

        sim.reset()

        _free_model_collide_arrays_and_churn(old_model, "cuda:0")

        # A hard device sync surfaces any deferred illegal access as an exception.
        sim.step(render=False)
        wp.synchronize_device("cuda:0")
