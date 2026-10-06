# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the FeatherPGS solver configuration and manager hooks."""

from __future__ import annotations

import inspect
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import isaaclab_newton.physics.feather_pgs_manager as feather_pgs_manager_module
import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import (
    FeatherPGSSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonFeatherPGSManager,
    NewtonManager,
    NewtonSolverCfg,
)
from newton.solvers import SolverFeatherPGS

from isaaclab.physics import PhysicsManager
from isaaclab.test.utils import DeviceScope, test_devices

_MANAGER_ONLY_FIELDS = {field.name for field in fields(NewtonSolverCfg)} | {
    "raise_on_constraint_overflow",
    "speculative_contact_gap_max",
}
_SOLVER_FIELDS = sorted({field.name for field in fields(FeatherPGSSolverCfg)} - _MANAGER_ONLY_FIELDS)
_SOLVER_PARAMETERS = inspect.signature(SolverFeatherPGS.__init__).parameters


def test_newton_cfg_dispatches_to_feather_pgs_manager():
    """The solver configuration selects the FeatherPGS manager."""
    cfg = NewtonCfg(solver_cfg=FeatherPGSSolverCfg())
    assert cfg.class_type == "isaaclab_newton.physics.feather_pgs_manager:NewtonFeatherPGSManager"


def test_every_solver_field_is_a_newton_solver_option_with_its_default():
    """No configured option is dropped by the solver-signature filter, and defaults match the solver's."""
    cfg = FeatherPGSSolverCfg()
    assert set(_SOLVER_FIELDS) <= set(_SOLVER_PARAMETERS), sorted(set(_SOLVER_FIELDS) - set(_SOLVER_PARAMETERS))
    defaults = {name: _SOLVER_PARAMETERS[name].default for name in _SOLVER_FIELDS}
    assert {name: getattr(cfg, name) for name in _SOLVER_FIELDS} == defaults


def test_manager_forwards_every_solver_field(monkeypatch):
    """Each solver field reaches the solver constructor with its configured value."""
    created = {}

    def fake_init(self, model, **kwargs):
        created.update(kwargs)

    fake_init.__signature__ = inspect.signature(SolverFeatherPGS.__init__)
    monkeypatch.setattr(feather_pgs_manager_module, "SolverFeatherPGS", type("FakeSolver", (), {"__init__": fake_init}))
    cfg = FeatherPGSSolverCfg(pgs_mode="split", pgs_iterations=7, contact_torsion_radius=0.01, enable_sleeping=True)

    NewtonFeatherPGSManager._create_solver(SimpleNamespace(), cfg)

    assert created == {name: getattr(cfg, name) for name in _SOLVER_FIELDS}


def _slider_model(device: str) -> newton.Model:
    """A box on a vertical prismatic joint resting on the ground."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    link = builder.add_link(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), label="slider")
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1, label="slider_box")
    joint = builder.add_joint_prismatic(-1, link, axis=newton.Axis.Z)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def test_shared_anchor_options_configure_the_solver():
    """Shared-anchor and friction-gap options reach a constructed solver."""
    cfg = FeatherPGSSolverCfg(
        pgs_mode="split",
        contact_shared_anchor=True,
        contact_friction_shared_anchor=True,
        contact_friction_gap_threshold=0.001,
    )

    solver = NewtonFeatherPGSManager._create_solver(_slider_model("cpu"), cfg)

    assert (solver.contact_shared_anchor, solver.contact_friction_shared_anchor) == (True, True)
    assert solver.contact_friction_gap_threshold == 0.001


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_contact_torsion_and_sleeping_options_configure_the_solver(device):
    """Contact-torsion and sleeping options reach a constructed solver."""
    cfg = FeatherPGSSolverCfg(
        contact_torsion_radius=0.01,
        contact_torsion_shape_patterns=("slider_box",),
        contact_torsion_device=True,
        enable_sleeping=True,
    )

    solver = NewtonFeatherPGSManager._create_solver(_slider_model(device), cfg)

    assert solver.contact_torsion_radius == 0.01
    assert solver.contact_torsion_shape_patterns == ("slider_box",)
    assert solver.contact_torsion_device
    assert solver.sleeping is not None


@pytest.mark.parametrize(
    ("collision_cfg", "expected"),
    [
        (None, -1),
        (NewtonCollisionPipelineCfg(rigid_contact_max=40), 40),
        (NewtonCollisionPipelineCfg(rigid_contacts_per_world=8), 32),
    ],
)
def test_contact_capacity_is_published_before_solver_construction(monkeypatch, collision_cfg, expected):
    """FeatherPGS sizes its contact scratch from the model, so the capacity is set before it is constructed."""
    model = SimpleNamespace(rigid_contact_max=-1, world_count=4)
    seen = []
    monkeypatch.setattr(NewtonManager, "_collision_cfg", collision_cfg)
    monkeypatch.setattr(
        NewtonFeatherPGSManager,
        "_create_solver",
        classmethod(lambda cls, model, solver_cfg: seen.append(model.rigid_contact_max) or SimpleNamespace()),
    )
    monkeypatch.setattr(NewtonManager, "_solver", None)

    NewtonFeatherPGSManager._build_solver(model, FeatherPGSSolverCfg())

    assert seen == [expected]


@pytest.mark.parametrize("raise_on_overflow", [False, True])
def test_status_check_validates_torsion_and_optionally_raises_on_overflow(monkeypatch, raise_on_overflow):
    """Contact torsion is validated after every step; the overflow status is read only when raising is requested."""
    solver = SimpleNamespace(
        validate_contact_torsion=Mock(), check_constraint_capacity=Mock(side_effect=RuntimeError("capacity"))
    )
    cfg = NewtonCfg(solver_cfg=FeatherPGSSolverCfg(raise_on_constraint_overflow=raise_on_overflow))
    monkeypatch.setattr(PhysicsManager, "_cfg", cfg)
    monkeypatch.setattr(NewtonManager, "_solver", solver)

    if raise_on_overflow:
        with pytest.raises(RuntimeError, match="capacity"):
            NewtonFeatherPGSManager._check_solver_status()
    else:
        NewtonFeatherPGSManager._check_solver_status()

    assert solver.validate_contact_torsion.call_count == 1
    assert solver.check_constraint_capacity.call_count == int(raise_on_overflow)


def test_contacts_are_generated_over_the_full_physics_step(monkeypatch):
    """The collision horizon spans all substeps of one physics step."""
    calls = []
    pipeline = SimpleNamespace(collide=lambda state, contacts, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", pipeline)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.005)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 4)

    NewtonFeatherPGSManager._collide("state", "contacts")

    assert calls == [{"dt": pytest.approx(0.02)}]


@pytest.fixture
def two_box_model(monkeypatch: pytest.MonkeyPatch):
    """A CPU model with one box near the ground per world, installed as the Newton manager's backend model."""
    builder = newton.ModelBuilder()
    for world in range(2):
        builder.begin_world()
        body = builder.add_body(xform=wp.transform((3.0 * world, 0.0, 0.2), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()
    builder.add_ground_plane()
    model = builder.finalize(device="cpu")
    monkeypatch.setattr(NewtonManager, "backend", SimpleNamespace(model=model))
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", True)
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", None)
    monkeypatch.setattr(NewtonManager, "_contacts", None)
    monkeypatch.setattr(NewtonManager, "_solver", SimpleNamespace())
    monkeypatch.setattr(NewtonManager, "_deterministic_mode", wp.DeterministicMode.NOT_GUARANTEED)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.005)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 4)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    return model


def test_predictive_contacts_need_the_feather_pgs_collision_horizon(monkeypatch, two_box_model):
    """The predictive-contact extension reaches the pipeline, whose collisions need FeatherPGS's step duration."""
    solver_cfg = FeatherPGSSolverCfg(speculative_contact_gap_max=0.05)
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(solver_cfg=solver_cfg))
    monkeypatch.setattr(NewtonManager, "_collision_cfg", NewtonCollisionPipelineCfg())
    state = two_box_model.state()

    NewtonFeatherPGSManager._initialize_contacts()
    with pytest.raises(ValueError, match="dt must be provided"):
        NewtonManager._collide(state, NewtonManager._contacts)
    NewtonFeatherPGSManager._collide(state, NewtonManager._contacts)

    assert int(NewtonManager._contacts.rigid_contact_count.numpy()[0]) > 0


def test_collision_only_determinism_and_sticky_matching_are_available_to_feather_pgs(monkeypatch, two_box_model):
    """FeatherPGS rejects the solver determinism guarantee but sorts and matches contacts in its pipeline."""
    with pytest.raises(ValueError, match="not supported by FeatherPGSSolverCfg"):
        NewtonManager._validate_deterministic_solver_cfg(FeatherPGSSolverCfg(), wp.DeterministicMode.RUN_TO_RUN)
    collision_cfg = NewtonCollisionPipelineCfg(deterministic=True, contact_matching="sticky")
    monkeypatch.setattr(
        PhysicsManager, "_cfg", NewtonCfg(solver_cfg=FeatherPGSSolverCfg(), collision_cfg=collision_cfg)
    )
    monkeypatch.setattr(NewtonManager, "_collision_cfg", collision_cfg)

    NewtonFeatherPGSManager._initialize_contacts()

    assert NewtonManager._collision_pipeline.deterministic
    assert NewtonManager._collision_pipeline.contact_matching == "sticky"


@pytest.mark.parametrize(
    ("radius", "device", "expected"), [(0.0, False, True), (0.01, False, False), (0.01, True, True)]
)
def test_host_contact_torsion_steps_without_a_cuda_graph(monkeypatch, radius, device, expected):
    """Host-prepared torsion reads contacts back every step, which a CUDA graph cannot capture."""
    solver_cfg = FeatherPGSSolverCfg(contact_torsion_radius=radius, contact_torsion_device=device)
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(solver_cfg=solver_cfg))

    assert NewtonFeatherPGSManager._supports_cuda_graph_capture() is expected


@pytest.mark.parametrize("capturing", [False, True])
def test_double_buffer_events_are_seeded_only_inside_capture(monkeypatch, capturing):
    """The solver's double-buffer waits are seeded inside a CUDA graph capture."""
    seed = Mock()
    monkeypatch.setattr(NewtonManager, "_solver", SimpleNamespace(seed_double_buffer_events=seed))
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0")
    monkeypatch.setattr(
        feather_pgs_manager_module.wp, "get_stream", lambda device: SimpleNamespace(is_capturing=capturing)
    )
    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(lambda cls: None))

    NewtonFeatherPGSManager._simulate_physics_only()

    assert seed.call_count == int(capturing)


def test_tendon_metadata_has_no_tendon_control():
    """Imported tendon metadata creates no tendon command path."""
    assert NewtonFeatherPGSManager.create_fixed_tendon_control(SimpleNamespace()) is None


@pytest.fixture(params=test_devices(DeviceScope.CUDA))
def torsion_manager(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """A FeatherPGS manager with device contact torsion whose graph steps one lifted slider."""
    device = request.param
    model = _slider_model(device)
    solver_cfg = FeatherPGSSolverCfg(contact_torsion_radius=0.01, contact_torsion_device=True, dense_max_constraints=1)
    solver = NewtonFeatherPGSManager._create_solver(model, solver_cfg)
    state_0, state_1, control = model.state(), model.state(), model.control()
    state_0.joint_q.assign(np.array([0.5], dtype=np.float32))
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()

    class Manager(NewtonFeatherPGSManager):
        @classmethod
        def _simulate_full(cls):
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            cls._collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 0.01)

    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(is_playing=lambda: True, has_gui=False))
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(solver_cfg=solver_cfg, use_cuda_graph=True))
    monkeypatch.setattr(PhysicsManager, "_device", device)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 1.5)
    monkeypatch.setattr(NewtonManager, "backend", SimpleNamespace(model=model, state_0=state_0, state_1=state_1))
    monkeypatch.setattr(NewtonManager, "_solver", solver)
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", pipeline)
    monkeypatch.setattr(NewtonManager, "_model_changes", set())
    monkeypatch.setattr(NewtonManager, "_graph", None)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.01)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 1)
    monkeypatch.setattr(NewtonManager, "_decimation", 1)
    monkeypatch.setattr(Manager, "forward", classmethod(lambda cls: None))
    monkeypatch.setattr(Manager, "_is_all_graphable", classmethod(lambda cls: True))
    monkeypatch.setattr(Manager, "_mark_transforms_changed", classmethod(lambda cls: None))
    return Manager, state_0


def test_replayed_contact_torsion_error_is_raised_before_time_advances(torsion_manager):
    """A device-torsion error inside a replayed graph is raised by the status check, before time advances."""
    manager, state_0 = torsion_manager
    manager.step()
    assert NewtonManager._graph is not None
    state_0.joint_q.assign(np.array([0.0], dtype=np.float32))

    with pytest.raises(RuntimeError, match="torsion"):
        manager.step()

    assert PhysicsManager._sim_time == pytest.approx(1.51)
