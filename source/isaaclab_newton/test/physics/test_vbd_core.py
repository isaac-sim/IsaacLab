# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the core Newton VBD integration."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from isaaclab_newton.physics import NewtonBackendCfg, NewtonBuilderCfg, NewtonCfg, NewtonManager, NewtonSoftContactCfg
from newton import ModelBuilder
from newton.solvers import SolverVBD

from isaaclab.sim import BackendCfg, SimulationContext
from isaaclab.utils import replace


# The soft-contact and simulation axes are independent, so each value is covered once.
@pytest.mark.parametrize(
    ("soft_contact_cfg", "expected", "simulation"),
    [
        pytest.param(None, (7.0, 8.0, 9.0), False, id="preserve-render"),
        pytest.param(
            NewtonSoftContactCfg(soft_contact_ke=11.0, soft_contact_kd=12.0, soft_contact_mu=13.0),
            (11.0, 12.0, 13.0),
            True,
            id="override-physics",
        ),
    ],
)
def test_soft_contact_cfg_updates_finalized_model(monkeypatch, soft_contact_cfg, expected, simulation):
    """Cfg-keyed construction shares the builder and applies physics settings before state allocation."""
    state_values = []

    class Model:
        soft_contact_ke = 7.0
        soft_contact_kd = 8.0
        soft_contact_mu = 9.0
        world_count = 0
        articulation_count = 0

        def state(self):
            state_values.append((self.soft_contact_ke, self.soft_contact_kd, self.soft_contact_mu))
            return object()

        def control(self):
            return object()

    model = Model()
    monkeypatch.setattr(ModelBuilder, "finalize", lambda self, device: model)
    physics_cfg = NewtonCfg(soft_contact_cfg=soft_contact_cfg) if simulation else object()
    builder_cfg = NewtonBuilderCfg(physics_cfg=physics_cfg)
    assert not isinstance(builder_cfg, BackendCfg)
    assert not hasattr(builder_cfg, "close")
    cfg = NewtonBackendCfg(physics_cfg=physics_cfg, device="cpu")
    sim = object.__new__(SimulationContext)
    sim._backend_registry = []
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim)
    builder = sim.get_or_create_backend(builder_cfg)
    backend = sim.get_or_create_backend(cfg)
    assert not {"_builder", "set_builder", "_model", "_state_0", "_state_1"}.intersection(vars(NewtonManager))
    assert not {"sync_transforms_to_fabric", "sync_transforms_to_usd"}.intersection(vars(NewtonManager))
    assert cfg.physics_cfg is physics_cfg
    assert backend is sim.get_or_create_backend(replace(cfg))
    assert builder is sim.get_or_create_backend(builder_cfg)
    assert (model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu) == expected
    assert state_values == [expected] * (2 if simulation else 1)
    assert (backend.state_1 is not None) == simulation
    assert (backend.control is not None) == simulation
    sim.close_backend(backend)
    assert backend.model is backend.state_0 is backend.state_1 is backend.control is None
    assert sim.get_or_create_backend(builder_cfg) is builder
    sim.close_backend(builder)
    assert sim._backend_registry == []


def test_vbd_colors_builder_before_finalization():
    """VBD colors the completed builder in the existing pre-finalization hook."""
    physics = importlib.import_module("isaaclab_newton.physics")
    events = []

    class Builder:
        def color(self, *, include_bending, balance_colors):
            events.append(("color", include_bending, balance_colors))

    physics.NewtonVBDManager._prepare_builder_for_finalize(Builder())
    assert events == [("color", True, False)]


def test_vbd_solver_force_input_capability(monkeypatch):
    """VBD rejects rigid forces when an external solver integrates rigid bodies.

    The default (VBD integrates rigid bodies itself) is covered end to end by
    ``test_initialize_solver_populates_canonical_state``.
    """
    physics = importlib.import_module("isaaclab_newton.physics")
    solver = object()
    monkeypatch.setattr(physics.NewtonVBDManager, "_create_solver", lambda model, cfg: solver)
    monkeypatch.setattr(NewtonManager, "_solver", None)
    monkeypatch.setattr(NewtonManager, "_use_single_state", True)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False)
    monkeypatch.setattr(NewtonManager, "_supports_rigid_body_force_input", True)

    solver_cfg = physics.VBDSolverCfg(integrate_with_external_rigid_solver=True)
    physics.NewtonVBDManager._build_solver(object(), solver_cfg)

    assert NewtonManager._solver is solver
    assert NewtonManager._supports_rigid_body_force_input is False


@pytest.mark.parametrize("overrides", [{}, {"rigid_compliant_alm": False}], ids=["defaults", "legacy"])
def test_vbd_rigid_solver_controls(overrides):
    """VBD preserves the default controls and explicit legacy-mode selection."""
    physics = importlib.import_module("isaaclab_newton.physics")
    kwargs = NewtonManager._filter_solver_kwargs(SolverVBD, physics.VBDSolverCfg(**overrides))
    assert kwargs["rigid_compliant_alm"] is overrides.get("rigid_compliant_alm")
    assert kwargs["rigid_body_contact_buffer_size"] == 64


def test_vbd_compliant_alm_cable_stiffness():
    """The manager's ALM solver retains finite cable stiffness under gravity."""
    physics = importlib.import_module("isaaclab_newton.physics")
    gravity = 9.81
    stretch_stiffness = 1.0e3
    builder = ModelBuilder(gravity=(0.0, 0.0, -gravity))
    body = builder.add_link()
    builder.add_shape_capsule(body=body, radius=0.01, half_height=0.1)
    mass = builder.body_mass[body]
    joint = builder.add_joint_rod(
        parent=-1,
        child=body,
        stretch_stiffness=stretch_stiffness,
        stretch_damping=2.0 * (mass * stretch_stiffness) ** 0.5,
        bend_stiffness=5.0,
        bend_damping=1.0,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device="cpu")
    solver_cfg = physics.VBDSolverCfg(rigid_compliant_alm=True, rigid_body_contact_buffer_size=256)
    solver = physics.NewtonVBDManager._create_solver(model, solver_cfg)
    assert solver.rigid_compliant_alm is True
    assert solver.body_body_contact_indices.size == model.body_count * 256

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(120):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0

    # At equilibrium the spring force balances the weight: k * extension = m * g.
    expected_extension = mass * gravity / stretch_stiffness
    assert state_0.body_q.numpy()[body, 2] == pytest.approx(-expected_extension, rel=0.01)


def test_vbd_rebuilds_particle_bvh_before_physics_step(monkeypatch):
    """VBD rebuilds its particle BVH before the base physics step."""
    physics = importlib.import_module("isaaclab_newton.physics")
    events = []
    state = object()

    class Solver:
        def rebuild_bvh(self, solver_state):
            events.append(("rebuild", solver_state))

    def simulate_physics_only(cls):
        events.append(("step", cls))

    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(simulate_physics_only))
    monkeypatch.setattr(
        physics.NewtonVBDManager, "backend", SimpleNamespace(model=SimpleNamespace(particle_count=1), state_0=state)
    )
    monkeypatch.setattr(physics.NewtonVBDManager, "_solver", Solver())

    physics.NewtonVBDManager._simulate_physics_only()

    assert events == [("rebuild", state), ("step", physics.NewtonVBDManager)]
