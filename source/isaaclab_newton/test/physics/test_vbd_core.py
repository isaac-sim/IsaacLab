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

from isaaclab.sim import BackendCfg, SimulationContext


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
    assert backend is sim.get_or_create_backend(cfg.replace())
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
        def color(self, *, balance_colors):
            events.append(("color", balance_colors))

    physics.NewtonVBDManager._prepare_builder_for_finalize(Builder())
    assert events == [("color", False)]


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
