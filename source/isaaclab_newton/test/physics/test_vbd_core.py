# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the core Newton VBD integration."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from isaaclab_newton.physics import NewtonBackendCfg, NewtonManager, NewtonSoftContactCfg
from newton import ModelBuilder

from isaaclab.sim import SimulationContext


# The soft-contact and simulation axes are independent, so each value is covered once.
@pytest.mark.parametrize(
    ("soft_contact_cfg", "expected", "simulation"),
    [
        pytest.param(None, (7.0, 8.0, 9.0), True, id="preserve-physics"),
        pytest.param(
            NewtonSoftContactCfg(soft_contact_ke=11.0, soft_contact_kd=12.0, soft_contact_mu=13.0),
            (11.0, 12.0, 13.0),
            False,
            id="override-render",
        ),
    ],
)
def test_soft_contact_cfg_updates_finalized_model(soft_contact_cfg, expected, simulation):
    """Registry construction shares the builder and applies model options before native allocation."""
    state_values = []

    class Model:
        soft_contact_ke = 7.0
        soft_contact_kd = 8.0
        soft_contact_mu = 9.0
        world_count = 0
        articulation_count = 0

        def set_gravity(self, gravity):
            pass

        def state(self):
            state_values.append((self.soft_contact_ke, self.soft_contact_kd, self.soft_contact_mu))
            return object()

        def control(self):
            return object()

    class Builder(ModelBuilder):
        def finalize(self, *, device):
            return model

        def __deepcopy__(self, memo):
            pytest.fail("A native builder must be borrowed without copying.")

    model = Model()
    builder = Builder()
    cfg = NewtonBackendCfg(builder=builder, device="cpu", soft_contact_cfg=soft_contact_cfg, simulation=simulation)
    sim = object.__new__(SimulationContext)
    sim._backend_registry = []
    backend = sim.get_or_create_backend(cfg)
    assert all(name not in vars(NewtonManager) for name in ("_backend", "_model", "_state_0", "_state_1", "_control"))
    assert cfg.builder is builder
    assert backend is sim.get_or_create_backend(cfg)
    assert (model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu) == expected
    assert state_values == [expected] * (2 if simulation else 1)
    assert (backend.state_1 is not None) == simulation
    assert (backend.control is not None) == simulation
    sim.close_backend(backend)
    assert backend.model is backend.state_0 is backend.state_1 is backend.control is None


def test_vbd_colors_prebuilt_builder_before_start(monkeypatch):
    """VBD colors a prebuilt builder before starting simulation."""
    physics = importlib.import_module("isaaclab_newton.physics")
    events = []

    class Builder:
        def color(self, *, balance_colors):
            events.append(("color", balance_colors))

    monkeypatch.setattr(physics.NewtonVBDManager, "_builder", Builder())
    monkeypatch.setattr(NewtonManager, "start_simulation", classmethod(lambda cls: events.append("start")))

    physics.NewtonVBDManager.start_simulation()

    assert events == [("color", False), "start"]


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
