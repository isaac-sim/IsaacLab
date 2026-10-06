# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the solver-status and collision hooks of :meth:`NewtonManager.step`."""

from __future__ import annotations

from types import SimpleNamespace

import newton
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonManager, NewtonMPMManager
from newton.solvers import SolverImplicitMPM

from isaaclab.physics import PhysicsManager
from isaaclab.test.utils import DeviceScope, test_devices


class _StatusError(RuntimeError):
    pass


@pytest.fixture
def stepping_manager(monkeypatch: pytest.MonkeyPatch):
    """A Newton manager whose step dispatch runs eagerly on stubs."""

    class Manager(NewtonManager):
        collided = []

        @classmethod
        def _collide(cls, state, contacts):
            cls.collided.append((state, contacts))

        @classmethod
        def _check_solver_status(cls):
            raise _StatusError

    sim = SimpleNamespace(is_playing=lambda: True)
    monkeypatch.setattr(PhysicsManager, "_sim", sim)
    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=False))
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    monkeypatch.setattr(PhysicsManager, "_sim_time", 1.5)
    monkeypatch.setattr(NewtonManager, "_model_changes", set())
    monkeypatch.setattr(NewtonManager, "_graph", None)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", False)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.01)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 1)
    monkeypatch.setattr(NewtonManager, "_decimation", 1)
    monkeypatch.setattr(NewtonManager, "_adapter", None)
    monkeypatch.setattr(NewtonManager, "_post_actuator_callbacks", [])
    monkeypatch.setattr(NewtonManager, "_post_step_callbacks", [])
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", True)
    monkeypatch.setattr(NewtonManager, "_contacts", "contacts")
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", SimpleNamespace(collide=lambda state, contacts: None))
    monkeypatch.setattr(NewtonManager, "backend", SimpleNamespace(state_0="state"))
    monkeypatch.setattr(Manager, "forward", classmethod(lambda cls: None))
    monkeypatch.setattr(Manager, "_run_solver_substeps", classmethod(lambda cls, contacts: None))
    monkeypatch.setattr(Manager, "_update_sensors", classmethod(lambda cls, contacts: None))
    monkeypatch.setattr(Manager, "_mark_transforms_changed", classmethod(lambda cls: None))
    return Manager


@pytest.mark.parametrize("all_graphable", [True, False])
def test_solver_status_is_checked_before_simulation_time_advances(stepping_manager, monkeypatch, all_graphable):
    """A failed status check leaves the published simulation time unchanged."""
    monkeypatch.setattr(stepping_manager, "_is_all_graphable", classmethod(lambda cls: all_graphable))

    with pytest.raises(_StatusError):
        stepping_manager.step()

    assert PhysicsManager._sim_time == 1.5


@pytest.mark.parametrize("all_graphable", [True, False])
def test_step_generates_contacts_through_the_collide_hook(stepping_manager, monkeypatch, all_graphable):
    """Both step paths route collision generation through the overridable hook."""
    monkeypatch.setattr(stepping_manager, "_is_all_graphable", classmethod(lambda cls: all_graphable))

    with pytest.raises(_StatusError):
        stepping_manager.step()

    assert stepping_manager.collided == [("state", "contacts")]


@pytest.fixture(params=test_devices(DeviceScope.CUDA))
def sparse_mpm_manager(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """An implicit-MPM manager whose sparse grid holds one active cell, stepped through the real status check."""
    device = request.param
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for position in ((0.01, 0.01, 0.01), (0.02, 0.02, 0.02)):
        builder.add_particle(wp.vec3(*position), wp.vec3(0.0), mass=1.0)
    model = builder.finalize(device=device)
    config = SolverImplicitMPM.Config(
        grid_type="sparse",
        voxel_size=0.1,
        max_active_cell_count=1,
        velocity_basis="Q1",
        strain_basis="P0",
        collider_basis="Q1",
        max_iterations=2,
        warmstart_mode="none",
    )
    solver = SolverImplicitMPM(model, config, verbose=False)
    state_in, state_out = model.state(), model.state()

    class Manager(NewtonMPMManager):
        @classmethod
        def _simulate_full(cls):
            solver.step(state_in, state_out, None, None, 0.001)

    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(is_playing=lambda: True, has_gui=False))
    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=True))
    monkeypatch.setattr(PhysicsManager, "_device", device)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 1.5)
    monkeypatch.setattr(NewtonManager, "_solver", solver)
    monkeypatch.setattr(NewtonManager, "_model_changes", set())
    monkeypatch.setattr(NewtonManager, "_graph", None)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.001)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 1)
    monkeypatch.setattr(NewtonManager, "_decimation", 1)
    monkeypatch.setattr(NewtonMPMManager, "_implicit_mpm_solver_root", None)
    monkeypatch.setattr(Manager, "forward", classmethod(lambda cls: None))
    monkeypatch.setattr(Manager, "_is_all_graphable", classmethod(lambda cls: True))
    monkeypatch.setattr(Manager, "_mark_transforms_changed", classmethod(lambda cls: None))
    return Manager, state_in


def _spread_particles(state) -> None:
    """Move the second particle into a different grid cell than the first."""
    positions = state.particle_q.numpy()
    positions[1] = (1.01, 1.01, 1.01)
    state.particle_q.assign(positions)


def test_first_eager_mpm_grid_overflow_is_rejected_before_time_advances(sparse_mpm_manager):
    """The eager dispatch that precedes graph capture reports a sparse-grid capacity overflow."""
    manager, state_in = sparse_mpm_manager
    _spread_particles(state_in)

    with pytest.raises(RuntimeError, match="capacity was exceeded"):
        manager.step()

    assert PhysicsManager._sim_time == 1.5
    assert NewtonManager._graph is None


def test_replayed_mpm_grid_overflow_is_rejected_before_time_advances(sparse_mpm_manager):
    """A sparse-grid capacity overflow inside a replayed graph is reported before time advances."""
    manager, state_in = sparse_mpm_manager
    manager.step()
    assert NewtonManager._graph is not None
    _spread_particles(state_in)

    with pytest.raises(RuntimeError, match="capacity was exceeded"):
        manager.step()

    assert PhysicsManager._sim_time == pytest.approx(1.501)
