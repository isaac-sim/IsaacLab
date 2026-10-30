# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Isaac Lab's use of Newton's shared reset-mask contract."""

from types import SimpleNamespace

import newton
import pytest
import warp as wp
from isaaclab_newton.physics import MPMSolver, MPMSolverCfg, reset_mpm_solver_state
from isaaclab_newton.physics import mpm_solver as mpm_solver_module
from newton.solvers import SolverImplicitMPM


@pytest.fixture(scope="module")
def cpu_mpm_solver_and_state():
    """Build a minimal two-world MPM solver through the Isaac Lab factory."""
    world_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(world_builder)
    world_builder.add_particle(
        pos=(0.0, 0.0, 0.0),
        vel=(0.0, 0.0, 0.0),
        mass=1.0,
        radius=0.02,
    )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(world_builder)
    builder.add_world(world_builder)
    model = builder.finalize(device="cpu")
    solver_cfg = MPMSolverCfg(
        separate_worlds=True,
        grid_type="fixed",
        grid_padding=1,
        max_iterations=1,
        solver="jacobi",
        warmstart_mode="none",
        transfer_scheme="pic",
    )
    solver = MPMSolver.build(solver_cfg, model, wp.DeterministicMode.NOT_GUARANTEED)
    return solver, model.state()


def test_mpm_manager_explicit_reset_accepts_canonical_mask(cpu_mpm_solver_and_state, monkeypatch):
    """The explicit task reset forwards Newton's canonical global-sentinel mask to the real solver."""
    solver, state = cpu_mpm_solver_and_state
    calls = []
    newton_reset = SolverImplicitMPM.reset

    def record_reset(self, state, world_mask=None, flags=None):
        calls.append((state, world_mask, flags))
        newton_reset(self, state, world_mask=world_mask, flags=flags)

    monkeypatch.setattr(SolverImplicitMPM, "reset", record_reset)
    backend = SimpleNamespace(model=solver.model, solver=solver)
    world_mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")

    reset_mpm_solver_state(backend, state=state, world_mask=world_mask, flags=0)

    assert len(calls) == 1
    reset_state, reset_mask, flags = calls[0]
    assert reset_state is state
    assert reset_mask.numpy().tolist() == [False, True, False]
    assert flags == 0


def test_mpm_manager_explicit_reset_rejects_local_only_mask(cpu_mpm_solver_and_state, monkeypatch):
    """The explicit API requires the final global-world entry."""
    solver, state = cpu_mpm_solver_and_state
    backend = SimpleNamespace(model=solver.model, solver=solver)
    local_only_mask = wp.array([False, True], dtype=wp.bool, device="cpu")

    with pytest.raises(ValueError, match=r"world_mask must have shape \(3,\); got \(2,\)"):
        reset_mpm_solver_state(backend, state=state, world_mask=local_only_mask, flags=0)


def test_mpm_manager_explicit_reset_promotes_selected_single_world(monkeypatch):
    """A selected one-world MPM grid resets all solver-owned history."""
    state = object()
    calls = []
    solver = SimpleNamespace(reset=lambda *args, **kwargs: calls.append((args, kwargs)))
    backend = SimpleNamespace(model=SimpleNamespace(world_count=1), solver=solver)
    monkeypatch.setattr(mpm_solver_module, "implicit_mpm_solvers", lambda root: (root,))
    world_mask = wp.array([True, False], dtype=wp.bool, device="cpu")

    reset_mpm_solver_state(backend, state=state, world_mask=world_mask, flags=0)

    assert calls == [((state,), {"world_mask": None, "flags": 0})]


def test_mpm_manager_explicit_reset_deduplicates_manager_states(monkeypatch):
    """An implicit MPM reset visits each distinct manager state once."""
    state_0 = object()
    state_1 = object()
    calls = []
    solver = SimpleNamespace(reset=lambda *args, **kwargs: calls.append((args, kwargs)))
    backend = SimpleNamespace(model=SimpleNamespace(world_count=2), state_0=state_0, state_1=state_1, solver=solver)
    monkeypatch.setattr(mpm_solver_module, "implicit_mpm_solvers", lambda root: (root,))
    world_mask = wp.array([True, False, False], dtype=wp.bool, device="cpu")

    reset_mpm_solver_state(backend, world_mask=world_mask, flags=17)

    assert calls == [
        ((state_1,), {"world_mask": world_mask, "flags": 17}),
        ((state_0,), {"world_mask": world_mask, "flags": 17}),
    ]

    calls.clear()
    backend.state_1 = state_0

    reset_mpm_solver_state(backend, world_mask=world_mask, flags=17)

    assert calls == [((state_0,), {"world_mask": world_mask, "flags": 17})]
