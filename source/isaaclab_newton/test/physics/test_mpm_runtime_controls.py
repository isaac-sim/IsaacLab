# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native CPU coverage for runtime MPM resolution changes."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
from isaaclab_newton.physics import MPMSolverCfg, NewtonMPMManager
from newton.solvers import SolverImplicitMPM
from newton.solvers.experimental.coupled import SolverCoupled

from isaaclab_contrib.coupling import NewtonCouplerManager


@pytest.mark.parametrize("coupled", (False, True), ids=("direct", "coupled"))
def test_runtime_resolution_preserves_current_native_state(coupled):
    """Rebuild the selected solver without losing state, mass, or material history."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.1, 0.0, 0.0), mass=0.01, radius=0.02)
    builder.add_particle(pos=(0.05, 0.0, 0.0), vel=(0.0, 0.2, 0.0), mass=0.02, radius=0.02)
    model = builder.finalize(device="cpu")
    config = MPMSolverCfg(grid_type="dense", grid_padding=1, voxel_size=0.1, max_iterations=1)
    if coupled:
        manager = NewtonCouplerManager()
        solver = SolverCoupled(
            model,
            [
                SolverCoupled.Entry(
                    name="media",
                    particles=(0, 1),
                    in_place=True,
                    solver=lambda view: NewtonMPMManager._create_solver(view, config),
                )
            ],
        )
        manager._cfg = SimpleNamespace(
            use_cuda_graph=False,
            solver_cfg=SimpleNamespace(entries=[SimpleNamespace(name="media", solver_cfg=config)]),
        )
        selected, state = solver.solver("media"), solver.entry_state("media")
        kwargs = {"entry_name": "media"}
    else:
        manager = NewtonMPMManager()
        solver = NewtonMPMManager._create_solver(model, config)
        manager._cfg = SimpleNamespace(use_cuda_graph=False, solver_cfg=config)
        selected, state = solver, model.state()
        kwargs = {}
    manager._solver = solver
    manager._sim = SimpleNamespace()
    manager.backend = SimpleNamespace(model=model, state_0=model.state() if coupled else state)
    # Allocate the old grid, then move the live local state far from its initial
    # positions. Coupled parent state deliberately stays at those old positions.
    selected.step(state, state, None, None, 0.001)
    state.particle_q.assign(state.particle_q.numpy() + np.array([1.0, 2.0, 3.0], dtype=np.float32))
    position, velocity = state.particle_q.numpy().copy(), state.particle_qd.numpy().copy()
    masses = model.particle_mass.numpy().copy()
    state.mpm.particle_Jp.fill_(0.9)

    manager._graph = object()
    with pytest.raises(ValueError):
        manager.set_voxel_size(float("nan"), **kwargs)
    assert manager._graph is not None
    assert config.voxel_size == 0.1

    manager.set_voxel_size(0.05, **kwargs)
    assert manager._graph is None
    assert selected.voxel_size == config.voxel_size == 0.05
    grid = selected._scratchpad.grid
    assert np.all(np.asarray(grid.origin) <= position.min(axis=0))
    assert np.all(np.asarray(grid.origin) + np.asarray(grid.res) * np.asarray(grid.cell_size) >= position.max(axis=0))
    np.testing.assert_array_equal(state.mpm.particle_Jp.numpy(), np.full(2, 0.9, dtype=np.float32))

    manager._graph = object()
    manager.set_particle_radius(0.015, **kwargs)
    assert manager._graph is None
    np.testing.assert_allclose(model.particle_radius.numpy(), 0.015)
    np.testing.assert_allclose(selected.model.particle_radius.numpy(), 0.015)
    assert model.particle_max_radius == selected.model.particle_max_radius == 0.015
    np.testing.assert_allclose(selected._mpm_model.particle_volume.numpy(), 8 * 0.015**3)
    np.testing.assert_allclose(selected._mpm_model.particle_density.numpy(), masses / (8 * 0.015**3))
    np.testing.assert_array_equal(state.mpm.particle_Jp.numpy(), np.full(2, 0.9, dtype=np.float32))
    np.testing.assert_array_equal(state.particle_q.numpy(), position)
    np.testing.assert_array_equal(state.particle_qd.numpy(), velocity)
    np.testing.assert_array_equal(model.particle_mass.numpy(), masses)
    selected.step(state, state, None, None, 0.001)
    assert np.isfinite(state.particle_q.numpy()).all()
    assert np.isfinite(state.particle_qd.numpy()).all()
    assert np.linalg.norm(state.particle_q.numpy() - position) > 0
