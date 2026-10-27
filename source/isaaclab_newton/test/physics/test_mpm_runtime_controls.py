# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native CPU coverage for runtime MPM resolution changes."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg, NewtonMPMManager, XPBDSolverCfg
from newton.solvers import SolverImplicitMPM

from isaaclab_contrib.coupling import (
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
    NewtonCouplerManager,
)


def _build_coupled_manager(model, entries, *, proxies=()):
    """Build the native solver through the manager's actual ownership resolution."""
    config = CouplerProxyCfg(entries=entries, proxies=list(proxies))
    manager = NewtonCouplerManager(NewtonCfg(use_cuda_graph=False, solver_cfg=config))
    manager._sim = SimpleNamespace(get_clone_plan=lambda: SimpleNamespace(env_template="/World/envs/env_0"))
    manager.backend = SimpleNamespace(model=model, state_0=model.state(), solver=None, contact_sensors={})
    manager._build_solver(model, manager._cfg.solver_cfg)
    return manager


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
        manager = _build_coupled_manager(
            model,
            [
                CouplerEntryCfg(
                    name="media",
                    particles=[1, 0],
                    in_place=True,
                    solver_cfg=config,
                )
            ],
        )
        solver = manager.backend.solver
        config = manager._cfg.solver_cfg.entries[0].solver_cfg
        selected, state = solver.solver("media"), solver.entry_state("media")
        kwargs = {"entry_name": "media"}
    else:
        manager = NewtonMPMManager(NewtonCfg(use_cuda_graph=False, solver_cfg=config))
        config = manager._cfg.solver_cfg
        solver = NewtonMPMManager._create_solver(model, config)
        selected, state = solver, model.state()
        kwargs = {}
    manager._sim = SimpleNamespace()
    manager.backend = SimpleNamespace(
        model=model, state_0=model.state() if coupled else state, solver=solver, contact_sensors={}
    )
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


def test_runtime_radius_rejects_foreign_particle_proxies_before_mutation():
    """Reject a partial owner even when foreign proxy rows fill its local view."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for x in (0.0, 0.1):
        builder.add_particle(pos=(x, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.01, radius=0.02)
    model = builder.finalize(device="cpu")
    config = MPMSolverCfg(grid_type="dense", grid_padding=1, voxel_size=0.1, max_iterations=1)
    manager = _build_coupled_manager(
        model,
        [
            CouplerEntryCfg(name="other", particles=[1], solver_cfg=XPBDSolverCfg(iterations=1)),
            CouplerEntryCfg(
                name="media",
                particles=[0],
                in_place=True,
                solver_cfg=config,
            ),
        ],
        proxies=[CouplerProxyMappingCfg(source="other", destination="media", particles=[1], collision_pipeline=None)],
    )
    solver = manager.backend.solver
    selected = solver.solver("media")
    # The second local row is a proxy, so local count alone cannot establish ownership.
    assert selected.model.particle_count == model.particle_count
    radii = [model.particle_radius, solver.view("media").particle_radius, solver.view("other").particle_radius]
    before = [array.numpy().copy() for array in radii]
    volume = selected._mpm_model.particle_volume.numpy().copy()
    graph = manager._graph = object()

    with pytest.raises(ValueError, match="own.*all|all.*particle|subset"):
        manager.set_particle_radius(0.03, entry_name="media")

    for array, expected in zip(radii, before, strict=True):
        np.testing.assert_array_equal(array.numpy(), expected)
    np.testing.assert_array_equal(selected._mpm_model.particle_volume.numpy(), volume)
    assert manager._graph is graph


def test_runtime_radius_updates_outgoing_particle_proxies():
    """Keep sibling proxy radii consistent when the MPM entry owns every particle."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for x in (0.0, 0.1):
        builder.add_particle(pos=(x, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.01, radius=0.02)
    model = builder.finalize(device="cpu")
    manager = _build_coupled_manager(
        model,
        [
            CouplerEntryCfg(
                name="media",
                particles=[1, 0],
                in_place=True,
                solver_cfg=MPMSolverCfg(grid_type="dense", grid_padding=1, voxel_size=0.1, max_iterations=1),
            ),
            CouplerEntryCfg(name="other", solver_cfg=XPBDSolverCfg(iterations=1)),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="media", destination="other", particles=[1], mode="staggered", collision_pipeline=None
            )
        ],
    )
    before = model.particle_mass.numpy().copy()
    manager._graph = object()

    manager.set_particle_radius(0.03, entry_name="media")

    root = manager.backend.solver
    for particle_model in (model, *(root.view(name) for name in root.entry_names())):
        np.testing.assert_allclose(particle_model.particle_radius.numpy(), 0.03)
        assert particle_model.particle_max_radius == 0.03
    np.testing.assert_array_equal(model.particle_mass.numpy(), before)
    assert manager._graph is None
