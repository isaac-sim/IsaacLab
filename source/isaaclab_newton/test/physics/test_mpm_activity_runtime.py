# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect native MPM reconfiguration and rollout-validation boundaries."""

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import (
    MPMSleepingCfg,
    MPMSolverCfg,
    MPMWorkingSetCfg,
    NewtonBackend,
    NewtonBackendCfg,
    NewtonCfg,
    NewtonManager,
    configure_mpm_sleeping,
    configure_mpm_working_set,
    defer_mpm_status_checks,
    reset_mpm_solver_state,
)
from newton.solvers import SolverImplicitMPM

from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.kitless


def _make_manager(device, *, guarded=False):
    """Use real native particles and solver, without starting a visualization runtime."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for x in (0.0125, 0.0375):
        builder.add_particle(pos=(x, 0.0125, 0.0125), vel=(0.0, 0.0, 0.0), mass=0.01, radius=0.01)
    cfg = MPMSolverCfg(
        grid_type="sparse" if guarded else "fixed",
        reuse_grid_topology=guarded,
        grid_padding=0 if guarded else 1,
        voxel_size=0.05,
        max_active_cell_count=128,
        max_iterations=1,
        solver="jacobi",
        warmstart_mode="none",
        transfer_scheme="pic",
        velocity_basis="Q1",
        strain_basis="P0",
        collider_basis="Q1",
    )
    physics_cfg = NewtonCfg(solver_cfg=cfg, use_cuda_graph=False)
    manager = NewtonManager(physics_cfg)
    manager.backend = NewtonBackend(NewtonBackendCfg(physics_cfg=physics_cfg, device=str(device)), builder, dt=0.005)
    manager.backend.init_solver()
    return manager


@pytest.fixture(scope="module")
def cpu_manager():
    return _make_manager("cpu")


@pytest.mark.parametrize(
    ("configure", "cfg", "native_method"),
    [
        (configure_mpm_sleeping, MPMSleepingCfg(), "set_particle_sleeping"),
        (configure_mpm_working_set, MPMWorkingSetCfg(particle_capacity=1), "set_particle_working_set"),
    ],
)
def test_runtime_configuration_opt_out_compatibility_and_graph_lifetime(
    cpu_manager, monkeypatch, configure, cfg, native_method
):
    """Opt-out preserves graphs; requested native reconfiguration releases them before replacing buffers."""
    manager = cpu_manager
    graph = object()
    manager.backend.graphs = {"step": graph}
    configure(manager, None)
    assert manager.backend.graphs == {"step": graph}

    backend = manager.backend
    try:
        manager.backend = None
        configure(manager, None)
        with pytest.raises(RuntimeError, match="Initialize the Newton backend"):
            configure(manager, cfg)
    finally:
        manager.backend = backend

    native_setter = getattr(SolverImplicitMPM, native_method, None)
    with monkeypatch.context() as missing_extension:
        missing_extension.delattr(SolverImplicitMPM, native_method, raising=False)
        with pytest.raises(RuntimeError, match=native_method):
            configure(manager, cfg)
        assert manager.backend.graphs == {"step": graph}

    if native_setter is None:
        return  # Compatibility checks above also run against Newton without the optional extension.
    entered_native_setter = []

    def check_graph_before_native_setter(self, *args, **kwargs):
        assert not manager.backend.graphs
        entered_native_setter.append(True)
        return native_setter(self, *args, **kwargs)

    monkeypatch.setattr(SolverImplicitMPM, native_method, check_graph_before_native_setter)
    with pytest.raises(ValueError, match="CUDA"):
        configure(manager, cfg)
    assert entered_native_setter == [True]


def test_deferred_validation_rejects_unguarded_cpu_solver(cpu_manager):
    """A scope must not suppress checks for a layout without a device-side failure guard."""
    with pytest.raises(RuntimeError, match="guarded CUDA MPM"):
        with defer_mpm_status_checks(cpu_manager):
            pytest.fail("An unguarded solver entered the deferred rollout window.")


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_captured_deferred_validation_restores_checks_and_rejects_real_overflow(device, monkeypatch):
    """A real captured capacity fault rejects the rollout and cannot be erased by reconfiguration."""
    if not hasattr(SolverImplicitMPM, "set_particle_working_set"):
        pytest.skip("Requires Newton's optional native sleeping and bounded-evaluation extension.")
    if not wp.is_mempool_enabled(device):
        pytest.skip("CUDA graph capture requires the Warp memory pool.")
    if not wp.is_conditional_graph_supported():
        pytest.skip("Guarded MPM requires Warp conditional graph support.")
    manager = _make_manager(device, guarded=True)
    solver, state = manager.backend.solver, manager.backend.state_0
    configure_mpm_sleeping(manager, MPMSleepingCfg(sleep_time=0.01))
    for _ in range(10):
        solver.step(state, state, None, None, 0.005)
    np.testing.assert_array_equal(solver.particle_activity.numpy(), 0)
    configure_mpm_working_set(manager, MPMWorkingSetCfg(particle_capacity=1))
    solver.step(state, state, None, None, 0.005)
    assert solver.step_status is not None
    solver.check_sparse_grid_rebuild_status()
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state, state, None, None, 0.005)

    checks = []
    native_check = solver.check_sparse_grid_rebuild_status

    def record_native_check():
        checks.append(True)
        native_check()

    monkeypatch.setattr(solver, "check_sparse_grid_rebuild_status", record_native_check)
    driver = manager.backend.solver_ops
    with defer_mpm_status_checks(manager):
        assert len(checks) == 1  # Entry validates the preceding window.
        wp.capture_launch(capture.graph)
        driver.check_step(captured=True, debug=False)
        assert len(checks) == 1
        driver.check_step(captured=True, debug=True)
        assert len(checks) == 2
        with defer_mpm_status_checks(manager):
            driver.check_step(captured=True, debug=False)
        assert len(checks) == 4  # The nested scope validates independently.
    assert len(checks) == 5
    driver.check_step(captured=True, debug=False)
    assert len(checks) == 6

    with pytest.raises(ValueError, match="controller failed"):
        with defer_mpm_status_checks(manager):
            raise ValueError("controller failed")
    assert len(checks) == 7  # Rejecting the body restores scope without accepting the window.
    driver.check_step(captured=True, debug=False)
    assert len(checks) == 8

    backend = manager.backend
    try:
        with pytest.raises(RuntimeError, match="backend changed"):
            with defer_mpm_status_checks(manager):
                manager.backend = None
    finally:
        manager.backend = backend
    driver.check_step(captured=True, debug=False)

    with pytest.raises(RuntimeError, match="working set requires 2 slots"):
        with defer_mpm_status_checks(manager):
            state.particle_f.assign(np.array([[0.0001, 0.0, 0.0]] * 2, dtype=np.float32))
            wp.capture_launch(capture.graph)
            driver.check_step(captured=True, debug=False)
            with pytest.raises(RuntimeError, match="working set requires 2 slots"):
                reset_mpm_solver_state(manager.backend, state=state, flags=0)
    with pytest.raises(RuntimeError, match="working set requires 2 slots"):
        driver.check_step(captured=True, debug=False)
    with pytest.raises(RuntimeError, match="working set requires 2 slots"):
        driver.check_step(captured=False, debug=False)
    with pytest.raises(RuntimeError, match="working set requires 2 slots"):
        with defer_mpm_status_checks(manager):
            pytest.fail("A pre-existing native fault entered a new rollout window.")

    manager.backend.graphs = {"step": capture.graph}
    with pytest.raises(RuntimeError, match="working set requires 2 slots"):
        configure_mpm_working_set(manager, MPMWorkingSetCfg(particle_capacity=2))
    assert manager.backend.graphs == {"step": capture.graph}
