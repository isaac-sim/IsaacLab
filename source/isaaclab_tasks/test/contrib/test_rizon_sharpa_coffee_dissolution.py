# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical wetting, conservation and captured-reset contracts for soluble coffee."""

import newton
import numpy as np
import pytest
import warp as wp
from newton.solvers import SolverImplicitMPM
from newton.solvers.experimental.coupled import SolverCoupled

from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.rizon_sharpa_coffee.dissolution import CoffeeDissolution
from isaaclab_tasks.contrib.rizon_sharpa_coffee.physics import CoffeeMPMSolver


def _make_solver(device, wetting_time, *, coupled=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for world in range(2):
        builder.begin_world()
        for i, position in enumerate(((0, 0, 0.02), (0.1, 0, 0.02), (float(world), 0.001, 0.02))):
            builder.add_particle(
                pos=wp.vec3(*position),
                vel=wp.vec3(0.01, 0.02, 0),
                mass=(1000.0 if i == 2 else 450.0) * 0.0015**3,
                radius=0.00075,
            )
        builder.end_world()
    model = builder.finalize(device=device)
    model.mpm.friction.assign([0.55, 0.55, 0.0] * 2)
    model.mpm.viscosity.assign([0.0, 0.0, 0.001] * 2)
    model.mpm.yield_pressure.assign([0.0, 0.0, 1.0e15] * 2)
    model.mpm.tensile_yield_ratio.assign([0.0, 0.0, 1.0] * 2)
    config = SolverImplicitMPM.Config(
        voxel_size=0.003,
        grid_type="fixed" if device == "cpu" else "sparse",
        grid_padding=0,
        max_active_cell_count=1024,
        velocity_basis="Q1",
        strain_basis="P1d",
        collider_basis="pic27",
        separate_worlds=True,
        max_iterations=10,
    )
    if coupled:
        stepper = SolverCoupled(
            model,
            entries=[
                SolverCoupled.Entry(
                    name="media",
                    solver=lambda view: CoffeeMPMSolver(view, config, grains_per_particle=0),
                    particles=list(range(model.particle_count)),
                    in_place=True,
                )
            ],
        )
        solver = stepper.solver("media")
    else:
        solver = stepper = CoffeeMPMSolver(model, config, grains_per_particle=0)
    solver.dissolution = CoffeeDissolution(
        solver,
        wp.array([0, 1, 3, 4], dtype=int, device=device),
        wp.array([2, 5], dtype=int, device=device),
        wetting_time=wetting_time,
        contact_radius=0.00255,
    )
    return solver, model.state(), stepper


def test_conversion_requires_own_world_water_and_preserves_mass_and_motion():
    """Wetted coffee becomes water material without deleting or moving particles."""
    solver, state, _ = _make_solver("cpu", 0.12)
    model, dissolution = solver.model, solver.dissolution
    mass, positions, velocities = model.particle_mass.numpy(), state.particle_q.numpy(), state.particle_qd.numpy()
    state.mpm.particle_stress.fill_(wp.mat33(0.2))
    for _ in range(2):
        dissolution.step(state, 0.04)
    np.testing.assert_array_equal(dissolution.liquid_flags.numpy(), [1, 1, 0, 0, 0, 0])
    dissolution.step(state, 0.04)
    # World 1 coffee overlaps world 0 water but must not dissolve in it.
    np.testing.assert_array_equal(dissolution.liquid_flags.numpy(), [1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(model.particle_mass.numpy(), mass)
    np.testing.assert_array_equal(state.particle_q.numpy(), positions)
    np.testing.assert_array_equal(state.particle_qd.numpy(), velocities)
    assert model.mpm.friction.numpy()[0] == 0.0
    assert model.mpm.viscosity.numpy()[0] == pytest.approx(0.001)
    assert model.mpm.friction.numpy()[1] == pytest.approx(0.55)
    assert model.mpm.friction.numpy()[3] == pytest.approx(0.55)
    assert 8.0 * model.particle_radius.numpy()[0] ** 3 == pytest.approx(mass[0] / 1000.0, rel=1.0e-6)
    assert solver._mpm_model.material_particle_volume.numpy()[0] == pytest.approx(mass[0] / 1000.0, rel=1.0e-6)
    np.testing.assert_array_equal(state.mpm.particle_stress.numpy()[0], np.zeros((3, 3)))
    np.testing.assert_array_equal(state.mpm.particle_elastic_strain.numpy()[0], np.eye(3))
    assert dissolution.converted_mass_kg == pytest.approx(float(mass[0]))
    # A grain touching the mixture, but beyond the original solvent's radius,
    # must also dissolve. This prevents a permanent crust above the first layer.
    positions[1] = [0, -0.002, 0.02]
    state.particle_q.assign(positions)
    for _ in range(3):
        dissolution.step(state, 0.04)
    np.testing.assert_array_equal(dissolution.liquid_flags.numpy(), [1, 1, 1, 1, 0, 0])
    assert dissolution.converted_mass_kg == pytest.approx(float(mass[0] + mass[1]))


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_native_capture_converts_and_partial_reset_restores_only_selected_material(device):
    """Graph replay performs real conversion and survives an in-place partial reset."""
    solver, state, stepper = _make_solver(device, 0.003, coupled=True)
    model, dissolution = solver.model, solver.dissolution
    mass = model.particle_mass.numpy()
    # Isaac Lab captures the first step without an eager physics warmup.
    with wp.ScopedCapture(device=device) as capture:
        stepper.step(state, state, None, None, 0.001)
    # Rendering builds a separate float grid between captured physics replays.
    # Its descriptor must never replace the wetting grid's captured descriptor.
    render_grid = wp.HashGrid(8, 8, 8, device=device)
    render_grid.build(state.particle_q, 0.006)
    for _ in range(5):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(dissolution.liquid_flags.numpy(), [1, 1, 1, 0, 0, 0])
    positions = state.particle_q.numpy()
    positions[5] = positions[3] + [0, 0.001, 0]
    state.particle_q.assign(positions)
    for _ in range(5):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(dissolution.liquid_flags.numpy(), [1, 1, 1, 0, 1, 0])
    flags_pointer = dissolution.liquid_flags.ptr
    stepper.reset(state, world_mask=wp.array([True, False, False], dtype=wp.bool, device=device))
    assert dissolution.liquid_flags.ptr == flags_pointer
    np.testing.assert_array_equal(dissolution.liquid_flags.numpy(), [1, 1, 0, 0, 1, 0])
    assert model.mpm.friction.numpy()[0] == pytest.approx(0.55)
    assert model.mpm.friction.numpy()[3] == 0.0
    assert 8.0 * model.particle_radius.numpy()[0] ** 3 == pytest.approx(mass[0] / 450.0, rel=1.0e-6)
    # Reuse the same graph after reset, keeping the reset grain out of water.
    positions = state.particle_q.numpy()
    positions[0] = [0.5, 0, 0.02]
    state.particle_q.assign(positions)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(dissolution.liquid_flags.numpy(), [1, 1, 0, 0, 1, 0])
    np.testing.assert_array_equal(model.particle_mass.numpy(), mass)
    assert np.isfinite(state.particle_q.numpy()).all()
    assert np.isfinite(state.particle_qd.numpy()).all()
    solver.check_sparse_grid_rebuild_status()
