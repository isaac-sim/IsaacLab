# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent physics and passive-tracer contracts for coffee presentation."""

import math
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import warp as wp
from newton._src.utils.texture import normalize_texture
from newton.geometry import ParticleSurface
from newton.solvers import SolverImplicitMPM

from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.rizon_sharpa_coffee.physics import CoffeeMPMSolver
from isaaclab_tasks.contrib.rizon_sharpa_coffee.presentation import (
    CoffeePresentation,
    _color_water_faces,
    _color_water_vertices,
    _diffuse_water_color,
    _release_converted_color,
    _water_diffusion_support,
)
from isaaclab_tasks.contrib.rizon_sharpa_coffee.surface import SurfaceHistory


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_grains_preserve_physics_and_follow_partial_reset_under_capture(device):
    """Twenty visual grains retain native physics and reset only the requested world."""
    prototype = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(prototype)
    prototype.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, 0.04),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.01, 0.0, 0.0),
        dim_x=3,
        dim_y=3,
        dim_z=3,
        cell_x=0.0025,
        cell_y=0.0025,
        cell_z=0.0025,
        mass=1.0e-5,
        radius_mean=0.00125,
        jitter=0.0,
    )
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(prototype)
    builder.add_world(prototype)
    model = builder.finalize(device=device)
    config = SolverImplicitMPM.Config(
        voxel_size=0.005,
        separate_worlds=True,
        grid_type="sparse",
        max_active_cell_count=1024,
        grid_padding=0,
        velocity_basis="Q1",
        strain_basis="P0",
        collider_basis="S2",
        max_iterations=10,
    )
    solver = CoffeeMPMSolver(model, config, grains_per_particle=20)
    reference_solver = SolverImplicitMPM(model, config=config)
    state, reference = model.state(), model.state()
    dt = 1.0 / 800.0
    for _ in range(4):
        solver.step(state, state, None, None, dt)
        reference_solver.step(reference, reference, None, None, dt)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state, state, None, None, dt)
    wp.capture_launch(capture.graph)
    reference_solver.step(reference, reference, None, None, dt)
    np.testing.assert_allclose(state.particle_q.numpy(), reference.particle_q.numpy(), rtol=0, atol=1.0e-8)
    np.testing.assert_allclose(state.particle_qd.numpy(), reference.particle_qd.numpy(), rtol=0, atol=1.0e-8)
    assert solver.grains.shape == (model.particle_count, 20)
    assert np.isfinite(solver.grains.numpy()).all()

    worlds = model.particle_world.numpy()
    selected = worlds == 0
    old_grains = solver.grains.numpy()
    positions = state.particle_q.numpy()
    positions[selected, 0] += 0.25
    wp.copy(state.particle_q, wp.array(positions, dtype=wp.vec3, device=device))
    buffer_ptr = solver.grains.ptr
    solver.reset(state, world_mask=wp.array([True, False, False], dtype=wp.bool, device=device))
    assert solver.grains.ptr == buffer_ptr  # CUDA graphs retain this allocation.
    reset_grains = solver.grains.numpy()
    np.testing.assert_array_equal(reset_grains[~selected], old_grains[~selected])
    assert np.max(np.abs(reset_grains[selected] - positions[selected, None, :])) <= 0.001251
    wp.capture_launch(capture.graph)
    assert np.isfinite(solver.grains.numpy()).all()


def test_liquid_color_is_released_once_and_dry_coffee_keeps_its_tracer():
    """Only physically converted coffee initializes a finite tracer on its carrier."""
    device = "cpu"
    masses = np.array([0.002, 0.001, 0.004], dtype=np.float32)
    mass = wp.array(masses, dtype=float, device=device)
    flags = wp.array([1, 1, 0], dtype=int, device=device)
    released = wp.zeros(2, dtype=int, device=device)
    dye = wp.zeros(3, dtype=float, device=device)
    ids = wp.array([0, 1, 2], dtype=int, device=device)
    for _ in range(2):
        wp.launch(
            _release_converted_color,
            dim=2,
            inputs=[ids, mass, 1, flags, released, dye],
            device=device,
        )
    np.testing.assert_allclose(dye.numpy(), [0.0, 0.001, 0.0], atol=1.0e-10)
    flags.assign([1, 1, 1])
    wp.launch(_release_converted_color, dim=2, inputs=[ids, mass, 1, flags, released, dye], device=device)
    np.testing.assert_allclose(dye.numpy(), [0.0, 0.001, 0.004], atol=1.0e-10)
    np.testing.assert_array_equal(mass.numpy(), masses)


def test_water_tint_diffuses_conservatively_without_crossing_air_gaps():
    """Unequal carrier masses mix to a common concentration; isolated water stays unchanged."""
    device = "cpu"
    water = wp.array([[0, 0, 0], [0.001, 0, 0], [0.0018, 0, 0], [1, 0, 0], [0.001, 0, 0]], dtype=wp.vec3, device=device)
    masses = np.array([1, 2, 3, 4, 100], dtype=np.float32)
    mass = wp.array(masses, device=device)
    flags = wp.array([1, 1, 1, 1, 0], dtype=int, device=device)
    dye = wp.array([0.005, 0, 0, 0.004, 0], dtype=float, device=device)
    output, support = wp.empty_like(dye), wp.empty_like(dye)
    grid = wp.HashGrid(8, 8, 8, device=device)
    grid.build(water, 0.002)
    wp.launch(
        _water_diffusion_support, dim=5, inputs=[water, mass, flags, grid.id, 0.002], outputs=[support], device=device
    )
    wp.launch(
        _diffuse_water_color,
        dim=5,
        inputs=[water, mass, flags, grid.id, 0.002, 0.0, support, dye],
        outputs=[output],
        device=device,
    )
    np.testing.assert_array_equal(output.numpy(), dye.numpy())
    for _ in range(100):
        wp.launch(
            _diffuse_water_color,
            dim=5,
            inputs=[water, mass, flags, grid.id, 0.002, 0.4, support, dye],
            outputs=[output],
            device=device,
        )
        dye, output = output, dye
    result = dye.numpy()
    assert result.sum() == pytest.approx(0.009, abs=1.0e-8)
    assert np.all(result >= 0.0)
    np.testing.assert_allclose(result[:3] / masses[:3], 0.005 / 6.0, atol=1.0e-7)
    assert result[3] == pytest.approx(0.004, abs=1.0e-9)
    assert result[4] == 0.0  # Nearby dry coffee cannot absorb the liquid's colour.

    # The recorded tracer metric must observe the colour field, independently
    # of the physical mass converted to liquid. A missing dye release should
    # remain detectable instead of reporting the source mass twice.
    presentation = CoffeePresentation.__new__(CoffeePresentation)
    presentation._water = object()
    presentation._dissolution = SimpleNamespace(converted_mass_kg=0.009)
    presentation._dye_mass = dye
    assert presentation.tracer_mass_kg == pytest.approx(0.009, abs=1.0e-8)
    dye.assign(result * 0.5)
    assert presentation.tracer_mass_kg == pytest.approx(0.0045, abs=1.0e-8)
    np.testing.assert_array_equal(mass.numpy(), masses)


def test_gl_surface_tint_stays_on_the_wet_faces_after_texture_upload():
    """A brown face in the second atlas row retains its spatial colour after GL's row flip."""
    device = "cpu"
    points = wp.array(
        [[-0.01, -0.01, 0], [0.01, -0.01, 0], [0, 0.02, 0], [0.99, -0.01, 0], [1.01, -0.01, 0], [1, 0.02, 0]],
        dtype=wp.vec3,
        device=device,
    )
    indices = wp.array([0, 1, 2] * 256 + [3, 4, 5], dtype=int, device=device)
    water = wp.array([[0, 0, 0], [1, 0, 0]], dtype=wp.vec3, device=device)
    mass = wp.ones(2, device=device)
    dye = wp.array([0, 1], dtype=float, device=device)
    grid = wp.HashGrid(8, 8, 8, device=device)
    grid.build(water, 0.1)
    clear, brown = wp.vec3(0.66, 0.83, 0.88), wp.vec3(0.12, 0.041, 0.014)
    colors = wp.full(512, clear, device=device)
    expanded = wp.empty(771, dtype=wp.vec3, device=device)
    normals, uvs = wp.empty_like(expanded), wp.empty(771, dtype=wp.vec2, device=device)
    vertex_colors = wp.empty(6, dtype=wp.vec3, device=device)
    wp.launch(
        _color_water_vertices,
        dim=6,
        inputs=[points, water, mass, wp.ones(2, dtype=int, device=device), dye, grid.id, 0.1, 0.1, clear, brown],
        outputs=[vertex_colors],
        device=device,
    )
    wp.launch(
        _color_water_faces,
        dim=257,
        inputs=[
            points,
            indices,
            wp.zeros(6, dtype=wp.vec3, device=device),
            vertex_colors,
            256,
            2,
        ],
        outputs=[colors, expanded, normals, uvs],
        device=device,
    )
    uploaded = normalize_texture(
        colors.numpy().reshape(2, 256, 3),
        flip_vertical=True,
        require_channels=True,
        scale_unit_range=True,
    )
    # Sample the actual backend upload at each face's texture coordinates.
    coordinates = uvs.numpy()[[0, 768]]
    pixels = uploaded[(coordinates[:, 1] * 2).astype(int), (coordinates[:, 0] * 256).astype(int)]
    np.testing.assert_allclose(pixels / 255.0, [list(clear), list(brown)], atol=1.0 / 255.0)


@wp.kernel
def _write_density(coordinates: wp.array[wp.vec3i], values: wp.array[float], grid: wp.uint64, density: wp.array[float]):
    i = wp.tid()
    c = coordinates[i]
    density[wp.volume_lookup_index(grid, c[0], c[1], c[2])] = values[i]


@wp.kernel
def _read_density(coordinates: wp.array[wp.vec3i], grid: wp.uint64, density: wp.array[float], values: wp.array[float]):
    i = wp.tid()
    c = coordinates[i]
    values[i] = density[wp.volume_lookup_index(grid, c[0], c[1], c[2])]


def test_surface_history_matches_spatial_voxels_and_clears_on_reset():
    """An existing voxel filters a step; new topology and an episode reset stay fresh."""

    def field(coordinates, values):
        coordinates = wp.array(coordinates, dtype=wp.vec3i, device="cpu")
        volume = wp.Volume.allocate_by_voxels(coordinates, voxel_size=1.0, device="cpu")
        data = wp.zeros(volume.get_voxel_count(), device="cpu")
        wp.launch(
            _write_density,
            dim=coordinates.size,
            inputs=[coordinates, wp.array(values, dtype=float, device="cpu"), volume.id, data],
            device="cpu",
        )
        return ParticleSurface.SparseField(
            volume, data, 0.0, wp.zeros(1, dtype=wp.vec3i, device="cpu"), wp.zeros(1, dtype=int, device="cpu")
        ), coordinates

    history = SurfaceHistory(0.06)
    old, _ = field([(0, 0, 0), (8, 0, 0)], [0.0, 0.8])
    old_values = old.voxel_data.numpy().copy()
    history.apply(old, 0.0)
    current, coordinates = field([(-8, 0, 0), (0, 0, 0)], [0.3, 1.0])
    history.apply(current, 0.06 * math.log(2.0))
    samples = wp.empty(2, device="cpu")
    wp.launch(
        _read_density,
        dim=2,
        inputs=[coordinates, current.volume.id, current.voxel_data],
        outputs=[samples],
        device="cpu",
    )
    np.testing.assert_allclose(samples.numpy(), [0.3, 0.5], atol=1.0e-6)
    np.testing.assert_array_equal(old.voxel_data.numpy(), old_values)
    history.reset()
    fresh, coordinates = field([(0, 0, 0)], [0.9])
    history.apply(fresh, 0.06 * math.log(2.0))
    wp.launch(
        _read_density, dim=1, inputs=[coordinates, fresh.volume.id, fresh.voxel_data], outputs=[samples], device="cpu"
    )
    assert samples.numpy()[0] == pytest.approx(0.9, abs=1.0e-6)


def test_mounted_spoon_transfers_com_twist_to_both_state_buffers():
    """A rotated palm with offset COM gives the spoon the correct point velocity."""
    from isaaclab_tasks.contrib.rizon_sharpa_coffee.attachment import _sync_mounted_spoon

    device = "cpu"
    sine = math.sqrt(0.5)
    poses = wp.array([[1, 2, 3, 0, 0, sine, sine], [0, 0, 0, 0, 0, 0, 1]], dtype=wp.transform, device=device)
    velocities = wp.array([[3, 4, 5, 0, 0, 2], [0, 0, 0, 0, 0, 0]], dtype=wp.spatial_vector, device=device)
    other_poses = wp.clone(poses)
    other_velocities = wp.clone(velocities)
    joint_q, other_joint_q = wp.zeros(10, device=device), wp.zeros(10, device=device)
    joint_qd, other_joint_qd = wp.zeros(8, device=device), wp.zeros(8, device=device)
    wp.launch(
        _sync_mounted_spoon,
        dim=1,
        inputs=[
            wp.array([0], dtype=int, device=device),
            wp.array([1], dtype=int, device=device),
            wp.array([2], dtype=int, device=device),
            wp.array([1], dtype=int, device=device),
            wp.transform(wp.vec3(1, 0, 0), wp.quat(1, 0, 0, 0)),
            wp.array([[0.1, 0, 0], [0.2, 0, 0]], dtype=wp.vec3, device=device),
            wp.ones(1, dtype=int, device=device),
            poses,
            velocities,
            other_poses,
            other_velocities,
            joint_q,
            joint_qd,
            other_joint_q,
            other_joint_qd,
        ],
        device=device,
    )
    np.testing.assert_allclose(poses.numpy()[1], [1, 3, 3, sine, sine, 0, 0], atol=1.0e-6)
    np.testing.assert_allclose(velocities.numpy()[1], [0.8, 4, 5, 0, 0, 2], atol=1.0e-6)
    np.testing.assert_array_equal(other_poses.numpy()[1], poses.numpy()[1])
    np.testing.assert_array_equal(other_velocities.numpy()[1], velocities.numpy()[1])
    np.testing.assert_allclose(velocities.numpy()[0], [3, 4, 5, 0, 0, 2])
    np.testing.assert_allclose(joint_q.numpy()[2:9], [1, 3, 3, sine, sine, 0, 0], atol=1.0e-6)
    np.testing.assert_allclose(joint_qd.numpy()[1:7], [0.8, 4, 5, 0, 0, 2], atol=1.0e-6)
    np.testing.assert_array_equal(other_joint_q.numpy(), joint_q.numpy())
    np.testing.assert_array_equal(other_joint_qd.numpy(), joint_qd.numpy())
