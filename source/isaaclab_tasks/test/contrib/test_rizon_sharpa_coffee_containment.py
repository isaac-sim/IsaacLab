# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native tool containment and liquid-level regressions, without a robot or renderer."""

import math

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics.mpm_manager import _make_solver_config
from newton._src.solvers.implicit_mpm.rasterized_collisions import Collider, _query_collider_sdf
from newton.solvers import SolverImplicitMPM
from scipy.spatial.transform import Rotation

from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.rizon_sharpa_coffee.assets import (
    CUP_INNER_RADIUS,
    SPOON_BOWL_DEPTH,
    SPOON_BOWL_LENGTH,
    SPOON_BOWL_THICKNESS,
    SPOON_BOWL_WIDTH,
    build_coffee_assets,
)
from isaaclab_tasks.contrib.rizon_sharpa_coffee.coffee_env_cfg import CoffeeEnvCfg, _particle_cfg
from isaaclab_tasks.contrib.rizon_sharpa_coffee.physics import CoffeeMPMSolver, configure_spoon_rim_margin

_SPOON_HALF_LENGTH = wp.constant(SPOON_BOWL_LENGTH / 2.0)
_SPOON_HALF_WIDTH = wp.constant(SPOON_BOWL_WIDTH / 2.0)
_SPOON_DEPTH = wp.constant(SPOON_BOWL_DEPTH)
_SPOON_THICKNESS = wp.constant(SPOON_BOWL_THICKNESS)


def _make_model(points, cfg, device):
    prototype = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(prototype)
    prototype.add_usd(
        build_coffee_assets()["spoon"],
        xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()),
        load_visual_shapes=False,
    )
    prototype.body_label[0] = "/Spoon"
    for index in range(len(prototype.shape_flags)):
        prototype.shape_flags[index] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        prototype.shape_flags[index] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        prototype.shape_margin[index] = cfg.voxel_size * cfg.spoon_margin_voxels
    spacing = cfg.voxel_size / 2.0
    prototype.add_particles(
        pos=points.tolist(),
        vel=np.zeros_like(points).tolist(),
        mass=[450.0 * spacing**3] * len(points),
        radius=[spacing / 2.0] * len(points),
    )
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(prototype)
    builder.particle_max_velocity = 4.0
    model = builder.finalize(device=device)
    model.mpm.friction.fill_(0.55)
    model.mpm.damping.fill_(0.001)
    return model


@wp.kernel
def _sample_sdf(
    points: wp.array[wp.vec3], collider: Collider, pose: wp.array[wp.transform], distances: wp.array[float]
):
    i = wp.tid()
    _found, distance, _normal, _velocity, _point, _material = _query_collider_sdf(points[i], collider, pose, 0)
    distances[i] = distance


def test_native_contact_margin_reduces_only_near_spoon_rim():
    """Actual signed contact clearance changes at the lip, preserving floor and handle."""
    cfg = CoffeeEnvCfg()
    cfg.voxel_size = 0.003
    model = _make_model(np.array([[0, 0, 0.2]]), cfg, "cpu")
    solver = CoffeeMPMSolver(
        model, SolverImplicitMPM.Config(voxel_size=0.003, grid_type="fixed", grid_padding=2), grains_per_particle=0
    )
    points = wp.array([[0, 0, 0.104], [-0.06, 0, 0.114], [0.027, 0.00003, 0.1091]], dtype=wp.vec3, device="cpu")
    distances = wp.empty(3, dtype=float, device="cpu")
    inputs = [points, solver._mpm_model.collider, model.body_q]
    wp.launch(_sample_sdf, dim=3, inputs=inputs, outputs=[distances], device="cpu")
    before = distances.numpy().copy()
    configure_spoon_rim_margin(solver, cfg.spoon_rim_margin_scale)
    wp.launch(_sample_sdf, dim=3, inputs=inputs, outputs=[distances], device="cpu")
    after = distances.numpy()
    assert np.isfinite(before).all() and np.isfinite(after).all()
    np.testing.assert_allclose(after[:2], before[:2], rtol=0, atol=1.0e-7)
    assert after[2] - before[2] == pytest.approx(0.0036, abs=1.0e-7)


@wp.func
def _blend(x: float):
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


@wp.func
def _blend_derivative(x: float):
    return 30.0 * x * x * (1.0 - x) * (1.0 - x)


@wp.kernel
def _move_spoon(
    pose: wp.array[wp.transform],
    velocity: wp.array[wp.spatial_vector],
    center_of_mass: wp.array[wp.vec3],
    tick: wp.array[int],
    dt: float,
):
    time = float(tick[0]) * dt
    progress = wp.clamp((time - 3.0) / 1.5, 0.0, 1.0)
    blend = _blend(progress)
    rate = _blend_derivative(progress) / 1.5
    position = wp.vec3(0.15 * blend, 0.20 * blend, 0.10 + 0.08 * blend)
    linear = wp.vec3(0.15 * rate, 0.20 * rate, 0.08 * rate)
    pitch = 0.13962634 * wp.sin(wp.pi * blend)
    angular = wp.vec3(0.0, 0.13962634 * wp.pi * wp.cos(wp.pi * blend) * rate, 0.0)
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), pitch)
    if time >= 4.5:
        progress = wp.clamp((time - 4.5) / 0.8, 0.0, 1.0)
        rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi * _blend(progress))
        angular = wp.vec3(0.5 * wp.pi * _blend_derivative(progress) / 0.8, 0.0, 0.0)
    pose[0] = wp.transform(position, rotation)
    velocity[0] = wp.spatial_vector(linear + wp.cross(angular, wp.quat_rotate(rotation, center_of_mass[0])), angular)
    tick[0] += 1


@wp.kernel
def _observe_floor(
    points: wp.array[wp.vec3],
    pose: wp.array[wp.transform],
    watched: wp.array[int],
    crossed: wp.array[int],
):
    index = wp.tid()
    if watched[index] != 0:
        point = wp.transform_point(wp.transform_inverse(pose[0]), points[index])
        radius_squared = (point[0] / _SPOON_HALF_LENGTH) ** 2.0 + (point[1] / _SPOON_HALF_WIDTH) ** 2.0
        # Once a particle leaves the central region, its later trajectory cannot
        # establish passage through the floor: it may have escaped over the rim.
        if radius_squared >= 0.8:
            watched[index] = 0
        elif point[2] < _SPOON_DEPTH * radius_squared - _SPOON_THICKNESS - 0.0005:
            crossed[index] = 1


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_spoon_holds_coffee_through_motion_and_releases_when_tipped(device):
    """The configured grid resolves the thin imported spoon without pinning coffee."""
    cfg = CoffeeEnvCfg()
    spacing = cfg.voxel_size / 2.0
    points = np.stack(
        np.meshgrid(
            np.arange(-0.022, 0.023, spacing),
            np.arange(-0.014, 0.015, spacing),
            np.arange(0.0028, 0.012, spacing),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    radius_squared = (points[:, 0] / (SPOON_BOWL_LENGTH / 2.0)) ** 2 + (points[:, 1] / (SPOON_BOWL_WIDTH / 2.0)) ** 2
    points = points[(radius_squared < 0.92) & (points[:, 2] > SPOON_BOWL_DEPTH * radius_squared + 0.0022)]
    points += np.random.default_rng(7).uniform(-0.0001, 0.0001, points.shape)
    radius_squared = (points[:, 0] / (SPOON_BOWL_LENGTH / 2.0)) ** 2 + (points[:, 1] / (SPOON_BOWL_WIDTH / 2.0)) ** 2
    points[:, 2] += 0.1

    model = _make_model(points, cfg, device)
    config = _make_solver_config(cfg.sim.physics.solver_cfg.entries[1].solver_cfg)
    config.voxel_size = cfg.voxel_size
    config.max_active_cell_count = max(1024, math.ceil(len(points) / 1024) * 1024)
    solver = CoffeeMPMSolver(model, config=config, grains_per_particle=0)
    configure_spoon_rim_margin(solver, cfg.spoon_rim_margin_scale)
    state = model.state()
    tick = wp.zeros(1, dtype=int, device=device)
    watched = wp.array((radius_squared < 0.65).astype(np.int32), dtype=int, device=device)
    crossed = wp.zeros(len(points), dtype=int, device=device)

    def step():
        wp.launch(
            _move_spoon,
            dim=1,
            inputs=[state.body_q, state.body_qd, model.body_com, tick, cfg.sim.dt],
            device=device,
        )
        solver.step(state, state, None, None, cfg.sim.dt)
        wp.launch(
            _observe_floor, dim=len(points), inputs=[state.particle_q, state.body_q, watched, crossed], device=device
        )

    def retained_count():
        pose = state.body_q.numpy()[0]
        positions = state.particle_q.numpy()
        assert np.isfinite(positions).all()
        assert np.isfinite(state.particle_qd.numpy()).all()
        local = Rotation.from_quat(pose[3:]).inv().apply(positions - pose[:3])
        radius_squared = (local[:, 0] / (SPOON_BOWL_LENGTH / 2.0)) ** 2 + (local[:, 1] / (SPOON_BOWL_WIDTH / 2.0)) ** 2
        return np.count_nonzero(
            (radius_squared < 1.0)
            & (local[:, 2] > SPOON_BOWL_DEPTH * radius_squared - 0.0005)
            & (local[:, 2] < SPOON_BOWL_DEPTH + 0.006)
        )

    step()
    with wp.ScopedCapture(device=device) as capture:
        for _ in range(8):
            step()
    for block in range(600):
        wp.capture_launch(capture.graph)
        if block in (299, 449):  # Three-second hold, then a 26 cm carry with eight degrees of pitch.
            assert not crossed.numpy().any(), "Coffee crossed the central spoon floor without first leaving the rim."
            assert retained_count() / len(points) > 0.99
    assert not crossed.numpy().any(), "Coffee crossed the floor during release."
    assert retained_count() / len(points) < 0.01, "Coffee should fall freely over the rim at ninety degrees."


@wp.kernel
def _stir_spoon(pose: wp.array[wp.transform], tick: wp.array[int], dt: float):
    time = float(tick[0]) * dt
    lower = _blend(wp.clamp((time - 4.0) / 2.0, 0.0, 1.0))
    raised = _blend(wp.clamp((time - 14.0) / 2.0, 0.0, 1.0))
    angle = 6.0 * wp.pi * _blend(wp.clamp((time - 6.0) / 8.0, 0.0, 1.0))
    position = wp.vec3(0.009 * wp.cos(angle), 0.009 * wp.sin(angle), 0.150 - 0.119 * lower + 0.119 * raised)
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * wp.pi) * wp.quat_from_axis_angle(
        wp.vec3(0.0, 1.0, 0.0), 65.0 * wp.pi / 180.0
    )
    pose[0] = wp.transform(position, rotation)
    tick[0] += 1


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize(
    ("voxel_size", "physics_hz", "iterations", "capacity"),
    ((0.003, 400, 500, 32768), (0.010, 100, 150, 2048)),
    ids=("fine", "coarse"),
)
def test_water_level_survives_stirring_with_native_profile(device, voxel_size, physics_hz, iterations, capacity):
    """A full-size fill retains its settled level after three turns and tool withdrawal."""
    cfg = CoffeeEnvCfg(water_enabled=True, voxel_size=voxel_size)
    cfg.spoon_margin_voxels = 0.0036 / voxel_size
    dt = 1.0 / physics_hz
    water = _particle_cfg(
        "Water", (0.0, 0.0, 0.0), CUP_INNER_RADIUS - 0.002, cfg.water_fill_height, voxel_size / 2.0, True
    )
    prototype = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(prototype)
    paths = build_coffee_assets()
    prototype.add_usd(paths["cup"], load_visual_shapes=False)
    prototype.add_usd(
        paths["spoon"], xform=wp.transform((0.009, 0.0, 0.150), wp.quat_identity()), load_visual_shapes=False
    )
    prototype.body_label[0] = "/Spoon"
    for index, body in enumerate(prototype.shape_body):
        prototype.shape_flags[index] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        prototype.shape_flags[index] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        prototype.shape_margin[index] = cfg.voxel_size * (cfg.spoon_margin_voxels if body >= 0 else 0.3)
    count = len(water.spawn.positions)
    prototype.add_particles(
        pos=water.spawn.positions,
        vel=[(0.0, 0.0, 0.0)] * count,
        mass=[water.spawn.mass] * count,
        radius=[water.spawn.radius] * count,
    )
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(prototype)
    builder.particle_max_velocity = 0.5 * cfg.voxel_size / dt
    model = builder.finalize(device=device)
    for name in (
        "young_modulus",
        "poisson_ratio",
        "damping",
        "friction",
        "viscosity",
        "yield_pressure",
        "tensile_yield_ratio",
        "yield_stress",
        "hardening",
        "hardening_rate",
        "softening_rate",
        "dilatancy",
    ):
        getattr(model.mpm, name).fill_(getattr(water.spawn.material, name))
    config = _make_solver_config(cfg.sim.physics.solver_cfg.entries[1].solver_cfg)
    config.max_active_cell_count = capacity
    config.max_iterations = iterations
    solver = CoffeeMPMSolver(model, config, grains_per_particle=0)
    configure_spoon_rim_margin(solver, cfg.spoon_rim_margin_scale)
    state = model.state()
    tick = wp.zeros(1, dtype=int, device=device)
    with wp.ScopedCapture(device=device) as capture:
        for _ in range(8):
            wp.launch(_stir_spoon, dim=1, inputs=[state.body_q, tick, dt], device=device)
            solver.step(state, state, None, None, dt)

    initial_level = None
    for block in range(round(18 * physics_hz / 8)):
        wp.capture_launch(capture.graph)
        if block + 1 == round(4 * physics_hz / 8):
            initial_level = np.quantile(state.particle_q.numpy()[:, 2], 0.999)
    positions = state.particle_q.numpy()
    assert np.isfinite(positions).all()
    contained = (
        (np.linalg.norm(positions[:, :2], axis=1) < CUP_INNER_RADIUS + 0.004)
        & (positions[:, 2] > 0.002)
        & (positions[:, 2] < 0.085)
    )
    assert contained.mean() > 0.995, "Stirring should retain the water inside the cup."
    final_level = np.quantile(positions[contained, 2], 0.999)
    assert abs(final_level - initial_level) < 0.002, "The full fill contracted during stirring despite retaining mass."
    solver.check_sparse_grid_rebuild_status()
