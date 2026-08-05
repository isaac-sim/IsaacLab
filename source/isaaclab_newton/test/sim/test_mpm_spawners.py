# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg, MPMPointsCfg
from isaaclab_newton.sim.spawners.mpm.mpm import emit_mpm_particles


class _RecordingBuilder:
    def __init__(self):
        self.grid_kwargs = None
        self.points_kwargs = None

    def add_particle_grid(self, **kwargs):
        self.grid_kwargs = kwargs

    def add_particles(self, **kwargs):
        self.points_kwargs = kwargs


def _emit_recorded_grid(
    cfg: MPMGridCfg,
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> dict:
    builder = _RecordingBuilder()
    emit_mpm_particles(builder, cfg, position=position, orientation=orientation)
    assert builder.grid_kwargs is not None
    return builder.grid_kwargs


def _emit_recorded_points(
    cfg: MPMPointsCfg,
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> dict:
    builder = _RecordingBuilder()
    emit_mpm_particles(builder, cfg, position=position, orientation=orientation)
    assert builder.points_kwargs is not None
    return builder.points_kwargs


def test_grid_default_preserves_boundary_particle_placement():
    cfg = MPMGridCfg(
        lower=(0.0, 0.0, 0.0),
        upper=(0.1, 0.2, 0.3),
        voxel_size=1.0,
    )

    emitted = _emit_recorded_grid(cfg)

    assert np.asarray(emitted["pos"]).tolist() == [0.0, 0.0, 0.0]
    assert (emitted["dim_x"], emitted["dim_y"], emitted["dim_z"]) == (2, 2, 2)
    assert emitted["radius_mean"] == pytest.approx(0.15)


@pytest.mark.parametrize("particles_per_cell", [0.7, 1.3, 2.25])
def test_grid_emits_cell_centers_with_exact_density_mass(particles_per_cell):
    lower = np.array((-0.1, 0.2, 1.0))
    upper = np.array((0.5, 1.1, 1.8))
    density = 1250.0
    voxel_size = 0.25
    cfg = MPMGridCfg(
        lower=lower,
        upper=upper,
        voxel_size=voxel_size,
        particles_per_cell=particles_per_cell,
        particle_placement="cell_center",
        material=MPMParticleMaterialCfg(density=density),
    )

    emitted = _emit_recorded_grid(cfg)

    extent = upper - lower
    resolution = np.maximum(np.ceil(particles_per_cell * extent / voxel_size), 1).astype(int)
    spacing = extent / resolution
    dimensions = np.array((emitted["dim_x"], emitted["dim_y"], emitted["dim_z"]))
    first_center = np.asarray(emitted["pos"])
    last_center = first_center + (dimensions - 1) * spacing
    particle_count = int(np.prod(dimensions))

    np.testing.assert_array_equal(dimensions, resolution)
    np.testing.assert_allclose(first_center, lower + 0.5 * spacing, rtol=1.0e-6)
    np.testing.assert_allclose(last_center, upper - 0.5 * spacing, rtol=1.0e-6)
    assert np.all(first_center > lower)
    assert np.all(last_center < upper)
    assert emitted["mass"] == pytest.approx(density * np.prod(spacing), rel=1.0e-6)
    assert particle_count * emitted["mass"] == pytest.approx(density * np.prod(extent), rel=1.0e-6)
    assert (2.0 * emitted["radius_mean"]) ** 3 == pytest.approx(np.prod(spacing), rel=1.0e-6)
    assert emitted["mass"] / (2.0 * emitted["radius_mean"]) ** 3 == pytest.approx(density, rel=1.0e-6)


@pytest.mark.parametrize("radius", [None, 0.007])
def test_grid_preserves_jitter_and_radius_semantics(radius):
    lower = np.array((-0.2, 0.1, 0.3))
    upper = np.array((0.3, 0.7, 1.0))
    position = np.array((4.0, -2.0, 0.5))
    orientation = (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))
    jitter = 0.013
    cfg = MPMGridCfg(
        lower=lower,
        upper=upper,
        voxel_size=0.21,
        particles_per_cell=1.4,
        particle_placement="cell_center",
        jitter=jitter,
        radius=radius,
    )

    emitted = _emit_recorded_grid(cfg, position=tuple(position), orientation=orientation)

    extent = upper - lower
    resolution = np.maximum(np.ceil(cfg.particles_per_cell * extent / cfg.voxel_size), 1).astype(int)
    spacing = extent / resolution
    local_first_center = lower + 0.5 * spacing
    expected_world_first_center = position + np.array(
        (-local_first_center[1], local_first_center[0], local_first_center[2])
    )
    expected_radius = 0.5 * np.cbrt(np.prod(spacing)) if radius is None else radius

    np.testing.assert_allclose(np.asarray(emitted["pos"]), expected_world_first_center, rtol=1.0e-6)
    assert emitted["jitter"] == pytest.approx(jitter)
    assert emitted["radius_mean"] == pytest.approx(expected_radius, rel=1.0e-6)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lower": (0.0, 0.0)}, "`lower` must have shape (3,)"),
        ({"upper": (1.0, np.nan, 1.0)}, "`upper` must contain only finite values"),
        ({"upper": (0.0, 1.0, 1.0)}, "upper corner must be greater than lower corner"),
        ({"voxel_size": np.nan}, "`voxel_size` must be finite and positive"),
        ({"particles_per_cell": np.inf}, "`particles_per_cell` must be finite and positive"),
        ({"jitter": -0.01}, "`jitter` must be finite and non-negative"),
        (
            {"material": MPMParticleMaterialCfg(density=0.0)},
            "`material.density` must be finite and positive when deriving particle mass",
        ),
        ({"mass": 0.0}, "`mass` must be finite and positive"),
        ({"radius": -0.01}, "`radius` must be finite and positive"),
    ],
)
def test_grid_rejects_invalid_physical_inputs(overrides, message):
    kwargs = {
        "lower": (0.0, 0.0, 0.0),
        "upper": (1.0, 1.0, 1.0),
        "voxel_size": 0.1,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=re.escape(message)):
        _emit_recorded_grid(MPMGridCfg(**kwargs))


def test_points_preserve_explicit_particle_values_and_pose_transform():
    cfg = MPMPointsCfg(
        positions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        velocities=((0.1, 0.0, 0.0), (0.0, 0.2, 0.0)),
        mass=(0.3, 0.4),
        radius=0.05,
    )

    emitted = _emit_recorded_points(
        cfg,
        position=(2.0, 3.0, 4.0),
        orientation=(0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)),
    )

    np.testing.assert_allclose(emitted["pos"], ((2.0, 4.0, 4.0), (1.0, 3.0, 4.0)), rtol=1.0e-6)
    np.testing.assert_allclose(emitted["vel"], ((0.0, 0.1, 0.0), (-0.2, 0.0, 0.0)), atol=1.0e-7)
    assert emitted["mass"] == [0.3, 0.4]
    assert emitted["radius"] == [0.05, 0.05]


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (
            MPMPointsCfg(positions=((0.0, np.nan, 0.0),)),
            "`positions` must contain only finite values",
        ),
        (
            MPMPointsCfg(positions=((0.0, 0.0, 0.0),), velocities=((0.0, np.inf, 0.0),)),
            "`velocities` must contain only finite values",
        ),
        (
            MPMPointsCfg(positions=((0.0, 0.0, 0.0),), mass=0.0),
            "`mass` must contain only finite positive values",
        ),
        (
            MPMPointsCfg(
                positions=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                mass=(1.0, np.inf),
            ),
            "`mass` must contain only finite positive values",
        ),
        (
            MPMPointsCfg(positions=((0.0, 0.0, 0.0),), radius=-0.1),
            "`radius` must contain only finite positive values",
        ),
    ],
)
def test_points_reject_invalid_physical_inputs(cfg, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        _emit_recorded_points(cfg)


def test_mpm_config_imports_do_not_load_pxr():
    code = textwrap.dedent(
        """
        import sys

        from isaaclab_newton.assets import MPMObjectCfg
        from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg, MPMPointsCfg

        MPMObjectCfg(
            prim_path="/World/envs/env_.*/Sand",
            spawn=MPMGridCfg(
                lower=(0.0, 0.0, 0.0),
                upper=(0.1, 0.1, 0.1),
                voxel_size=0.1,
                material=MPMParticleMaterialCfg(),
            ),
        )

        loaded_pxr_modules = [module for module in sys.modules if module == "pxr" or module.startswith("pxr.")]
        if loaded_pxr_modules:
            raise SystemExit("pxr loaded before SimulationApp: " + ", ".join(loaded_pxr_modules[:20]))
        """
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module",
    [
        "scripts.demos.mpm.newton_mpm_granular",
        "scripts.demos.mpm.snowball_smash",
        "scripts.demos.mpm.teapot_fill",
    ],
)
def test_mpm_demo_configs_do_not_load_pxr_before_simulation_launch(module):
    code = textwrap.dedent(
        f"""
        import importlib
        import sys

        sys.argv = ["demo.py", "--max_steps", "0", "--visualizer", "none", "--device", "cuda:0"]
        demo = importlib.import_module({module!r})
        demo.create_sim_cfg()

        loaded_pxr_modules = [name for name in sys.modules if name == "pxr" or name.startswith("pxr.")]
        if loaded_pxr_modules:
            raise SystemExit("pxr loaded before simulation launch: " + ", ".join(loaded_pxr_modules[:20]))
        """
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
