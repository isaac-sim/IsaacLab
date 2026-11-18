# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coffee surface publication at the native GL rendering boundary."""

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg

from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.visualizers import VisualizerCfg, WindowCfg

from isaaclab_tasks.contrib.rizon_sharpa_coffee.rendering import (
    BREW_COLOR,
    WATER_COLOR,
    CoffeeGLVisualizer,
    CoffeeRendering,
)

pytestmark = [pytest.mark.rendering, pytest.mark.kitless]


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_coffee_surface_draw_handles_empty_and_changing_color(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An initially empty surface must become visible, change tint, hide, and return."""
    visualizer_cfg = VisualizerCfg(
        backend=NewtonGLBackendCfg(class_type=CoffeeGLVisualizer),
        headless=True,
        eye=(0.0, -3.0, 0.0),
        lookat=(0.0, 0.0, 0.0),
        background_color=(0.0, 0.0, 0.0),
        window=WindowCfg(size=(128, 128)),
    )
    sim_cfg = SimulationCfg(
        device=device, physics=NewtonCfg(solver_cfg=XPBDSolverCfg()), visualizer_cfgs=[visualizer_cfg]
    )
    with wp.ScopedDevice(device), build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim.reset()
        (visualizer,) = sim.visualizers
        # Supply the prepared mesh at the MPM-to-presentation boundary.
        points = wp.array(
            ((-0.75, 0.0, -0.75), (0.75, 0.0, -0.75), (0.75, 0.0, 0.75), (-0.75, 0.0, 0.75)),
            dtype=wp.vec3,
        )
        indices = wp.array((0, 1, 2, 0, 2, 3), dtype=wp.int32)
        normals = wp.full(4, wp.vec3(0.0, -1.0, 0.0), dtype=wp.vec3)
        water = wp.full(4, wp.vec3(WATER_COLOR), dtype=wp.vec3)
        brew = wp.full(4, wp.vec3(BREW_COLOR), dtype=wp.vec3)
        presentation = CoffeeRendering.__new__(CoffeeRendering)
        presentation._model = sim.backend_manager.backend.model
        presentation._surface = object()
        presentation._gl_surfaces = {}
        presentation._points = presentation._colors = wp.empty(0, dtype=wp.vec3)
        presentation._radii = wp.empty(0, dtype=float)
        presentation._empty_points = presentation._points
        presentation._empty_indices = wp.empty(0, dtype=wp.int32)
        monkeypatch.setattr(presentation, "update", lambda: None)
        visualizer.rendering = presentation

        frames = []
        for colors in (None, water, brew, None, water):
            # Hold opacity fixed so a color failure cannot pass through an opacity change.
            surface = (None, None, None, None, 0.5) if colors is None else (points, indices, normals, colors, 0.5)
            monkeypatch.setattr(presentation, "_water_surface_arrays", lambda data=surface: data)
            sim.render()
            frames.append(visualizer.render_rgb_array().copy())
        visible = np.any(frames[1] != frames[0], axis=-1)
        assert visible.sum() > 100, "The initially empty coffee surface never became visible"
        water_rgb = frames[1][visible].astype(np.float32).mean(axis=0)
        brew_rgb = frames[2][visible].astype(np.float32).mean(axis=0)
        assert water_rgb[2] > brew_rgb[2] + 20.0, "The brewed-coffee tint did not reach rendered pixels"
        np.testing.assert_array_equal(frames[0], frames[3], err_msg="The empty surface remained visible")
        np.testing.assert_array_equal(frames[1], frames[4], err_msg="The restored surface lost its color")
