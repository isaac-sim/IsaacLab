# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coffee surface publication at the native GL rendering boundary."""

import numpy as np
import pytest
import warp as wp
from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerGL

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.visualizers import WindowCfg

from isaaclab_tasks.contrib.rizon_sharpa_coffee.rendering import BREW_COLOR, WATER_COLOR, _GLWaterSurface

pytestmark = [pytest.mark.rendering, pytest.mark.kitless]


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_coffee_surface_draw_handles_empty_and_changing_color(device: str) -> None:
    """An initially empty surface must become visible, change tint, hide, and return."""
    with wp.ScopedDevice(device):
        viewer = NewtonViewerGL(width=128, height=128, headless=True, window_cfg=WindowCfg())
        try:
            viewer.renderer.draw_grid = viewer.renderer.draw_sky = False
            viewer.renderer.sky_upper = (0.0, 0.0, 0.0)
            viewer.set_camera((0.0, -3.0, 0.0))
            viewer.camera.look_at((0.0, 0.0, 0.0))
            points = wp.array(
                ((-0.75, 0.0, -0.75), (0.75, 0.0, -0.75), (0.75, 0.0, 0.75), (-0.75, 0.0, 0.75)),
                dtype=wp.vec3,
            )
            indices = wp.array((0, 1, 2, 0, 2, 3), dtype=wp.int32)
            normals = wp.full(4, wp.vec3(0.0, -1.0, 0.0), dtype=wp.vec3)
            water = wp.full(4, wp.vec3(WATER_COLOR), dtype=wp.vec3)
            brew = wp.full(4, wp.vec3(BREW_COLOR), dtype=wp.vec3)
            presentation = _GLWaterSurface(device)
            frames = []
            for frame, colors in enumerate((None, water, brew, None, water)):
                # Hold opacity fixed so a color failure cannot pass through an opacity change.
                surface = (None, None, None, None, 0.5) if colors is None else (points, indices, normals, colors, 0.5)
                viewer.begin_frame(frame / 60.0)
                presentation.draw(viewer, *surface)
                viewer.end_frame()
                frames.append(viewer.get_frame().numpy())
            visible = np.any(frames[1] != frames[0], axis=-1)
            assert visible.sum() > 100, "The initially empty coffee surface never became visible"
            water_rgb = frames[1][visible].astype(np.float32).mean(axis=0)
            brew_rgb = frames[2][visible].astype(np.float32).mean(axis=0)
            assert water_rgb[2] > brew_rgb[2] + 20.0, "The brewed-coffee tint did not reach rendered pixels"
            np.testing.assert_array_equal(frames[0], frames[3], err_msg="The empty surface remained visible")
            np.testing.assert_array_equal(frames[1], frames[4], err_msg="The restored surface lost its color")
        finally:
            viewer.close()
