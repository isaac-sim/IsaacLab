# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamic geometry remains live when independent GL viewers alternate frames."""

import numpy as np
import pytest
import warp as wp
from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerGL

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.visualizers import WindowCfg


@pytest.mark.rendering
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_dynamic_mesh_moves_after_growth_between_viewers(device: str) -> None:
    """A growing surface must move in rendered pixels after another viewer has rendered."""
    viewers = []
    with wp.ScopedDevice(device):
        try:
            for _ in range(2):
                viewer = NewtonViewerGL(width=256, height=192, headless=True, window_cfg=WindowCfg())
                viewers.append(viewer)
                viewer.renderer.draw_grid = viewer.renderer.draw_sky = False
                viewer.renderer.sky_upper = (0.0, 0.0, 0.0)
                viewer.set_camera((0.0, -3.0, 0.0))
                viewer.camera.look_at((0.0, 0.0, 0.0))
            primary, surface = viewers
            centers = []
            for frame, x in enumerate((-0.65, 0.65, -0.65)):
                # Grow both meshes so closing one viewer must preserve the other's replacement VAO.
                points = np.array(((x - 0.25, 0.0, -0.3), (x + 0.25, 0.0, -0.3), (x, 0.0, 0.3)))
                points = np.tile(points, (1 if frame == 0 else 2, 1))
                if frame == 2:
                    primary.close()
                    primary.close()
                    viewers.remove(primary)
                for viewer in viewers:
                    viewer.begin_frame(frame / 60.0)
                    viewer.log_mesh(
                        "/surface",
                        wp.array(points, dtype=wp.vec3),
                        wp.array(np.arange(len(points)), dtype=wp.int32),
                        color=(0.05, 0.2, 1.0),
                        backface_culling=False,
                        dynamic=True,
                    )
                    viewer.end_frame()
                pixels = surface.get_frame().numpy().astype(np.int16)
                blue = (pixels[..., 2] > 3 * pixels[..., 1]) & (pixels[..., 2] > 70)
                assert blue.any(), "The dynamic surface disappeared"
                centers.append(np.argwhere(blue)[:, 1].mean())
            assert centers[1] > centers[0] + 50.0, "The replacement surface remained at its old position"
            assert abs(centers[2] - centers[0]) < 2.0, "The replacement surface stopped updating"
        finally:
            for viewer in viewers:
                viewer.close()
