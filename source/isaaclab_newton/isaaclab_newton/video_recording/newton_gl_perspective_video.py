# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton GL perspective RGB capture via headless ``newton.viewer.ViewerGL``."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .newton_gl_perspective_video_cfg import NewtonGlPerspectiveVideoCfg

logger = logging.getLogger(__name__)


class NewtonGlPerspectiveVideo:
    """Lazy-initialised ViewerGL; one RGB frame per :meth:`render_rgb_array` call."""

    def __init__(self, cfg: NewtonGlPerspectiveVideoCfg):
        self.cfg = cfg
        self._viewer = None
        self._init_attempted = False

    def _ensure_viewer(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        from isaaclab.sim import SimulationContext

        sdp = SimulationContext.instance().initialize_scene_data_provider()
        model = sdp.get_newton_model()
        if model is None:
            raise RuntimeError(
                "Newton GL perspective video requires a Newton model on the scene data provider. "
                "Do not use --video for this setup."
            )

        import pyglet

        pyglet.options["headless"] = True
        from newton.viewer import ViewerGL

        w, h = self.cfg.window_width, self.cfg.window_height
        viewer = ViewerGL(width=w, height=h, headless=True)
        viewer.set_model(model)
        viewer.set_world_offsets((0.0, 0.0, 0.0))
        viewer.up_axis = 2

        import warp as wp

        ex, ey, ez = self.cfg.camera_position
        lx, ly, lz = self.cfg.camera_target
        dx, dy, dz = lx - ex, ly - ey, lz - ez
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        dx, dy, dz = dx / length, dy / length, dz / length
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, dz))))
        yaw = math.degrees(math.atan2(dy, dx))
        aspect = w / h
        h_fov = math.radians(self.cfg.horiz_fov_deg)
        v_fov_deg = math.degrees(2.0 * math.atan(math.tan(h_fov / 2.0) / aspect))
        viewer.camera.fov = v_fov_deg
        viewer.set_camera(pos=wp.vec3(ex, ey, ez), pitch=pitch, yaw=yaw)

        self._viewer = viewer
        logger.info("[NewtonGlPerspectiveVideo] ViewerGL ready (%dx%d).", w, h)

    def render_rgb_array(self) -> np.ndarray:
        """Return one RGB frame from the Newton GL viewer. Raises on failure."""
        self._ensure_viewer()
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        sdp = sim.initialize_scene_data_provider()
        state = sdp.get_newton_state()
        dt = sim.get_physics_dt()

        viewer = self._viewer
        viewer.begin_frame(dt)
        viewer.log_state(state)
        viewer.end_frame()
        return viewer.get_frame().numpy()


def create_newton_gl_perspective_video(cfg: NewtonGlPerspectiveVideoCfg) -> NewtonGlPerspectiveVideo:
    """Instantiate the Newton GL perspective capture from ``cfg.class_type``."""
    ct = cfg.class_type
    if isinstance(ct, type):
        return ct(cfg)
    from isaaclab.utils.string import string_to_callable

    cls = string_to_callable(str(ct))
    return cls(cfg)
