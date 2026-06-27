# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton GL perspective RGB capture via headless ``newton.viewer.ViewerGL``."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from newton.viewer import ViewerGL

    from .newton_gl_perspective_video_cfg import NewtonGlPerspectiveVideoCfg

logger = logging.getLogger(__name__)


class NewtonGlPerspectiveVideo:
    """Lazy-initialised ViewerGL; one RGB frame per :meth:`render_rgb_array` call."""

    def __init__(self, cfg: NewtonGlPerspectiveVideoCfg):
        self.cfg = cfg
        self._viewer = None
        self._init_attempted = False
        self._visible_worlds: tuple[int, ...] | None = None
        self._world_spacing: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._frame_overlay_callback: Callable[[ViewerGL], None] | None = None

    def _ensure_viewer(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        if model is None:
            raise RuntimeError(
                "Newton GL perspective video requires a Newton model from NewtonManager. "
                "Do not use --video for this setup."
            )

        import pyglet

        pyglet.options["headless"] = True
        from newton.viewer import ViewerGL

        w, h = self.cfg.window_width, self.cfg.window_height
        viewer = ViewerGL(width=w, height=h, headless=True)
        viewer.set_model(model)
        set_visible_worlds = getattr(viewer, "set_visible_worlds", None)
        if self._visible_worlds is not None and callable(set_visible_worlds):
            set_visible_worlds(list(self._visible_worlds))
        viewer.set_world_offsets(self._world_spacing)
        viewer.up_axis = 2

        aspect = w / h
        h_fov = math.radians(self.cfg.horiz_fov_deg)
        v_fov_deg = math.degrees(2.0 * math.atan(math.tan(h_fov / 2.0) / aspect))
        viewer.camera.fov = v_fov_deg

        self._viewer = viewer
        self._apply_camera(self.cfg.eye, self.cfg.lookat)
        logger.info("[NewtonGlPerspectiveVideo] ViewerGL ready (%dx%d).", w, h)

    def _apply_camera(
        self,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> None:
        """Point the recorder's ViewerGL at ``position`` looking toward ``target``."""
        if self._viewer is None:
            return
        import warp as wp

        ex, ey, ez = position
        lx, ly, lz = target
        dx, dy, dz = lx - ex, ly - ey, lz - ez
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        if length > 1e-9:
            dx, dy, dz = dx / length, dy / length, dz / length
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, dz))))
        yaw = math.degrees(math.atan2(dy, dx))
        self._viewer.set_camera(pos=wp.vec3(ex, ey, ez), pitch=pitch, yaw=yaw)

    def update_camera(
        self,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> None:
        """Update the recorder camera to match ``position`` / ``target``.

        Safe to call before the first :meth:`render_rgb_array` (the viewer is
        created lazily; the values will be applied immediately after creation).
        When the viewer is already live the camera is repositioned in-place so
        the next frame reflects the new viewpoint.

        Args:
            position: Camera eye position ``(x, y, z)``.
            target: Camera look-at target ``(x, y, z)``.
        """
        self._ensure_viewer()
        self._apply_camera(position, target)

    def set_visible_worlds(self, world_indices: Sequence[int] | None) -> None:
        """Select the Newton simulation worlds included in recorded frames.

        Repeated selections are ignored because Newton rebuilds its GL shape caches when
        visibility changes. The selection may be set before the lazy viewer is initialized.

        Args:
            world_indices: World indices to render, or None to render all worlds.
        """
        visible_worlds = None if world_indices is None else tuple(int(index) for index in world_indices)
        if visible_worlds == self._visible_worlds:
            return

        self._visible_worlds = visible_worlds
        set_visible_worlds = getattr(self._viewer, "set_visible_worlds", None)
        if callable(set_visible_worlds):
            set_visible_worlds(None if visible_worlds is None else list(visible_worlds))

    def set_world_offsets(self, spacing: Sequence[float]) -> None:
        """Set visual spacing between recorded Newton worlds.

        The spacing may be set before the lazy viewer is initialized.

        Args:
            spacing: Visual spacing along the x, y, and z axes [m]. Non-zero axes
                arrange visible worlds in a compact grid.

        Raises:
            ValueError: If spacing does not contain exactly three values.
        """
        if len(spacing) != 3:
            raise ValueError(f"Expected world spacing to contain three values, received {len(spacing)}.")
        world_spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        if world_spacing == self._world_spacing:
            return

        self._world_spacing = world_spacing
        set_world_offsets = getattr(self._viewer, "set_world_offsets", None)
        if callable(set_world_offsets):
            set_world_offsets(world_spacing)

    def set_frame_overlay_callback(self, callback: Callable[[ViewerGL], None] | None) -> None:
        """Set a callback that renders viewer-side overlays into each recorded frame.

        Args:
            callback: Function invoked with the capture viewer after Newton state logging and
                before the frame ends, or ``None`` to render no overlays.
        """
        self._frame_overlay_callback = callback

    def render_rgb_array(self) -> np.ndarray:
        """Return one RGB frame from the Newton GL viewer. Raises on failure."""

        self._ensure_viewer()
        from isaaclab.sim import SimulationContext

        from isaaclab_newton.physics import NewtonManager

        sim = SimulationContext.instance()
        state = NewtonManager.get_state()
        dt = sim.get_physics_dt()

        viewer = self._viewer
        viewer.begin_frame(dt)
        try:
            viewer.log_state(state)
            if self._frame_overlay_callback is not None:
                self._frame_overlay_callback(viewer)
        finally:
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
