# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recorder implementation.

Captures a single wide-angle perspective view of the scene:

* **Newton backends** — uses the Newton GL viewer (``newton.viewer.ViewerGL``).
* **Kit backends** — captures the ``/OmniverseKit_Persp`` viewport via ``omni.replicator.core``.

See :mod:`video_recorder_cfg` for configuration.
"""

from __future__ import annotations

import logging
import math
import traceback
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from .video_recorder_cfg import VideoRecorderCfg

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Records perspective video frames from the scene's active renderer.

    Args:
        cfg: Recorder configuration.
        scene: The interactive scene that owns the sensors.
    """

    def __init__(self, cfg: VideoRecorderCfg, scene: InteractiveScene):
        self.cfg = cfg
        self._scene = scene
        self._gl_viewer = None
        self._gl_viewer_init_attempted = False

        if cfg.render_mode == "rgb_array":
            # enable EGL headless rendering for pyglet before any pyglet.window import.
            try:
                import pyglet

                if not pyglet.options.get("headless", False):
                    pyglet.options["headless"] = True
            except ImportError:
                pass

    def render_rgb_array(self) -> np.ndarray | None:
        """Return an RGB frame, or ``None`` when neither GL viewer nor Kit runtime is available."""
        if not self._gl_viewer_init_attempted:
            self._try_init_gl_viewer()
        if self._gl_viewer is not None:
            return self._render_newton_gl_rgb_array()
        return self._render_kit_perspective_rgb_array()

    def _try_init_gl_viewer(self) -> None:
        """Lazy-initialise the Newton GL viewer on the first render call.

        Called after ``sim.reset()`` so the Newton model is fully built.
        Leaves ``_gl_viewer`` as ``None`` on Kit backends; ``render_rgb_array`` then
        calls ``_render_kit_perspective_rgb_array`` instead.
        """
        self._gl_viewer_init_attempted = True
        try:
            from isaaclab.sim import SimulationContext

            sdp = SimulationContext.instance().initialize_scene_data_provider()
            model = sdp.get_newton_model()
            if model is None:
                return

            import pyglet

            pyglet.options["headless"] = True
            from newton.viewer import ViewerGL

            viewer = ViewerGL(width=self.cfg.gl_viewer_width, height=self.cfg.gl_viewer_height, headless=True)
            viewer.set_model(model)
            viewer.set_world_offsets((0.0, 0.0, 0.0))  # world positions already in body_q
            viewer.up_axis = 2  # Z-up
            self._gl_viewer = viewer

            # place camera to match Kit /OmniverseKit_Persp (same eye/lookat as ViewerCfg).
            try:
                import warp as wp

                ex, ey, ez = self.cfg.camera_eye
                lx, ly, lz = self.cfg.camera_lookat
                dx, dy, dz = lx - ex, ly - ey, lz - ez
                length = math.sqrt(dx**2 + dy**2 + dz**2)
                dx, dy, dz = dx / length, dy / length, dz / length
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, dz))))
                yaw = math.degrees(math.atan2(dy, dx))

                # Kit uses horizontal FOV (60°); pyglet/Newton GL uses vertical FOV.
                aspect = self.cfg.gl_viewer_width / self.cfg.gl_viewer_height
                v_fov_deg = math.degrees(2.0 * math.atan(math.tan(math.radians(60.0) / 2.0) / aspect))
                viewer.camera.fov = v_fov_deg  # ≈ 36° for 1280×720
                viewer.set_camera(pos=wp.vec3(ex, ey, ez), pitch=pitch, yaw=yaw)
            except Exception as exc:
                logger.warning("[VideoRecorder] GL viewer camera setup failed: %s", exc)

            logger.info(
                "[VideoRecorder] Newton GL viewer ready (%dx%d).",
                self.cfg.gl_viewer_width,
                self.cfg.gl_viewer_height,
            )
        except Exception as exc:
            logger.warning("[VideoRecorder] Newton GL viewer unavailable: %s", exc)

    def _render_newton_gl_rgb_array(self) -> np.ndarray:
        """Return one RGB frame from the Newton GL viewer, or a blank frame on error."""
        try:
            from isaaclab.sim import SimulationContext

            sim = SimulationContext.instance()
            sdp = sim.initialize_scene_data_provider()
            state = sdp.get_newton_state()
            dt = sim.get_physics_dt()

            viewer = self._gl_viewer
            viewer.begin_frame(dt)
            viewer.log_state(state)
            viewer.end_frame()
            return viewer.get_frame().numpy()
        except Exception as exc:
            logger.warning("[VideoRecorder] GL frame capture failed: %s", exc)
            return np.zeros((self.cfg.gl_viewer_height, self.cfg.gl_viewer_width, 3), dtype=np.uint8)

    def _render_kit_perspective_rgb_array(self) -> np.ndarray:
        """Return one RGB frame from the Kit /OmniverseKit_Persp camera via omni.replicator.

        On the first call the viewport camera is positioned to match ``cfg.camera_eye`` /
        ``cfg.camera_lookat`` (the same values used by the Newton GL viewer), so both
        backends produce a consistent framing.

        Returns a blank frame during warmup or on any error.
        """
        try:
            import omni.kit.app
            import omni.replicator.core as rep

            # Drive the Kit app loop to produce a fresh RTX viewport frame.
            omni.kit.app.get_app().update()

            if not hasattr(self, "_rgb_annotator"):
                try:
                    import isaacsim.core.utils.viewports as isaacsim_viewports

                    # set the camera view to the Kit /OmniverseKit_Persp camera.
                    # commit da2983e switched active viewport views
                    isaacsim_viewports.set_camera_view(
                        eye=list(self.cfg.camera_eye),
                        target=list(self.cfg.camera_lookat),
                        camera_prim_path="/OmniverseKit_Persp",
                    )
                except Exception as exc:
                    logger.warning("[VideoRecorder] Kit perspective camera positioning failed: %s", exc)

                self._render_product = rep.create.render_product(
                    "/OmniverseKit_Persp", (1280, 720)
                )
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])

            rgb_data = self._rgb_annotator.get_data()
            if isinstance(rgb_data, dict):
                rgb_data = rgb_data.get("data", np.array([], dtype=np.uint8))
            rgb_data = np.asarray(rgb_data, dtype=np.uint8)
            if rgb_data.size == 0:
                # renderer is warming up; return blank frame
                return np.zeros((720, 1280, 3), dtype=np.uint8)
            if rgb_data.ndim == 1:
                rgb_data = rgb_data.reshape(720, 1280, -1)
            return rgb_data[:, :, :3]
        except Exception as exc:
            logger.warning("[VideoRecorder] Kit perspective capture failed: %s\n%s", exc, traceback.format_exc())
            return np.zeros((720, 1280, 3), dtype=np.uint8)
