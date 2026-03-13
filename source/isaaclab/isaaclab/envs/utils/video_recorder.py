# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recorder implementation.

* **Perspective view** (``video_mode="perspective"``) — captures a single wide-angle
  view of the scene using the Newton GL viewer (Newton backends) or the Kit viewport
  camera ``/OmniverseKit_Persp`` via ``omni.replicator.core`` (Kit backends).
* **Camera sensor / tiled** (``video_mode="tiled"``) — reads pixel data from a
  :class:`~isaaclab.sensors.camera.TiledCamera` sensor, producing a grid of per-agent
  views.

See :mod:`video_recorder_cfg` for configuration and full mode descriptions.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from .video_recorder_cfg import VideoRecorderCfg

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Records video frames from the scene's active renderer.

    See :class:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg` for the full
    description of ``video_mode`` and the fallback priority chain.

    Args:
        cfg: Recorder configuration.
        scene: The interactive scene that owns the sensors.
    """

    def __init__(self, cfg: VideoRecorderCfg, scene: InteractiveScene):
        self.cfg = cfg
        self._scene = scene
        self._fallback_tiled_camera = None
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

            # pre-spawn fallback TiledCamera; must exist in USD stage before physics initialises.
            # whether it is actually used is decided lazily in _find_video_camera().
            if cfg.fallback_camera_cfg is not None and cfg.video_mode == "tiled":
                self._fallback_tiled_camera = self._spawn_fallback_cameras(cfg, scene)

    def render_rgb_array(self) -> np.ndarray | None:
        """Return an RGB frame for video recording, or ``None`` on transient Kit warmup."""
        if self.cfg.video_mode == "perspective":
            if not self._gl_viewer_init_attempted:
                self._try_init_gl_viewer()
            if self._gl_viewer is not None:
                return self._render_newton_gl_rgb_array()
            return self._render_kit_perspective_rgb_array()

        # tiled mode: use observation TiledCamera if available, then fallback.
        video_camera = self._find_video_camera()
        if video_camera is None:
            raise RuntimeError(
                "Cannot record video in tiled mode: no TiledCamera sensor with RGB output was found"
                " in the scene. Add a TiledCamera sensor or switch to perspective mode (--video=perspective)."
            )
        if video_camera is not self._fallback_tiled_camera:
            logger.debug("[VideoRecorder] tiled source: observation TiledCamera")
        else:
            logger.debug("[VideoRecorder] tiled source: fallback TiledCamera")
        return self._render_tiled_camera_rgb_array()

    def _try_init_gl_viewer(self) -> None:
        """Lazy-initialise the Newton GL viewer on the first render call.

        Called after ``sim.reset()`` so the Newton model is fully built.
        Leaves ``_gl_viewer`` as ``None`` on failure so callers fall through gracefully.
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

            max_worlds = (
                None if self.cfg.video_num_tiles < 0 else min(self.cfg.video_num_tiles, model.world_count)
            )

            viewer = ViewerGL(width=self.cfg.gl_viewer_width, height=self.cfg.gl_viewer_height, headless=True)
            viewer.set_model(model, max_worlds=max_worlds)
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
                "[VideoRecorder] Newton GL viewer ready (%dx%d, max_worlds=%s).",
                self.cfg.gl_viewer_width,
                self.cfg.gl_viewer_height,
                max_worlds,
            )
        except Exception as exc:
            logger.warning("[VideoRecorder] Newton GL viewer unavailable: %s", exc)

    def _render_newton_gl_rgb_array(self) -> np.ndarray | None:
        """Return one RGB frame from the Newton GL viewer, or ``None`` on error."""
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
            return None

    def _render_kit_perspective_rgb_array(self) -> np.ndarray | None:
        """Return one RGB frame from the Kit /OmniverseKit_Persp camera via omni.replicator.

        Returns ``None`` during the initial warmup frames when the renderer returns empty data.
        """
        try:
            import omni.replicator.core as rep

            from isaaclab.sim import SimulationContext

            # /OmniverseKit_Persp is not an RTX sensor; always force a render pass for fresh data.
            SimulationContext.instance().render()

            if not hasattr(self, "_rgb_annotator"):
                self._render_product = rep.create.render_product(
                    self.cfg.kit_cam_prim_path, self.cfg.kit_resolution
                )
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])

            rgb_data = self._rgb_annotator.get_data()
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            if rgb_data.size == 0:
                # renderer is warming up; return blank frame
                h, w = self.cfg.kit_resolution[1], self.cfg.kit_resolution[0]
                return np.zeros((h, w, 3), dtype=np.uint8)
            return rgb_data[:, :, :3]
        except Exception as exc:
            logger.warning("[VideoRecorder] Kit perspective capture failed: %s", exc)
            return None

    @staticmethod
    def _spawn_fallback_cameras(cfg: VideoRecorderCfg, scene: InteractiveScene):
        """Spawn one video camera prim per environment and return a single TiledCamera.

        Must be called **before** ``sim.reset()`` so the prims exist when the TiledCamera
        registers for its ``PHYSICS_READY`` callback.
        """
        import torch

        from isaaclab.sensors.camera import TiledCamera
        from isaaclab.utils.math import convert_camera_frame_orientation_convention

        camera_cfg = cfg.fallback_camera_cfg
        n_total_envs = scene.num_envs

        rot = torch.tensor(camera_cfg.offset.rot, dtype=torch.float32, device="cpu").unsqueeze(0)
        rot_offset = convert_camera_frame_orientation_convention(
            rot, origin=camera_cfg.offset.convention, target="opengl"
        ).squeeze(0).cpu().numpy()

        spawn_cfg = camera_cfg.spawn
        if spawn_cfg.vertical_aperture is None:
            spawn_cfg = spawn_cfg.replace(
                vertical_aperture=spawn_cfg.horizontal_aperture * camera_cfg.height / camera_cfg.width
            )

        for i in range(n_total_envs):
            spawn_cfg.func(f"/World/envs/env_{i}/VideoCamera", spawn_cfg,
                           translation=camera_cfg.offset.pos, orientation=rot_offset)

        tiled_cfg = camera_cfg.replace(prim_path="/World/envs/env_.*/VideoCamera", spawn=None)
        return TiledCamera(tiled_cfg)

    def _find_video_camera(self):
        """Locate and cache the TiledCamera to use for video recording.

        Priority: (1) observation TiledCamera already in the scene, (2) fallback camera.
        Returns ``None`` if neither is available.
        """
        if not hasattr(self, "_video_camera"):
            from isaaclab.sensors.camera import TiledCamera

            self._video_camera = None

            for sensor in self._scene.sensors.values():
                if isinstance(sensor, TiledCamera):
                    output = sensor.data.output
                    if "rgb" in output or "rgba" in output:
                        self._video_camera = sensor
                        break

            if self._video_camera is None and self._fallback_tiled_camera is not None:
                if self._fallback_tiled_camera.is_initialized:
                    output = self._fallback_tiled_camera.data.output
                    if "rgb" in output or "rgba" in output:
                        self._video_camera = self._fallback_tiled_camera

            if self._video_camera is not None:
                output = self._video_camera.data.output
                self._video_rgb_key = "rgb" if "rgb" in output else "rgba"
                n_total = int(output[self._video_rgb_key].shape[0])
                n_envs = n_total if self.cfg.video_num_tiles < 0 else min(self.cfg.video_num_tiles, n_total)
                self._video_n_envs = n_envs
                self._video_grid_size = math.ceil(math.sqrt(n_envs))
                n_slots = self._video_grid_size ** 2
                H = int(output[self._video_rgb_key].shape[1])
                W = int(output[self._video_rgb_key].shape[2])
                self._video_H = H
                self._video_W = W
                pad = n_slots - n_envs
                self._video_pad = np.zeros((pad, H, W, 3), dtype=np.uint8) if pad > 0 else None

        return self._video_camera

    def _render_tiled_camera_rgb_array(self) -> np.ndarray:
        """Return a square tile-grid ``(G*H, G*W, 3)`` from the cached TiledCamera."""
        if self._video_camera is self._fallback_tiled_camera:
            self._fallback_tiled_camera.update(dt=0.0, force_recompute=True)

        rgb_all = self._video_camera.data.output[self._video_rgb_key]
        if self._video_rgb_key == "rgba":
            rgb_all = rgb_all[..., :3]

        tiles = rgb_all[: self._video_n_envs].contiguous().cpu().numpy()
        if self._video_pad is not None:
            tiles = np.concatenate([tiles, self._video_pad], axis=0)

        g, H, W = self._video_grid_size, self._video_H, self._video_W
        return tiles.reshape(g, g, H, W, 3).transpose(0, 2, 1, 3, 4).reshape(g * H, g * W, 3)
