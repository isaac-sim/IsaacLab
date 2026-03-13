# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recorder for capturing video frames from either a Newton OpenGL perspective
viewer or a :class:`~isaaclab.sensors.camera.TiledCamera` sensor."""

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

    The recording strategy is determined by :attr:`~VideoRecorderCfg.video_mode`:

    **``video_mode = "perspective"`` (default)**

    The TiledCamera is **bypassed** entirely, even when one is in the scene.

    * **Newton backends** - headless :class:`newton.viewer.ViewerGL` renders an isometric
      wide-angle view of all environments (limited to ``video_num_tiles`` when set).
    * **Kit backends** - returns ``None`` so that the environment's ``render()`` method
      falls through to the ``omni.replicator.core`` Kit viewport camera path
      (``/OmniverseKit_Persp``).

    **``video_mode = "tiled"``**

    Frame sources are tried in priority order on every :meth:`render_rgb_array` call:

    1. **Observation** :class:`~isaaclab.sensors.camera.TiledCamera` already present in
       the scene; vision-based env path. Reuses the agent's own camera sensor at zero
       extra cost and produces a square tile-grid of per-agent views.

    2. **Newton OpenGL perspective viewer** - Newton backends with no observation
       ``TiledCamera``. A headless :class:`newton.viewer.ViewerGL` is lazy-initialised
       on the first call and renders an isometric perspective of all environments
       (limited to ``video_num_tiles`` when that field is set).

    3. **Fallback** :class:`~isaaclab.sensors.camera.TiledCamera` - state-based env path
       with Kit-based backends.  A camera prim is spawned per environment before
       ``sim.reset()``.

    For fallback cameras to initialise correctly they **must** be created before
    ``sim.reset()`` is called; the environment base classes handle this.

    Args:
        cfg: Configuration for this recorder.
        scene: The interactive scene that owns the sensors.
    """

    def __init__(self, cfg: VideoRecorderCfg, scene: InteractiveScene):
        self.cfg = cfg
        self._scene = scene
        self._fallback_tiled_camera = None

        # Newton GL perspective viewer; lazy-initialised on first render call.
        self._gl_viewer = None
        self._gl_viewer_initialized = False  # True once _try_init_gl_viewer() has run

        if cfg.render_mode == "rgb_array":
            # Enable EGL-backed headless rendering for pyglet before ViewerGL is ever
            # imported.  Must be set before the first 'import pyglet.window'.  This is a
            # no-op when pyglet is not installed (GL viewer simply stays None).
            try:
                import pyglet

                if not pyglet.options.get("headless", False):
                    pyglet.options["headless"] = True
            except ImportError:
                pass

            # Skip spawning fallback TiledCameras when:
            #   (a) a Newton backend is active; the GL perspective viewer handles state-based
            #       rendering so creating per-env camera prims would waste GPU resources, or
            #   (b) perspective mode is requested; TiledCamera is not used in that path.
            _newton_backend = self._is_newton_backend()
            if cfg.fallback_camera_cfg is not None and not _newton_backend and cfg.video_mode == "tiled":
                self._fallback_tiled_camera = self._spawn_fallback_cameras(cfg, scene)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_rgb_array(self) -> np.ndarray | None:
        """Return an RGB frame for video recording, or ``None`` when unavailable.

        The frame source depends on :attr:`~VideoRecorderCfg.video_mode`:

        **``"tiled"`` mode** (default):

        * Source 1 - observation :class:`~isaaclab.sensors.camera.TiledCamera`:
          returns a square tile-grid ``(G*H, G*W, 3)`` uint8 array,
          where ``G = ceil(sqrt(video_num_tiles))``.
        * Source 2 - Newton GL perspective viewer (state-based + Newton backend):
          returns ``(gl_viewer_height, gl_viewer_width, 3)`` uint8.
        * Source 3 - fallback :class:`~isaaclab.sensors.camera.TiledCamera`
          (state-based + Kit backend): same tile-grid shape as source 1.

        **``"perspective"`` mode**:

        * Newton backends: Newton GL perspective viewer (same shape as source 2).
        * Kit backends: returns ``None`` so the environment's ``render()`` method
          falls through to the ``omni.replicator.core`` viewport camera path.
        """
        if self.cfg.video_mode == "perspective":
            # Perspective mode: bypass TiledCamera entirely.
            # Newton backends → GL viewer; Kit backends → return None (env render() continues).
            if not self._gl_viewer_initialized:
                self._try_init_gl_viewer()
            if self._gl_viewer is not None:
                return self._render_newton_gl_rgb_array()
            # No GL viewer (Kit backend) → signal the env to use its Kit perspective path.
            return None

        # --- Tiled mode (default) - priority chain. ---------------------------------

        # Source 1: observation TiledCamera (vision-based path).
        # _find_video_camera() sets self._video_camera and caches grid constants.
        video_camera = self._find_video_camera()
        has_obs_camera = video_camera is not None and video_camera is not self._fallback_tiled_camera
        if has_obs_camera:
            return self._render_tiled_camera_rgb_array()

        # Source 2: Newton GL perspective viewer (state-based + Newton backend).
        if not self._gl_viewer_initialized:
            self._try_init_gl_viewer()
        if self._gl_viewer is not None:
            return self._render_newton_gl_rgb_array()

        # Source 3: fallback TiledCamera (state-based + Kit backend).
        if video_camera is None:
            return None
        return self._render_tiled_camera_rgb_array()

    # ------------------------------------------------------------------
    # Internal helpers - Newton GL viewer
    # ------------------------------------------------------------------

    @staticmethod
    def _is_newton_backend() -> bool:
        """Return ``True`` when the active scene data provider is Newton-based.

        Detected by duck-typing: Newton providers expose ``get_newton_model()``,
        while PhysX providers do not.  Safe to call before ``sim.reset()`` since
        the provider is registered during scene setup.
        """
        try:
            from isaaclab.sim import SimulationContext

            sdp = SimulationContext.instance().initialize_scene_data_provider()
            return hasattr(sdp, "get_newton_model")
        except Exception:
            return False

    def _try_init_gl_viewer(self) -> None:
        """Lazy-initialise the Newton OpenGL perspective viewer.

        Called once on the first :meth:`render_rgb_array` invocation, at which point
        ``sim.reset()`` has already been called so the Newton model is fully built.
        On failure the viewer stays ``None`` and the caller falls through to the next
        source: source 3 (fallback TiledCamera) in tiled mode, or ``None`` (Kit
        viewport path) in perspective mode.
        """
        self._gl_viewer_initialized = True
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

            viewer = ViewerGL(
                width=self.cfg.gl_viewer_width,
                height=self.cfg.gl_viewer_height,
                headless=True,
            )
            # set_model() auto-computes per-world visual offsets from body positions.
            viewer.set_model(model, max_worlds=max_worlds)
            # Zero additional spacing - world positions are already in model body_q.
            viewer.set_world_offsets((0.0, 0.0, 0.0))
            viewer.up_axis = 2  # Z-up

            self._gl_viewer = viewer

            # Position the camera to match the Kit /OmniverseKit_Persp viewport.
            # Convert cfg.camera_eye / cfg.camera_lookat (same defaults as ViewerCfg)
            # into Newton GL pitch/yaw (Z-up convention, degrees).
            try:
                import warp as wp

                ex, ey, ez = self.cfg.camera_eye
                lx, ly, lz = self.cfg.camera_lookat
                dx, dy, dz = lx - ex, ly - ey, lz - ez
                length = math.sqrt(dx**2 + dy**2 + dz**2)
                dx, dy, dz = dx / length, dy / length, dz / length
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, dz))))
                yaw = math.degrees(math.atan2(dy, dx))

                # Kit's /OmniverseKit_Persp uses a *horizontal* FOV of 60° (derived
                # from its default focal_length=18.15 mm / horizontal_aperture=20.955 mm).
                # pyglet / Newton GL use *vertical* FOV.  Convert so both cameras see
                # the same scene extent.
                aspect = self.cfg.gl_viewer_width / self.cfg.gl_viewer_height
                kit_h_fov_rad = math.radians(60.0)
                v_fov_deg = math.degrees(2.0 * math.atan(math.tan(kit_h_fov_rad / 2.0) / aspect))
                viewer.camera.fov = v_fov_deg  # ≈ 36° for 1280×720
                viewer.set_camera(pos=wp.vec3(ex, ey, ez), pitch=pitch, yaw=yaw)
            except Exception as frame_exc:
                logger.warning("[VideoRecorder] GL viewer camera setup failed: %s", frame_exc)

            logger.info(
                "[VideoRecorder] Newton GL perspective viewer ready (%dx%d, max_worlds=%s).",
                self.cfg.gl_viewer_width,
                self.cfg.gl_viewer_height,
                max_worlds,
            )
        except Exception as exc:
            logger.warning("[VideoRecorder] Newton GL viewer unavailable: %s", exc)

    def _render_newton_gl_rgb_array(self) -> np.ndarray | None:
        """Render one perspective frame from the Newton OpenGL viewer.

        Returns:
            RGB array of shape ``(gl_viewer_height, gl_viewer_width, 3)`` and
            dtype ``uint8``, or ``None`` on error.
        """
        try:
            from isaaclab.sim import SimulationContext

            sim = SimulationContext.instance()
            sdp = sim.initialize_scene_data_provider()
            state = sdp.get_newton_state()

            # Use the actual physics timestep so that the viewer does not treat
            # dt=0 as a no-op and skip drawing geometry on frames after the first.
            dt = sim.get_physics_dt()

            viewer = self._gl_viewer
            viewer.begin_frame(dt)
            viewer.log_state(state)
            viewer.end_frame()  # renders scene geometry to the off-screen FBO
            frame = viewer.get_frame()  # wp.array (H, W, 3) uint8 - GPU readback via PBO
            return frame.numpy()
        except Exception as exc:
            logger.warning("[VideoRecorder] GL frame capture failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers - TiledCamera (sources 1 and 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _spawn_fallback_cameras(cfg: VideoRecorderCfg, scene: InteractiveScene):
        """Spawn one video camera prim per environment (up to ``cfg.video_num_tiles``) and
        return a single :class:`~isaaclab.sensors.camera.TiledCamera` covering all of them.

        Camera prims are spawned at ``/World/envs/env_{i}/VideoCamera`` for
        ``i in range(n_cameras)``, then a ``TiledCamera`` with the regex prim path
        ``/World/envs/env_.*/VideoCamera`` is created so that all spawned prims are
        discovered and rendered as tiles.

        This must be called **before** ``sim.reset()`` so the prims exist in the USD stage
        and the ``TiledCamera`` can register for the ``PHYSICS_READY`` callback.
        """
        import torch

        from isaaclab.sensors.camera import TiledCamera
        from isaaclab.utils.math import convert_camera_frame_orientation_convention

        camera_cfg = cfg.fallback_camera_cfg

        # Pre-compute the OpenGL rotation offset (mirrors Camera.__init__ logic).
        n_total_envs = scene.num_envs
        rot = torch.tensor(camera_cfg.offset.rot, dtype=torch.float32, device="cpu").unsqueeze(0)
        rot_offset = convert_camera_frame_orientation_convention(
            rot, origin=camera_cfg.offset.convention, target="opengl"
        )
        rot_offset = rot_offset.squeeze(0).cpu().numpy()

        # Ensure vertical_aperture is set before calling the spawn func.
        spawn_cfg = camera_cfg.spawn
        if spawn_cfg.vertical_aperture is None:
            spawn_cfg = spawn_cfg.replace(
                vertical_aperture=spawn_cfg.horizontal_aperture * camera_cfg.height / camera_cfg.width
            )

        # TiledCamera requires exactly one camera prim per environment (count == num_envs).
        # We must therefore spawn cameras for ALL environments, not just video_num_tiles of them.
        # The video_num_tiles limit is applied at render time in _render_tiled_camera_rgb_array,
        # which only reads the first N tiles - the same behaviour as vision-based observation cameras.
        for i in range(n_total_envs):
            prim_path_i = f"/World/envs/env_{i}/VideoCamera"
            spawn_cfg.func(prim_path_i, spawn_cfg, translation=camera_cfg.offset.pos, orientation=rot_offset)

        # Create one TiledCamera that discovers all spawned prims via the regex path.
        # spawn=None tells Camera.__init__ to skip re-spawning; it will verify the prims exist.
        tiled_cfg = camera_cfg.replace(
            prim_path="/World/envs/env_.*/VideoCamera",
            spawn=None,
        )
        return TiledCamera(tiled_cfg)

    def _find_video_camera(self):
        """Locate and cache the TiledCamera to use for video recording.

        Search order:
          1. Observation TiledCamera already in the scene (vision-based env path, zero extra cost).
          2. Dedicated fallback TiledCamera from ``cfg.fallback_camera_cfg`` (state-based env path).

        Returns ``None`` if neither source is available.

        Previously used the omni.replicator viewer camera which had RGB output only for
        Kit-based backends (``physx`` / ``newton,isaacsim_rtx_renderer``).
        """
        if not hasattr(self, "_video_camera"):
            from isaaclab.sensors.camera import TiledCamera

            self._video_camera = None

            # Priority 1: observation TiledCamera in the scene (vision-based env path).
            for sensor in self._scene.sensors.values():
                if isinstance(sensor, TiledCamera):
                    output = sensor.data.output
                    if "rgb" in output or "rgba" in output:
                        self._video_camera = sensor
                        break

            # Priority 2: fallback video camera (state-based env path).
            if self._video_camera is None and self._fallback_tiled_camera is not None:
                if self._fallback_tiled_camera.is_initialized:
                    output = self._fallback_tiled_camera.data.output
                    if "rgb" in output or "rgba" in output:
                        self._video_camera = self._fallback_tiled_camera

            # Cache all grid constants - these are fixed for the lifetime of the env.
            if self._video_camera is not None:
                output = self._video_camera.data.output
                self._video_rgb_key = "rgb" if "rgb" in output else "rgba"
                n_total = int(output[self._video_rgb_key].shape[0])
                n_envs = n_total if self.cfg.video_num_tiles < 0 else min(self.cfg.video_num_tiles, n_total)
                self._video_n_envs = n_envs
                self._video_grid_size = math.ceil(math.sqrt(n_envs))
                n_slots = self._video_grid_size * self._video_grid_size
                H = int(output[self._video_rgb_key].shape[1])
                W = int(output[self._video_rgb_key].shape[2])
                self._video_H = H
                self._video_W = W
                # Pre-allocate the black padding block (zero-copy when pad == 0).
                pad = n_slots - n_envs
                self._video_pad = np.zeros((pad, H, W, 3), dtype=np.uint8) if pad > 0 else None

        return self._video_camera

    def _render_tiled_camera_rgb_array(self) -> np.ndarray:
        """Return a square tile-grid of RGB frames from the TiledCamera.

        Create a square grid of tiles. This method reads directly from the
        TiledCamera sensor buffer to generate the tiles.

        If using the dedicated fallback video cameras (not observation sensors),
        this method calls ``update()`` on them first to trigger a fresh render pass.
        Observation TiledCameras are updated by ``scene.update()`` during the
        environment step and do not need an extra update here.

        Returns:
            RGB image of shape ``(G*H, G*W, 3)`` and dtype ``uint8``, where
            ``G = ceil(sqrt(num_envs))`` and ``(H, W)`` is the per-tile resolution.
        """
        # Fallback cameras are not updated by scene.update(), so drive them manually.
        if self._video_camera is self._fallback_tiled_camera:
            self._fallback_tiled_camera.update(dt=0.0, force_recompute=True)

        rgb_all = self._video_camera.data.output[self._video_rgb_key]
        # Drop alpha channel once on GPU before the CPU transfer.
        if self._video_rgb_key == "rgba":
            rgb_all = rgb_all[..., :3]

        # .contiguous() ensures the reshape below returns a zero-copy view.
        tiles = rgb_all[: self._video_n_envs].contiguous().cpu().numpy()  # [n_envs, H, W, 3]
        if self._video_pad is not None:
            tiles = np.concatenate([tiles, self._video_pad], axis=0)
        # [grid_size, grid_size, H, W, 3] → [grid_size*H, grid_size*W, 3]
        g, H, W = self._video_grid_size, self._video_H, self._video_W
        grid = tiles.reshape(g, g, H, W, 3)
        grid = grid.transpose(0, 2, 1, 3, 4)
        # after transpose the strides are non-standard; reshape must copy here.
        return grid.reshape(g * H, g * W, 3)
