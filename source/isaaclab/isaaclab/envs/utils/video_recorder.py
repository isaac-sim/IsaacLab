# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viewport recorder for capturing video frames from a :class:`~isaaclab.sensors.camera.TiledCamera`."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from .video_recorder_cfg import VideoRecorderCfg


class VideoRecorder:
    """Records video frames from the scene's :class:`~isaaclab.sensors.camera.TiledCamera`.

    On the first :meth:`render_rgb_array` call this class searches the scene for the first
    ``TiledCamera`` sensor with ``"rgb"`` or ``"rgba"`` output and caches the camera reference
    together with all grid-layout constants so subsequent calls are allocation-free (except for
    the unavoidable GPU-to-CPU transfer and the final tile-stitch reshape).

    The default implementation reads *all* ``num_envs`` frames from the TiledCamera buffer on
    the GPU and slices the first ``cfg.video_num_tiles`` on the CPU (Option A).  Swap
    ``cfg.class_type`` for a custom subclass to change this behaviour without touching any
    environment code.

    **Camera selection priority:**

    1. An existing :class:`~isaaclab.sensors.camera.TiledCamera` found in the scene sensors
       (vision-based env path — the observation camera is reused for free).
    2. A dedicated video camera grid instantiated from ``cfg.fallback_camera_cfg``
       (state-based env path — no observation camera exists, so one camera per environment
       is spawned, up to ``cfg.video_num_tiles``).

    For the fallback cameras to be initialised correctly they **must** be created before
    ``sim.reset()`` is called, so :class:`VideoRecorder` must be instantiated before
    ``sim.reset()`` in the environment setup. The environment base classes handle this.

    Args:
        cfg: Configuration for this recorder.
        scene: The interactive scene that owns the sensors.
    """

    def __init__(self, cfg: VideoRecorderCfg, scene: InteractiveScene):
        self.cfg = cfg
        self._scene = scene
        self._fallback_tiled_camera = None

        # Spawn fallback cameras only when video recording is actually requested.
        # cfg.render_mode is set to "rgb_array" by the env base class when --video is active
        # (forwarded from the render_mode argument of gym.make / the env constructor).
        # Gating here avoids GPU overhead in ordinary training runs that don't record video.
        if cfg.fallback_camera_cfg is not None and cfg.render_mode == "rgb_array":
            self._fallback_tiled_camera = self._spawn_fallback_cameras(cfg, scene)

    def render_rgb_array(self) -> np.ndarray | None:
        """Return a square tile-grid RGB frame, or ``None`` if no suitable camera exists.

        Returns:
            RGB image of shape ``(G*H, G*W, 3)`` and dtype ``uint8``, where
            ``G = ceil(sqrt(video_num_tiles))`` and ``(H, W)`` is the per-tile resolution,
            or ``None`` when no :class:`~isaaclab.sensors.camera.TiledCamera` with RGB output
            is present in the scene or configured as a fallback.
        """
        if self._find_video_camera() is None:
            return None
        return self._render_tiled_camera_rgb_array()

    # ------------------------------------------------------------------
    # Internal helpers
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
        # which only reads the first N tiles — the same behaviour as vision-based observation cameras.
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

            # Cache all grid constants — these are fixed for the lifetime of the env.
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
