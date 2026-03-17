# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tiled-camera grid video: RGB tiles from :class:`~isaaclab.sensors.camera.TiledCamera`.

Used by :mod:`isaaclab_physx.video_recording.isaacsim_tiled_camera_video` and
:mod:`isaaclab_newton.video_recording.newton_tiled_camera_video` factories so Kit and Newton backends
each expose a dedicated entry point (mirroring perspective recording).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene


class TiledCameraGridVideoCapture:
    """Capture a square grid of per-environment RGB frames from a TiledCamera sensor.

    Priority: (1) first scene :class:`~isaaclab.sensors.camera.TiledCamera` with rgb/rgba
    output; (2) optional fallback camera spawned at construction time.
    """

    def __init__(
        self,
        scene: InteractiveScene,
        *,
        video_num_tiles: int,
        fallback_camera_cfg: object | None,
    ):
        self._scene = scene
        self._video_num_tiles = video_num_tiles
        self._fallback_tiled_camera = None
        if fallback_camera_cfg is not None:
            self._fallback_tiled_camera = self._spawn_fallback_cameras(fallback_camera_cfg, scene)

    @staticmethod
    def _spawn_fallback_cameras(camera_cfg: object, scene: InteractiveScene):
        """Spawn one video camera prim per environment and return a single TiledCamera."""
        import torch

        from isaaclab.sensors.camera import TiledCamera
        from isaaclab.utils.math import convert_camera_frame_orientation_convention

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
            spawn_cfg.func(
                f"/World/envs/env_{i}/VideoCamera",
                spawn_cfg,
                translation=camera_cfg.offset.pos,
                orientation=rot_offset,
            )

        tiled_cfg = camera_cfg.replace(prim_path="/World/envs/env_.*/VideoCamera", spawn=None)
        return TiledCamera(tiled_cfg)

    def _find_video_camera(self):
        if hasattr(self, "_video_camera"):
            return self._video_camera

        from isaaclab.sensors.camera import TiledCamera

        camera = None
        for sensor in self._scene.sensors.values():
            if isinstance(sensor, TiledCamera):
                output = sensor.data.output
                if "rgb" in output or "rgba" in output:
                    camera = sensor
                    break

        if camera is None and self._fallback_tiled_camera is not None:
            if self._fallback_tiled_camera.is_initialized:
                output = self._fallback_tiled_camera.data.output
                if "rgb" in output or "rgba" in output:
                    camera = self._fallback_tiled_camera

        if camera is None:
            return None

        self._video_camera = camera
        output = camera.data.output
        self._video_rgb_key = "rgb" if "rgb" in output else "rgba"
        n_total = int(output[self._video_rgb_key].shape[0])
        n_envs = n_total if self._video_num_tiles < 0 else min(self._video_num_tiles, n_total)
        self._video_n_envs = n_envs
        self._video_grid_size = math.ceil(math.sqrt(n_envs))
        n_slots = self._video_grid_size**2
        h = int(output[self._video_rgb_key].shape[1])
        w = int(output[self._video_rgb_key].shape[2])
        self._video_H = h
        self._video_W = w
        pad = n_slots - n_envs
        self._video_pad = np.zeros((pad, h, w, 3), dtype=np.uint8) if pad > 0 else None
        return self._video_camera

    def render_rgb_array(self) -> np.ndarray:
        video_camera = self._find_video_camera()
        if video_camera is None:
            raise RuntimeError(
                "Cannot record video in tiled mode: no TiledCamera sensor with RGB output was found "
                "in the scene. Add a TiledCamera sensor or switch to perspective mode (--video=perspective)."
            )
        if video_camera is self._fallback_tiled_camera:
            self._fallback_tiled_camera.update(dt=0.0, force_recompute=True)

        rgb_all = self._video_camera.data.output[self._video_rgb_key]
        if self._video_rgb_key == "rgba":
            rgb_all = rgb_all[..., :3]

        tiles = rgb_all[: self._video_n_envs].contiguous().cpu().numpy()
        if self._video_pad is not None:
            tiles = np.concatenate([tiles, self._video_pad], axis=0)

        g, h, w = self._video_grid_size, self._video_H, self._video_W
        return tiles.reshape(g, g, h, w, 3).transpose(0, 2, 1, 3, 4).reshape(g * h, g * w, 3)
