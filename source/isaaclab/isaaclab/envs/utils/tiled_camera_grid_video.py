# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tiled-camera grid video: square RGB tile grid from :class:`~isaaclab.sensors.camera.Camera` sensors."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene


def _tiled_camera_renderer_type(sensor) -> str:
    cfg = getattr(sensor, "cfg", None)
    rc = getattr(cfg, "renderer_cfg", None) if cfg is not None else None
    return getattr(rc, "renderer_type", "default") if rc is not None else "default"


def _tiled_camera_has_rgb_cfg(sensor) -> bool:
    cfg = getattr(sensor, "cfg", None)
    if cfg is None:
        return False
    dt = getattr(cfg, "data_types", None) or []
    return "rgb" in dt or "rgba" in dt


class TiledCameraGridVideoCapture:
    """Capture a square grid of per-environment RGB frames from a :class:`~isaaclab.sensors.camera.Camera`.

    Cameras are filtered by ``preferred_renderer_types`` when set; falls back to an auto-spawned camera if none match.
    """

    def __init__(
        self,
        scene: InteractiveScene,
        *,
        video_num_tiles: int,
        fallback_camera_cfg: object | None,
        preferred_renderer_types: tuple[str, ...] | None = None,
    ):
        self._scene = scene
        self._video_num_tiles = video_num_tiles
        self._preferred_renderer_types = preferred_renderer_types
        self._fallback_tiled_camera = None
        if fallback_camera_cfg is not None:
            self._fallback_tiled_camera = self._spawn_fallback_cameras(fallback_camera_cfg, scene)

    @staticmethod
    def _spawn_fallback_cameras(camera_cfg: object, scene: InteractiveScene):
        """Spawn one video camera prim per environment and return a single Camera."""

        import torch

        from isaaclab.sensors.camera import Camera
        from isaaclab.utils.math import convert_camera_frame_orientation_convention

        rot = torch.tensor(camera_cfg.offset.rot, dtype=torch.float32, device="cpu").unsqueeze(0)
        rot_offset = (
            convert_camera_frame_orientation_convention(rot, origin=camera_cfg.offset.convention, target="opengl")
            .squeeze(0)
            .cpu()
            .numpy()
        )

        spawn_cfg = camera_cfg.spawn
        if spawn_cfg.vertical_aperture is None:
            spawn_cfg = spawn_cfg.replace(
                vertical_aperture=spawn_cfg.horizontal_aperture * camera_cfg.height / camera_cfg.width
            )
        for i in range(scene.num_envs):
            spawn_cfg.func(
                f"/World/envs/env_{i}/VideoCamera",
                spawn_cfg,
                translation=camera_cfg.offset.pos,
                orientation=rot_offset,
            )
        return Camera(camera_cfg.replace(prim_path="/World/envs/env_.*/VideoCamera", spawn=None))

    def _find_video_camera(self):
        if hasattr(self, "_video_camera"):
            return self._video_camera
        from isaaclab.sensors.camera import Camera

        pref = self._preferred_renderer_types

        def _has_rgb(s):
            return "rgb" in s.data.output or "rgba" in s.data.output

        if pref is None:
            camera = next((s for s in self._scene.sensors.values() if isinstance(s, Camera) and _has_rgb(s)), None)
            fb = self._fallback_tiled_camera
            if camera is None and fb is not None and fb.is_initialized and _has_rgb(fb):
                camera = fb
            if camera is None:
                return None
        else:
            candidates = [
                s for s in self._scene.sensors.values() if isinstance(s, Camera) and _tiled_camera_has_rgb_cfg(s)
            ]
            if self._fallback_tiled_camera is not None and _tiled_camera_has_rgb_cfg(self._fallback_tiled_camera):
                candidates.append(self._fallback_tiled_camera)
            candidates = [c for c in candidates if _tiled_camera_renderer_type(c) in frozenset(pref)]
            camera = next(
                (s for s in candidates if s.is_initialized and _has_rgb(s)),
                candidates[0] if candidates else None,
            )
            if camera is None:
                raise RuntimeError(
                    f"No Camera with RGB matching renderers {sorted(pref)}. "
                    "Add a matching Camera or enable fallback_camera_cfg."
                )
            if camera is self._fallback_tiled_camera:
                out = camera.data.output if camera.is_initialized else {}
                if "rgb" not in out and "rgba" not in out:
                    self._fallback_tiled_camera.update(dt=0.0, force_recompute=True)

        self._video_camera = camera
        output = camera.data.output
        if "rgb" in output:
            self._video_rgb_key = "rgb"
        elif "rgba" in output:
            self._video_rgb_key = "rgba"
        else:
            raise RuntimeError(
                f"Camera {camera} has no 'rgb' or 'rgba' in data.output after initialization. "
                "Ensure data_types includes 'rgb' or 'rgba'."
            )
        n_total = int(output[self._video_rgb_key].shape[0])
        n_envs = n_total if self._video_num_tiles < 0 else min(self._video_num_tiles, n_total)
        self._video_n_envs = n_envs
        self._video_grid_size = math.ceil(math.sqrt(n_envs))
        self._video_H, self._video_W = (
            int(output[self._video_rgb_key].shape[1]),
            int(output[self._video_rgb_key].shape[2]),
        )
        pad = self._video_grid_size**2 - n_envs
        self._video_pad = np.zeros((pad, self._video_H, self._video_W, 3), dtype=np.uint8) if pad > 0 else None
        return self._video_camera

    def render_rgb_array(self) -> np.ndarray:
        video_camera = self._find_video_camera()
        if video_camera is None:
            raise RuntimeError(
                "Cannot record video in tiled mode: no Camera sensor with RGB output found. "
                "Add a Camera sensor or use --video=perspective."
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
