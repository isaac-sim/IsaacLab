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

    Args:
        cfg: Configuration for this recorder.
        scene: The interactive scene that owns the sensors.
    """

    def __init__(self, cfg: VideoRecorderCfg, scene: InteractiveScene):
        self.cfg = cfg
        self._scene = scene

    def render_rgb_array(self) -> np.ndarray | None:
        """Return a square tile-grid RGB frame, or ``None`` if no suitable camera exists.

        Returns:
            RGB image of shape ``(G*H, G*W, 3)`` and dtype ``uint8``, where
            ``G = ceil(sqrt(video_num_tiles))`` and ``(H, W)`` is the per-tile resolution,
            or ``None`` when no :class:`~isaaclab.sensors.camera.TiledCamera` with RGB output
            is present in the scene.
        """
        if self._find_video_camera() is None:
            return None
        return self._render_tiled_camera_rgb_array()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_video_camera(self):
        """
            Locates and caches the first TiledCamera sensor with RGB output.
            Previously used the omni.replicator viewer camera which had RGB output.
            Returns ``None`` if absent.
        """
        if not hasattr(self, "_video_camera"):
            from isaaclab.sensors.camera import TiledCamera

            self._video_camera = None
            for sensor in self._scene.sensors.values():
                if isinstance(sensor, TiledCamera):
                    output = sensor.data.output
                    if "rgb" in output or "rgba" in output:
                        self._video_camera = sensor
                        self._video_rgb_key = "rgb" if "rgb" in output else "rgba"
                        # Cache all grid constants — these are fixed for the lifetime of the env.
                        n_total = int(sensor.data.output[self._video_rgb_key].shape[0])
                        n_envs = n_total if self.cfg.video_num_tiles < 0 else min(self.cfg.video_num_tiles, n_total)
                        self._video_n_envs = n_envs
                        self._video_grid_size = math.ceil(math.sqrt(n_envs))
                        n_slots = self._video_grid_size * self._video_grid_size
                        H = int(sensor.data.output[self._video_rgb_key].shape[1])
                        W = int(sensor.data.output[self._video_rgb_key].shape[2])
                        self._video_H = H
                        self._video_W = W
                        # Pre-allocate the black padding block (zero-copy when pad == 0).
                        pad = n_slots - n_envs
                        self._video_pad = np.zeros((pad, H, W, 3), dtype=np.uint8) if pad > 0 else None
                        break
        return self._video_camera

    def _render_tiled_camera_rgb_array(self) -> np.ndarray:
        """Return a square tile-grid of RGB frames from the scene's TiledCamera.

        Create a square grid of tiles. This method reads directly from the
        TiledCamera sensor buffer to generate the tiles.

        Returns:
            RGB image of shape ``(G*H, G*W, 3)`` and dtype ``uint8``, where
            ``G = ceil(sqrt(num_envs))`` and ``(H, W)`` is the per-tile resolution.
        """
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
