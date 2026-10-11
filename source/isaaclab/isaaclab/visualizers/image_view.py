# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared device images for presentation and recording."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from matplotlib import colormaps

from ..utils import validate
from ..utils.buffers import TimestampedBuffer
from ..utils.images import compose_image, sensor_key_for_gt_type
from .visualizer_cfg import ImageViewCfg

if TYPE_CHECKING:
    from ..sensors import Camera
    from ..sensors.ray_caster.base_ray_caster_camera import BaseRayCasterCamera


def _image_grid_shape(num_envs: int, num_channels: int, height: int, width: int, aspect: float) -> tuple[int, int]:
    """Size a grid with each environment's channels side by side; inputs must be positive."""
    columns, best_score = 1, math.inf
    for candidate in range(1, num_envs + 1):
        rows = math.ceil(num_envs / candidate)
        empty = rows * candidate - num_envs
        ratio = candidate * num_channels * width / (rows * height)
        # Penalize empty cells and aspect distortion; prefer wider layouts when scores otherwise tie.
        score = empty * 10.0 + abs(math.log(ratio / aspect)) - candidate * 1e-6
        if score < best_score:
            columns, best_score = candidate, score
    return math.ceil(num_envs / columns) * height, columns * num_channels * width


class ImageView:
    """Own one composed GPU image shared by its display and recording consumers.

    The scene retains ownership of sensors and their capture cadence. Perspective producers write
    device pixels through ``render``. Reading twice for the same frame reuses the composed image;
    only :meth:`read_rgb` transfers pixels to the host.
    """

    def __init__(self, cfg: ImageViewCfg, *, camera: Camera | BaseRayCasterCamera | None = None) -> None:
        """Bind a view declaration to a scene sensor or an initially unbound perspective producer."""
        validate(cfg)
        self.cfg = cfg
        if isinstance(cfg.source, str) and camera is None:
            raise ValueError(f"No scene camera named {cfg.source!r} was bound to the image view.")
        # Perspective pixels come from the visualizer callback, not a scene sensor.
        self.camera = camera
        self.render: Callable[[wp.array | None], wp.array] | None = None
        self.aspect = 1.0
        self.frame = TimestampedBuffer()
        self._host_frame: np.ndarray | None = None
        self._env_ids: wp.array | None = None
        self._depth_colors: wp.array | None = None
        self._selection = (tuple(cfg.envs), tuple(cfg.channels), cfg.depth_range, self.aspect)
        self._layout = None
        available = frozenset(camera.cfg.data_types) if camera is not None else frozenset({"rgb"})
        self._keys = tuple(sensor_key_for_gt_type(channel, available) for channel in cfg.channels)

    def read(self, frame_id: int | float) -> wp.array | None:
        """Return native RGB/RGBA perspective pixels or composed uint8 [H, W, 4] sensor tiles; None for no rows."""
        cfg, camera = self.cfg, self.camera
        if not cfg.envs:
            return None
        selection = (tuple(cfg.envs), tuple(cfg.channels), cfg.depth_range, self.aspect)
        if selection != self._selection:
            validate(cfg)
            available = frozenset(camera.cfg.data_types) if camera is not None else frozenset({"rgb"})
            self._keys = tuple(sensor_key_for_gt_type(channel, available) for channel in cfg.channels)
            self._selection, self._layout = selection, None
            self.invalidate()
        if self.frame.timestamp == frame_id:
            return self.frame.data

        if camera is None:
            if self.render is None:
                raise RuntimeError("Bind a perspective image view to a visualizer before reading or recording it.")
            self.frame.data = self.render(self.frame.data)
        else:
            outputs = camera.data.output
            sources = tuple(outputs[key].warp for key in self._keys)

            layout = tuple((source.shape, source.dtype, source.device) for source in sources)
            if layout != self._layout:
                for source, channel in zip(sources, cfg.channels, strict=True):
                    components = 3 if channel in ("rgb", "normals") else 1
                    if source.ndim != 4 or min(source.shape) < 1 or source.shape[3] < components:
                        raise ValueError(
                            f"Channel {channel!r} requires nonempty [N, H, W, C] arrays with C >= {components}."
                        )
                    if source.shape[:3] != sources[0].shape[:3] or source.device != sources[0].device:
                        raise ValueError("Image channels must have the same batch, resolution, and device.")
                device = sources[0].device
                count, height, width, _ = sources[0].shape
                if max(cfg.envs) >= count:
                    raise ValueError(f"Image row selection is outside the source batch of {count} rows.")
                shape = _image_grid_shape(len(cfg.envs), len(sources), height, width, self.aspect)
                colors = np.empty((0, 3), dtype=np.uint8)
                if "depth" in cfg.channels:
                    colors = (colormaps["turbo"](np.arange(256) / 255.0)[..., :3] * 255).astype(np.uint8)
                self._env_ids = wp.array(cfg.envs, dtype=wp.int32, device=device)
                self._depth_colors = wp.array(colors, dtype=wp.uint8, device=device)
                self.frame.data = wp.empty((*shape, 4), dtype=wp.uint8, device=device)
                self._layout = layout

            compose_image(
                self.frame.data, sources, self._env_ids, cfg.channels, self._depth_colors,
                depth_min=cfg.depth_range[0], depth_max=cfg.depth_range[1],
            )  # fmt: skip
        self.frame.timestamp = frame_id
        self._host_frame = None
        return self.frame.data

    def read_rgb(self, frame_id: int | float) -> np.ndarray | None:
        """Download the composed frame once for CPU encoders or network transports."""
        image = self.read(frame_id)
        if image is None:
            return None
        if self._host_frame is None:
            # CUDA readback owns fresh host storage; CPU arrays still alias the reusable source.
            self._host_frame = np.array(image.numpy()[..., :3], copy=not image.device.is_cuda)
        return self._host_frame

    def invalidate(self) -> None:
        """Invalidate cached pixels after camera movement, a source update, or a simulation reset."""
        self.frame.timestamp = -1.0
        self._host_frame = None
