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

from ..envs.utils.camera_view import image_grid_columns, sensor_key_for_gt_type
from ..utils import validate
from ..utils.buffers import TimestampedBuffer
from ..utils.images import _resize_rgba_image, compose_image
from .visualizer_cfg import ImageViewCfg

if TYPE_CHECKING:
    from ..sensors import Camera


class ImageView:
    """Own one composed GPU image shared by its display and recording consumers.

    The scene retains ownership of sensors and their capture cadence. Perspective producers write
    device pixels through ``render``. Reading twice for the same frame reuses the composed image;
    only :meth:`read_rgb` transfers pixels to the host.
    """

    def __init__(self, cfg: ImageViewCfg, camera: Camera | None = None) -> None:
        """Bind a view declaration to a scene sensor or an initially unbound perspective producer."""
        validate(cfg)
        self.cfg = cfg
        self.camera = camera
        self.render: Callable[[wp.array | None], wp.array] | None = None
        self.aspect = 1.0
        self.frame = TimestampedBuffer()
        self._host_frame: np.ndarray | None = None
        self._render_buffer: wp.array | None = None
        self._tiles: wp.array | None = None
        self._env_ids: wp.array | None = None
        self._depth_colors: wp.array | None = None
        self._selection = self._layout = None
        self._keys: tuple[str, ...] = ()

    def read(self, frame_id: int | float) -> wp.array | None:
        """Return borrowed uint8 [H, W, 4] pixels, or None for an empty row selection."""
        cfg, camera = self.cfg, self.camera
        if not cfg.envs:
            return None
        selection = (tuple(cfg.envs), tuple(cfg.channels), cfg.size, cfg.depth_range, self.aspect)
        if selection != self._selection:
            validate(cfg)
            available = frozenset(camera.cfg.data_types) if camera is not None else frozenset({"rgb"})
            self._keys = tuple(sensor_key_for_gt_type(channel, available) for channel in cfg.channels)
            self._selection, self._layout = selection, None
            self.invalidate()
        if self.frame.timestamp == frame_id:
            return self.frame.data

        if camera is not None:
            outputs = camera.data.output
            sources = tuple(outputs[key].warp for key in self._keys)
        else:
            if self.render is None:
                raise RuntimeError("Bind a perspective image view to a visualizer before reading or recording it.")
            self._render_buffer = self.render(self._render_buffer)
            pixels = self._render_buffer
            sources = (pixels.reshape((1, *pixels.shape)),)

        layout = tuple((source.shape, source.dtype, source.device) for source in sources)
        if layout != self._layout:
            for source, channel in zip(sources, cfg.channels, strict=True):
                components = 3 if channel in ("rgb", "normals") else 1
                if source.ndim != 4 or min(source.shape) < 1 or source.shape[3] < components:
                    raise ValueError(
                        f"Channel {channel!r} requires nonempty [N, H, W, C] arrays with C >= {components}."
                    )
            device = sources[0].device
            count, height, width, _ = sources[0].shape
            if any(source.device != device or source.shape[:3] != (count, height, width) for source in sources):
                raise ValueError("Image channels must have the same batch size, resolution, and device.")
            if max(cfg.envs) >= count:
                raise ValueError(f"Image row selection is outside the source batch of {count} rows.")
            aspect = cfg.size[0] / cfg.size[1] if cfg.size is not None else self.aspect
            columns = image_grid_columns(len(cfg.envs), len(sources), height, width, aspect)
            shape = (math.ceil(len(cfg.envs) / columns) * height, columns * len(sources) * width, 4)
            colors = np.empty((0, 3), dtype=np.uint8)
            if "depth" in cfg.channels:
                colors = (colormaps["turbo"](np.arange(256) / 255.0)[..., :3] * 255).astype(np.uint8)
            self._env_ids = wp.array(cfg.envs, dtype=wp.int32, device=device)
            self._depth_colors = wp.array(colors, dtype=wp.uint8, device=device)
            self._tiles = wp.empty(shape, dtype=wp.uint8, device=device)
            output_shape = (cfg.size[1], cfg.size[0], 4) if cfg.size is not None else shape
            if output_shape == shape:
                self.frame.data = self._tiles
            elif self.frame.data is None or self.frame.data.shape != output_shape or self.frame.data.device != device:
                self.frame.data = wp.empty(output_shape, dtype=wp.uint8, device=device)
            self._layout = layout

        compose_image(
            self._tiles, sources, self._env_ids, cfg.channels, self._depth_colors,
            depth_min=cfg.depth_range[0], depth_max=cfg.depth_range[1],
        )  # fmt: skip
        if self.frame.data is not self._tiles:
            output = self.frame.data
            wp.launch(_resize_rgba_image, dim=output.shape[:2], inputs=[self._tiles, output], device=output.device)
        self.frame.timestamp = frame_id
        self._host_frame = None
        return self.frame.data

    def read_rgb(self, frame_id: int | float) -> np.ndarray | None:
        """Download the composed frame once for CPU encoders or network transports."""
        image = self.read(frame_id)
        if image is None:
            return None
        if self._host_frame is None:
            self._host_frame = np.ascontiguousarray(image.numpy()[..., :3])
        return self._host_frame

    def invalidate(self) -> None:
        """Invalidate cached pixels after camera movement, a source update, or a simulation reset."""
        self.frame.timestamp = -1.0
        self._host_frame = None

    def close(self) -> None:
        """Release source references and owned storage after the consumers have closed."""
        self.camera = self.render = self._render_buffer = None
        self.frame = TimestampedBuffer()
        self._host_frame = None
        self._tiles = self._env_ids = self._depth_colors = None
        self._layout = self._selection = None
