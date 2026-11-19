# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit foveated ray grids for Newton Warp and OptiX."""

from __future__ import annotations

import warp as wp

from .foveated_camera import FoveatedCamera
from .foveated_camera_cfg import FoveatedCameraCfg


class FoveatedNewtonCamera(FoveatedCamera):
    """Render the shared retinal grid with Newton's explicit ray buffer."""

    def _initialize_impl(self):
        renderer = self.cfg.renderer_cfg
        if renderer.render_order == "tiled":
            tile_width, tile_height = renderer.tile_rendering_width, renderer.tile_rendering_height
            if tile_width <= 0 or tile_height <= 0:
                raise ValueError("Newton render tile dimensions must be positive.")
            if self.cfg.width % tile_width or self.cfg.height % tile_height:
                raise ValueError(
                    f"Foveated grid {self.cfg.width}x{self.cfg.height} must be divisible by Newton tile size "
                    f"{tile_width}x{tile_height}; adjust foveated_res or use a non-tiled render order."
                )

        super()._initialize_impl()
        self._initialize_gaze()
        rays = self._retinal_rays.new_zeros((self.cfg.height, self.cfg.width, 1, 2, 3))
        rays[..., 1, :] = self._retinal_rays.view(self.cfg.height, self.cfg.width, 1, 3)
        # Newton allocates its default rays only once. Replace them before the first capture.
        self._render_data.camera_rays = wp.from_torch(rays, dtype=wp.vec3f)
        self._render_output = self._data.output
        self._prepare_grid_publish({name: value.torch for name, value in self._render_output.items()})

    def _update_buffers_impl(self, env_mask: wp.array):
        # Newton's render may redraw every environment. Only publish the rows due for update.
        published = self._data.output
        self._data._output = self._render_output
        try:
            super()._update_buffers_impl(env_mask)
        finally:
            self._data._output = published
        self._publish_grid(env_mask)

    @staticmethod
    def _pose_update_turns_gaze(cfg: FoveatedCameraCfg) -> bool:
        return True
