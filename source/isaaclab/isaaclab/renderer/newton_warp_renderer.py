# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

from newton.warp_raytrace import RenderContext

from .newton_warp_renderer_cfg import NewtonWarpRendererCfg
from .renderer import RendererBase


class NewtonWarpRenderer(RendererBase):
    """Newton Warp Renderer implementation."""

    def __init__(self, cfg: NewtonWarpRendererCfg):
        super().__init__(cfg)
        self._render_context = RenderContext(width=self._width, height=self._height, num_worlds=self._num_envs)

    def initialize(self):
        """Initialize the renderer."""
        pass

    def step(self):
        """Step the renderer."""
