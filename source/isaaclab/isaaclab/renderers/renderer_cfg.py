# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for renderers."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .base_renderer import BaseRenderer


@configclass
class RendererCfg:
    """Configuration for a renderer."""

    renderer_type: str = "default"
    """Type identifier (e.g. 'isaac_rtx', 'newton_warp')."""

    data_types: list[str] = MISSING
    """List of data types to use for rendering (synced from camera config when needed)."""

    def get_renderer_type(self) -> str:
        return self.renderer_type

    def create_renderer(self) -> "BaseRenderer":
        """Create a renderer instance from this config. Uses the Renderer factory."""
        from isaaclab.renderers import Renderer
        return Renderer(self)
