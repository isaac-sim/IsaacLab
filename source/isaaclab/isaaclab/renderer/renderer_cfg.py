# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for renderers."""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .renderer import RendererBase


@configclass
class RendererCfg:
    """Configuration for a renderer."""

    renderer_type: str = "base"
    """Type Identifier (e.g., 'newton_warp', 'ov_rtx', 'kit_app')."""

    height: int = 1024
    """Height of the renderer. Defaults to 1024."""

    width: int = 1024
    """Width of the renderer. Defaults to 1024."""

    num_envs: int = 1
    """Number of environments to use for rendering. Defaults to 1."""

    num_cameras: int = 1
    """Number of cameras to use for rendering. Defaults to 1."""

    data_types: list[str] = field(default_factory=list)
    """List of data types to use for rendering. Overridden by the camera at runtime when used with TiledCameraCfg."""

    def get_renderer_type(self) -> str:
        """Get the type identifier of the renderer."""
        return self.renderer_type

    def get_height(self) -> int:
        """Get the height of the renderer."""
        return self.height

    def get_width(self) -> int:
        """Get the width of the renderer."""
        return self.width

    def get_num_envs(self) -> int:
        """Get the number of environments to use for rendering."""
        return self.num_envs

    def get_data_types(self) -> list[str]:
        """Get the list of data types to use for rendering."""
        return self.data_types

    def get_num_cameras(self) -> int:
        """Get the number of cameras to use for rendering."""
        return self.num_cameras

    def create_renderer(self) -> RendererBase:
        """Create a renderer instance from this config."""
        from . import get_renderer_class

        return get_renderer_class(self)(self)
