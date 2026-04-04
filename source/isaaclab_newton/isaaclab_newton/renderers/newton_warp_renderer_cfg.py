# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton Warp Renderer."""

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import configclass


@configclass
class NewtonWarpRendererCfg(RendererCfg):
    """Configuration for Newton Warp Renderer."""

    renderer_type: str = "newton_warp"
    """Type identifier for Newton Warp renderer."""

    enable_textures: bool = True
    """Enable texture-mapped rendering for meshes."""

    enable_shadows: bool = False
    """Enable shadow rays for directional lights."""

    enable_ambient_lighting: bool = True
    """Enable ambient lighting for the scene."""

    enable_backface_culling: bool = True
    """Cull back-facing triangles."""

    max_distance: float = 1000.0
    """Maximum ray distance [m]."""

    create_default_light: bool = True
    """Create a default directional light source in the scene."""
