# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for renderers."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RendererCfg:
    """Configuration for a renderer."""

    renderer_type: str = "default"
    """Type identifier (e.g. 'isaac_rtx', 'newton_warp')."""

    # required by Hydra overrides
    # Overrides like env.scene.base_camera.renderer_type=newton_warp 
    #   only work if the composed config has that attribute.
    data_types: list[str] = MISSING
    """List of data types to use for rendering (synced from camera config when needed)."""
