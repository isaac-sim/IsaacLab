# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for renderers."""

from __future__ import annotations

from dataclasses import field

from isaaclab.utils import configclass


@configclass
class RendererCfg:
    """Configuration for a renderer."""

    renderer_type: str = "default"
    """Type identifier (e.g. 'isaac_rtx', 'newton_warp')."""

    data_types: list[str] = field(default_factory=list)
    """Data types to render (e.g. 'rgb', 'depth'). Set by the camera at use time; default empty."""
