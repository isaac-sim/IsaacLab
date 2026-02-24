# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stub renderer config classes.

These are intentionally lightweight placeholders so the rendering domain can
stabilize around explicit renderer config names before full implementation.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class RendererCfg:
    """Base configuration for all renderer backends."""

    renderer_type: str | None = None
    """Type identifier (e.g., 'rtx', 'warp'). Must be overridden by subclasses."""

    rendering_quality: str | None = None
    """Name of the rendering quality profile to use with this renderer."""
