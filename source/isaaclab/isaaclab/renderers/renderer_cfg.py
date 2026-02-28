# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

<<<<<<< mtrepte/add_rendering_quality_cfg
"""Stub renderer config classes.

These are intentionally lightweight placeholders so the rendering domain can
stabilize around explicit renderer config names before full implementation.
"""

from __future__ import annotations
=======
"""Base configuration for renderers."""
>>>>>>> develop

from isaaclab.utils import configclass


@configclass
class RendererCfg:
<<<<<<< mtrepte/add_rendering_quality_cfg
    """Base configuration for all renderer backends."""

    renderer_type: str | None = None
    """Type identifier (e.g., 'rtx', 'warp'). Must be overridden by subclasses."""

    rendering_mode: str | None = None
    """Name of the rendering mode profile to use with this renderer."""
=======
    """Configuration for a renderer."""

    renderer_type: str = "default"
>>>>>>> develop
