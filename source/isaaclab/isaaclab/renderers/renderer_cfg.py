# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for renderers."""

from isaaclab.utils import configclass


@configclass
class RendererCfg:
    """Configuration for a renderer."""

    renderer_type: str | None = None
    """Type identifier for selecting a renderer backend implementation."""

    rendering_mode: str | None = None
    """Name of the Kit/RTX rendering mode profile (``default``, ``isaac_rtx``, ``rtx`` only).

    Ignored for other renderer types; configure those via their dedicated cfg classes.
    """
