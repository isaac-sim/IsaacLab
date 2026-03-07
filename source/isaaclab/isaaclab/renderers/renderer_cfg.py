# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for renderers."""

from isaaclab.utils import configclass


@configclass
class RendererCfg:
    """Configuration for a renderer."""

    renderer_type: str = "default"

    requires_newton_model: bool = False
    """Internal requirement flag for scene-data setup; avoid overriding in user configs."""

    requires_usd_stage: bool = False
    """Internal requirement flag for scene-data setup; avoid overriding in user configs."""
