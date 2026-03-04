# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Isaac RTX (Replicator) Renderer."""

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import configclass


@configclass
class IsaacRtxRendererCfg(RendererCfg):
    """Configuration for Isaac RTX renderer using Omniverse Replicator."""

    renderer_type: str = "isaac_rtx"
    """Type identifier for Isaac RTX renderer."""
