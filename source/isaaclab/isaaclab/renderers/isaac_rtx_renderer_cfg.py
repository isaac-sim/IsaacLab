# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Isaac RTX (Replicator) renderer.

When used as TiledCamera's renderer_cfg, the camera uses the built-in Omniverse RTX
tiled rendering path (Replicator annotators).
"""

from isaaclab.utils import configclass

from .renderer_cfg import RendererCfg


@configclass
class IsaacRtxRendererCfg(RendererCfg):
    """Configuration for the built-in Isaac RTX (Replicator) tiled renderer.

    Use with ``TiledCameraCfg(renderer_cfg=IsaacRtxRendererCfg(), ...)`` for the
    default Omniverse RTX path.
    """

    renderer_type: str = "rtx"
    """Type identifier for the Isaac RTX renderer."""
