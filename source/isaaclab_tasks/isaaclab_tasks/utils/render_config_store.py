# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer config presets for Hydra ConfigStore.

Register renderer backend configs that can be selected via the ``render`` config
group (e.g. ``render=isaac_rtx`` or ``render=newton_warp``). The selected config
is applied to all cameras in the scene.
"""

from hydra.core.config_store import ConfigStore
from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

try:
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    NEWTON_WARP_AVAILABLE = True
except ImportError:
    NewtonWarpRendererCfg = None
    NEWTON_WARP_AVAILABLE = False


def register_render_configs() -> None:
    """Register renderer config presets in Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="isaac_rtx", group="render", node=IsaacRtxRendererCfg)
    if NewtonWarpRendererCfg is not None:
        cs.store(name="newton_warp", group="render", node=NewtonWarpRendererCfg)
