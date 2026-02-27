# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer config presets for Hydra ConfigStore.

Register renderer backend configs that can be selected via the ``render_cfg`` config
group (e.g. ``render_cfg=isaac_rtx`` or ``render_cfg=newton_warp``). The selected config
is applied to all cameras in the scene.
"""

from hydra.core.config_store import ConfigStore

from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg


def register_render_configs() -> None:
    """Register renderer config presets in Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="isaac_rtx", group="render_cfg", node=IsaacRtxRendererCfg)
    cs.store(name="newton_warp", group="render_cfg", node=NewtonWarpRendererCfg)
