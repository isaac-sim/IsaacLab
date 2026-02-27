# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer config presets for Hydra ConfigStore.

Register renderer backend configs that can be selected via the ``renderer`` config
group (e.g. ``renderer=isaac_rtx`` or ``renderer=newton_warp``). The selected config
is applied to all cameras in the scene.
"""

from hydra.core.config_store import ConfigStore

from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg


def register_render_configs() -> None:
    """Register renderer config presets in Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="isaac_rtx", group="renderer", node=IsaacRtxRendererCfg)
    cs.store(name="newton_warp", group="renderer", node=NewtonWarpRendererCfg)
