# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class _AutoRtxRendererCfg(RendererCfg):
    renderer_type: str = "auto_rtx"


@configclass
class MultiBackendRendererCfg(PresetCfg):
    default: IsaacRtxRendererCfg = IsaacRtxRendererCfg()
    rtx: _AutoRtxRendererCfg = _AutoRtxRendererCfg()
    newton_renderer: NewtonWarpRendererCfg = NewtonWarpRendererCfg()
    ovrtx_renderer: OVRTXRendererCfg = OVRTXRendererCfg()
    isaacsim_rtx_renderer = default


def set_isaac_rtx_global_settings(renderer_cfg: Any, **settings: Any) -> None:
    """Set Isaac RTX settings on direct or preset-wrapped renderer configs."""
    visited: set[int] = set()

    def _visit(cfg: Any) -> None:
        if cfg is None or id(cfg) in visited:
            return
        visited.add(id(cfg))
        if getattr(cfg, "renderer_type", None) == "isaac_rtx" and hasattr(cfg, "global_settings"):
            for key, value in settings.items():
                setattr(cfg.global_settings, key, value)
        for attr_name in ("default", "isaacsim_rtx_renderer"):
            _visit(getattr(cfg, attr_name, None))

    _visit(renderer_cfg)
