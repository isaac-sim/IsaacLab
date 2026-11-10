# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, ClassVar

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_visualizers.kit import KitBackendCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg, NewtonRTXBackendCfg
from isaaclab_visualizers.rerun import RerunBackendCfg
from isaaclab_visualizers.viser import ViserBackendCfg

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import VISUALIZER_ALIASES, VisualizerBackendCfg

from .hydra import PresetCfg


@configclass
class _AutoRtxRendererCfg(RendererCfg):
    renderer_type: str = "auto_rtx"


@configclass
class MultiBackendRendererCfg(PresetCfg):
    rtx: _AutoRtxRendererCfg = _AutoRtxRendererCfg()
    ovrtx: OVRTXRendererCfg = OVRTXRendererCfg()
    isaacsim_rtx: IsaacRtxRendererCfg = IsaacRtxRendererCfg()
    newton_renderer: NewtonWarpRendererCfg = NewtonWarpRendererCfg()
    default: NewtonWarpRendererCfg = NewtonWarpRendererCfg()


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
        for attr_name in ("default", "isaacsim_rtx"):
            _visit(getattr(cfg, attr_name, None))

    _visit(renderer_cfg)


@configclass
class MultiBackendVisualizerCfg(PresetCfg):
    """Built-in visualizers selected with ``visualizer=NAME``; no viewer opens by default."""

    kit: KitBackendCfg = KitBackendCfg()
    newton_gl: NewtonGLBackendCfg = NewtonGLBackendCfg()
    newton_rtx: NewtonRTXBackendCfg = NewtonRTXBackendCfg()
    rerun: RerunBackendCfg = RerunBackendCfg()
    viser: ViserBackendCfg = ViserBackendCfg()
    none: list[VisualizerBackendCfg] = []
    default: list[VisualizerBackendCfg] = []

    _aliases: ClassVar[dict[str, str]] = VISUALIZER_ALIASES
