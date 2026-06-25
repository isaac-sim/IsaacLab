# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
