# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util

from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

# Backend-specific renderer imports — each is optional depending on the installation.
_HAS_NEWTON = importlib.util.find_spec("isaaclab_newton") is not None
_HAS_OV = importlib.util.find_spec("isaaclab_ov") is not None
_HAS_PHYSX = importlib.util.find_spec("isaaclab_physx") is not None

if _HAS_PHYSX:
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

if _HAS_NEWTON:
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

if _HAS_OV:
    from isaaclab_ov.renderers import OVRTXRendererCfg


@configclass
class MultiBackendRendererCfg(PresetCfg):
    if _HAS_PHYSX:
        default: IsaacRtxRendererCfg = IsaacRtxRendererCfg()
        isaacsim_rtx_renderer: IsaacRtxRendererCfg = IsaacRtxRendererCfg()

    if _HAS_NEWTON:
        newton_renderer: NewtonWarpRendererCfg = NewtonWarpRendererCfg()

    if _HAS_OV:
        ovrtx_renderer: OVRTXRendererCfg = OVRTXRendererCfg()
