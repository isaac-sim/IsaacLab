# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.hydra import preset


@configclass
class MultiBackendRendererCfg(PresetCfg):
    default: IsaacRtxRendererCfg = IsaacRtxRendererCfg()
    newton_renderer: NewtonWarpRendererCfg = NewtonWarpRendererCfg()
    ovrtx_renderer: OVRTXRendererCfg = OVRTXRendererCfg()
    isaacsim_rtx_renderer = default


@configclass
class MultiBackendCameraCfg(CameraCfg):
    """CameraCfg with multi-backend renderer and automatic Newton frame stacking.

    When ``presets=newton`` is used, ``frame_stack`` is automatically set to 2.
    Newton's energy-conserving (symplectic) integrator produces dynamics that
    require velocity information for effective camera-based control. Frame stacking
    provides this temporal information by concatenating consecutive frames along
    the channel dimension, enabling the policy to infer velocity from pixel
    differences between frames.
    """

    renderer_cfg: MultiBackendRendererCfg = MultiBackendRendererCfg()
    frame_stack = preset(default=1, newton=2)
