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


@configclass
class MultiBackendRendererCfg(PresetCfg):
    default: IsaacRtxRendererCfg = IsaacRtxRendererCfg()
    newton_renderer: NewtonWarpRendererCfg = NewtonWarpRendererCfg()
    ovrtx_renderer: OVRTXRendererCfg = OVRTXRendererCfg()
    isaacsim_rtx_renderer = default


@configclass
class _FrameStackPolicyByRenderer(PresetCfg):
    """Renderer-keyed inner preset; ``default=0`` is the sentinel ``Camera`` reads via ``max(1, ...)``."""

    default: int = 0
    newton_renderer: int = 2


@configclass
class _FrameStackPolicyBranch:
    """Intermediate regular configclass between the two PresetCfg layers: forces ``collect_presets``
    to extend the path with ``.by_renderer`` so the global broadcast doesn't clobber and over-fire."""

    by_renderer: _FrameStackPolicyByRenderer = _FrameStackPolicyByRenderer()


@configclass
class FrameStackPolicyCfg(PresetCfg):
    """``frame_stack`` policy keyed on physics + renderer; resolves to 2 only for Newton + Warp."""

    default: int = 0
    newton_mjwarp: _FrameStackPolicyBranch = _FrameStackPolicyBranch()


@configclass
class MultiBackendCameraCfg(CameraCfg):
    """``frame_stack`` defaults to 0 (sentinel); ``launch_simulation`` auto-applies the policy.
    Any user-set value (including 1) is respected."""

    renderer_cfg: MultiBackendRendererCfg = MultiBackendRendererCfg()
    frame_stack: int = 0
    frame_stack_policy: FrameStackPolicyCfg = FrameStackPolicyCfg()
