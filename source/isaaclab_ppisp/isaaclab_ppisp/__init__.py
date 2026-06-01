# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-render PPISP (Physically Plausible Image Signal Processing) for IsaacLab.

Provides the renderer-backend-agnostic ISP pipeline that converts the renderer's
HDR scene-linear AOV to LDR RGBA at the end of a render tick. Renderer backends
(:class:`~isaaclab_physx.renderers.IsaacRtxRenderer`,
:class:`~isaaclab_ov.renderers.OVRTXRenderer`,
:class:`~isaaclab_newton.renderers.NewtonWarpRenderer`) compose
:class:`PpispPipeline` internally when the camera's
:attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg` is set.
"""

import importlib.metadata

from isaaclab.utils.module import lazy_export

try:
    __version__ = importlib.metadata.version("isaaclab_ppisp")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

lazy_export()
