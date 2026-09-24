# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-render PPISP (Physically Plausible Image Signal Processing) for IsaacLab.

Provides the ISP processor that converts rendered scene-linear HDR to LDR
RGB/RGBA. Configure :class:`PpispProcessorCfg` in a
:class:`~isaaclab.envs.mdp.visual_observations.processed_image` observation term,
or use the compatible
:attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg` entry point.
"""

import importlib.metadata

from isaaclab.utils.module import lazy_export

try:
    __version__ = importlib.metadata.version("isaaclab_ppisp")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

lazy_export()
