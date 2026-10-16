# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-render PPISP (Physically Plausible Image Signal Processing) for IsaacLab.

Provides a dictionary processor that converts scene-linear radiance to RGB or RGBA.
Camera and observation pipelines use :class:`PpispProcessorCfg` with concrete settings. USD discovery
helpers remain explicit, and renderer backends do not own PPISP execution or controller workspace.
"""

import importlib.metadata

from isaaclab.utils.module import lazy_export

try:
    __version__ = importlib.metadata.version("isaaclab_ppisp")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

lazy_export()
