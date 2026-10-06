# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-render PPISP (Physically Plausible Image Signal Processing) for IsaacLab.

Provides the ISP processor that converts rendered scene-linear HDR to LDR
RGB/RGBA. Configure :class:`PpispModifierCfg` in an observation term's modifier list,
or apply :class:`PpispPipeline` directly to radiance buffers.
"""

import importlib.metadata

from isaaclab.utils.module import lazy_export

try:
    __version__ = importlib.metadata.version("isaaclab_ppisp")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

lazy_export()
