# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-render PPISP (Physically Plausible Image Signal Processing) for IsaacLab.

Add :class:`PpispProcessorCfg` to :attr:`~isaaclab.sensors.camera.CameraCfg.tensor_processors`
to convert a camera's scene-linear HDR output to LDR RGB(A) after each capture.
"""

import importlib.metadata

from isaaclab.utils.module import lazy_export

try:
    __version__ = importlib.metadata.version("isaaclab_ppisp")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

lazy_export()
