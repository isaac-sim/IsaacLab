# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discovery modes for ``isaaclab_ppisp.PpispProcessorCfg.isp_cfg``.

Camera construction resolves these modes before scene import.
"""

from __future__ import annotations

from enum import StrEnum


class CameraISPMode(StrEnum):
    """Modes that discover ``isaaclab_ppisp.PpispProcessorCfg.isp_cfg`` from ``ppisp:*`` camera attributes.

    The PPISP processor resolves the mode during camera construction, once for the whole Camera sensor
    batch, using the first matched camera prim. When discovery finds no PPISP attributes, the camera
    removes the processor and the renderer produces the requested outputs directly.
    """

    AUTO_CAMERA = "auto_camera"
    """Read the PPISP attributes authored on the batch's first matched camera prim."""

    AUTO_ANY = "auto_any"
    """Same as :attr:`AUTO_CAMERA`, then fall back to the first camera on the stage with PPISP attributes."""
