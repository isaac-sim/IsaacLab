# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy ISP discovery names retained for migration of saved configurations.

``CameraCfg.isp_cfg`` no longer executes processing. Use ``CameraCfg.pipeline`` and
the discovery configuration provided by the image-processing package instead.
"""

from __future__ import annotations

from enum import StrEnum


class CameraISPMode(StrEnum):
    """Retired camera ISP discovery modes; use the processing package's discovery configuration."""

    AUTO_CAMERA = "auto_camera"
    """Discover an ISP cfg bound to the batch's first matched camera prim.

    The renderer walks the stage for a ``RenderProduct`` whose ``camera``
    relationship targets the first matched camera prim in the sensor batch. If
    found and it has an ISP shader child, parses the shader. Resolves to
    ``None`` (no ISP) if no match.
    """

    AUTO_ANY = "auto_any"
    """Same as :attr:`AUTO_CAMERA`, then fall back to the first ISP shader anywhere on the stage.

    Used to honour a stage-wide ISP authoring even when no ``RenderProduct``
    binds the shader explicitly to the batch's first matched camera prim.
    Resolves to ``None`` only when the stage contains no ISP shader at all.
    """
