# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sentinel enum for discovering ISP settings from the USD stage.

Camera construction resolves these sentinels before scene import, through the
``prepare_camera`` hook of the ISP processor, such as :class:`isaaclab_ppisp.PpispProcessor`.
"""

from __future__ import annotations

from enum import StrEnum


class CameraISPMode(StrEnum):
    """Sentinel modes for :attr:`isaaclab_ppisp.PpispProcessorCfg.isp_cfg`.

    Selects how camera construction discovers the ISP configuration from the
    USD stage at camera initialization time. Discovery resolves one cfg for
    the whole Camera sensor batch, using the first matched camera prim as the
    camera-bound lookup target. The deprecated
    :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg` accepts the same values.
    """

    AUTO_CAMERA = "auto_camera"
    """Read ``ppisp:*`` attributes from the batch's first matched camera prim.

    Resolves to no ISP when that camera has no PPISP attributes.
    """

    AUTO_ANY = "auto_any"
    """Same as :attr:`AUTO_CAMERA`, then fall back to the first camera with PPISP attributes on the stage.

    Resolves to no ISP only when no camera on the stage has PPISP attributes.
    """
