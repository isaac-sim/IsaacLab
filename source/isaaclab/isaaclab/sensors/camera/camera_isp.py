# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sentinel enum for :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg`.

The renderer backend (which depends on the appropriate ISP implementation
package) is responsible for resolving these sentinels to a concrete cfg and
applying the ISP pipeline. :mod:`isaaclab.sensors.camera` carries the sentinel
through to the renderer without knowing about any specific ISP implementation.
"""

from __future__ import annotations

from enum import StrEnum


class CameraISPMode(StrEnum):
    """Sentinel modes for :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg`.

    Selects how the renderer backend discovers the ISP configuration from the
    USD stage at camera initialization time. Discovery resolves one cfg for
    the whole Camera sensor batch, using the first matched camera prim as the
    camera-bound lookup target. ``isp_cfg=None`` disables ISP entirely; this
    enum carries only the *active* discovery strategies.
    """

    AUTO_CAMERA = "auto_camera"
    """Discover an ISP cfg bound to the batch's first matched camera prim.

    The renderer checks the first matched camera prim in the sensor batch for
    recognized ``ppisp:*`` attributes, including optional controller weights.
    It parses those attributes into an ISP cfg, or resolves to ``None`` when
    the camera has no PPISP attributes.
    """

    AUTO_ANY = "auto_any"
    """Same as :attr:`AUTO_CAMERA`, then fall back to the first PPISP camera anywhere on the stage.

    Used to honor stage-wide camera authoring when the batch's first matched
    camera has no recognized ``ppisp:*`` attributes. Resolves to ``None`` only
    when the stage contains no camera with PPISP attributes.
    """
