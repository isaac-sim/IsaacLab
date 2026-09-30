# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated PPISP discovery modes, available without importing the optional PPISP package."""

from enum import StrEnum


class CameraISPMode(StrEnum):
    """Compatibility modes for :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg`.

    Use :class:`isaaclab_ppisp.PpispDiscoveryMode` for new processor configurations.
    The PPISP resolver warns when a legacy mode is used. Discovery reads camera-authored
    ``ppisp:*`` attributes once, before renderer setup, and applies the result to the camera batch.
    """

    AUTO_CAMERA = "auto_camera"
    """Read PPISP attributes from the batch's first matched camera."""

    AUTO_ANY = "auto_any"
    """Read the source camera first, then fall back to the first PPISP camera on the stage."""
