# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass


@configclass
class CameraExposureCfg:
    """Explicit RTX camera exposure settings, applied before renderer stage export.

    Defaults disable automatic exposure and make HDR neutral for PPISP. Settings affect every
    output from the camera prim. Newton does not consume RTX exposure attributes; its HDR output
    is already scene-linear. Leave ``CameraCfg.exposure`` unset to preserve authored settings.
    """

    exposure: float = 0.0
    """Exposure compensation in stops."""

    f_stop: float = 1.0
    """Exposure aperture ratio (separate from the lens depth-of-field aperture)."""

    iso: float = 0.0
    """RTX sensor ISO; zero selects neutral exposure."""

    responsivity: float = 1.0
    """Sensor response multiplier."""

    time: float = 1.0
    """Exposure duration in seconds."""

    auto_exposure: bool = False
    """Whether RTX automatic exposure is enabled."""
