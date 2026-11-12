# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX exposure settings authored during camera construction, before scene import."""

import warnings

from pxr import Usd

from isaaclab.sensors.camera.camera_exposure import apply_neutral_exposure


def apply_rtx_exposure_overrides(stage: Usd.Stage, prim_paths: list[str]) -> None:
    """Neutralize RTX exposure when PPISP owns image processing.

    .. deprecated::
        Cameras do this themselves with ``CameraCfg.exposure = ExposureCfg(mode="neutral")``; use that, or
        :func:`isaaclab.sensors.camera.camera_exposure.apply_neutral_exposure` on other prims.

    Args:
        stage: USD stage containing the cameras.
        prim_paths: Resolved camera prim paths (typically one per env).
    """
    warnings.warn(
        'apply_rtx_exposure_overrides is deprecated. Set CameraCfg.exposure = ExposureCfg(mode="neutral") or call'
        " isaaclab.sensors.camera.camera_exposure.apply_neutral_exposure.",
        FutureWarning,
        stacklevel=2,
    )
    apply_neutral_exposure(stage, prim_paths)
