# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX exposure settings authored during camera construction, before scene import."""

from pxr import Sdf, Usd


def apply_rtx_exposure_overrides(stage: Usd.Stage, prim_paths: list[str]) -> None:
    """Neutralize RTX exposure when PPISP owns image processing.

    Adding the schema tokens directly preserves other applied APIs without requiring RTX's
    schema plugin to be registered in the host USD process.

    Args:
        stage: USD stage containing the cameras.
        prim_paths: Resolved camera prim paths (typically one per env).
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
        prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
        for name, value in (
            ("exposure", 0.0),
            ("exposure:fStop", 1.0),
            ("exposure:iso", 0.0),
            ("exposure:responsivity", 1.0),
            ("exposure:time", 1.0),
        ):
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(value)
        prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool, custom=False).Set(False)
