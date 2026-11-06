# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera exposure configuration, authored on the camera prims before the scene is cloned."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ...utils import configclass

if TYPE_CHECKING:
    from pxr import Usd


@configclass
class ExposureCfg:
    """Camera exposure.

    Isaac RTX and OVRTX apply the exposure authored on the camera prim to every color output of the camera.
    The Newton Warp renderer applies no exposure, so it ignores this setting.
    """

    mode: Literal["default", "neutral"] = "default"
    """Exposure behavior. Defaults to ``"default"``.

    - ``"default"``: keep the exposure authored on the camera prim, or the renderer's own default.
    - ``"neutral"``: disable automatic exposure and author unit exposure on the camera prims, so color outputs keep
      the scene's linear radiance. Image processors that apply their own exposure, such as PPISP, require it.
    """


NEUTRAL_EXPOSURE: tuple[tuple[str, str, float | bool], ...] = (
    ("exposure", "Float", 0.0),
    ("exposure:fStop", "Float", 1.0),
    ("exposure:iso", "Float", 0.0),
    ("exposure:responsivity", "Float", 1.0),
    ("exposure:time", "Float", 1.0),
    ("omni:rtx:autoExposure:enabled", "Bool", False),
)
"""Camera prim attributes that make RTX exposure an identity, as ``(name, USD value type, value)``."""


def apply_neutral_exposure(stage: Usd.Stage, prim_paths: list[str]) -> None:
    """Author neutral exposure on camera prims.

    Adding the schema tokens directly preserves other applied APIs without requiring RTX's schema plugin to be
    registered in the host USD process.

    Args:
        stage: USD stage containing the cameras.
        prim_paths: Camera prim paths.
    """
    from pxr import Sdf  # noqa: PLC0415

    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
        prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
        for name, value_type, value in NEUTRAL_EXPOSURE:
            prim.CreateAttribute(name, getattr(Sdf.ValueTypeNames, value_type), custom=False).Set(value)


def has_neutral_exposure(prim: Usd.Prim) -> bool:
    """Return whether a camera prim has every neutral exposure attribute authored at its neutral value."""
    for name, _, value in NEUTRAL_EXPOSURE:
        attr = prim.GetAttribute(name)
        if not attr or not attr.HasAuthoredValue() or attr.Get() != value:
            return False
    return True
