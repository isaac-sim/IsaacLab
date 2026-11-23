# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neutral camera exposure, authored on the camera prims before the scene is cloned."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd


NEUTRAL_EXPOSURE: tuple[tuple[str, str, float | bool], ...] = (
    ("exposure", "Float", 0.0),
    ("exposure:fStop", "Float", 1.0),
    ("exposure:iso", "Float", 0.0),
    ("exposure:responsivity", "Float", 1.0),
    ("exposure:time", "Float", 1.0),
    ("omni:rtx:autoExposure:enabled", "Bool", False),
)
"""Camera prim attributes that make RTX exposure an identity, as ``(name, USD value type, value)``."""

NEUTRAL_EXPOSURE_SCHEMAS: tuple[str, ...] = ("OmniRtxCameraAutoExposureAPI_1", "OmniRtxCameraExposureAPI_1")
"""RTX API schemas that route the exposure attributes into the camera's render settings."""


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
        for schema in NEUTRAL_EXPOSURE_SCHEMAS:
            prim.AddAppliedSchema(schema)
        for name, value_type, value in NEUTRAL_EXPOSURE:
            prim.CreateAttribute(name, getattr(Sdf.ValueTypeNames, value_type), custom=False).Set(value)


def has_neutral_exposure(prim: Usd.Prim) -> bool:
    """Return whether a camera prim has every neutral exposure attribute authored at its neutral value, with the
    RTX schemas that apply them."""
    # The authored list, since USD leaves RTX's schemas out of GetAppliedSchemas() without RTX's schema plugin.
    schemas = prim.GetMetadata("apiSchemas")
    applied = schemas.GetAppliedItems() if schemas is not None else []
    if any(schema not in applied for schema in NEUTRAL_EXPOSURE_SCHEMAS):
        return False
    for name, _, value in NEUTRAL_EXPOSURE:
        attr = prim.GetAttribute(name)
        if not attr or not attr.HasAuthoredValue() or attr.Get() != value:
            return False
    return True
