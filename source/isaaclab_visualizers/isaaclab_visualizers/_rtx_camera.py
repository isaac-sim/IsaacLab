# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared camera setup for Isaac RTX and Newton RTX visualizers."""

from __future__ import annotations

import math
from typing import Any

_RTX_CAMERA_API_SCHEMAS = (
    "OmniRtxCameraAutoExposureAPI_1",
    "OmniRtxCameraExposureAPI_1",
)

# Isaac Sim's default perspective camera values. Authoring these on both RTX visualizers
# keeps exposure compensation comparable instead of relying on USD's much brighter fallbacks.
_RTX_CAMERA_F_STOP = 5.0
_RTX_CAMERA_ISO = 100.0
_RTX_CAMERA_RESPONSIVITY = 1.1026709079742432
_RTX_CAMERA_EXPOSURE_TIME = 0.02


def apply_rtx_camera_settings(
    stage: Any,
    camera_path: str,
    *,
    focal_length: float | None,
    exposure: float,
) -> bool:
    """Apply common lens and exposure settings to an RTX USD camera.

    Args:
        stage: USD stage containing the camera.
        camera_path: Camera prim path.
        focal_length: Optional focal length [mm]. ``None`` preserves the authored lens.
        exposure: Exposure compensation [EV].

    Returns:
        ``True`` when the camera exists and the settings were applied.

    Raises:
        ValueError: If ``focal_length`` is non-positive or ``exposure`` is not finite.
    """
    from pxr import Sdf, UsdGeom

    if focal_length is not None and focal_length <= 0.0:
        raise ValueError(f"focal_length must be positive, got {focal_length}.")
    if not math.isfinite(exposure):
        raise ValueError(f"exposure must be finite, got {exposure}.")

    camera = UsdGeom.Camera.Get(stage, camera_path)
    if not camera:
        return False

    prim = camera.GetPrim()
    current_schemas = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    existing_schemas = [
        *current_schemas.prependedItems,
        *current_schemas.explicitItems,
        *current_schemas.appendedItems,
    ]
    missing_schemas = [schema for schema in _RTX_CAMERA_API_SCHEMAS if schema not in existing_schemas]
    if missing_schemas:
        prim.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.Create(prependedItems=[*missing_schemas, *existing_schemas]),
        )

    if focal_length is not None:
        camera.GetFocalLengthAttr().Set(float(focal_length))
    camera.GetExposureAttr().Set(float(exposure))
    camera.GetExposureFStopAttr().Set(_RTX_CAMERA_F_STOP)
    camera.GetExposureIsoAttr().Set(_RTX_CAMERA_ISO)
    camera.GetExposureResponsivityAttr().Set(_RTX_CAMERA_RESPONSIVITY)
    camera.GetExposureTimeAttr().Set(_RTX_CAMERA_EXPOSURE_TIME)
    prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(False)
    return True
