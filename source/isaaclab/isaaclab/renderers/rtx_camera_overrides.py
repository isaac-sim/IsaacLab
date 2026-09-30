# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared RTX camera exposure overrides for scene-linear sensor inputs."""

from __future__ import annotations

from typing import Any

# Both RTX backends consume these schemas without requiring the schema plugin at authoring time.
_CAMERA_EXPOSURE_SCHEMAS = ("OmniRtxCameraAutoExposureAPI_1", "OmniRtxCameraExposureAPI_1")
_NEUTRAL_CAMERA_EXPOSURE: tuple[tuple[str, str, Any], ...] = (
    ("exposure", "Float", 0.0),
    ("exposure:fStop", "Float", 1.0),
    ("exposure:iso", "Float", 0.0),
    ("exposure:responsivity", "Float", 1.0),
    ("exposure:time", "Float", 1.0),
    ("omni:rtx:autoExposure:enabled", "Bool", False),
)


def apply_rtx_exposure_overrides(stage: Any, prim_paths: list[str]) -> None:
    """Disable RTX camera exposure so downstream processors receive scene-linear HDR.

    Existing API schemas are preserved. Schema metadata is authored directly so the RTX schema
    plugin does not need to be loaded yet.

    Args:
        stage: USD stage containing the camera prims.
        prim_paths: Resolved camera prim paths.
    """
    from pxr import Sdf

    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        current = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
        existing = list(current.prependedItems) + list(current.explicitItems) + list(current.appendedItems)
        missing = [schema for schema in _CAMERA_EXPOSURE_SCHEMAS if schema not in existing]
        if missing:
            prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=[*missing, *existing]))
        for attr_name, sdf_type_name, value in _NEUTRAL_CAMERA_EXPOSURE:
            attr = prim.GetAttribute(attr_name)
            if not attr:
                attr = prim.CreateAttribute(attr_name, getattr(Sdf.ValueTypeNames, sdf_type_name), custom=False)
            attr.Set(value)
