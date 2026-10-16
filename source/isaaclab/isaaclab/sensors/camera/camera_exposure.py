# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author explicitly configured exposure before renderer stage export."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .camera_exposure_cfg import CameraExposureCfg

_CAMERA_EXPOSURE_SCHEMAS = ("OmniRtxCameraAutoExposureAPI_1", "OmniRtxCameraExposureAPI_1")


def apply_camera_exposure(stage: Any, prim_paths: list[str], cfg: CameraExposureCfg) -> None:
    """Apply camera exposure independently of requested render outputs.

    Existing API schemas are preserved. Schema metadata is authored directly so the RTX schema
    plugin does not need to be loaded yet.

    Args:
        stage: USD stage containing the camera prims.
        prim_paths: Resolved camera prim paths.
        cfg: Explicit exposure settings shared by these camera prims.
    """
    from pxr import Sdf

    settings = (
        ("exposure", "Float", cfg.exposure),
        ("exposure:fStop", "Float", cfg.f_stop),
        ("exposure:iso", "Float", cfg.iso),
        ("exposure:responsivity", "Float", cfg.responsivity),
        ("exposure:time", "Float", cfg.time),
        ("omni:rtx:autoExposure:enabled", "Bool", cfg.auto_exposure),
    )
    value_types = {"Bool": Sdf.ValueTypeNames.Bool, "Float": Sdf.ValueTypeNames.Float}
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        current = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
        existing = list(current.prependedItems) + list(current.explicitItems) + list(current.appendedItems)
        missing = [schema for schema in _CAMERA_EXPOSURE_SCHEMAS if schema not in existing]
        if missing:
            prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=[*missing, *existing]))
        for attr_name, sdf_type_name, value in settings:
            attr = prim.GetAttribute(attr_name)
            if not attr:
                attr = prim.CreateAttribute(attr_name, value_types[sdf_type_name], custom=False)
            attr.Set(value)
