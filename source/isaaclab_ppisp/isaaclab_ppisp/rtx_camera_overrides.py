# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX-specific per-camera-prim USD overrides authored when PPISP is active.

Kept in a Kit-free module (no ``omni.*`` imports) so it can be unit-tested
without booting Isaac Sim. Both Kit-driven Isaac RTX and Kit-less OVRTX embed
the same RTX core and author the same exposure schema, so the helper lives
here in :mod:`isaaclab_ppisp` and is called from both backends'
``prepare_cameras`` hook.
"""

from __future__ import annotations

from typing import Any

# Neutral RTX-side values authored on the camera prim when PPISP is the ISP
# authority. RTX otherwise compounds its own physical-camera exposure model
# on top of PPISP, leaving the HDR AOV under- or over-exposed. PPISP supplies
# its own ``responsivity`` / ``exposureOffset``; pinning these stage-side
# values (``iso=0`` plus ``autoExposure:enabled=False``) keeps RTX's exposure
# stage a no-op.
#
# Each entry is ``(attribute_name, sdf_type_name, value)`` where
# ``sdf_type_name`` is the suffix of ``Sdf.ValueTypeNames``; values are
# coerced inside :func:`apply_rtx_exposure_overrides`.
_NEUTRAL_CAMERA_EXPOSURE: tuple[tuple[str, str, Any], ...] = (
    ("exposure", "Float", 0.0),
    ("exposure:fStop", "Float", 1.0),
    ("exposure:iso", "Float", 0.0),
    ("exposure:responsivity", "Float", 1.0),
    ("exposure:time", "Float", 1.0),
    ("omni:rtx:autoExposure:enabled", "Bool", False),
)

# API schemas applied (prepended) on each camera prim so Kit's USD watcher
# routes the camera's ``exposure:*`` and ``omni:rtx:autoExposure:*``
# attributes into the per-camera carb settings RTX consumes.
_PPISP_CAMERA_API_SCHEMAS: tuple[str, ...] = (
    "OmniRtxCameraAutoExposureAPI_1",
    "OmniRtxCameraExposureAPI_1",
)


def apply_rtx_exposure_overrides(stage: Any, prim_paths: list[str]) -> None:
    """Pin RTX-side exposure to neutral and apply the API schemas on each camera prim.

    On every prim in ``prim_paths``:

    1. Author the neutral values from :data:`_NEUTRAL_CAMERA_EXPOSURE` —
       ``exposure:*`` knobs disabling RTX's physical-camera exposure stage
       plus ``omni:rtx:autoExposure:enabled=False``.
    2. Apply the :data:`_PPISP_CAMERA_API_SCHEMAS` API schemas via
       ``SetMetadata("apiSchemas", ...)``. In Kit, ``omni.kit``'s USD
       watcher pumps the authored attributes into the per-camera carb
       settings RTX consumes. ``Sdf.TokenListOp`` metadata is used rather
       than ``Usd.Prim.ApplyAPI`` so this does not require the USD
       ``PlugRegistry`` to have discovered the RTX-settings schema plugin
       yet.

    Existing API schemas on the prim are preserved — the RTX schemas are
    prepended, not replaced.

    Args:
        stage: USD stage containing the camera prim(s).
        prim_paths: Resolved camera prim paths (typically one per env).
    """
    from pxr import Sdf

    sdf_value_types = {
        "Bool": Sdf.ValueTypeNames.Bool,
        "Float": Sdf.ValueTypeNames.Float,
    }

    def coerce(value: Any, sdf_type_name: str) -> Any:
        if sdf_type_name == "Float":
            return float(value)
        return value

    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        current = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
        existing = list(current.prependedItems) + list(current.explicitItems) + list(current.appendedItems)
        missing = [s for s in _PPISP_CAMERA_API_SCHEMAS if s not in existing]
        if missing:
            merged = Sdf.TokenListOp.Create(prependedItems=[*missing, *existing])
            prim.SetMetadata("apiSchemas", merged)
        for attr_name, sdf_type_name, value in _NEUTRAL_CAMERA_EXPOSURE:
            attr = prim.GetAttribute(attr_name)
            if not attr:
                attr = prim.CreateAttribute(attr_name, sdf_value_types[sdf_type_name], custom=False)
            attr.Set(coerce(value, sdf_type_name))
