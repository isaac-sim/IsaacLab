# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the RTX-side camera-exposure overrides applied when PPISP is active."""

from isaaclab_ppisp import apply_rtx_exposure_overrides

from pxr import Sdf, Usd, UsdGeom


def test_apply_rtx_exposure_overrides_resets_authored_exposure_values():
    """When PPISP is the ISP authority, the camera prim's RTX-side exposure must be neutral."""
    stage = Usd.Stage.CreateInMemory()
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    prim = cam.GetPrim()
    # Author non-neutral values to confirm the helper overrides them.
    prim.CreateAttribute("exposure", Sdf.ValueTypeNames.Float).Set(2.0)
    prim.CreateAttribute("exposure:fStop", Sdf.ValueTypeNames.Float).Set(2.8)
    prim.CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float).Set(800.0)
    prim.CreateAttribute("exposure:responsivity", Sdf.ValueTypeNames.Float).Set(0.5)
    prim.CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(1.0 / 60.0)
    prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(True)

    apply_rtx_exposure_overrides(stage, ["/World/Camera"])

    assert prim.GetAttribute("exposure").Get() == 0.0
    assert prim.GetAttribute("exposure:fStop").Get() == 1.0
    assert prim.GetAttribute("exposure:iso").Get() == 0.0
    assert prim.GetAttribute("exposure:responsivity").Get() == 1.0
    assert prim.GetAttribute("exposure:time").Get() == 1.0
    assert prim.GetAttribute("omni:rtx:autoExposure:enabled").Get() is False
    # The PPISP API schemas must also be applied — that is what makes Kit's
    # USD watcher route the camera prim's ``exposure:*`` and
    # ``omni:rtx:autoExposure:*`` attributes into the per-camera carb
    # settings RTX consumes.
    api_schemas = prim.GetMetadata("apiSchemas")
    assert api_schemas is not None
    applied = list(api_schemas.prependedItems) + list(api_schemas.explicitItems) + list(api_schemas.appendedItems)
    assert "OmniRtxCameraAutoExposureAPI_1" in applied
    assert "OmniRtxCameraExposureAPI_1" in applied
