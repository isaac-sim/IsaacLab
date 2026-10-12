# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The deprecated PPISP entry point to the RTX camera exposure overrides."""

import importlib

import pytest

from pxr import Sdf, Usd, UsdGeom


@pytest.mark.parametrize("module", ["isaaclab_ppisp", "isaaclab_ppisp.rtx_camera_overrides"])
def test_the_deprecated_exposure_override_warns_and_still_neutralizes_exposure(module):
    """Callers of the former public helper keep working from both import paths, with a warning to remove the call."""
    apply = importlib.import_module(module).apply_rtx_exposure_overrides
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Camera.Define(stage, "/World/Camera").GetPrim()
    prim.CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float).Set(800.0)

    with pytest.warns(DeprecationWarning, match="renderers apply it themselves"):
        apply(stage, ["/World/Camera"])

    assert prim.GetAttribute("exposure:iso").Get() == 0.0
    assert prim.GetAttribute("omni:rtx:autoExposure:enabled").Get() is False
