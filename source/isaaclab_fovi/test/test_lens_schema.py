# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native LUT metadata and assets survive USD export without schema registration."""

import pytest

Usd = pytest.importorskip("pxr.Usd")
from isaaclab_fovi.sensors.lens_lut import (  # noqa: E402
    apply_lens_lut_to_prim,
)

from pxr import UsdGeom  # noqa: E402

pytestmark = pytest.mark.kitless


def test_lut_schema_has_explicit_model_and_assets_after_export() -> None:
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Camera.Define(stage, "/Camera").GetPrim()
    apply_lens_lut_to_prim(prim, "/tmp/direction.exr", "/tmp/exit.exr")
    restored = Usd.Stage.CreateInMemory()
    restored.GetRootLayer().ImportFromString(stage.GetRootLayer().ExportToString())
    prim = restored.GetPrimAtPath("/Camera")
    assert "OmniLensDistortionLutAPI" in prim.GetMetadata("apiSchemas").GetAppliedItems()
    assert prim.GetAttribute("omni:lensdistortion:model").Get() == "lut"
    assert prim.GetAttribute("omni:lensdistortion:lut:rayEnterDirectionTexture").Get().path == "/tmp/direction.exr"
    assert prim.GetAttribute("omni:lensdistortion:lut:rayExitPositionTexture").Get().path == "/tmp/exit.exr"
    for name in (
        "cameraProjectionType",
        "generalizedProjectionDirectionTexturePath",
        "generalizedProjectionNDCTexturePath",
    ):
        assert not prim.HasProperty(name), name
