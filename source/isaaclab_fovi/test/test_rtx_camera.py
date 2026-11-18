# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX camera configuration contracts that need no renderer."""

from types import SimpleNamespace

import pytest
from isaaclab_fovi.sensors.foveated_rtx_camera import FoveatedRTXCamera

pytestmark = pytest.mark.kitless


def test_async_ovrtx_with_the_sensor_mask_is_rejected(monkeypatch) -> None:
    # The camera never initializes, so its destructor has nothing to release.
    monkeypatch.setattr(FoveatedRTXCamera, "__del__", lambda self: None, raising=False)
    renderer = SimpleNamespace(renderer_type="ovrtx", async_rendering=True)
    with pytest.raises(ValueError, match="async_rendering"):
        FoveatedRTXCamera(SimpleNamespace(renderer_cfg=renderer, mask_to_sensor_fov=True))


def test_near_clip_comes_from_spawn_or_else_the_camera_prim() -> None:
    pxr = pytest.importorskip("pxr")
    stage = pxr.Usd.Stage.CreateInMemory()
    camera = pxr.UsdGeom.Camera.Define(stage, "/Camera")
    camera.GetClippingRangeAttr().Set(pxr.Gf.Vec2f(0.05, 100.0))
    prim = camera.GetPrim()

    existing = SimpleNamespace(cfg=SimpleNamespace(spawn=None))
    assert FoveatedRTXCamera._near_clipping_distance(existing, prim) == pytest.approx(0.05)
    spawned = SimpleNamespace(cfg=SimpleNamespace(spawn=SimpleNamespace(clipping_range=(0.01, 20.0))))
    assert FoveatedRTXCamera._near_clipping_distance(spawned, prim) == pytest.approx(0.01)
