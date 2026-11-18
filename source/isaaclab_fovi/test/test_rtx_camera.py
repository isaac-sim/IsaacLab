# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX camera contracts that need no renderer."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_fovi.sensors.foveated_rtx_camera import FoveatedRTXCamera

from isaaclab.sensors.camera.camera import Camera

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


@pytest.mark.parametrize("native_grid", [False, True])
def test_partial_capture_extracts_only_the_updated_environments(native_grid) -> None:
    """Environments outside the update mask keep their samples, and OVRTX's near-plane offset is added once."""
    texture = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4, 1)
    camera = SimpleNamespace(
        cfg=SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="ovrtx")),
        _native_grid=native_grid,
        _near_clip=0.25,
        _texture_output={"distance_to_camera": texture},
        _foveated_output={"distance_to_camera": torch.full((2, 16, 1, 1), -1.0)},
        _extract_py=torch.arange(16) // 4,
        _extract_px=torch.arange(16) % 4,
    )
    updated = wp.array([True, False], dtype=wp.bool, device="cpu")
    for _ in range(2):
        FoveatedRTXCamera._extract_foveated_samples(camera, updated)
    samples = camera._foveated_output["distance_to_camera"]
    torch.testing.assert_close(samples[0, :, 0, 0], texture[0].flatten() + 0.25)
    assert bool((samples[1] == -1.0).all())


def test_releasing_the_camera_removes_its_lens_textures(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(Camera, "__del__", lambda self, *args, **kwargs: None)
    camera = object.__new__(FoveatedRTXCamera)
    camera._lut_dir = tempfile.TemporaryDirectory(dir=tmp_path)
    directory = Path(camera._lut_dir.name)
    (directory / "foveated_ray_enter_direction.exr").write_bytes(b"")
    camera.__del__()
    assert not directory.exists() and camera._lut_dir is None


def test_construction_unlocks_spawned_cameras_without_changing_the_config(monkeypatch) -> None:
    """The gaze moves every foveated camera, so RTX must not treat a spawned camera as static."""
    pxr = pytest.importorskip("pxr")
    stage = pxr.Usd.Stage.CreateInMemory()
    prim = pxr.UsdGeom.Camera.Define(stage, "/World/envs/env_0/Camera").GetPrim()
    prim.CreateAttribute("omni:kit:cameraLock", pxr.Sdf.ValueTypeNames.Bool).Set(True)

    def construct(self, cfg):
        self.cfg = cfg
        self.stage = stage

    monkeypatch.setattr(Camera, "__init__", construct)
    monkeypatch.setattr(FoveatedRTXCamera, "__del__", lambda self, *args, **kwargs: None)
    spawn = SimpleNamespace(lock_camera=True)
    cfg = SimpleNamespace(
        prim_path="/World/envs/env_.*/Camera",
        spawn=spawn,
        mask_to_sensor_fov=False,
        renderer_cfg=SimpleNamespace(renderer_type="isaac_rtx"),
    )
    FoveatedRTXCamera(cfg)
    assert prim.GetAttribute("omni:kit:cameraLock").Get() is False
    assert cfg.spawn is spawn and spawn.lock_camera
