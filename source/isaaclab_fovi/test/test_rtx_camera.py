# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX camera contracts that need no renderer: construction, lens textures and their USD schema, and capture."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from isaaclab_fovi.sensors.foveated_camera_cfg import FoveatedCameraCfg
from isaaclab_fovi.sensors.foveated_rtx_camera import FoveatedRTXCamera
from isaaclab_fovi.sensors.lens_lut import apply_lens_lut_to_prim
from isaaclab_ov.renderers import OVRTXRendererCfg

from isaaclab.sensors.camera.camera import Camera
from isaaclab.sim import PinholeCameraCfg

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


def test_releasing_the_camera_removes_its_lens_textures(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(Camera, "__del__", lambda self, *args, **kwargs: None)
    camera = object.__new__(FoveatedRTXCamera)
    camera._lut_dir = tempfile.TemporaryDirectory(dir=tmp_path)
    directory = Path(camera._lut_dir.name)
    (directory / "foveated_ray_enter_direction.exr").write_bytes(b"")
    camera.__del__()
    assert not directory.exists() and camera._lut_dir is None


def test_construction_authors_the_lens_and_unlocks_cameras_without_changing_the_config(monkeypatch):
    """Construction authors the lens lookup on every camera prim and unlocks it, since the gaze moves every foveated
    camera; the caller's configuration, spawn and texture size included, stays as it was."""
    pxr = pytest.importorskip("pxr")
    stage = pxr.Usd.Stage.CreateInMemory()
    prims = [pxr.UsdGeom.Camera.Define(stage, f"/World/envs/env_{i}/Camera").GetPrim() for i in range(2)]
    for prim in prims:
        prim.CreateAttribute("omni:kit:cameraLock", pxr.Sdf.ValueTypeNames.Bool).Set(True)

    def construct(self, cfg):
        self.cfg = cfg
        self.stage = stage

    def prepare(self):
        self._dir_tex_path, self._exit_tex_path = "/textures/enter.exr", "/textures/exit.exr"

    monkeypatch.setattr(Camera, "__init__", construct)
    monkeypatch.setattr(FoveatedRTXCamera, "_prepare_projection", prepare)
    monkeypatch.setattr(FoveatedRTXCamera, "__del__", lambda self, *args, **kwargs: None)
    cfg = FoveatedCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        spawn=PinholeCameraCfg(),
        renderer_cfg=OVRTXRendererCfg(),
        data_types=["rgb"],
        foveated_res=64,
    )
    assert cfg.spawn.lock_camera
    camera = FoveatedRTXCamera(cfg)
    for prim in prims:
        assert prim.GetAttribute("omni:kit:cameraLock").Get() is False
        assert prim.GetAttribute("omni:lensdistortion:model").Get() == "lut"
        assert prim.GetAttribute("omni:lensdistortion:lut:rayEnterDirectionTexture").Get().path == "/textures/enter.exr"
    assert camera.cfg is not cfg and camera.cfg.height == 64
    assert cfg.spawn.lock_camera and (cfg.height, cfg.width) == (1, 1)


def test_lens_lut_survives_usd_export_without_schema_registration() -> None:
    pxr = pytest.importorskip("pxr")
    stage = pxr.Usd.Stage.CreateInMemory()
    prim = pxr.UsdGeom.Camera.Define(stage, "/Camera").GetPrim()
    apply_lens_lut_to_prim(prim, "/tmp/direction.exr", "/tmp/exit.exr")
    restored = pxr.Usd.Stage.CreateInMemory()
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
