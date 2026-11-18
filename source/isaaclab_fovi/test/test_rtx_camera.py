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
from isaaclab_fovi.sensors.foveated_camera_cfg import FoveatedCameraCfg
from isaaclab_fovi.sensors.foveated_rtx_camera import FoveatedRTXCamera
from isaaclab_ov.renderers import OVRTXRendererCfg

from isaaclab.sensors.camera.camera import Camera
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.warp import ProxyArray

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
    """Environments outside the update mask keep their samples, and each camera's near-plane offset is added once."""
    height, width = 4, 5
    texture = torch.arange(3 * height * width, dtype=torch.float32).reshape(3, height, width, 1)
    if native_grid:
        texels = [(py, px) for py in range(height) for px in range(width)]
    else:
        texels = [(3, 1), (0, 4), (2, 0), (1, 3), (3, 4), (0, 2), (2, 2)]
    camera = SimpleNamespace(
        cfg=SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="ovrtx")),
        _native_grid=native_grid,
        _near_clip=torch.tensor([0.25, 0.5, 0.75]).view(-1, 1, 1, 1),
        # The renderer-bound texture buffers are ProxyArrays, as the parent camera allocates them.
        _texture_output={"distance_to_camera": ProxyArray(wp.from_torch(texture))},
        _foveated_output={"distance_to_camera": torch.full((3, len(texels), 1, 1), -1.0)},
        _extract_py=torch.tensor([py for py, _ in texels]),
        _extract_px=torch.tensor([px for _, px in texels]),
    )
    updated = wp.array([True, False, True], dtype=wp.bool, device="cpu")
    for _ in range(2):
        FoveatedRTXCamera._extract_foveated_samples(camera, updated)
    samples = camera._foveated_output["distance_to_camera"]
    for env, near_clip in ((0, 0.25), (2, 0.75)):
        expected = torch.tensor([float(texture[env, py, px, 0]) + near_clip for py, px in texels])
        torch.testing.assert_close(samples[env, :, 0, 0], expected, atol=0, rtol=0)
    assert bool((samples[1] == -1.0).all())


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
        self.cfg.height = self.cfg.width = 64
        self._dir_tex_path, self._exit_tex_path = "/textures/enter.exr", "/textures/exit.exr"

    monkeypatch.setattr(Camera, "__init__", construct)
    monkeypatch.setattr(FoveatedRTXCamera, "_prepare_projection", prepare)
    monkeypatch.setattr(FoveatedRTXCamera, "__del__", lambda self, *args, **kwargs: None)
    cfg = FoveatedCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        spawn=PinholeCameraCfg(),
        renderer_cfg=OVRTXRendererCfg(),
        data_types=["rgb"],
    )
    assert cfg.spawn.lock_camera
    camera = FoveatedRTXCamera(cfg)
    for prim in prims:
        assert prim.GetAttribute("omni:kit:cameraLock").Get() is False
        assert prim.GetAttribute("omni:lensdistortion:model").Get() == "lut"
        assert prim.GetAttribute("omni:lensdistortion:lut:rayEnterDirectionTexture").Get().path == "/textures/enter.exr"
    assert camera.cfg is not cfg and camera.cfg.height == 64
    assert cfg.spawn.lock_camera and (cfg.height, cfg.width) == (1, 1)
