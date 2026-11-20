# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX camera contracts that need no renderer: construction, lens textures and their USD schema, and capture."""

import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_fovi.sensors import foveated_rtx_camera, lens_lut
from isaaclab_fovi.sensors.foveated_camera_cfg import FoveatedCameraCfg
from isaaclab_fovi.sensors.foveated_rtx_camera import FoveatedRTXCamera
from isaaclab_fovi.sensors.lens_lut import apply_lens_lut_to_prim, build_lut_textures
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
        # An OVRTX camera with distance_to_camera output adds each camera's near clip.
        _offset_distance=True,
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

    build_directions = lens_lut.native_grid_directions
    ray_builds = []

    def measure_directions(*args, **kwargs):
        rays = build_directions(*args, **kwargs)
        ray_builds.append(rays)
        return rays

    texture_paths = ("/textures/enter.exr", "/textures/exit.exr")
    monkeypatch.setattr(Camera, "__init__", construct)
    # Count at both import locations so evaluating a second ray grid in the camera fails this check.
    monkeypatch.setattr(lens_lut, "native_grid_directions", measure_directions)
    monkeypatch.setattr(foveated_rtx_camera, "native_grid_directions", measure_directions, raising=False)
    monkeypatch.setattr(lens_lut, "save_lut_textures", lambda *args: texture_paths)
    monkeypatch.setattr(FoveatedRTXCamera, "__del__", lambda self, *args, **kwargs: None)
    cfg = FoveatedCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        spawn=PinholeCameraCfg(),
        renderer_cfg=OVRTXRendererCfg(),
        data_types=["rgb"],
        foveated_res=16,
    )
    assert cfg.spawn.lock_camera
    camera = FoveatedRTXCamera(cfg)
    for prim in prims:
        assert prim.GetAttribute("omni:kit:cameraLock").Get() is False
        assert prim.GetAttribute("omni:lensdistortion:model").Get() == "lut"
        assert prim.GetAttribute("omni:lensdistortion:lut:rayEnterDirectionTexture").Get().path == "/textures/enter.exr"
    assert camera.cfg is not cfg and (camera.cfg.height, camera.cfg.width) == (16, 16)
    assert len(ray_builds) == 1
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


def test_sample_set_lens_textures_place_samples_and_mark_rays_outside_the_field() -> None:
    # A sample set with one sample per texel, in shuffled order, puts each sample's ray at its own texel.
    size = 4
    axis = (torch.arange(size) + 0.5) / size
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    centres = torch.stack((col, row), -1).reshape(-1, 2)
    order = torch.randperm(size * size, generator=torch.Generator().manual_seed(0))
    directions = torch.nn.functional.normalize(torch.randn(size * size, 3, generator=torch.Generator().manual_seed(1)))
    direction_texture, _ = build_lut_textures(centres[order], directions, size, size, 16, 16, device="cpu")
    texels = (centres[order] * size).long()
    torch.testing.assert_close(direction_texture[texels[:, 1], texels[:, 0]], directions)

    # The inverse lookup sends each in-field ray to its nearest sample, and every other ray to the invalid UV, never
    # to the fovea. Three samples: the optical axis (OpenGL -z) and two rays 10 degrees off it.
    angle = math.radians(10.0)
    directions = torch.tensor(
        [[0.0, 0.0, -1.0], [math.sin(angle), 0.0, -math.cos(angle)], [0.0, math.sin(angle), -math.cos(angle)]]
    )
    uv = torch.tensor([[0.5, 0.5], [0.9, 0.5], [0.5, 0.9]])
    _, exit_texture = build_lut_textures(uv, directions, 8, 8, 32, 32, device="cpu")
    texels = exit_texture.reshape(-1, exit_texture.shape[-1])[:, :2]
    invalid = (texels == -1.0).all(dim=-1)
    # Rays outside the 10 degree field outnumber those inside it on the hemisphere's octahedral grid.
    assert invalid.float().mean() > 0.5 and (~invalid).any()
    assert all(any(torch.equal(texel, sample) for sample in uv) for texel in texels[~invalid])
    # The octahedral centre is the optical axis, which reaches the axis sample.
    torch.testing.assert_close(exit_texture[exit_texture.shape[0] // 2, exit_texture.shape[1] // 2, :2], uv[0])
