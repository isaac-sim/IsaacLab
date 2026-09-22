# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX renderer output contract."""

import contextlib
import importlib.util
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg, SimulationContext

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers import OVRTXBackendCfg, OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_compat import RENDER_VAR_FRAME_KEYS  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import (  # noqa: E402
        _DISABLE_LINUX_CUDA_CPU_SYNC_ENV,
        OVRTXBackend,
        OVRTXCameraRenderData,
        OVRTXRenderer,
        _gpu_side_render_var_sync_enabled,
        ovrtx_use_ovstage_enabled,
    )
else:
    OVRTXCameraRenderData = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None
    ovrtx_use_ovstage_enabled = None
    _DISABLE_LINUX_CUDA_CPU_SYNC_ENV = None
    _gpu_side_render_var_sync_enabled = None
    RENDER_VAR_FRAME_KEYS = None

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=data_types,
    )


def _make_ovrtx_camera_render_data() -> OVRTXCameraRenderData:
    rd = OVRTXCameraRenderData.__new__(OVRTXCameraRenderData)
    rd.render_product_path = None
    rd.camera_xform_binding = None
    rd.camera_xform_query = None
    rd.resources = contextlib.ExitStack()
    rd.width = 16
    rd.height = 8
    rd.num_envs = 2
    rd.warp_buffers = {}
    rd.intrinsic_bindings = []
    rd.renderer_info = {}
    rd.ppisp_pipeline = None
    return rd


def _make_ovrtx_renderer_without_backend() -> OVRTXRenderer:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer.backend = OVRTXBackend.__new__(OVRTXBackend)
    cfg = OVRTXBackendCfg(renderer_cfg=renderer.cfg, use_ovstage=False, read_gpu_transforms=True)
    renderer.backend.stage = renderer.backend.paths = None
    renderer.backend._resources = contextlib.ExitStack()
    SimulationContext.instance()._backend_registry.append((cfg, renderer.backend))
    renderer._camera_render_data = []
    return renderer


@pytest.fixture(autouse=True)
def _simulation_registry(monkeypatch):
    sim = types.SimpleNamespace(_backend_registry=[])
    sim.get_or_create_backend = SimulationContext.get_or_create_backend.__get__(sim)
    sim.close_backend = SimulationContext.close_backend.__get__(sim)
    monkeypatch.setattr(SimulationContext, "_instance", sim)


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_ovrtx_renderer_config_enables_supported_runtime_options(monkeypatch: pytest.MonkeyPatch, use_ovstage):
    """Equal cfgs share one native resource; closing borrowers leaves it owned by the registry."""
    config_kwargs: dict[str, object] = {}
    destroyed = []

    class RecordingRendererConfig:
        def __init__(self, **kwargs):
            config_kwargs.update(kwargs)

    monkeypatch.setattr(ovrtx_renderer_module, "RendererConfig", RecordingRendererConfig)
    monkeypatch.setattr(
        ovrtx_renderer_module, "Renderer", lambda cfg: types.SimpleNamespace(destroy=lambda: destroyed.append(cfg))
    )
    monkeypatch.setattr(ovrtx_renderer_module, "ovrtx_use_ovstage_enabled", lambda: use_ovstage)
    monkeypatch.setattr(ovrtx_renderer_module, "create_ovstage", lambda _name: contextlib.nullcontext(object()))
    monkeypatch.setattr(ovrtx_renderer_module.ovstage, "PathDictionary", lambda _: contextlib.nullcontext(object()))

    renderer = OVRTXRenderer(OVRTXRendererCfg())
    shared = OVRTXRenderer(renderer.cfg)

    assert not {"_backend", "_renderer", "_stage", "_stage_paths", "_ovstage_exit_stack"}.intersection(vars(renderer))
    assert shared.backend is renderer.backend
    assert renderer.backend.renderer is not None
    assert config_kwargs["suppress_deprecation_warnings"] is True
    assert config_kwargs["texture_streaming_mode"] is ovrtx_renderer_module.TextureStreamingMode.SYNCHRONOUS
    assert len(SimulationContext.instance()._backend_registry) == 1
    other = OVRTXRenderer(renderer.cfg.replace(enable_shadows=True))
    assert other.backend is not renderer.backend
    renderer.close()
    renderer.close()
    assert not destroyed
    assert shared.backend.renderer is not None
    shared.close()
    other.close()
    assert not destroyed
    SimulationContext.instance().close_backend(renderer.backend)
    SimulationContext.instance().close_backend(other.backend)
    assert len(destroyed) == 2
    assert not SimulationContext.instance()._backend_registry


def test_ovrtx_supported_output_types_key_set():
    """OVRTX publishes the documented key set and per-output spec."""
    renderer = _make_ovrtx_renderer_without_backend()
    specs = renderer.supported_output_types()

    assert specs == renderer.cfg.supported_output_types()
    assert set(specs.keys()) == {
        RenderBufferKind.RGB,
        RenderBufferKind.RGBA,
        RenderBufferKind.RGB_HDR,
        RenderBufferKind.ALBEDO,
        RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE,
        RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL,
        RenderBufferKind.SIMPLE_SHADING_FULL_MDL,
        RenderBufferKind.SEMANTIC_SEGMENTATION,
        RenderBufferKind.INSTANCE_SEGMENTATION,
        RenderBufferKind.DEPTH,
        RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
        RenderBufferKind.DISTANCE_TO_CAMERA,
        RenderBufferKind.NORMALS,
        RenderBufferKind.MOTION_VECTORS,
    }
    assert specs[RenderBufferKind.RGBA] == RenderBufferSpec(4, wp.uint8)
    assert specs[RenderBufferKind.RGB_HDR] == RenderBufferSpec(3, wp.float32)
    assert specs[RenderBufferKind.DEPTH] == RenderBufferSpec(1, wp.float32)
    assert specs[RenderBufferKind.MOTION_VECTORS] == RenderBufferSpec(2, wp.float32)


@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.parametrize("use_ovstage", [False, True])
def test_ovrtx_multiple_cameras_render_independent_views(monkeypatch, use_ovstage, tmp_path):
    """Check independent camera batches and save RGB/depth frames under pytest's temporary directory.

    Use ``--basetemp=/tmp/ovrtx-camera-frames`` to choose where pytest writes the captures.
    Depth PNGs use a shared 0-8 m range (near is white); NPY files retain the raw depths [m].
    """
    from isaaclab_newton.physics import NewtonManager
    from PIL import Image

    from pxr import Gf, Usd, UsdGeom, UsdLux

    from isaaclab.cloner.clone_plan import ClonePlan
    from isaaclab.renderers.camera_render_spec import CameraRenderSpec
    from isaaclab.utils.math import convert_camera_frame_orientation_convention
    from isaaclab.utils.warp import ProxyArray

    if not torch.cuda.is_available():
        pytest.skip("OVRTX rendering requires CUDA")
    # This static USD scene has no physics model or scene-data provider.
    monkeypatch.setattr(NewtonManager, "get_model", classmethod(lambda cls: None))
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", str(int(use_ovstage)))
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    if not use_ovstage:
        UsdGeom.Xform.Define(stage, "/World/envs/env_1")
    for index, x in enumerate((0.0, 2.0)):
        camera = UsdGeom.Camera.Define(stage, f"/World/envs/env_0/cam{index}")
        camera.CreateProjectionAttr("orthographic")
        camera.CreateHorizontalApertureAttr(20.0)
        camera.CreateVerticalApertureAttr(20.0)
        camera.AddTranslateOp().Set(Gf.Vec3d(x, 0, 5))
        cube = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/cube{index}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([(0.8, 0.1, 0.1) if index == 0 else (0.1, 0.8, 0.1)])
        cube.AddTranslateOp().Set(Gf.Vec3d(x, 0, index))

    renderer = OVRTXRenderer(OVRTXRendererCfg())
    renderer._exported_usd_string = stage.ExportToString()
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2, dtype=np.int64),
        positions=np.zeros((2, 3), dtype=np.float32),
    )
    cameras = []

    def camera_scope_exists(rd):
        scope = rd.render_product_path.rsplit("/", 1)[0] + "/"
        if use_ovstage:
            import ovstage

            camera_filter = ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.PREFIX, [scope])])
            with renderer.backend.stage.query(filter=camera_filter) as query:
                return query.result().total_prim_count > 0
        return any(path.startswith(scope) for path in renderer.backend.renderer.query_prims())

    def check_depth(rd, data, expected, label):
        renderer.render(rd)
        depth = data.output["distance_to_image_plane"].torch
        for env_id in range(rd.num_envs):
            prefix = tmp_path / f"{label}_env{env_id}"
            rgb = data.output["rgb"].torch[env_id].cpu().numpy()
            depth_m = depth[env_id, ..., 0].cpu().numpy()
            Image.fromarray(rgb).save(f"{prefix}_rgb.png")
            np.save(f"{prefix}_depth.npy", depth_m)
            preview = ((1.0 - np.clip(depth_m / 8.0, 0.0, 1.0)) * 255).astype(np.uint8)
            Image.fromarray(preview).save(f"{prefix}_depth.png")
        assert depth.shape == (2, rd.height, rd.width, 1)
        torch.testing.assert_close(
            depth[:, rd.height // 2, rd.width // 2, 0],
            torch.full((2,), expected, device="cuda:0"),
            atol=0.02,
            rtol=0,
        )

    try:
        for index, (height, width) in enumerate(((480, 640), (384, 512))):
            cfg = _make_camera_cfg(
                ["rgb", "distance_to_image_plane"] if index == 0 else ["rgb", "distance_to_image_plane", "normals"]
            )
            cfg.height, cfg.width = height, width
            spec = CameraRenderSpec(
                cfg=cfg,
                device="cuda:0",
                num_instances=2,
                camera_prim_paths=tuple(f"/World/envs/env_{i}/cam{index}" for i in range(2)),
                view_count=2,
                camera_path_relative_to_env_0=f"cam{index}",
            )
            rd = renderer.create_render_data(spec)
            data = CameraData.allocate(
                data_types=cfg.data_types,
                height=height,
                width=width,
                num_views=2,
                device="cuda:0",
                supported_specs=renderer.supported_output_types(),
            )
            renderer.set_outputs(rd, data.output)
            cameras.append((rd, data))
            # Register the next camera after rendering has already started.
            check_depth(rd, data, 5.0 - index - 0.5, f"initial_cam{index}")
            if index == 1:
                normals = data.output["normals"].torch[:, height // 2, width // 2, :3]
                torch.testing.assert_close(
                    torch.linalg.vector_norm(normals, dim=-1), torch.ones(2, device="cuda:0"), atol=0.02, rtol=0
                )

        # Move only cam1 farther away. cam0 must retain its original depth.
        positions = ProxyArray(wp.array([[2.0, 0, 7]] * 2, dtype=wp.vec3f, device="cuda:0"))
        quats = convert_camera_frame_orientation_convention(
            torch.tensor([[0.0, 0, 0, 1.0]] * 2, device="cuda:0"), origin="opengl", target="world"
        )
        orientations = ProxyArray(wp.from_torch(quats, dtype=wp.quatf))
        renderer.update_camera(cameras[1][0], positions, orientations, cameras[1][1].intrinsic_matrices)
        check_depth(*cameras[0], 4.5, "after_move_cam0")
        check_depth(*cameras[1], 5.5, "after_move_cam1")
        assert all(camera_scope_exists(rd) for rd, _ in cameras)
        renderer.cleanup(cameras[0][0])
        assert not camera_scope_exists(cameras[0][0])
        check_depth(*cameras[1], 5.5, "after_cleanup_cam1")
        renderer.cleanup(cameras[1][0])
        renderer.cleanup(cameras[1][0])
        assert not camera_scope_exists(cameras[1][0])
    finally:
        renderer.close()
        SimulationContext.instance().close_backend(renderer.backend)


def test_ovrtx_set_outputs_wraps_caller_torch_zero_copy():
    """OVRTXRenderer.set_outputs publishes warp views over the caller's warp storage."""
    renderer = _make_ovrtx_renderer_without_backend()

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    cfg = _make_camera_cfg(["rgb", "rgba", "depth"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device=device,
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_camera_render_data()
    renderer.set_outputs(render_data, data.output)

    assert set(render_data.warp_buffers.keys()) >= {"rgba", "depth"}
    assert render_data.warp_buffers["rgba"].ptr == data.output["rgba"].warp.ptr
    assert render_data.warp_buffers["depth"].ptr == data.output["depth"].warp.ptr
    assert "rgb" not in render_data.warp_buffers


def test_ovrtx_set_outputs_wraps_requested_rgb_hdr_output():
    """OVRTXRenderer.set_outputs publishes a zero-copy view for requested RGB_HDR."""
    renderer = _make_ovrtx_renderer_without_backend()

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    cfg = _make_camera_cfg(["rgb_hdr"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device=device,
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_camera_render_data()
    renderer.set_outputs(render_data, data.output)

    assert render_data.warp_buffers["rgb_hdr"].ptr == data.output["rgb_hdr"].warp.ptr


def test_ovrtx_set_outputs_routes_ppisp_buffers_through_warp_buffers():
    """OVRTXRenderer.set_outputs stores PPISP source/destination in warp_buffers."""
    renderer = _make_ovrtx_renderer_without_backend()

    cfg = _make_camera_cfg(["rgb"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device="cpu",
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_camera_render_data()
    render_data.ppisp_pipeline = object()
    renderer.set_outputs(render_data, data.output)

    assert render_data.warp_buffers["rgba"].ptr == data.output["rgba"].warp.ptr
    assert "rgb_hdr" in render_data.warp_buffers
    assert render_data.warp_buffers["rgb_hdr"].shape == (2, 8, 16, 3)
    assert render_data.warp_buffers["rgb_hdr"].dtype is wp.float32


def test_ovrtx_process_frame_skips_ldr_rgba_when_ppisp_is_active():
    """PPISP owns RGBA output, so OVRTX LdrColor should not pre-fill it."""

    class FailingRenderVar:
        def map(self, *args, **kwargs):
            raise AssertionError("PPISP RGBA output must not read OVRTX LdrColor")

    class Frame:
        render_vars = {RENDER_VAR_FRAME_KEYS["LdrColor"]: FailingRenderVar()}

    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_camera_render_data()
    render_data.ppisp_pipeline = object()

    renderer._process_render_frame(render_data, Frame(), {"rgba": object()})


@pytest.mark.parametrize("stale_key", ["LdrColor", "/Render/Vars/LdrColor"])
def test_ovrtx_process_frame_reads_only_the_installed_ldr_color_key(monkeypatch: pytest.MonkeyPatch, stale_key: str):
    """Frames are keyed by source name on OVRTX 0.4 and by prim path on 0.5; only one form is read."""
    installed_key = RENDER_VAR_FRAME_KEYS["LdrColor"]

    mapped = []

    @contextlib.contextmanager
    def fake_map(self, render_var):
        mapped.append(render_var)
        yield object()

    monkeypatch.setattr(OVRTXRenderer, "_map_render_var_to_dlpack", fake_map)
    monkeypatch.setattr(OVRTXRenderer, "_extract_rgba_tiles", lambda *args, **kwargs: None)

    class Frame:
        render_vars = {stale_key: "stale", installed_key: "installed"}

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._process_render_frame(_make_ovrtx_camera_render_data(), Frame(), {"rgba": object()})
    assert mapped == ["installed"]


def test_ovrtx_ppisp_hdr_source_is_cloned_to_output_device(monkeypatch):
    """PPISP HdrColor source is moved to the HDR output buffer device."""

    class FakeArray:
        device = "cuda:1"

    class OutputArray:
        device = "cuda:0"

    cloned = object()
    clone_calls = []

    def fake_clone(src, *, device):
        clone_calls.append((src, device))
        return cloned

    monkeypatch.setattr(wp, "clone", fake_clone)

    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_camera_render_data()
    render_data.ppisp_pipeline = object()
    source = FakeArray()

    assert renderer._prepare_ppisp_hdr_source(render_data, source, {"rgb_hdr": OutputArray()}) is cloned
    assert clone_calls == [(source, "cuda:0")]


class _FakeArray:
    def __init__(self, shape):
        self.shape = shape


def test_launch_extract_all_tiles_rejects_wider_output_channels():
    """An output wider than the tiled input would read out of bounds, so it must raise before launching."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_camera_render_data()

    with pytest.raises(ValueError, match="out of bounds"):
        renderer._launch_extract_all_tiles(render_data, _FakeArray((8, 16, 3)), _FakeArray((2, 8, 16, 4)))


def test_launch_extract_all_tiles_launches_kernel_when_channels_are_compatible(monkeypatch):
    """Equal or narrower output channel counts pass validation and reach the kernel launch."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_camera_render_data()
    render_data.num_cols = 2

    launch_calls = []
    monkeypatch.setattr(wp, "launch", lambda **kwargs: launch_calls.append(kwargs))

    tiled_buffer = _FakeArray((8, 16, 4))
    output_buffer = _FakeArray((2, 8, 16, 3))
    renderer._launch_extract_all_tiles(render_data, tiled_buffer, output_buffer)

    assert len(launch_calls) == 1
    assert launch_calls[0]["inputs"][:2] == [tiled_buffer, output_buffer]


# Tile counts spanning the small cases and the large ones a camera-per-environment training run reaches.
# 4096 and its neighbours are called out because that is where the renderer stops supplying a tile per
# camera, so the grid the extraction assumes has to be checked on both sides of it.
_TILE_COUNTS = [1, 2, 7, 1000, 4095, 4096, 4097, 8192, 16384]


def _make_render_spec(num_envs: int, tile: int, device: str):
    """Build a render spec for ``num_envs`` cameras of ``tile`` x ``tile`` px."""
    from isaaclab.renderers.camera_render_spec import CameraRenderSpec

    return CameraRenderSpec(
        cfg=CameraCfg(height=tile, width=tile, prim_path="/World/envs/env_.*/Camera", spawn=_SPAWN, data_types=["rgb"]),
        device=device,
        num_instances=num_envs,
        camera_prim_paths=tuple(f"/World/envs/env_{index}/Camera" for index in range(num_envs)),
        view_count=num_envs,
        camera_path_relative_to_env_0="Camera",
    )


@pytest.mark.parametrize("num_envs", _TILE_COUNTS)
def test_render_data_tile_grid_matches_the_authored_atlas(num_envs):
    """The extraction grid and the authored atlas are derived separately and must stay in agreement.

    :class:`OVRTXRenderData` computes the tile grid the extraction kernel indexes with, while
    :func:`build_render_product_as_string` independently sizes the atlas the renderer draws into. They
    are two copies of the same layout: if they ever disagree, every tile is read from the wrong offset
    even though both the render and the returned tensor keep their expected shapes.
    """
    import re

    from isaaclab_ov.renderers.ovrtx_usd import build_render_product_as_string

    if not torch.cuda.is_available():
        pytest.skip("OVRTX render data requires a CUDA device")
    device = "cuda"

    tile = 96
    render_data = OVRTXRenderData(_make_render_spec(num_envs, tile, device), device=device)
    render_product, _ = build_render_product_as_string(width=tile, height=tile, num_envs=num_envs, data_types=["rgb"])

    match = re.search(r"uniform int2 resolution = \((\d+), (\d+)\)", render_product)
    assert match is not None
    tiled_width, tiled_height = int(match.group(1)), int(match.group(2))

    assert render_data.num_cols == tiled_width // tile
    assert render_data.num_rows == tiled_height // tile
    assert render_data.num_cols * render_data.num_rows >= num_envs


@pytest.mark.skip(
    reason=(
        "_launch_extract_all_tiles validates only the channel count, so a tiled buffer holding fewer"
        " tiles than num_envs is reshaped into wrong per-environment images instead of raising."
        " Remove this skip once the extraction validates tile capacity."
    )
)
def test_launch_extract_all_tiles_rejects_a_tiled_buffer_missing_tiles():
    """A tiled buffer too small to hold every tile must raise rather than reshape into wrong images.

    This is the failure the >4096-tile bug produces: the renderer hands back a buffer covering only the
    tiles it actually drew, and the extraction happily indexes past it, so the trailing environments get
    another environment's pixels with no error anywhere.
    """
    if not torch.cuda.is_available():
        pytest.skip("OVRTX tile extraction requires a CUDA device")
    device = "cuda"

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = device
    render_data = _make_ovrtx_render_data()
    render_data.width = 96
    render_data.height = 96
    render_data.num_envs = 8192
    render_data.num_cols = 91

    # An atlas holding only 4096 tiles (64 x 64) while 8192 were requested.
    undersized = _FakeArray((64 * 96, 64 * 96, 4))
    output_buffer = _FakeArray((8192, 96, 96, 3))

    with pytest.raises(ValueError, match="out of bounds"):
        renderer._launch_extract_all_tiles(render_data, undersized, output_buffer)


def test_ovrtx_read_output_copies_no_pixel_data():
    """OVRTXRenderer.read_output copies no pixel data; with empty renderer_info it leaves info untouched."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_camera_render_data()
    camera_data = CameraData()
    camera_data.info = {}
    camera_data._output = {}

    result = renderer.read_output(render_data, camera_data)
    assert result is None
    assert render_data.warp_buffers == {}
    assert camera_data.info == {}
    assert camera_data.output == {}


def test_ovrtx_read_output_forwards_renderer_info():
    """OVRTXRenderer.read_output forwards render_data.renderer_info (e.g. semantic idToLabels) into info."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_camera_render_data()
    id_to_labels = {"2": {"class": "cartpole"}}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": id_to_labels}}

    camera_data = CameraData()
    camera_data.info = {"semantic_segmentation": None}
    camera_data._output = {}

    renderer.read_output(render_data, camera_data)
    assert camera_data.info["semantic_segmentation"] == {"idToLabels": id_to_labels}


def test_ovrtx_read_output_clears_stale_metadata_and_keeps_seeded_keys():
    """read_output replaces (not merges): a dropped render var resets its info entry, seeded keys persist."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_camera_render_data()

    # ``camera_data.info`` is seeded with one key per output (mirrors ``camera_data.output``); both start None.
    camera_data = CameraData()
    camera_data.info = {"rgb": None, "semantic_segmentation": None}
    camera_data._output = {}

    # Frame 1: the SemanticIdMap render var is present, so its metadata lands in info.
    id_to_labels = {"2": {"class": "cartpole"}}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": id_to_labels}}
    renderer.read_output(render_data, camera_data)
    assert camera_data.info["semantic_segmentation"] == {"idToLabels": id_to_labels}

    # Frame 2: render() rebuilds renderer_info from scratch and the SemanticIdMap is gone this frame.
    render_data.renderer_info = {}
    renderer.read_output(render_data, camera_data)

    # The stale idToLabels must be cleared, and the seeded keys (rgb, semantic_segmentation) must remain.
    assert camera_data.info == {"rgb": None, "semantic_segmentation": None}


def test_ovrtx_semantic_spec_follows_colorize_flag():
    """Semantic segmentation output spec is colorized RGBA (uint8) or raw int32 IDs per the cfg flag."""
    colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    colorized.cfg = OVRTXRendererCfg(colorize_semantic_segmentation=True)
    assert colorized.supported_output_types()[RenderBufferKind.SEMANTIC_SEGMENTATION] == RenderBufferSpec(4, wp.uint8)

    non_colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    non_colorized.cfg = OVRTXRendererCfg(colorize_semantic_segmentation=False)
    assert non_colorized.supported_output_types()[RenderBufferKind.SEMANTIC_SEGMENTATION] == RenderBufferSpec(
        1, wp.int32
    )


def test_ovrtx_instance_segmentation_spec_follows_colorize_flag():
    """Instance segmentation output spec is colorized RGBA (uint8) or raw int32 IDs per the cfg flag."""
    colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    colorized.cfg = OVRTXRendererCfg(colorize_instance_segmentation=True)
    assert colorized.supported_output_types()[RenderBufferKind.INSTANCE_SEGMENTATION] == RenderBufferSpec(4, wp.uint8)

    non_colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    non_colorized.cfg = OVRTXRendererCfg(colorize_instance_segmentation=False)
    assert non_colorized.supported_output_types()[RenderBufferKind.INSTANCE_SEGMENTATION] == RenderBufferSpec(
        1, wp.int32
    )


def test_ovrtx_use_ovstage_defaults_to_disabled(monkeypatch):
    """The ovstage path is off unless explicitly opted into, so existing deployments are unaffected."""
    monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
    assert ovrtx_use_ovstage_enabled() is False

    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "0")
    assert ovrtx_use_ovstage_enabled() is False


def test_ovrtx_use_ovstage_enabled_when_requested(monkeypatch):
    """Setting the variable to 1 selects the ovstage path."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    assert ovrtx_use_ovstage_enabled() is True


def test_ovrtx_use_ovstage_rejects_non_boolean_values(monkeypatch):
    """Values other than 0/1 are a configuration error, not a silent disable."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "true")

    with pytest.raises(ValueError, match="Expected 0 or 1"):
        ovrtx_use_ovstage_enabled()


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_ovrtx_render_var_sync_is_gpu_side_off_linux(monkeypatch, platform):
    """Everywhere but Linux the mapping is ordered by a GPU-side wait on the Warp stream."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, raising=False)
    assert _gpu_side_render_var_sync_enabled() is True


def test_ovrtx_render_var_sync_waits_on_host_on_linux(monkeypatch):
    """Linux blocks the calling thread instead, which measures faster there."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, raising=False)
    assert _gpu_side_render_var_sync_enabled() is False


def test_ovrtx_render_var_sync_is_gpu_side_on_linux_when_disabled(monkeypatch):
    """Opting out of the host wait puts Linux on the same GPU-side wait as every other platform."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, "1")
    assert _gpu_side_render_var_sync_enabled() is True


def test_ovrtx_render_var_sync_keeps_host_wait_when_explicitly_enabled(monkeypatch):
    """``0`` is the default, so setting it explicitly must not change anything."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, "0")
    assert _gpu_side_render_var_sync_enabled() is False


@pytest.mark.parametrize("value", ["", "true", "yes", "2"])
def test_ovrtx_render_var_sync_rejects_non_boolean_values(monkeypatch, value):
    """Values other than 0/1 are a configuration error, not a silent fallback to the host wait."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, value)
    with pytest.raises(ValueError, match="Expected 0 or 1"):
        _gpu_side_render_var_sync_enabled()


class _RecordingRenderVar:
    """Stand-in for an OVRTX ``RenderVarOutput`` that records how the read was ordered.

    Any of OVRTX's ordering mechanisms counts, so the test stays about *whether* the read is
    ordered rather than which call carries it.
    """

    def __init__(self):
        self.ordering: list[str] = []

    def map(self, *, device, sync_stream):
        if sync_stream:
            self.ordering.append("gpu")
        recorder = self

        class _Mapping:
            def wait(self):
                recorder.ordering.append("host")

            def wait_on(self, stream):
                recorder.ordering.append("gpu")

        return contextlib.nullcontext(_Mapping())


@pytest.mark.parametrize(("gpu_side", "expected"), [(True, "gpu"), (False, "host")])
def test_ovrtx_map_render_var_orders_the_read_against_render_completion(monkeypatch, gpu_side, expected):
    """The read is ordered exactly once -- by a GPU-side barrier or a host block, never by neither.

    Ordering by neither is a silent race on half-written render output rather than a failure, so
    this asserts which mechanism ran and not which API call carries it.
    """
    sentinel = object()
    render_var = _RecordingRenderVar()
    monkeypatch.setattr(ovrtx_renderer_module, "_gpu_side_render_var_sync_enabled", lambda: gpu_side)
    monkeypatch.setattr(ovrtx_renderer_module.wp, "from_dlpack", lambda mapping: sentinel)

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cuda:0"
    renderer._warp_device = types.SimpleNamespace(stream=types.SimpleNamespace(cuda_stream=99))
    with renderer._map_render_var_to_dlpack(render_var) as array:
        assert array is sentinel

    assert render_var.ordering == [expected]


@pytest.mark.parametrize("cleanup_directly", [False, True])
@pytest.mark.parametrize("use_ovstage", [False, True])
def test_ovrtx_cleanup_releases_only_the_given_render_data(cleanup_directly, use_ovstage):
    """Release one camera's pose and calibration resources once, keeping other cameras usable."""
    events = []
    renderer = (
        _make_ovstage_renderer_with_backend(events) if use_ovstage else _make_legacy_renderer_with_backend(events)
    )
    other_camera = renderer._camera_render_data[0]
    render_data = _make_ovrtx_camera_render_data()
    render_data.render_product_path = "/Render/RenderProduct_to_remove"
    renderer._render_product_paths.append(render_data.render_product_path)
    renderer._camera_render_data.append(render_data)
    render_data.warp_buffers = {"rgba": wp.zeros((8, 16, 4), dtype=wp.uint8, device="cpu")}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": {}}}
    render_data.ppisp_pipeline = object()
    if use_ovstage:
        render_data.camera_xform_query = "to_remove"
        render_data.resources.callback(renderer.backend.paths.destroy_path_list, "to_remove")
        render_data.resources.callback(lambda: renderer.backend.stage.release_query("to_remove").wait())
    else:
        render_data.camera_xform_binding = _RecordingBinding(events, "pose")
        render_data.resources.callback(render_data.camera_xform_binding.unbind)
        render_data.intrinsic_bindings = [_RecordingBinding(events, "intrinsics")]
        render_data.resources.callback(render_data.intrinsic_bindings[0].unbind)

    if cleanup_directly:
        render_data.cleanup()
    renderer.cleanup(render_data)
    renderer.cleanup(render_data)

    expected = (
        ["release_query:to_remove", "destroy_path_list:to_remove"]
        if use_ovstage
        else ["unbind:intrinsics", "unbind:pose"]
    )
    assert events == expected
    assert renderer._camera_render_data == [other_camera]
    assert render_data.camera_xform_binding is None
    assert render_data.camera_xform_query is None
    assert render_data.intrinsic_bindings == []
    assert render_data.warp_buffers == {}
    assert render_data.renderer_info == {}
    assert render_data.ppisp_pipeline is None
    assert renderer._render_product_paths == ["/Render/RenderProduct_camera"]
    assert renderer._initialized_scene is True


def test_ovrtx_cleanup_without_render_data_keeps_renderer_state():
    """``cleanup(None)`` has nothing to release and must not disturb the renderer."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._initialized_scene = True

    renderer.cleanup(None)

    assert renderer._render_product_paths == ["/Render/RenderProduct_camera"]
    assert renderer._initialized_scene is True


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_intrinsic_updates_target_the_given_camera(monkeypatch, use_ovstage):
    """Cameras sharing a renderer must bind and update distinct native camera paths."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._initialized_scene = True
    renderer._device = "cpu"
    renderer._next_camera_id = 0
    renderer._render_product_paths = []
    renderer._use_ovstage = use_ovstage
    renderer._current_ordinal = 1
    renderer.backend.renderer = MagicMock()
    renderer.backend.renderer.bind_attribute.side_effect = lambda **kwargs: MagicMock()
    renderer.backend.stage = MagicMock()
    renderer.backend.paths = MagicMock()
    renderer.backend.paths.create_path_list_from_strings.side_effect = tuple
    renderer.backend.stage.query_from_path_list.side_effect = lambda paths: contextlib.nullcontext(object())
    for name in ("add_usd_reference_from_string", "apply_usd_changes", "remove_usd"):
        monkeypatch.setattr(ovrtx_renderer_module.ovstage.population, name, MagicMock())
    paths = [[f"/World/envs/env_{i}/{name}" for i in range(2)] for name in ("CameraA", "CameraB")]
    cameras = [
        renderer.create_render_data(
            types.SimpleNamespace(
                cfg=_make_camera_cfg(["depth"]),
                device="cpu",
                num_instances=2,
                camera_prim_paths=camera_paths,
                camera_path_relative_to_env_0=camera_paths[0].rsplit("/", 1)[1],
            )
        )
        for camera_paths in paths
    ]
    monkeypatch.setattr(wp, "get_stream", lambda device: types.SimpleNamespace(cuda_stream=99))
    parameters = wp.zeros((5, 2), dtype=wp.float32, device="cpu")
    renderer.update_camera_intrinsics(cameras[1], wp.zeros(2, dtype=wp.mat33f, device="cpu"), parameters)
    # Keep native resources on the backend and reuse the camera-owned query for calibration.
    assert not {"_renderer", "_stage", "_stage_paths", "_camera_intrinsic_bindings"}.intersection(vars(renderer))
    assert all(not {"intrinsic_query", "intrinsic_paths"}.intersection(vars(camera)) for camera in cameras)
    if use_ovstage:
        assert renderer.backend.stage.write_attributes.call_args.args[0] is cameras[1].camera_xform_query
        bound_paths = [call.args[0] for call in renderer.backend.paths.create_path_list_from_strings.call_args_list]
        assert bound_paths == [[cameras[0].render_product_path], paths[0], [cameras[1].render_product_path], paths[1]]
    else:
        bound_paths = [
            call.kwargs["prim_paths"]
            for call in renderer.backend.renderer.bind_attribute.call_args_list
            if call.kwargs["attribute_name"] in ovrtx_renderer_module._CAMERA_INTRINSIC_ATTRIBUTES
        ]
        assert bound_paths == [paths[0]] * 5 + [paths[1]] * 5
        assert all(not binding.write_async.called for binding in cameras[0].intrinsic_bindings)
        for row, binding in enumerate(cameras[1].intrinsic_bindings):
            assert binding.write_async.call_args.args[0].ptr == parameters[row].ptr
        bindings = list(cameras[1].intrinsic_bindings)
        cameras[1].cleanup()
        cameras[1].cleanup()
        assert all(binding.unbind.call_count == 1 for binding in bindings)
        assert all(not binding.unbind.called for binding in cameras[0].intrinsic_bindings)


class _RecordingBinding:
    def __init__(self, events: list[str], name: str):
        self._events = events
        self._name = name

    def unbind(self) -> None:
        self._events.append(f"unbind:{self._name}")


def _make_legacy_renderer_with_backend(events: list[str]) -> OVRTXRenderer:
    """Build a legacy-path renderer whose backend calls are recorded into ``events``."""

    class Backend:
        def destroy(self) -> None:
            events.append("destroy_renderer")

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = False
    render_data = _make_ovrtx_camera_render_data()
    render_data.render_product_path = "/Render/RenderProduct_camera"
    render_data.camera_xform_binding = _RecordingBinding(events, "camera")
    render_data.resources.callback(render_data.camera_xform_binding.unbind)
    render_data.renderer_info = {"rgb": object()}
    renderer._camera_render_data.append(render_data)
    renderer._camera_xform_binding = None
    renderer._object_xform_binding = _RecordingBinding(events, "object")
    renderer._deformable_points_binding = _RecordingBinding(events, "deformable")
    renderer._particle_points_binding = _RecordingBinding(events, "particle")
    renderer._cable_points_binding = _RecordingBinding(events, "cable")
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [1]
    renderer._particle_visual_offsets = [0]
    renderer._particle_visual_counts = [1]
    renderer._particle_workaround_applied = True
    renderer._cable_segment_counts = [1]
    renderer.backend.renderer = Backend()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._output_id_color_buffers = {"semantic_segmentation": object()}
    renderer._initialized_scene = True
    return renderer


def _make_ovstage_renderer_with_backend(events: list[str]) -> OVRTXRenderer:
    """Build an ovstage-path renderer whose backend calls are recorded into ``events``."""

    class Completion:
        def wait(self) -> None:
            return

    class Stage:
        def release_query(self, query):
            events.append(f"release_query:{query}")
            return Completion()

    class StagePaths:
        def destroy_path_list(self, path_list) -> None:
            events.append(f"destroy_path_list:{path_list}")

    class Backend:
        def destroy(self) -> None:
            events.append("detach_ovstage")
            events.append("destroy_renderer")

    class ExitStack:
        def close(self) -> None:
            events.append("exit_stack_close")

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer.backend.stage = Stage()
    renderer.backend.paths = StagePaths()
    render_data = _make_ovrtx_camera_render_data()
    render_data.render_product_path = "/Render/RenderProduct_camera"
    render_data.camera_xform_query = "camera"
    render_data.resources.callback(renderer.backend.paths.destroy_path_list, "camera")
    render_data.resources.callback(lambda: renderer.backend.stage.release_query("camera").wait())
    render_data.renderer_info = {"rgb": object()}
    renderer._camera_render_data.append(render_data)
    renderer._camera_xform_query = None
    renderer._camera_paths_list = None
    renderer._object_xform_query = "object"
    renderer._object_paths_list = "object"
    renderer._deformable_points_query = "deformable"
    renderer._deformable_paths_list = "deformable"
    renderer._particle_points_query = "particle"
    renderer._particle_paths_list = "particle"
    renderer._cable_points_query = "cable"
    renderer._cable_paths_list = "cable"
    renderer._object_newton_indices = object()
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [1]
    renderer._particle_visual_offsets = [0]
    renderer._particle_visual_counts = [1]
    renderer.backend.renderer = Backend()
    renderer.backend._resources = ExitStack()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._output_id_color_buffers = {"semantic_segmentation": object()}
    renderer._initialized_scene = True
    renderer._current_ordinal = 7
    return renderer


def test_ovrtx_close_releases_legacy_renderer_state():
    """Borrowers unbind their tensor bindings before the registry closes the native engine."""
    events: list[str] = []
    renderer = _make_legacy_renderer_with_backend(events)
    render_data = renderer._camera_render_data[0]

    renderer.close()
    assert "destroy_renderer" not in events
    SimulationContext.instance().close_backend(renderer.backend)

    assert events == [
        "unbind:camera",
        "unbind:object",
        "unbind:deformable",
        "unbind:particle",
        "unbind:cable",
        "destroy_renderer",
    ]
    assert renderer._camera_xform_binding is None
    assert renderer._camera_render_data == []
    assert render_data.camera_xform_binding is None
    assert render_data.renderer_info == {}
    assert renderer._object_xform_binding is None
    assert renderer._object_transform_buffer is None
    assert renderer._deformable_points_binding is None
    assert renderer._particle_points_binding is None
    assert renderer._cable_points_binding is None
    assert renderer._particle_workaround_applied is False
    assert renderer.backend.renderer is None
    assert renderer._render_product_paths == []
    assert renderer._output_id_color_buffers == {}
    assert renderer._initialized_scene is False


def test_ovrtx_close_releases_ovstage_renderer_state():
    """Queries release before the native engine, which must detach before stage resources close."""
    events: list[str] = []
    renderer = _make_ovstage_renderer_with_backend(events)
    render_data = renderer._camera_render_data[0]

    renderer.close()
    assert "destroy_renderer" not in events
    SimulationContext.instance().close_backend(renderer.backend)

    assert events == [
        "release_query:camera",
        "destroy_path_list:camera",
        "release_query:object",
        "destroy_path_list:object",
        "release_query:deformable",
        "destroy_path_list:deformable",
        "release_query:particle",
        "destroy_path_list:particle",
        "release_query:cable",
        "destroy_path_list:cable",
        "detach_ovstage",
        "destroy_renderer",
        "exit_stack_close",
    ]
    assert renderer._camera_xform_query is None
    assert renderer._camera_render_data == []
    assert render_data.camera_xform_query is None
    assert render_data.renderer_info == {}
    assert renderer._particle_paths_list is None
    assert renderer._cable_points_query is None
    assert renderer._cable_paths_list is None
    assert renderer._object_newton_indices is None
    assert renderer.backend.renderer is None
    assert renderer.backend.stage is None
    assert renderer.backend.paths is None
    assert renderer._render_product_paths == []
    assert renderer._output_id_color_buffers == {}
    assert renderer._initialized_scene is False
    assert renderer._current_ordinal == 0
    events.clear()
    renderer.close()
    assert events == []
