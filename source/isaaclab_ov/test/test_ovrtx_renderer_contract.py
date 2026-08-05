# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX renderer output contract."""

import importlib.util

import pytest
import torch
import warp as wp

from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import (  # noqa: E402
        OVRTXRenderData,
        OVRTXRenderer,
        ovrtx_use_ovstage_enabled,
    )
else:
    OVRTXRenderData = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None
    ovrtx_use_ovstage_enabled = None

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


def _make_ovrtx_render_data() -> OVRTXRenderData:
    rd = OVRTXRenderData.__new__(OVRTXRenderData)
    rd.width = 16
    rd.height = 8
    rd.num_envs = 2
    rd.warp_buffers = {}
    rd.renderer_info = {}
    rd.ppisp_pipeline = None
    return rd


def _make_ovrtx_renderer_without_backend() -> OVRTXRenderer:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    return renderer


def test_ovrtx_supported_output_types_key_set():
    """OVRTX publishes the documented key set and per-output spec."""
    renderer = _make_ovrtx_renderer_without_backend()
    specs = renderer.supported_output_types()

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
    render_data = _make_ovrtx_render_data()
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
    render_data = _make_ovrtx_render_data()
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
    render_data = _make_ovrtx_render_data()
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
        render_vars = {"LdrColor": FailingRenderVar()}

    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()

    renderer._process_render_frame(render_data, Frame(), {"rgba": object()})


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
    render_data = _make_ovrtx_render_data()
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
    render_data = _make_ovrtx_render_data()

    with pytest.raises(ValueError, match="out of bounds"):
        renderer._launch_extract_all_tiles(render_data, _FakeArray((8, 16, 3)), _FakeArray((2, 8, 16, 4)))


def test_launch_extract_all_tiles_launches_kernel_when_channels_are_compatible(monkeypatch):
    """Equal or narrower output channel counts pass validation and reach the kernel launch."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_render_data()
    render_data.num_cols = 2

    launch_calls = []
    monkeypatch.setattr(wp, "launch", lambda **kwargs: launch_calls.append(kwargs))

    tiled_buffer = _FakeArray((8, 16, 4))
    output_buffer = _FakeArray((2, 8, 16, 3))
    renderer._launch_extract_all_tiles(render_data, tiled_buffer, output_buffer)

    assert len(launch_calls) == 1
    assert launch_calls[0]["inputs"][:2] == [tiled_buffer, output_buffer]


def test_ovrtx_read_output_copies_no_pixel_data():
    """OVRTXRenderer.read_output copies no pixel data; with empty renderer_info it leaves info untouched."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
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
    render_data = _make_ovrtx_render_data()
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
    render_data = _make_ovrtx_render_data()

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


def test_ovrtx_use_ovstage_enabled_when_requested_and_available(monkeypatch):
    """Setting the variable to 1 selects the ovstage path when ovstage is importable."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    monkeypatch.setattr(ovrtx_renderer_module, "_OVSTAGE_AVAILABLE", True)
    assert ovrtx_use_ovstage_enabled() is True


def test_ovrtx_use_ovstage_raises_when_requested_but_unavailable(monkeypatch):
    """An explicit opt-in must fail loudly rather than silently falling back to the legacy path."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    monkeypatch.setattr(ovrtx_renderer_module, "_OVSTAGE_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="uv run --extra ovrtx"):
        ovrtx_use_ovstage_enabled()


def test_ovrtx_use_ovstage_rejects_non_boolean_values(monkeypatch):
    """Values other than 0/1 are a configuration error, not a silent disable."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "true")
    monkeypatch.setattr(ovrtx_renderer_module, "_OVSTAGE_AVAILABLE", True)

    with pytest.raises(ValueError, match="Expected 0 or 1"):
        ovrtx_use_ovstage_enabled()


def test_ovrtx_cleanup_releases_only_the_given_render_data():
    """``cleanup`` releases the render data's own buffers and leaves the renderer usable.

    The stage queries, tensor bindings and render products the renderer holds are shared with
    every other camera that resolved to it, so a single camera's cleanup must not take them.
    """
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._initialized_scene = True

    render_data = _make_ovrtx_render_data()
    render_data.warp_buffers = {"rgba": wp.zeros((8, 16, 4), dtype=wp.uint8, device="cpu")}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": {}}}
    render_data.ppisp_pipeline = object()

    renderer.cleanup(render_data)

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


class _RecordingBinding:
    def __init__(self, events: list[str], name: str):
        self._events = events
        self._name = name

    def unbind(self) -> None:
        self._events.append(f"unbind:{self._name}")


def _make_legacy_renderer_with_backend(events: list[str]) -> OVRTXRenderer:
    """Build a legacy-path renderer whose backend calls are recorded into ``events``."""

    class Backend:
        def reset_stage(self) -> None:
            events.append("reset_stage")

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = False
    renderer._camera_xform_binding = _RecordingBinding(events, "camera")
    renderer._object_xform_binding = _RecordingBinding(events, "object")
    renderer._deformable_points_binding = _RecordingBinding(events, "deformable")
    renderer._particle_points_binding = _RecordingBinding(events, "particle")
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [1]
    renderer._particle_visual_offsets = [0]
    renderer._particle_visual_counts = [1]
    renderer._particle_workaround_applied = True
    renderer._renderer = Backend()
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
        def detach_ovstage(self) -> None:
            events.append("detach_ovstage")

    class ExitStack:
        def close(self) -> None:
            events.append("exit_stack_close")

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._stage = Stage()
    renderer._stage_paths = StagePaths()
    renderer._camera_xform_query = "camera"
    renderer._camera_paths_list = "camera"
    renderer._object_xform_query = "object"
    renderer._object_paths_list = "object"
    renderer._deformable_points_query = "deformable"
    renderer._deformable_paths_list = "deformable"
    renderer._particle_points_query = "particle"
    renderer._particle_paths_list = "particle"
    renderer._object_newton_indices = object()
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [1]
    renderer._particle_visual_offsets = [0]
    renderer._particle_visual_counts = [1]
    renderer._env_root_xforms = object()
    renderer._renderer = Backend()
    renderer._ovstage_exit_stack = ExitStack()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._output_id_color_buffers = {"semantic_segmentation": object()}
    renderer._initialized_scene = True
    renderer._current_ordinal = 7
    return renderer


def test_ovrtx_close_releases_legacy_renderer_state():
    """``close`` unbinds the tensor bindings and resets the stage the renderer owns."""
    events: list[str] = []
    renderer = _make_legacy_renderer_with_backend(events)

    renderer.close()

    assert events == [
        "unbind:camera",
        "unbind:object",
        "unbind:deformable",
        "unbind:particle",
        "reset_stage",
    ]
    assert renderer._camera_xform_binding is None
    assert renderer._object_xform_binding is None
    assert renderer._deformable_points_binding is None
    assert renderer._particle_points_binding is None
    assert renderer._particle_workaround_applied is False
    assert renderer._renderer is None
    assert renderer._render_product_paths == []
    assert renderer._output_id_color_buffers == {}
    assert renderer._initialized_scene is False


def test_ovrtx_close_releases_ovstage_renderer_state():
    """``close`` releases the queries and path lists, then detaches before closing the ExitStack.

    The ExitStack owns the ovstage ``Stage`` and ``PathDictionary`` as context managers, so it is the
    only thing that releases them — ``ExitStack`` has no finalizer, and garbage collection never
    invokes ``__exit__``. Detaching first avoids a use-after-free while the renderer still references
    the stage.
    """
    events: list[str] = []
    renderer = _make_ovstage_renderer_with_backend(events)

    renderer.close()

    assert events == [
        "release_query:camera",
        "destroy_path_list:camera",
        "release_query:object",
        "destroy_path_list:object",
        "release_query:deformable",
        "destroy_path_list:deformable",
        "release_query:particle",
        "destroy_path_list:particle",
        "detach_ovstage",
        "exit_stack_close",
    ]
    assert renderer._camera_xform_query is None
    assert renderer._particle_paths_list is None
    assert renderer._object_newton_indices is None
    assert renderer._env_root_xforms is None
    assert renderer._renderer is None
    assert renderer._ovstage_exit_stack is None
    assert renderer._stage is None
    assert renderer._stage_paths is None
    assert renderer._render_product_paths == []
    assert renderer._output_id_color_buffers == {}
    assert renderer._initialized_scene is False
    assert renderer._current_ordinal == 0


def test_ovrtx_close_is_idempotent():
    """A second ``close`` releases nothing again, so a repeated teardown cannot double-free."""
    events: list[str] = []
    renderer = _make_ovstage_renderer_with_backend(events)

    renderer.close()
    events.clear()
    renderer.close()

    assert events == []
