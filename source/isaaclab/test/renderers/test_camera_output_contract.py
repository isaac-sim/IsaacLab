# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the renderer→camera output contract."""

import builtins
import warnings
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

pytest.importorskip("isaaclab_physx")

from isaaclab.sensors.camera import CameraCfg, TiledCameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import clone, validate
from isaaclab.utils.visual_processing import (
    VisualProcessingPipeline,
    VisualProcessor,
    VisualProcessorCfg,
    VisualProcessorContext,
)

pytestmark = [pytest.mark.integration, pytest.mark.rendering]

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)


@pytest.mark.parametrize(
    "field_name,deprecated_value",
    [
        ("colorize_semantic_segmentation", False),
        ("colorize_instance_segmentation", False),
        ("colorize_instance_id_segmentation", False),
        ("semantic_filter", ["class"]),
        ("semantic_segmentation_mapping", {"class:cube": (1, 2, 3, 4)}),
        ("depth_clipping_behavior", "max"),
    ],
)
def test_camera_cfg_forwards_deprecated_fields_to_renderer_cfg(field_name, deprecated_value):
    """Deprecated CameraCfg field is forwarded to renderer_cfg with a warning."""
    kwargs = {
        "height": 64,
        "width": 64,
        "prim_path": "/World/Camera",
        "spawn": _SPAWN,
        "data_types": ["rgb"],
        field_name: deprecated_value,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = CameraCfg(**kwargs)

    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any(f"CameraCfg.{field_name}" in str(w.message) for w in deprecation_warnings)
    assert getattr(cfg.renderer_cfg, field_name) == deprecated_value


def test_camera_cfg_default_does_not_warn_or_forward():
    """Default-valued deprecated fields stay silent."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = CameraCfg(
            height=64,
            width=64,
            prim_path="/World/Camera",
            spawn=_SPAWN,
            data_types=["rgb"],
        )

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning) and "CameraCfg." in str(w.message)
    ]
    assert deprecation_warnings == []
    assert cfg.renderer_cfg.colorize_semantic_segmentation is True


def test_camera_cfg_copy_does_not_reforward_deprecated_fields():
    """Copying a cfg (as ``SensorBase`` does) keeps a renderer_cfg value set after forwarding."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = CameraCfg(
            height=64,
            width=64,
            prim_path="/World/Camera",
            spawn=_SPAWN,
            data_types=["rgb"],
            colorize_semantic_segmentation=False,
        )
    cfg.renderer_cfg.colorize_semantic_segmentation = True

    assert clone(cfg).renderer_cfg.colorize_semantic_segmentation is True


def test_camera_cfg_preserves_deprecated_isp_configuration():
    """Legacy configuration remains accepted and explains where processed pixels move."""
    from isaaclab_ppisp import PpispCfg

    with pytest.warns(DeprecationWarning, match=r"CameraCfg.isp_cfg.*processed_image"):
        cfg = CameraCfg(height=2, width=3, prim_path="/World/Camera", spawn=_SPAWN, isp_cfg=PpispCfg())
    assert isinstance(cfg.isp_cfg, PpispCfg)
    assert not hasattr(cfg.renderer_cfg, "isp_cfg")


def test_camera_cfg_post_construction_mutation_is_silent_no_op():
    """Mutating a deprecated field after construction does not propagate to renderer_cfg."""
    cfg = CameraCfg(
        height=64,
        width=64,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=["rgb"],
    )
    assert cfg.renderer_cfg.colorize_semantic_segmentation is True
    cfg.colorize_semantic_segmentation = False
    assert cfg.renderer_cfg.colorize_semantic_segmentation is True


def test_tiled_camera_cfg_does_not_forward_deprecated_fields():
    """TiledCameraCfg skips CameraCfg's per-field forwarder."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = TiledCameraCfg(
            height=64,
            width=64,
            prim_path="/World/Camera",
            spawn=_SPAWN,
            data_types=["rgb"],
            colorize_semantic_segmentation=False,
        )

    tiled_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning) and "TiledCameraCfg" in str(w.message)
    ]
    assert tiled_warnings

    field_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning) and "CameraCfg.colorize_" in str(w.message)
    ]
    assert field_warnings == []

    assert cfg.renderer_cfg.colorize_semantic_segmentation is True


def test_newton_warp_supported_output_types_key_set():
    """Newton renderer and config publish one shared output contract."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)
    renderer.cfg = NewtonWarpRendererCfg()
    specs = renderer.supported_output_types()

    assert specs == renderer.cfg.supported_output_types()
    assert set(specs.keys()) == {
        RenderBufferKind.RGB,
        RenderBufferKind.RGBA,
        RenderBufferKind.RGB_HDR,
        RenderBufferKind.RGB_RADIANCE,
        RenderBufferKind.ALBEDO,
        RenderBufferKind.DEPTH,
        RenderBufferKind.DISTANCE_TO_CAMERA,
        RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
        RenderBufferKind.NORMALS,
        RenderBufferKind.SEMANTIC_SEGMENTATION,
        RenderBufferKind.INSTANCE_SEGMENTATION,
    }
    assert specs[RenderBufferKind.RGB_HDR] == RenderBufferSpec(3, wp.float32, color_space="scene_linear")


def test_camera_cfg_rejects_outputs_unsupported_by_renderer():
    """Camera config validation rejects output types absent from the renderer contract and accepts supported ones."""
    pytest.importorskip("isaaclab_newton")
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    def make_cfg(data_type: str) -> CameraCfg:
        return CameraCfg(
            height=64,
            width=64,
            prim_path="/World/Camera",
            spawn=_SPAWN,
            data_types=[data_type],
            renderer_cfg=NewtonWarpRendererCfg(),
        )

    validate(make_cfg("rgb_hdr"))
    with pytest.raises(ValueError, match="simple_shading_full_mdl"):
        validate(make_cfg("simple_shading_full_mdl"))


@pytest.mark.parametrize("colorize", [True, False])
def test_newton_warp_segmentation_spec_follows_colorize_flags(colorize):
    """Segmentation specs are RGBA uint8 when colorized, else single-channel int32 (matching RTX)."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)
    renderer.cfg = NewtonWarpRendererCfg(
        colorize_semantic_segmentation=colorize,
        colorize_instance_segmentation=colorize,
    )
    specs = renderer.supported_output_types()

    expected = RenderBufferSpec(4, wp.uint8) if colorize else RenderBufferSpec(1, wp.int32)
    for kind in (
        RenderBufferKind.SEMANTIC_SEGMENTATION,
        RenderBufferKind.INSTANCE_SEGMENTATION,
    ):
        assert specs[kind] == expected


@pytest.mark.parametrize("data_type", [RenderBufferKind.RGB_HDR, RenderBufferKind.RGB_RADIANCE])
def test_newton_warp_wraps_requested_hdr_output(data_type):
    """Newton wires both linear color signals to its HDR output slot."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    wp.init()
    from isaaclab_newton.renderers.newton_warp_renderer import RenderData

    from isaaclab.utils.warp.proxy_array import ProxyArray

    fake_sensor = SimpleNamespace(model=SimpleNamespace(world_count=2, device="cpu"))
    spawn = SimpleNamespace(distortion=None)
    camera_cfg = SimpleNamespace(width=4, height=3, spawn=spawn)
    render_data = RenderData(fake_sensor, SimpleNamespace(cfg=camera_cfg))
    hdr_proxy = ProxyArray(wp.zeros((2, 3, 4, 3), dtype=wp.float32, device="cpu"))

    render_data.set_outputs({str(data_type): hdr_proxy})

    assert render_data.outputs.hdr_color_image is not None
    assert render_data.get_output(data_type) is render_data.outputs.hdr_color_image


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=data_types,
    )


def test_camera_data_allocates_supported_subset_and_aliases_color():
    """CameraData aliases RGB/RGBA and the active HDR signal without duplicate storage."""
    cfg = _make_camera_cfg(["rgb", "rgba", "rgb_hdr", "rgb_radiance", "depth"])
    specs = {
        RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
        RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
        RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32, color_space="scene_linear"),
        RenderBufferKind.RGB_RADIANCE: RenderBufferSpec(3, wp.float32, color_space="scene_linear"),
        RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
        RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
    }
    data = CameraData.allocate(
        data_types=cfg.data_types, height=8, width=16, num_views=2, device="cpu", supported_specs=specs
    )

    assert set(data.output.keys()) == {"rgba", "rgb", "rgb_hdr", "rgb_radiance", "depth"}
    assert data.output["rgba"].shape == (2, 8, 16, 4)
    assert data.output["rgba"].dtype == wp.uint8
    assert data.output["depth"].shape == (2, 8, 16, 1)
    assert data.output["depth"].dtype == wp.float32
    assert data.output["rgb"].warp.ptr == data.output["rgba"].warp.ptr
    assert data.output["rgb_hdr"] is data.output["rgb_radiance"]
    assert data.output["rgb_radiance"].shape == (2, 8, 16, 3)
    assert data.output["rgb_radiance"].dtype == wp.float32
    assert data.image_shape == (8, 16)
    assert data.info == dict.fromkeys(data.output)


def test_camera_data_allocate_raises_on_unknown_name():
    """An unknown data_types name raises ValueError naming the offender."""
    supported_specs = {RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8)}
    with pytest.raises(ValueError) as exc_info:
        CameraData.allocate(
            data_types=["not_a_real_type"],
            height=4,
            width=4,
            num_views=1,
            device="cpu",
            supported_specs=supported_specs,
        )
    assert "not_a_real_type" in str(exc_info.value)
    assert "RenderBufferKind" in str(exc_info.value)


def _make_increment_processor(cfg, context, events):
    """Build an observable processor with state local to each sensor."""
    bindings = {}
    name = cfg.params["name"]

    def initialize(inputs, outputs):
        bindings["input"] = inputs[next(iter(cfg.inputs))]
        bindings["output"] = outputs[next(iter(cfg.outputs))]
        if "rgba" in outputs:
            bindings["rgba_owner"] = weakref.ref(outputs["rgba"].warp)
        events.append((name, "initialize", bindings))

    def process(mask):
        events.append((name, "process", mask))
        selected = wp.to_torch(mask)
        source = bindings["input"].torch
        output = bindings["output"].torch
        output[selected] = (source[selected] + cfg.params.get("increment", 1)).to(output.dtype)

    return VisualProcessor(
        inputs=cfg.inputs,
        outputs=cfg.outputs,
        initialize=initialize,
        process=process,
        reset=lambda mask: events.append((name, "reset", mask)),
        close=lambda: events.append((name, "close", None)),
        in_place=cfg.params.get("in_place", False),
    )


def _processor_cfg(name, inputs, outputs, events, **params):
    return VisualProcessorCfg(
        func=lambda cfg, context: _make_increment_processor(cfg, context, events),
        inputs=inputs,
        outputs=outputs,
        params={"name": name, **params},
    )


def _processing_context():
    return VisualProcessorContext(
        stage=None,
        camera_prim_paths=("/World/envs/env_0/Camera", "/World/envs/env_1/Camera"),
        num_views=2,
        height=2,
        width=3,
        device="cpu",
    )


def test_visual_pipeline_preserves_rgb_only_renderer_contract():
    """An empty chain does not request RGBA from a renderer that only supplies RGB."""
    pipeline = VisualProcessingPipeline([], _processing_context(), {"rgb": RenderBufferSpec(3, wp.uint8)}, ["rgb"])
    outputs = pipeline.allocate()
    assert pipeline.render_data_types == ("rgb",)
    assert set(pipeline.render_outputs) == set(outputs) == {"rgb"}
    assert outputs["rgb"].shape == (2, 2, 3, 3)
    pipeline.close()


def test_visual_processors_order_intermediates_and_persistent_aliases():
    """Stages consume earlier results, keep their bindings, and expose only public outputs."""
    events = []
    hdr = RenderBufferSpec(3, wp.float32, color_space="scene_linear")
    rgb = RenderBufferSpec(3, wp.uint8, color_space="srgb")
    configs = [
        _processor_cfg("tone", {"rgb_hdr": hdr}, {"tone_mapped": rgb}, events),
        _processor_cfg("color", {"tone_mapped": rgb}, {"rgb": rgb}, events, increment=2),
        _processor_cfg("finish", {"rgb": rgb}, {"rgb": rgb}, events, increment=3, in_place=True),
    ]
    pipeline = VisualProcessingPipeline(configs, _processing_context(), {"rgb_hdr": hdr}, ["rgb", "rgba"])
    outputs = pipeline.allocate()
    assert set(pipeline.render_data_types) == {"rgb_hdr"}
    assert set(pipeline.render_outputs) == {"rgb_hdr"}
    assert set(outputs) == {"rgb", "rgba"}
    assert outputs["rgb"].warp.ptr == outputs["rgba"].warp.ptr
    assert outputs["rgb"].shape == (2, 2, 3, 3)
    assert outputs["rgba"].shape == (2, 2, 3, 4)

    bindings = {name: value for name, event, value in events if event == "initialize"}
    assert bindings["tone"]["input"] is pipeline.render_outputs["rgb_hdr"]
    assert bindings["color"]["input"] is bindings["tone"]["output"]
    assert bindings["finish"]["input"].warp.ptr == bindings["finish"]["output"].warp.ptr
    pointers = {name: value.warp.ptr for name, value in outputs.items()}

    pipeline.render_outputs["rgb_hdr"].torch.fill_(10)
    mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    pipeline.process(mask)
    np.testing.assert_array_equal(outputs["rgb"].warp.numpy()[0], np.full((2, 3, 3), 10 + 1 + 2 + 3))
    np.testing.assert_array_equal(outputs["rgb"].warp.numpy()[1], np.zeros((2, 3, 3)))
    assert [(name, event) for name, event, _ in events if event == "process"] == [
        ("tone", "process"),
        ("color", "process"),
        ("finish", "process"),
    ]

    pipeline.process(mask)
    assert {name: value.warp.ptr for name, value in outputs.items()} == pointers
    assert sum(event == "initialize" for _, event, _ in events) == 3
    pipeline.reset(mask)
    assert all(value is mask for _, event, value in events if event == "reset")
    assert sum(event == "reset" for _, event, _ in events) == 3
    pipeline.close()
    assert sum(event == "close" for _, event, _ in events) == 3


def test_visual_processors_reject_incompatible_order():
    """A consumer cannot read an intermediate that is produced later in the chain."""
    hdr = RenderBufferSpec(3, wp.float32, color_space="scene_linear")
    rgb = RenderBufferSpec(3, wp.uint8, color_space="srgb")
    configs = [
        _processor_cfg("consumer", {"tone_mapped": rgb}, {"rgb": rgb}, []),
        _processor_cfg("producer", {"rgb_hdr": hdr}, {"tone_mapped": rgb}, []),
    ]
    with pytest.raises(ValueError, match="tone_mapped"):
        VisualProcessingPipeline(configs, _processing_context(), {"rgb_hdr": hdr}, ["rgb"])


@pytest.mark.parametrize(
    "requirement",
    [
        RenderBufferSpec(4, wp.float32, color_space="scene_linear"),
        RenderBufferSpec(3, wp.float16, color_space="scene_linear"),
        RenderBufferSpec(3, wp.float32, layout="NCHW", color_space="scene_linear"),
        RenderBufferSpec(3, wp.float32, device="cuda:0", color_space="scene_linear"),
        RenderBufferSpec(3, wp.float32, color_space="srgb"),
    ],
)
def test_visual_processors_reject_incompatible_input_contract(requirement):
    """Initialization identifies unsupported layout, precision, device, and color requirements."""
    hdr = RenderBufferSpec(3, wp.float32, color_space="scene_linear")
    config = _processor_cfg("invalid", {"rgb_hdr": requirement}, {"rgb": RenderBufferSpec(3, wp.uint8)}, [])
    with pytest.raises(ValueError, match="rgb_hdr|NHWC|device|layout"):
        VisualProcessingPipeline([config], _processing_context(), {"rgb_hdr": hdr}, ["rgb"])


def test_visual_processors_keep_intermediate_rgb_storage_alive():
    """RGB views retain valid intermediate RGBA storage after another stage replaces the output."""
    events = []
    hdr = RenderBufferSpec(3, wp.float32)
    rgb = RenderBufferSpec(3, wp.uint8)
    configs = [
        _processor_cfg("first", {"rgb_hdr": hdr}, {"rgb": rgb}, events),
        _processor_cfg("second", {"rgb": rgb}, {"rgb": rgb}, events),
    ]
    pipeline = VisualProcessingPipeline(configs, _processing_context(), {"rgb_hdr": hdr}, ["rgb"])
    outputs = pipeline.allocate()
    first_bindings = events[0][2]
    assert first_bindings["rgba_owner"]() is not None
    assert first_bindings["output"].warp.ptr != outputs["rgb"].warp.ptr
    pipeline.render_outputs["rgb_hdr"].torch.fill_(10)
    pipeline.process(wp.ones(2, dtype=wp.bool, device="cpu"))
    np.testing.assert_array_equal(outputs["rgb"].warp.numpy(), np.full((2, 2, 3, 3), 12))
    pipeline.close()


def test_visual_processor_cleanup_after_initialization_failure():
    """All resolved processors release resources even if binding a later stage fails."""
    closed = []

    def make_processor(cfg, context):
        def initialize(inputs, outputs):
            if cfg.params["fail"]:
                raise RuntimeError("processor initialization failed")

        return VisualProcessor(
            inputs={},
            outputs={},
            initialize=initialize,
            process=lambda mask: None,
            close=lambda: closed.append(cfg.params["name"]),
        )

    configs = [
        VisualProcessorCfg(func=make_processor, params={"name": "first", "fail": False}),
        VisualProcessorCfg(func=make_processor, params={"name": "second", "fail": True}),
    ]
    pipeline = VisualProcessingPipeline(configs, _processing_context(), {}, [])
    with pytest.raises(RuntimeError, match="processor initialization failed"):
        pipeline.allocate()
    assert closed == ["second", "first"]


def test_visual_processor_cleanup_continues_after_callback_failure():
    """One failing close callback must not leak other processors' state."""
    closed = []

    def make_processor(cfg, context):
        def close():
            closed.append(cfg.params["name"])
            if cfg.params["fail"]:
                raise RuntimeError("processor cleanup failed")

        return VisualProcessor(inputs={}, outputs={}, process=lambda mask: None, close=close)

    configs = [
        VisualProcessorCfg(func=make_processor, params={"name": "first", "fail": False}),
        VisualProcessorCfg(func=make_processor, params={"name": "second", "fail": True}),
    ]
    pipeline = VisualProcessingPipeline(configs, _processing_context(), {}, [])
    pipeline.allocate()
    with pytest.raises(RuntimeError) as exc_info:
        pipeline.close()
    assert "processor cleanup failed" in str(exc_info.value.__cause__ or exc_info.value)
    assert closed == ["second", "first"]
    pipeline.close()
    assert closed == ["second", "first"]


@pytest.mark.parametrize("fail_cleanup", [None, "processor", "renderer"])
def test_camera_initialization_failure_releases_renderer_state(fail_cleanup):
    """A partial camera failure closes every resource and preserves the original diagnostic."""
    from isaaclab.sensors.camera import Camera

    camera = Camera.__new__(Camera)
    camera._clear_callbacks = lambda: None
    closed = []
    render_data = object()

    def close_processor():
        closed.append("processor")
        if fail_cleanup == "processor":
            raise ValueError("processor cleanup failed")

    def cleanup_renderer(data):
        closed.append(data)
        if fail_cleanup == "renderer":
            raise ValueError("cleanup failed")

    def initialize_camera():
        camera._legacy_isp = SimpleNamespace(close=close_processor)
        camera._renderer = SimpleNamespace(cleanup=cleanup_renderer)
        camera._view = SimpleNamespace(close=lambda: closed.append("view"))
        camera._render_data = render_data
        raise RuntimeError("camera initialization failed")

    camera._initialize_camera = initialize_camera
    with pytest.raises(RuntimeError, match="camera initialization failed"):
        camera._initialize_impl()
    assert closed == ["processor", render_data, "view"]
    assert camera._legacy_isp is None
    assert camera._render_data is None
    assert camera._renderer is None
    assert camera._view is None
    camera.__del__()
    assert closed == ["processor", render_data, "view"]


@pytest.mark.parametrize("source", ["public", "explicit", "discovered", "disabled"])
@pytest.mark.parametrize("supports_rgba", [False, True])
def test_all_camera_signals_prepare_before_shared_stage_export(monkeypatch, supports_rgba, source):
    """Public and private inputs reach shared renderer setup without changing public output layouts."""
    from pxr import Sdf, Usd, UsdGeom

    from isaaclab.physics import PhysicsEvent, PhysicsManager
    from isaaclab.renderers.rtx_camera_overrides import apply_rtx_exposure_overrides
    from isaaclab.sensors.camera import Camera
    from isaaclab.sensors.camera import camera as camera_module
    from isaaclab.sensors.sensor_base import SensorBase
    from isaaclab.sim import SimulationContext

    if source != "public":
        from isaaclab_ppisp import PpispCfg, PpispPipeline

        from isaaclab.sensors.camera import CameraISPMode
    else:
        original_import = builtins.__import__

        def without_ppisp(name, *args, **kwargs):
            if name == "isaaclab_ppisp" or name.startswith("isaaclab_ppisp."):
                raise AssertionError("A camera without legacy ISP must not import PPISP.")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_ppisp)
    device = "cpu"
    if source in {"explicit", "discovered"}:
        if not wp.is_cuda_available():
            pytest.skip("PPISP camera output validation requires CUDA.")
        device = "cuda:0"

    class CameraPhysicsManager(PhysicsManager):
        _callbacks = {}

    stage = Usd.Stage.CreateInMemory()
    prims = [UsdGeom.Camera.Define(stage, path).GetPrim() for path in ("/World/First", "/World/Second")]
    prims[1].CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float).Set(100.0)
    specs = {
        RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
        RenderBufferKind.RGB_RADIANCE: RenderBufferSpec(3, wp.float32, color_space="scene_linear"),
    }
    if supports_rgba:
        specs[RenderBufferKind.RGBA] = RenderBufferSpec(4, wp.uint8)
    exports = []
    prepared = []
    bound_outputs = {}

    def prepare_cameras(stage, spec):
        prepared.append(spec)
        if "rgb_radiance" in spec.data_types:
            apply_rtx_exposure_overrides(stage, list(spec.camera_prim_paths))

    def export_stage(stage, num_envs):
        if not exports:
            assert {spec.camera_prim_paths for spec in prepared} == {(str(prim.GetPath()),) for prim in prims}
            assert prims[0].GetAttribute("exposure:iso").Get() == 0.0
            assert prims[1].GetAttribute("exposure:iso").Get() == (100.0 if source == "disabled" else 0.0)
            exports.append(stage.ExportToString())

    renderer = SimpleNamespace(
        supported_output_types=lambda: specs,
        prepare_cameras=prepare_cameras,
        create_render_data=lambda spec: SimpleNamespace(spec=spec),
        set_outputs=lambda data, outputs: bound_outputs.update({data.spec.camera_prim_paths[0]: outputs}),
        cleanup=lambda data: None,
    )
    sim = SimpleNamespace(
        device=device,
        physics_manager=CameraPhysicsManager,
        get_clone_plan=lambda: SimpleNamespace(topology=SimpleNamespace(world_prototype_layout=np.zeros(2))),
        render_context=SimpleNamespace(ensure_prepare_stage=export_stage),
        vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda sensor: None),
    )
    monkeypatch.setattr(SimulationContext, "instance", staticmethod(lambda: sim))
    monkeypatch.setattr(SensorBase, "_initialize_impl", lambda self: None)
    monkeypatch.setattr(Camera, "_initialize_intrinsics", lambda self: None)
    monkeypatch.setattr(Camera, "_update_poses", lambda self: None)
    monkeypatch.setattr(
        camera_module,
        "FrameView",
        lambda path, **kwargs: SimpleNamespace(count=2, prims=[stage.GetPrimAtPath(path)] * 2, close=lambda: None),
    )
    cameras = []
    for index, prim in enumerate(prims):
        camera = Camera.__new__(Camera)
        camera.cfg = SimpleNamespace(
            prim_path=str(prim.GetPath()),
            data_types=["rgb_radiance"] if index == 1 and source == "public" else ["rgb"],
            isp_cfg=None,
            height=2,
            width=3,
            renderer_cfg=SimpleNamespace(renderer_type="newton"),
        )
        camera.stage = stage
        camera._device = device
        camera._num_envs = 2
        camera._is_initialized = False
        camera._legacy_isp = None
        camera.render_generation = 0
        camera._requested_render_inputs = ()
        camera._sensor_prims = []
        camera._renderer = renderer
        camera._render_data = None
        camera._view = None
        camera._register_callbacks()
        cameras.append(camera)

    if source != "public":
        cameras[1].cfg.isp_cfg = PpispCfg() if source == "explicit" else CameraISPMode.AUTO_CAMERA
    if source == "discovered":
        # Discovery must happen after prestartup authors camera attributes.
        prims[1].CreateAttribute("ppisp:exposureOffset", Sdf.ValueTypeNames.Float).Set(1.0)

    try:
        first = cameras[0]
        with pytest.raises(ValueError, match="unsupported"):
            first.request_render_inputs(("unsupported",))
        first.request_render_inputs(("rgb_radiance",))
        first.request_render_inputs(("rgb_radiance",))
        CameraPhysicsManager.dispatch_event(PhysicsEvent.PHYSICS_READY)
        assert len(exports) == 1
        assert all(camera.is_initialized for camera in cameras)
        assert all(spec.num_instances == spec.view_count == 2 for spec in prepared)
        outputs = bound_outputs[first.cfg.prim_path]
        public_names = {"rgb", "rgba"} if supports_rgba else {"rgb"}
        assert first.cfg.data_types == ["rgb"]
        assert first._render_data.spec.data_types == ("rgb", "rgb_radiance")
        assert set(first._data.output) == public_names
        assert set(outputs) == public_names | {"rgb_radiance"}
        assert outputs["rgb"] is first._data.output["rgb"]
        if supports_rgba:
            assert outputs["rgb"].warp.ptr == outputs["rgba"].warp.ptr
        if source != "public":
            legacy = cameras[1]
            raw = bound_outputs[legacy.cfg.prim_path]
            if source == "disabled":
                assert "rgb_radiance" not in raw
                assert legacy._data.output["rgb"] is raw["rgb"]
            else:
                radiance = raw["rgb_radiance"]
                radiance.warp.fill_(0.5)
                pointers = {name: output.warp.ptr for name, output in legacy._data.output.items()}
                assert set(pointers) == {"rgb", "rgba"}
                mask = wp.ones(2, dtype=wp.bool, device=device)
                legacy._finish_capture(mask)
                assert legacy.render_generation == 1
                assert np.any(legacy._data.output["rgb"].warp.numpy())
                reference = PpispPipeline(PpispCfg(inputs={"exposureOffset": 1.0 if source == "discovered" else 0.0}))
                expected = wp.empty_like(legacy._data.output["rgba"].warp)
                try:
                    reference.apply(radiance.warp, expected)
                    np.testing.assert_array_equal(legacy._data.output["rgba"].warp.numpy(), expected.numpy())
                finally:
                    reference.close()
                assert pointers["rgb"] == pointers["rgba"] != radiance.warp.ptr
                np.testing.assert_array_equal(radiance.warp.numpy(), 0.5)
                legacy._finish_capture(mask)
                assert {name: output.warp.ptr for name, output in legacy._data.output.items()} == pointers
        with pytest.raises(RuntimeError, match="before sensor initialization"):
            first.request_render_inputs(("rgb_radiance",))
    finally:
        for camera in cameras:
            camera.__del__()
    assert not CameraPhysicsManager._callbacks
