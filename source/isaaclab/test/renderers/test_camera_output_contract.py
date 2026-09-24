# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the renderer→camera output contract."""

import warnings
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

pytest.importorskip("isaaclab_physx")

from isaaclab.sensors.camera import CameraCfg, TiledCameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sensors.camera.post_processing import (
    VisualProcessingPipeline,
    VisualProcessor,
    VisualProcessorCfg,
    VisualProcessorContext,
)
from isaaclab.sim import PinholeCameraCfg

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
        RenderBufferKind.ALBEDO,
        RenderBufferKind.DEPTH,
        RenderBufferKind.DISTANCE_TO_CAMERA,
        RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
        RenderBufferKind.NORMALS,
        RenderBufferKind.SEMANTIC_SEGMENTATION,
        RenderBufferKind.INSTANCE_SEGMENTATION,
    }
    assert specs[RenderBufferKind.RGB_HDR] == RenderBufferSpec(3, wp.float32, color_space="scene_linear")


@pytest.mark.parametrize("data_type", ["simple_shading_full_mdl", "not_a_render_buffer_kind"])
def test_camera_cfg_rejects_outputs_unsupported_by_renderer(data_type):
    """Camera config validation rejects output types absent from the renderer contract."""
    pytest.importorskip("isaaclab_newton")
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    cfg = CameraCfg(
        height=64,
        width=64,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=[data_type],
        renderer_cfg=NewtonWarpRendererCfg(),
    )

    with pytest.raises(ValueError, match=data_type):
        cfg.validate()


@pytest.mark.parametrize("data_type", ["rgba", "rgb_hdr", "albedo"])
def test_camera_cfg_accepts_supported_newton_outputs(data_type):
    """Camera config validation accepts every formerly omitted Newton color output."""
    pytest.importorskip("isaaclab_newton")
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    cfg = CameraCfg(
        height=64,
        width=64,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=[data_type],
        renderer_cfg=NewtonWarpRendererCfg(),
    )

    cfg.validate()


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


def test_newton_warp_wraps_requested_rgb_hdr_output():
    """NewtonWarpRenderer wires requested RGB_HDR proxies to the Newton HDR output slot."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    wp.init()
    from isaaclab_newton.renderers.newton_warp_renderer import RenderData

    from isaaclab.utils.warp.proxy_array import ProxyArray

    fake_sensor = SimpleNamespace(model=SimpleNamespace(world_count=2, device="cpu"))
    spawn = SimpleNamespace(distortion=None)
    camera_cfg = SimpleNamespace(width=4, height=3, spawn=spawn, isp_cfg=None)
    render_data = RenderData(fake_sensor, SimpleNamespace(cfg=camera_cfg))
    hdr_proxy = ProxyArray(wp.zeros((2, 3, 4, 3), dtype=wp.float32, device="cpu"))

    render_data.set_outputs({str(RenderBufferKind.RGB_HDR): hdr_proxy})

    assert render_data.outputs.hdr_color_image is not None
    assert render_data.get_output(RenderBufferKind.RGB_HDR) is render_data.outputs.hdr_color_image


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=data_types,
    )


def test_camera_data_allocates_supported_subset_and_aliases_rgb():
    """CameraData allocates the intersection of requested + supported and aliases rgb into rgba."""
    cfg = _make_camera_cfg(["rgb", "rgba", "depth"])
    specs = {
        RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
        RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
        RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
        RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
    }
    data = CameraData.allocate(
        data_types=cfg.data_types, height=8, width=16, num_views=2, device="cpu", supported_specs=specs
    )

    assert set(data.output.keys()) == {"rgba", "rgb", "depth"}
    assert data.output["rgba"].shape == (2, 8, 16, 4)
    assert data.output["rgba"].dtype == wp.uint8
    assert data.output["depth"].shape == (2, 8, 16, 1)
    assert data.output["depth"].dtype == wp.float32
    assert data.output["rgb"].warp.ptr == data.output["rgba"].warp.ptr
    assert data.image_shape == (8, 16)
    assert data.info == {"rgba": None, "rgb": None, "depth": None}


def test_camera_data_drops_requested_types_not_in_supported_specs():
    """Requested types absent from supported_specs are absent from data.output."""
    cfg = _make_camera_cfg(["rgb", "normals"])
    specs = {
        RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
        RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
    }
    data = CameraData.allocate(
        data_types=cfg.data_types, height=4, width=4, num_views=1, device="cpu", supported_specs=specs
    )

    assert "normals" not in data.output
    assert {"rgb", "rgba"} <= set(data.output.keys())


def test_camera_data_no_arg_construction_yields_empty_container():
    """Bare CameraData() produces an all-None container."""
    data = CameraData()
    assert data.pos_w is None
    assert data.quat_w_world is None
    assert data.intrinsic_matrices is None
    assert data.output is None
    assert data.info is None
    assert data.image_shape is None


def test_camera_data_segmentation_dtype_follows_supported_spec():
    """CameraData consumes the layout dtype declared by the renderer spec."""
    cfg = _make_camera_cfg(["instance_segmentation"])
    raw_specs = {RenderBufferKind.INSTANCE_SEGMENTATION: RenderBufferSpec(1, wp.int32)}
    colorized_specs = {RenderBufferKind.INSTANCE_SEGMENTATION: RenderBufferSpec(4, wp.uint8)}

    raw = CameraData.allocate(
        data_types=cfg.data_types, height=4, width=4, num_views=1, device="cpu", supported_specs=raw_specs
    )
    colorized = CameraData.allocate(
        data_types=cfg.data_types, height=4, width=4, num_views=1, device="cpu", supported_specs=colorized_specs
    )

    assert raw.output["instance_segmentation"].dtype == wp.int32
    assert raw.output["instance_segmentation"].shape == (1, 4, 4, 1)
    assert colorized.output["instance_segmentation"].dtype == wp.uint8
    assert colorized.output["instance_segmentation"].shape == (1, 4, 4, 4)


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


def test_visual_processor_state_is_independent_per_camera():
    """Reusing a configuration creates separate bindings and buffers for each sensor."""
    events = []
    rgb = RenderBufferSpec(3, wp.uint8)
    config = _processor_cfg("increment", {"rgb": rgb}, {"rgb": rgb}, events, in_place=True)
    specs = {"rgb": rgb, "rgba": RenderBufferSpec(4, wp.uint8)}
    first = VisualProcessingPipeline([config], _processing_context(), specs, ["rgb"])
    second = VisualProcessingPipeline([config], _processing_context(), specs, ["rgb"])
    first_output = first.allocate()["rgb"]
    second_output = second.allocate()["rgb"]
    assert first_output.warp.ptr != second_output.warp.ptr
    first.render_outputs["rgb"].torch.fill_(7)
    second.render_outputs["rgb"].torch.fill_(20)
    first.process(wp.ones(2, dtype=wp.bool, device="cpu"))
    np.testing.assert_array_equal(first_output.warp.numpy(), np.full((2, 2, 3, 3), 8))
    np.testing.assert_array_equal(second_output.warp.numpy(), np.full((2, 2, 3, 3), 20))
    first.close()
    second.close()


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


def test_camera_processing_respects_cached_reads_and_partial_resets(monkeypatch):
    """Camera reads run processors once per fresh frame and forward each reset selection."""
    from isaaclab.sensors.camera import Camera
    from isaaclab.sim import SimulationContext

    processed = []
    reset = []
    rendered = []
    rgb = RenderBufferSpec(3, wp.uint8)
    config = VisualProcessorCfg(
        func=lambda cfg, context: VisualProcessor(
            inputs={"rgb": rgb},
            outputs={"rgb": rgb},
            process=lambda mask: processed.append(mask.numpy().copy()),
            reset=lambda mask: reset.append(mask.numpy().copy()),
            in_place=True,
        )
    )
    pipeline = VisualProcessingPipeline([config], _processing_context(), {"rgb": rgb}, ["rgb"])
    camera = Camera.__new__(Camera)
    camera._clear_callbacks = lambda: None
    camera._view = None
    camera.cfg = SimpleNamespace(update_period=0.1, update_latest_camera_pose=False)
    camera._device = "cpu"
    camera._num_envs = 2
    camera._is_initialized = True
    camera._is_visualizing = False
    camera._data_generation = 0
    camera._data_generation_last_update = -1
    camera._is_outdated = wp.ones(2, dtype=wp.bool, device="cpu")
    camera._timestamp = wp.zeros(2, dtype=wp.float32, device="cpu")
    camera._timestamp_last_update = wp.zeros_like(camera._timestamp)
    camera._ALL_ENV_MASK = wp.ones(2, dtype=wp.bool, device="cpu")
    camera._reset_mask = wp.zeros(2, dtype=wp.bool, device="cpu")
    camera._reset_mask_torch = wp.to_torch(camera._reset_mask)
    camera._post_processing = pipeline
    camera._data = SimpleNamespace(output=pipeline.allocate(), info={})
    camera._render_camera_data = SimpleNamespace(output=pipeline.render_outputs, info={})
    camera._render_data = object()
    camera._renderer = SimpleNamespace(
        render=lambda data: rendered.append(data),
        read_output=lambda data, camera_data: None,
        cleanup=lambda data: None,
    )
    camera._update_camera_state = lambda **kwargs: None
    camera._update_poses = lambda *args, **kwargs: None
    monkeypatch.setattr(SimulationContext, "instance", staticmethod(lambda: None))

    first_data = camera.data
    assert camera.data is first_data
    assert len(processed) == len(rendered) == 1
    camera.update(0.05)
    assert camera.data is first_data
    assert len(processed) == 1
    camera.update(0.05)
    assert camera.data is first_data
    assert len(processed) == len(rendered) == 2
    camera.reset(env_ids=[1])
    np.testing.assert_array_equal(reset[-1], [False, True])
    assert camera.data is first_data
    np.testing.assert_array_equal(processed[-1], [False, True])
    camera.reset(env_mask=wp.array([True, False], dtype=wp.bool, device="cpu"))
    np.testing.assert_array_equal(reset[-1], [True, False])
    assert camera.data is first_data
    assert len(processed) == len(rendered) == 4
    del camera


@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_camera_initialization_failure_releases_processor_and_renderer_state(fail_cleanup):
    """A partial camera failure closes every resource and preserves the original diagnostic."""
    from isaaclab.sensors.camera import Camera

    camera = Camera.__new__(Camera)
    camera._clear_callbacks = lambda: None
    closed = []
    render_data = object()

    def close_processor():
        closed.append("processor")
        if fail_cleanup:
            raise ValueError("cleanup failed")

    def initialize_camera():
        camera._post_processing = SimpleNamespace(close=close_processor)
        camera._renderer = SimpleNamespace(cleanup=lambda data: closed.append(data))
        camera._view = SimpleNamespace(close=lambda: closed.append("view"))
        camera._render_data = render_data
        raise RuntimeError("camera initialization failed")

    camera._initialize_camera = initialize_camera
    with pytest.raises(RuntimeError, match="camera initialization failed"):
        camera._initialize_impl()
    assert closed == ["processor", render_data, "view"]
    assert camera._post_processing is None
    assert camera._render_data is None
    assert camera._renderer is None
    assert camera._view is None
    camera.__del__()
    assert closed == ["processor", render_data, "view"]
