# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the renderer→camera output contract."""

import builtins
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

pytest.importorskip("isaaclab_physx")

from isaaclab.sensors.camera import CameraCfg, TiledCameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import clone, validate

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


@pytest.mark.parametrize(
    ("data_types", "message"),
    [
        (("rgb", "depth"), "does not support the following requested data types: \\['depth'\\]"),
        (("rgb", "not_a_type"), "Unknown camera data types: \\['not_a_type'\\]"),
    ],
)
def test_camera_buffers_reject_outputs_outside_a_deferred_renderer_contract(data_types, message):
    """A renderer config may defer its contract; the created renderer still validates requested outputs."""
    from isaaclab.sensors.camera import Camera

    camera = Camera.__new__(Camera)
    camera.cfg = SimpleNamespace(renderer_cfg=SimpleNamespace(supported_output_types=lambda: None))
    camera._renderer = SimpleNamespace(
        supported_output_types=lambda: {RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8, color_space="srgb")}
    )
    camera._render_data_types = data_types
    with pytest.raises(ValueError, match=message):
        camera._create_buffers()


@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_camera_initialization_failure_releases_renderer_state(fail_cleanup):
    """A partial camera failure closes every resource and preserves the original diagnostic."""
    from isaaclab.sensors.camera import Camera

    camera = Camera.__new__(Camera)
    camera._clear_callbacks = lambda: None
    closed = []
    render_data = object()

    def cleanup_renderer(data):
        closed.append(data)
        if fail_cleanup:
            raise ValueError("cleanup failed")

    def initialize_camera():
        camera._renderer = SimpleNamespace(cleanup=cleanup_renderer)
        camera._view = SimpleNamespace(close=lambda: closed.append("view"))
        camera._render_data = render_data
        raise RuntimeError("camera initialization failed")

    camera._initialize_camera = initialize_camera
    with pytest.raises(RuntimeError, match="camera initialization failed"):
        camera._initialize_impl()
    assert closed == [render_data, "view"]
    assert camera._render_data is None
    assert camera._renderer is None
    assert camera._view is None
    camera.__del__()
    assert closed == [render_data, "view"]


@pytest.mark.parametrize("supports_rgba", [False, True])
def test_all_camera_signals_prepare_before_shared_stage_export(monkeypatch, supports_rgba):
    """Public and private inputs reach shared renderer setup without changing public output layouts."""
    from pxr import Sdf, Usd, UsdGeom

    from isaaclab.physics import PhysicsEvent, PhysicsManager
    from isaaclab.renderers.rtx_camera_overrides import apply_rtx_exposure_overrides
    from isaaclab.sensors.camera import Camera
    from isaaclab.sensors.camera import camera as camera_module
    from isaaclab.sensors.sensor_base import SensorBase
    from isaaclab.sim import SimulationContext

    original_import = builtins.__import__

    def without_ppisp(name, *args, **kwargs):
        if name == "isaaclab_ppisp" or name.startswith("isaaclab_ppisp."):
            raise AssertionError("Preparing raw camera inputs must not import PPISP.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_ppisp)

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
            assert all(prim.GetAttribute("exposure:iso").Get() == 0.0 for prim in prims)
            exports.append(stage.ExportToString())

    renderer = SimpleNamespace(
        supported_output_types=lambda: specs,
        prepare_cameras=prepare_cameras,
        create_render_data=lambda spec: SimpleNamespace(spec=spec),
        set_outputs=lambda data, outputs: bound_outputs.update({data.spec.camera_prim_paths[0]: outputs}),
        cleanup=lambda data: None,
    )
    sim = SimpleNamespace(
        device="cpu",
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
            data_types=["rgb_radiance"] if index == 1 else ["rgb"],
            height=2,
            width=3,
            renderer_cfg=SimpleNamespace(renderer_type="newton"),
        )
        camera.stage = stage
        camera._device = "cpu"
        camera._num_envs = 2
        camera._is_initialized = False
        camera._requested_render_inputs = ()
        camera._sensor_prims = []
        camera._renderer = renderer
        camera._render_data = None
        camera._view = None
        camera._register_callbacks()
        cameras.append(camera)

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
        with pytest.raises(RuntimeError, match="before sensor initialization"):
            first.request_render_inputs(("rgb_radiance",))
    finally:
        for camera in cameras:
            camera.__del__()
    assert not CameraPhysicsManager._callbacks
