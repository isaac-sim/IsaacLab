# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the renderer→camera output contract."""

import warnings
from builtins import ExceptionGroup
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

pytest.importorskip("isaaclab_physx")

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg, TiledCameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import clone, validate
from isaaclab.utils.tensor_processors import DictTensorProcessor, DictTensorProcessorCfg
from isaaclab.utils.warp import ProxyArray

pytestmark = [pytest.mark.integration, pytest.mark.rendering]

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)


@pytest.mark.parametrize("background", [None, (0.25, 0.5, 0.75)])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_newton_hdr_background_preserves_unlit_geometry(background, device):
    """HDR misses use linear background color while unlit geometry remains black, including graph replay."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    sim_cfg = sim_utils.SimulationCfg(
        device=device, gravity=(0.0, 0.0, 0.0), physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())
    )
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
        scene_cfg.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.3),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                mass_props=sim_utils.MassCfg(mass=1.0),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            ),
        )
        scene_cfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            width=32,
            height=32,
            data_types=["rgb_hdr"],
            background_color=background,
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), convention="opengl"),
            spawn=PinholeCameraCfg(clipping_range=(0.1, 10.0)),
            renderer_cfg=NewtonWarpRendererCfg(enable_ambient_lighting=False, create_default_light=False),
        )
        scene = InteractiveScene(scene_cfg)
        camera = scene["camera"]
        sim.reset()
        expected = (0.0, 0.0, 0.0) if background is None else (0.050876088, 0.21404114, 0.52252155)
        for _ in range(2):
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            camera.update(sim.get_physics_dt(), force_recompute=True)
            pixels = camera.data.output["rgb_hdr"].warp.numpy()[0]
            np.testing.assert_allclose(pixels[16, 16], 0.0, atol=1e-6)
            np.testing.assert_allclose(pixels[0, 0], expected, atol=1e-6)


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
    assert specs[RenderBufferKind.RGB_HDR] == RenderBufferSpec(3, wp.float32)


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


@pytest.mark.parametrize("data_type", [RenderBufferKind.RGB_HDR])
def test_newton_warp_wraps_requested_hdr_output(data_type):
    """Newton wires both linear color outputs to its HDR output slot."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    wp.init()
    from isaaclab_newton.renderers.newton_warp_renderer import RenderData

    from isaaclab.utils.warp.proxy_array import ProxyArray

    fake_sensor = SimpleNamespace(model=SimpleNamespace(world_count=2, device="cpu"))
    camera_cfg = CameraCfg(width=4, height=3, prim_path="/World/Camera", spawn=_SPAWN)
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
    """CameraData allocates requested + supported outputs, aliasing rgb into rgba."""
    cfg = _make_camera_cfg(["rgb", "rgba", "rgb_hdr", "depth", "instance_segmentation"])
    specs = {
        RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
        RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
        RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
        RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
        RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
    }
    data = CameraData.allocate(
        data_types=cfg.data_types, height=8, width=16, num_views=2, device="cpu", supported_specs=specs
    )

    assert set(data.output.keys()) == {"rgba", "rgb", "rgb_hdr", "depth"}
    assert data.output["rgba"].shape == (2, 8, 16, 4)
    assert data.output["rgba"].dtype == wp.uint8
    assert data.output["depth"].shape == (2, 8, 16, 1)
    assert data.output["depth"].dtype == wp.float32
    assert data.output["rgb"].warp.ptr == data.output["rgba"].warp.ptr
    assert data.output["rgb_hdr"].shape == (2, 8, 16, 3)
    assert data.output["rgb_hdr"].dtype == wp.float32
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


def test_every_camera_prepares_before_the_shared_stage_export(monkeypatch):
    """Every live camera prepares before stage export, including after stop and with callback error forwarding."""
    import weakref

    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    from isaaclab_newton.physics import NewtonCfg, NewtonManager

    from pxr import Sdf, UsdGeom

    from isaaclab.physics import PhysicsEvent
    from isaaclab.renderers import RendererCfg
    from isaaclab.sensors.camera import Camera, CameraExposureCfg
    from isaaclab.sensors.camera import camera as camera_module
    from isaaclab.sensors.sensor_base import SensorBase

    specs = {
        RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
        RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
        RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
    }
    exported_iso = []
    prepare_error = None

    def prepare_cameras(stage, spec):
        if prepare_error is not None:
            raise prepare_error

    def export_stage(stage, num_envs):
        exported_iso.append(prims[1].GetAttribute("exposure:iso").Get())

    renderer = SimpleNamespace(
        initialize=lambda: None,
        supported_output_types=lambda: specs,
        prepare_cameras=prepare_cameras,
        prepare_stage=export_stage,
        create_render_data=lambda spec: SimpleNamespace(spec=spec),
        set_outputs=lambda data, outputs: None,
        cleanup=lambda data: None,
        close=lambda: None,
    )
    renderer_cfg = RendererCfg(class_type=lambda cfg: renderer, renderer_type="contract")
    monkeypatch.setattr(SensorBase, "_initialize_impl", lambda self: None)
    monkeypatch.setattr(Camera, "_initialize_intrinsics", lambda self: None)
    monkeypatch.setattr(Camera, "_update_poses", lambda self: None)
    monkeypatch.setattr(
        camera_module,
        "FrameView",
        lambda path, **kwargs: SimpleNamespace(count=2, prims=[stage.GetPrimAtPath(path)] * 2, close=lambda: None),
    )

    class Identity(DictTensorProcessor):
        def __init__(self, cfg, data):
            super().__init__(cfg, data)
            self.outputs["processed_radiance"] = ProxyArray(wp.empty_like(data["rgb_hdr"].warp))

        def __call__(self, data):
            wp.copy(self.outputs["processed_radiance"].warp, data["rgb_hdr"].warp)
            return dict(self.outputs)

    def make_camera(prim, data_types):
        camera = Camera.__new__(Camera)
        pipeline = (
            [DictTensorProcessorCfg(Identity, ["rgb_hdr"], ["processed_radiance"])]
            if data_types == ["processed_radiance"]
            else []
        )
        camera.cfg = CameraCfg(
            prim_path=str(prim.GetPath()),
            data_types=data_types,
            height=2,
            width=3,
            spawn=None,
            pipeline=pipeline,
            exposure=CameraExposureCfg() if pipeline else None,
            renderer_cfg=renderer_cfg,
        )
        camera._render_cfg = replace(camera.cfg, data_types=camera.cfg.render_data_types(), pipeline=[])
        camera.stage = stage
        camera._device = "cpu"
        camera._num_envs = 2
        camera._is_initialized = False
        camera._sensor_prims = []
        camera._renderer = renderer
        camera._render_data = None
        camera._view = None
        camera._register_callbacks()
        return camera

    sim_cfg = sim_utils.SimulationCfg(device="cpu", physics=NewtonCfg())
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        stage = sim.stage
        prims = [UsdGeom.Camera.Define(stage, path).GetPrim() for path in ("/World/First", "/World/Second")]
        iso = prims[1].CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float)
        monkeypatch.setattr(
            sim, "get_clone_plan", lambda: SimpleNamespace(topology=SimpleNamespace(world_prototype_layout=np.zeros(2)))
        )
        sim.get_or_create_backend(renderer_cfg)
        cameras = [make_camera(prim, names) for prim, names in zip(prims, (["rgb"], ["processed_radiance"]))]
        discarded = make_camera(prims[0], ["rgb"])
        discarded_ref = weakref.ref(discarded)
        del discarded
        assert discarded_ref() is None

        try:
            for _ in range(2):
                iso.Set(100.0)
                sim.physics_manager.dispatch_event(PhysicsEvent.PHYSICS_READY)
                assert all(camera.is_initialized for camera in cameras)
                sim.physics_manager.dispatch_event(PhysicsEvent.STOP)
            assert exported_iso == [0.0, 0.0]

            # External event buses store callback failures instead of propagating them directly.
            errors = []
            monkeypatch.setattr(NewtonManager, "store_callback_exception", errors.append, raising=False)
            prepare_error = ValueError("camera preparation failed")
            sim.physics_manager.dispatch_event(PhysicsEvent.PHYSICS_READY)
            assert errors and errors[0] is prepare_error
            assert exported_iso == [0.0, 0.0]
        finally:
            for camera in cameras:
                camera.__del__()


def test_camera_pipeline_keeps_raw_inputs_and_processes_only_new_captures(monkeypatch):
    """Camera publication preserves raw pixels, capture metadata and calibration-compatible geometry."""
    from isaaclab.sensors.camera import Camera
    from isaaclab.sensors.sensor_base import SensorBase
    from isaaclab.sim import SimulationContext

    events = []
    output_shape = (2, 1, 1, 4)
    close_fails = False

    class Color(DictTensorProcessor):
        def __init__(self, cfg, data):
            super().__init__(cfg, data)
            self.outputs["rgba"] = ProxyArray(wp.empty(output_shape, dtype=wp.uint8, device=data["rgb_hdr"].device))

        def __call__(self, data):
            events.append("apply")
            output = self.outputs["rgba"].torch
            output[..., :3].copy_(100 * (data["rgb_hdr"].torch + data["depth"].torch))
            output[..., 3] = 255
            return dict(self.outputs)

        def reset(self, env_ids=None):
            events.append(tuple(env_ids))

        def close(self):
            events.append("close")
            if close_fails:
                raise RuntimeError("processor close failed")

    cfg = CameraCfg(
        prim_path="/World/Camera",
        spawn=None,
        height=1,
        width=1,
        data_types=["rgb", "depth"],
        pipeline=[DictTensorProcessorCfg(Color, ["rgb_hdr", "depth"], ["rgba"])],
    )
    specs = {
        RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
        RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
    }
    capture = {"frame": 1}
    radiance = 0.2

    def read_output(render_data, camera_data):
        camera_data.output["rgb_hdr"].torch.fill_(radiance)
        camera_data.output["depth"].torch.fill_(0.5)
        for name in camera_data.info:
            camera_data.info[name] = {"capture": capture}

    camera = Camera.__new__(Camera)
    camera.cfg = cfg
    camera._render_cfg = replace(cfg, data_types=cfg.render_data_types(), pipeline=[])
    camera._view = SimpleNamespace(count=2, close=lambda: None)
    camera._device = "cpu"
    camera._num_envs = 2
    camera._is_initialized = True
    camera._render_data = None
    camera._clear_callbacks = lambda: None
    camera._renderer = SimpleNamespace(
        supported_output_types=lambda: specs,
        set_outputs=lambda data, outputs: None,
        render=lambda data: None,
        read_output=read_output,
        reset=lambda data, ids: None,
        cleanup=lambda data: None,
    )
    monkeypatch.setattr(SimulationContext, "instance", staticmethod(lambda: None))
    monkeypatch.setattr(SensorBase, "reset", lambda self, ids, mask: None)
    monkeypatch.setattr(camera, "_initialize_intrinsics", lambda: None)
    monkeypatch.setattr(camera, "_update_poses", lambda **kwargs: None)
    monkeypatch.setattr(camera, "_prepare_camera", lambda mask: None)
    monkeypatch.setattr(camera, "_update_outdated_buffers", lambda: None)
    camera._create_buffers()
    mask = wp.ones(2, dtype=wp.bool, device="cpu")
    camera._update_buffers_impl(mask)

    assert set(camera.data.output) == {"rgb", "depth"}
    assert set(camera.render_outputs) == {"rgb_hdr", "depth"}
    torch.testing.assert_close(camera.data.output["rgb"].torch, torch.full((2, 1, 1, 3), 70, dtype=torch.uint8))
    torch.testing.assert_close(camera.render_outputs["rgb_hdr"].torch, torch.full((2, 1, 1, 3), 0.2))
    assert camera.data.output["depth"] is camera.render_outputs["depth"]
    assert camera.data.info["rgb"]["capture"] is capture

    camera._update_buffers_impl(mask)
    assert events == ["apply"]
    capture = {"frame": 2}
    radiance = 0.4
    camera._update_buffers_impl(mask)
    torch.testing.assert_close(camera.data.output["rgb"].torch, torch.full((2, 1, 1, 3), 90, dtype=torch.uint8))
    assert events == ["apply", "apply"]
    reset_mask = wp.array([False, True], dtype=wp.bool, device="cpu")
    camera.reset(env_mask=reset_mask)
    assert events[-1] == (1,)
    # A rebuilt pipe cannot publish resized pixels against unchanged camera calibration.
    output_shape = (2, 2, 1, 4)
    with pytest.raises(ValueError, match="camera calibration"):
        camera._create_buffers()
    assert camera._buffer_pipe is None
    assert events[-2:] == ["close", "close"]
    output_shape = (2, 1, 1, 4)
    monkeypatch.setattr(Color, "reset", DictTensorProcessor.reset)
    camera._create_buffers()
    # Stateless operations must not materialize indices: CUDA nonzero would synchronize the host.
    with monkeypatch.context() as patch:
        patch.setattr(
            torch.Tensor, "nonzero", lambda self: pytest.fail("Stateless camera reset materialized mask indices")
        )
        camera.reset(env_mask=reset_mask)
    camera._render_data = object()
    camera._renderer.cleanup = lambda data: events.append("renderer-close")
    camera._view.close = lambda: events.append("view-close")
    close_fails = True
    with pytest.raises(ExceptionGroup, match="close dictionary processors"):
        camera._invalidate_initialize_callback(None)
    assert events[-3:] == ["close", "renderer-close", "view-close"]
    assert not camera.is_initialized
    assert camera._renderer is camera._render_data is camera._view is camera._buffer_pipe is None
    camera.__del__()


def test_camera_requests_private_observation_inputs_before_initialization():
    """An observation may request raw inputs before reset, while late additions fail clearly."""
    from isaaclab.sensors.camera import Camera

    camera = Camera.__new__(Camera)
    camera.cfg = _make_camera_cfg(["rgb"])
    camera._render_cfg = replace(camera.cfg, pipeline=[])
    camera._render_outputs = {}
    camera._is_initialized = False
    camera._clear_callbacks = lambda: None
    camera.request_render_inputs(["depth", "rgb_hdr"])
    assert camera.cfg.data_types == ["rgb"]
    assert set(camera._render_cfg.data_types) == {"rgb", "depth", "rgb_hdr"}
    camera._is_initialized = True
    camera.request_render_inputs(["depth"])
    with pytest.raises(RuntimeError, match="before initialization"):
        camera.request_render_inputs(["normals"])


def test_camera_retired_isp_setting_reports_pipeline_migration():
    """Legacy camera ISP configuration fails before it can silently skip image processing."""
    with pytest.raises(ValueError, match="CameraCfg.pipeline"):
        CameraCfg(isp_cfg="auto_any")
