# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX renderer output contract."""

import contextlib
import ctypes
import importlib.util
import logging
import sys
import types
from builtins import ExceptionGroup
from collections.abc import Callable
from typing import Literal
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.assets import AssetBaseCfg
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg, SimulationContext
from isaaclab.utils import replace, validate
from isaaclab.utils.warp import ProxyArray

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
    pytest.mark.ov,
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers import OVRTXBackendCfg, OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import (  # noqa: E402
        _DISABLE_LINUX_CUDA_CPU_SYNC_ENV,
        OVRTXBackend,
        OVRTXCameraRenderData,
        OVRTXRenderer,
        OVRTXRenderProduct,
        _AsyncWriteBuffers,
        _gpu_side_render_var_sync_enabled,
    )
else:
    OVRTXCameraRenderData = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None
    _DISABLE_LINUX_CUDA_CPU_SYNC_ENV = None
    _gpu_side_render_var_sync_enabled = None

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)
_SPLIT_IMAGE_SIZE = 64
_SPLIT_CUBE_SIZE = 1.0
_SPLIT_FAR_CUBE_X = 10.0
_SPLIT_CUBES = (
    ("near", 0.0, "near_cube", (0.8, 0.1, 0.1)),
    ("far", _SPLIT_FAR_CUBE_X, "far_cube", (0.1, 0.8, 0.1)),
)
_SPLIT_CAMERA_POSITIONS = (
    (0.0, 0.0, 5.0),
    (0.45, 0.45, 6.0),
    (0.3, -0.1, 7.0),
    (0.45, 0.2, 8.0),
    (_SPLIT_FAR_CUBE_X, 0.0, 9.0),
)
# Tolerances (atol, rtol) of the color means. HDR color scales with the camera exposure, so its tolerance is relative.
_SPLIT_COLOR_MEAN_TOLERANCES = {"rgb": (8.0, 0.0), "rgba": (8.0, 0.0), "rgb_hdr": (0.0, 0.05)}


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/envs/env_0/Camera",
        spawn=_SPAWN,
        data_types=data_types,
    )


def _make_ovrtx_camera_render_data(render_scope_name: str = "RenderCamera_0") -> OVRTXCameraRenderData:
    spec = CameraRenderSpec(_make_camera_cfg(["rgb"]), "cpu", 2, ("/World/envs/env_0/Camera",), 2)
    return OVRTXCameraRenderData(spec, render_scope_name, [f"/World/envs/env_{i}/Camera" for i in range(2)], "auto")


def _single_render_product(render_data: OVRTXCameraRenderData):
    """Return the render product of a sensor that must have exactly one.

    A test that reads only the first render product would not check the environments of a second one.
    """
    assert len(render_data.products) == 1, f"Expected one render product, got {render_data.render_product_paths}"
    return render_data.products[0]


def _make_ovrtx_renderer_without_backend() -> OVRTXRenderer:
    renderer = OVRTXRenderer(OVRTXRendererCfg(), sim=SimulationContext.instance())
    renderer.backend = OVRTXBackend.__new__(OVRTXBackend)
    renderer.scene = renderer.backend
    from isaaclab_ov.stage import OvstageBackend

    renderer.scene.commit = OvstageBackend.commit.__get__(renderer.scene)
    renderer.scene.query = OvstageBackend.query.__get__(renderer.scene)
    renderer.scene.next_camera_id = 0
    renderer.scene.clone_copies = []
    renderer.scene.clone_env_paths = ["/World/envs/env_0", "/World/envs/env_1"]
    renderer.backend.attached = False
    renderer.backend.render_products = set()
    renderer.backend.unrendered = {}
    cfg = OVRTXBackendCfg(scene_key=renderer.cfg, use_ovstage=False, read_gpu_transforms=True, device="cpu")
    renderer.scene.stage = renderer.scene.paths = None
    renderer.backend.cfg = cfg
    SimulationContext.instance()._backend_registry.append((cfg, renderer.backend))
    return renderer


def _camera_scope_exists(renderer: OVRTXRenderer, render_scope_name: str, use_ovstage: bool) -> bool:
    scope = f"/{render_scope_name}/"
    if use_ovstage:
        import ovstage

        camera_filter = ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.PREFIX, [scope])])
        with renderer.scene.stage.query(filter=camera_filter) as query:
            return query.result().total_prim_count > 0
    return any(path.startswith(scope) for path in renderer.backend.renderer.query_prims())


def _make_mocked_scene_renderer(monkeypatch, use_ovstage: bool, clone_env_paths: list[str]) -> OVRTXRenderer:
    """Return a renderer whose scene and backend are mocks, so render data can be created without OVRTX."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._initialized_scene = True
    renderer._device = "cpu"
    renderer.scene.clone_env_paths = clone_env_paths
    renderer._use_ovstage = use_ovstage
    renderer.scene.ordinal = 1
    renderer.backend.renderer = MagicMock()
    renderer.scene.stage = MagicMock()
    renderer.scene.paths = MagicMock()
    renderer.scene.paths.create_path_list_from_strings.side_effect = tuple
    renderer.scene.stage.query_from_path_list.side_effect = lambda paths: contextlib.nullcontext(object())
    for name in ("add_usd_reference_from_string", "apply_usd_changes", "remove_usd"):
        monkeypatch.setattr(ovrtx_renderer_module.ovstage.population, name, MagicMock())
    return renderer


@pytest.fixture(autouse=True)
def _simulation_registry(monkeypatch):
    from pxr import Usd

    from isaaclab.renderers import RenderContext

    for name in ("ISAAC_LAB_OVRTX_USE_OVSTAGE", "ISAAC_LAB_SHARE_OVSTAGE"):
        monkeypatch.delenv(name, raising=False)
    registry = []
    sim = types.SimpleNamespace(
        device="cpu",
        cfg=types.SimpleNamespace(physics=types.SimpleNamespace(mesh_variant_source_root=None)),
        _backend_registry=registry,
        _render_context=RenderContext(registry),
        backend_manager=types.SimpleNamespace(clone_context_type=None, backend=None, _clone_recipes=[]),
        clone_contexts={},
        stage=Usd.Stage.CreateInMemory(),
        plan=None,
    )
    sim.render_context = sim._render_context
    sim.get_clone_plan = lambda: sim.plan
    sim.get_scene_data_provider = lambda: types.SimpleNamespace(
        backend=types.SimpleNamespace(transform_paths=[]), get_geometry_points=lambda: {}
    )
    sim.get_backends = SimulationContext.get_backends.__get__(sim)
    sim.get_or_create_backend = SimulationContext.get_or_create_backend.__get__(sim)
    sim.close_backend = SimulationContext.close_backend.__get__(sim)
    monkeypatch.setattr(SimulationContext, "_instance", sim)


@pytest.mark.parametrize(
    "settings, with_physics, use_ovstage, shared_physics",
    [
        pytest.param((None, None), False, False, False, id="other-default"),
        pytest.param((None, None), True, False, False, id="ovphysx-default"),
        pytest.param(("1", None), False, True, False, id="other-ovstage"),
        pytest.param(("1", None), True, True, False, id="ovphysx-render-only"),
        pytest.param(("0", None), True, False, False, id="ovphysx-native"),
        pytest.param((None, "1"), True, True, True, id="shared"),
        pytest.param(("0", "1"), True, True, True, id="shared-override"),
    ],
)
def test_ovrtx_renderer_config_enables_supported_runtime_options(
    monkeypatch, tmp_path, settings, with_physics, use_ovstage, shared_physics
):
    """Preparation resolves routes before native allocation; registry ownership outlives borrowers."""
    from isaaclab_ov.cloner import OvPhysxReplicateContext, OvrtxReplicateContext, OvstageReplicateContext

    from pxr import UsdGeom

    from isaaclab.cloner import make_clone_plan, replicate

    monkeypatch.setenv("ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS", "1")
    config_kwargs: dict[str, object] = {}
    destroyed, redirected, stage_releases = [], [], []
    dependency = tmp_path / "bin/plugins/libosdCPU.so.3.6.0"
    dependency.parent.mkdir(parents=True)
    dependency.touch()
    loaded = []
    monkeypatch.setattr(ovrtx_renderer_module.ovstage, "__file__", str(tmp_path / "__init__.py"))
    monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))
    # Cache redirection can load the SDK too; isolate it with the other native entry points.
    monkeypatch.setenv("OVRTX_SHADER_CACHE_PATH", str(tmp_path / "shader-cache"))
    monkeypatch.setattr(ovrtx_renderer_module, "redirect_shader_cache", redirected.append)

    class RecordingRendererConfig:
        def __init__(self, **kwargs):
            assert loaded, "The renderer must load its native dependencies without viewer setup."
            config_kwargs.update(kwargs)

    monkeypatch.setattr(ovrtx_renderer_module, "RendererConfig", RecordingRendererConfig)
    monkeypatch.setattr(
        ovrtx_renderer_module, "Renderer", lambda cfg: types.SimpleNamespace(destroy=lambda: destroyed.append(cfg))
    )
    SimulationContext.instance().device = "cuda:1" if use_ovstage else "cpu"
    devices = {
        "cpu": types.SimpleNamespace(is_cuda=False),
        "cuda:1": types.SimpleNamespace(is_cuda=True, ordinal=1),
    }
    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_device", devices.__getitem__)
    for name, value in zip(("ISAAC_LAB_OVRTX_USE_OVSTAGE", "ISAAC_LAB_SHARE_OVSTAGE"), settings):
        if value is not None:
            monkeypatch.setenv(name, value)

    @contextlib.contextmanager
    def stage_resource(label):
        yield object()
        stage_releases.append(label)

    def create_stage(name, config):
        assert (
            config.runtime_default_hierarchy_computation_model
            is ovrtx_renderer_module.ovstage.HierarchyComputationModel.GPU_INCREMENTAL
        )
        return stage_resource("stage")

    monkeypatch.setattr(ovrtx_renderer_module.ovstage, "Stage", create_stage)
    monkeypatch.setattr(ovrtx_renderer_module.ovstage, "PathDictionary", lambda _: stage_resource("paths"))

    sim = SimulationContext.instance()
    if with_physics:
        sim.backend_manager.clone_context_type = OvPhysxReplicateContext
    if use_ovstage and with_physics:
        # An asset route alone does not configure a renderer or authorize changing physics ownership.
        routes = {OvPhysxReplicateContext: {0}, OvrtxReplicateContext: {1}}
        OvrtxReplicateContext.prepare(sim, routes)
        assert routes == {OvPhysxReplicateContext: {0}, OvrtxReplicateContext: {1}}
    cfg = OVRTXRendererCfg()
    renderer = sim.get_or_create_backend(cfg)
    shared = sim.get_or_create_backend(replace(cfg))
    assert shared is renderer
    assert renderer.backend is None
    assert renderer.scene is None
    assert not loaded and not redirected and not config_kwargs
    renderer.close()  # Teardown is also valid before preparation.

    assets = [AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Visual", cloning_contexts=(OvrtxReplicateContext,))]
    if with_physics:
        assets.append(AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Body", cloning_contexts=(OvPhysxReplicateContext,)))
    for name in ("Visual", "Body"):
        UsdGeom.Xform.Define(sim.stage, f"/World/envs/env_0/{name}")
    sim.plan = make_clone_plan(assets, (tuple(range(len(assets))),), 2, positions=np.zeros((2, 3), dtype=np.float32))
    replicate(sim.plan)
    sim.render_context.ensure_initialize()
    # Adding a consumer must retain the route selected before native construction.
    monkeypatch.setenv("ISAAC_LAB_SHARE_OVSTAGE", str(int(not shared_physics)))
    other = sim.get_or_create_backend(replace(cfg, enable_shadows=True))
    assert other.backend is not None
    assert (other.backend is renderer.backend) is shared_physics
    assert (other.scene is renderer.scene) is shared_physics
    assert loaded == [str(dependency)] * (1 if shared_physics else 2)
    assert len(redirected) == (1 if shared_physics else 2)
    assert renderer.backend.renderer is not None
    assert config_kwargs.get("active_cuda_gpus") == ("1" if use_ovstage else None)
    # Camera compatibility must retain the requested GPU scene-transform path,
    # including when simulation tensors originate on the CPU.
    assert config_kwargs["read_gpu_transforms"] is True
    assert config_kwargs["suppress_deprecation_warnings"] is True
    assert config_kwargs["texture_streaming_mode"] is ovrtx_renderer_module.TextureStreamingMode.SYNCHRONOUS
    expected_contexts = {OvstageReplicateContext if use_ovstage else OvrtxReplicateContext}
    if with_physics:
        expected_contexts.add(OvPhysxReplicateContext)
    assert set(sim.clone_contexts) == expected_contexts
    assert {source for source, _ in renderer.scene.clone_copies} == {
        "/World/envs/env_0/Visual",
        *(["/World/envs/env_0/Body"] if shared_physics else []),
    }
    assert bool(sim.backend_manager._clone_recipes) is with_physics
    # Repeated preparation retains the selected resource identity.
    resources = tuple(sim._backend_registry)
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", str(int(not use_ovstage)))
    OvrtxReplicateContext.prepare(sim, {OvrtxReplicateContext: set()})
    assert tuple(sim._backend_registry) == resources
    if use_ovstage:
        assert renderer.scene.cfg.population_domains == ovrtx_renderer_module.ovstage.PopulationDomain.RENDERING
        assert renderer.scene.cfg.populate_physics is shared_physics
        assert renderer.scene.cfg.defer_render_copies is shared_physics
    if shared_physics:
        with pytest.raises(ValueError, match="one OVRTX engine"):
            sim.get_or_create_backend(replace(renderer.cfg, log_level="verbose"))
        conflict = next(
            resource
            for cfg, resource in sim._backend_registry
            if isinstance(cfg, OVRTXRendererCfg) and cfg.log_level == "verbose"
        )
        sim.close_backend(conflict)
    renderer.close()
    renderer.close()
    assert not destroyed
    assert shared.backend.renderer is not None
    shared.close()
    other.close()
    assert not destroyed
    for backend in dict.fromkeys((renderer.backend, other.backend)):
        SimulationContext.instance().close_backend(backend)
    assert len(destroyed) == (1 if shared_physics else 2)
    assert redirected == destroyed
    assert not stage_releases
    if use_ovstage:
        for scene in dict.fromkeys((renderer.scene, other.scene)):
            SimulationContext.instance().close_backend(scene)
        assert stage_releases == ["paths", "stage"] * (1 if shared_physics else 2)
    sim.close_backend(renderer)
    sim.close_backend(other)
    assert not sim._backend_registry


# Each missing output runs with and without batching; ``use_ovstage`` only changes the ordinal bookkeeping.
@pytest.mark.parametrize(
    ("missing_output", "batch", "use_ovstage"),
    [
        (None, False, False),
        (None, True, True),
        ("product", False, True),
        ("product", True, False),
        ("frame", False, False),
        ("frame", True, True),
    ],
)
def test_ovrtx_render_submits_every_product_and_routes_requested_outputs(
    monkeypatch, use_ovstage, missing_output, batch
):
    """A submission covers every registered product, fills each requested camera, and rejects gaps."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = use_ovstage
    renderer._initialized_scene = True
    renderer._visual_material_writer_ref = None
    renderer.scene.ordinal = 7
    cameras = [_make_ovrtx_camera_render_data(f"RenderCamera_{index}") for index in range(2 if batch else 1)]
    products = {}
    processed = []
    submissions = []
    published_ordinals = []
    for camera in cameras:
        camera.warp_buffers = {str(RenderBufferKind.RGB_HDR): object(), str(RenderBufferKind.RGBA): object()}
        products[_single_render_product(camera).path] = types.SimpleNamespace(frames=[object()])
    renderer._camera_render_data = cameras
    other = OVRTXRenderer(OVRTXRendererCfg(enable_shadows=True), sim=SimulationContext.instance())
    other.backend = renderer.backend
    other._camera_render_data = [types.SimpleNamespace(render_product_paths=("/Render/UnrequestedCamera",))]
    renderer.backend.render_products.update(
        path for camera in (*cameras, *other._camera_render_data) for path in camera.render_product_paths
    )
    if missing_output == "product":
        del products[_single_render_product(cameras[-1]).path]
    elif missing_output == "frame":
        products[_single_render_product(cameras[-1]).path].frames.clear()

    def step(**kwargs):
        submissions.append(kwargs)
        return products

    def advance_write_floor(*, ordinal):
        published_ordinals.append(ordinal)
        return types.SimpleNamespace(wait=lambda: None)

    renderer.backend.renderer = types.SimpleNamespace(step=step)
    renderer.scene.stage = types.SimpleNamespace(advance_write_floor=advance_write_floor)
    monkeypatch.setattr(renderer, "_process_render_frames", lambda *args: processed.append(args))

    render = renderer.render_batch if batch else renderer.render
    request = cameras if batch else cameras[0]
    if missing_output is None:
        render(request)
        assert processed == [
            (camera, [products[_single_render_product(camera).path].frames[0]], camera.warp_buffers)
            for camera in cameras
        ]
    else:
        with pytest.raises(RuntimeError, match=_single_render_product(cameras[-1]).path):
            render(request)
        assert not processed

    assert len(submissions) == 1
    # Unrequested products are submitted too, and only the requested cameras are read back.
    assert submissions[0]["render_products"] == {
        *[path for camera in cameras for path in camera.render_product_paths],
        "/Render/UnrequestedCamera",
    }
    if use_ovstage:
        assert submissions[0]["ordinal"] == 7
        assert published_ordinals == [7]
        assert renderer.scene.ordinal == 8
    else:
        assert "ordinal" not in submissions[0]
        assert not published_ordinals


def test_lidar_and_camera_captures_use_separate_steps(monkeypatch):
    """A lidar capture submits only the requested product at its own sampling interval."""
    from isaaclab_ov.renderers import OVRTXLidarCfg

    from isaaclab.sensors import LidarData

    renderer = _make_mocked_scene_renderer(monkeypatch, use_ovstage=False, clone_env_paths=[])
    renderer._needs_transform_initialization = False
    camera = _make_ovrtx_camera_render_data()
    renderer._camera_render_data.append(camera)
    renderer.backend.render_products.update(camera.render_product_paths)
    cfg = OVRTXLidarCfg(
        prim_path="/Lidar", update_period=0.1, azimuth_deg=[0.0], elevation_deg=[0.0], fire_time_ns=[0], channel_id=[0]
    )
    lidar = renderer.create_lidar_render_data(cfg, [cfg.prim_path], 1)
    other = replace(cfg, prim_path="/OtherLidar", update_period=0.2)
    renderer.create_lidar_render_data(other, [other.prim_path], 1)
    native = renderer.backend.renderer
    native.step.return_value = {_single_render_product(camera).path: types.SimpleNamespace(frames=[object()])}
    monkeypatch.setattr(renderer, "_process_render_frames", MagicMock())
    publish = MagicMock()
    monkeypatch.setattr(renderer, "_publish_lidar_output", publish)

    renderer.render(camera)
    assert native.step.call_args.kwargs["render_products"] == set(camera.render_product_paths)

    data = LidarData(1, 1, "cpu")
    renderer.render_lidar(lidar, data)
    native.step.assert_called_with(render_products={lidar.render_product_path}, delta_time=0.1)
    publish.assert_called_once_with(lidar, native.step.return_value, data)


def test_lidar_readback_propagates_cpu_map_failure(monkeypatch):
    """A failed fallback map must reach the caller rather than publish an all-miss scan."""
    from isaaclab.sensors import LidarData

    renderer = _make_mocked_scene_renderer(monkeypatch, use_ovstage=False, clone_env_paths=[])
    renderer._warp_device = types.SimpleNamespace(stream=types.SimpleNamespace(cuda_stream=1))
    lidar = ovrtx_renderer_module.OVRTXLidarRenderData("Lidar", 0.1, ["/Lidar"])
    cloud = MagicMock()
    cloud.map.side_effect = [
        RuntimeError("Failed to map CUDA point cloud"),
        RuntimeError("Failed to map CPU point cloud"),
    ]
    frame = types.SimpleNamespace(render_vars={lidar.point_cloud_path: cloud})
    products = {lidar.render_product_path: types.SimpleNamespace(frames=[frame])}
    with pytest.raises(RuntimeError, match="Failed to map CPU point cloud"):
        renderer._publish_lidar_output(lidar, products, LidarData(1, 1, "cpu"))


@pytest.mark.parametrize("missing", ["product", "renderer", "backend"])
def test_lidar_capture_fails_when_unavailable(monkeypatch, missing):
    from isaaclab_ov.renderers import OVRTXLidarCfg

    from isaaclab.sensors import LidarData

    renderer = _make_mocked_scene_renderer(monkeypatch, use_ovstage=False, clone_env_paths=[])
    cfg = OVRTXLidarCfg(
        prim_path="/Lidar", update_period=0.1, azimuth_deg=[0.0], elevation_deg=[0.0], fire_time_ns=[0], channel_id=[0]
    )
    lidar = renderer.create_lidar_render_data(cfg, [cfg.prim_path], 1)
    if missing == "product":
        renderer.cleanup_lidar(lidar)
    elif missing == "renderer":
        renderer.backend.renderer = None
    else:
        renderer.backend = None

    with pytest.raises(RuntimeError, match=lidar.render_product_path):
        renderer.render_lidar(lidar, LidarData(1, 1, "cpu"))


def test_ovrtx_render_batch_empty_sequence_does_not_require_initialized_backend():
    """An empty render request has no backend work or initialization precondition."""
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.render_batch([])


@pytest.mark.parametrize("mode", ["sync", "ovstage", "async"])
@pytest.mark.parametrize("failed_frame", [True, False], ids=["failed-frame", "other-error"])
def test_ovrtx_failed_frame_error_names_the_render_product_split(mode, failed_frame):
    """A failed frame gets one note about the render product split. Other step errors stay unchanged."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._initialized_scene = True
    renderer._visual_material_writer_ref = None
    renderer._use_ovstage = mode == "ovstage"
    renderer.cfg.async_rendering = mode == "async"
    renderer.scene.ordinal = 1
    renderer.scene.stage = types.SimpleNamespace(advance_write_floor=lambda *, ordinal: MagicMock())
    unsplit = _make_ovrtx_camera_render_data()
    wrist_paths = [f"/World/envs/env_{i}/Wrist" for i in range(2)]
    wrist_spec = CameraRenderSpec(_make_camera_cfg(["rgb"]), "cpu", 2, tuple(wrist_paths), 2)
    split = OVRTXCameraRenderData(wrist_spec, "RenderCamera_1", wrist_paths, 1)
    renderer._camera_render_data = [unsplit, split]
    for camera in renderer._camera_render_data:
        renderer.backend.render_products.update(camera.render_product_paths)
    reason = "The renderer failed to produce a frame (render_products='...')." if failed_frame else "device lost"
    error = RuntimeError(f"Operation 'step' failed:\n  {reason}")
    operation = MagicMock()
    operation.wait.side_effect = error
    renderer.backend.renderer = types.SimpleNamespace(
        step=MagicMock(side_effect=error), step_async=lambda **kwargs: operation
    )

    # Under asynchronous rendering, the step failure surfaces when read_output waits for the frame.
    with pytest.raises(RuntimeError) as raised:
        renderer.render(unsplit)
        renderer.read_output(unsplit)
    assert raised.value is error
    if mode == "async":
        # The drain gets the same error object again. The note must not repeat.
        assert renderer.drain_pending_renders() == [error]
    notes = getattr(error, "__notes__", [])
    if failed_frame:
        assert len(notes) == 1
        assert "OVRTXRendererCfg.max_envs_per_render_product" in notes[0]
        assert "'/World/envs/env_0/Camera' (2 environments)" in notes[0]
        assert "Wrist" not in notes[0]
    else:
        assert notes == []


@pytest.mark.parametrize(
    "max_envs, valid",
    [("auto", True), ("all", True), (3, True), (0, False), (-2, False), (True, False), (None, False), ("ALL", False)],
)
def test_ovrtx_renderer_cfg_validates_render_product_split(max_envs, valid):
    """The split is "auto", "all" or an integer >= 1. A camera rejects other values when it validates its config."""
    cfg = replace(_make_camera_cfg(["rgb"]), renderer_cfg=OVRTXRendererCfg(max_envs_per_render_product=max_envs))
    if valid:
        validate(cfg)
    else:
        with pytest.raises(ValueError, match='max_envs_per_render_product must be "auto", "all" or an integer >= 1'):
            validate(cfg)


_FAILED_FRAME = (
    "Operation 'step(dt=0.0)' failed:\n  Failed to step the stage: The renderer failed to produce a frame"
    " (render_products='/RenderCamera_0/RenderProduct'). See the preceding errors."
)


def _add_search_camera(
    renderer: OVRTXRenderer, name: str, num_envs: int, max_envs: int | Literal["auto", "all"], width: int
):
    """Register one square ``rgb`` sensor of ``num_envs`` environments with ``renderer``.

    The renderer reads its split value when it registers a sensor. So each sensor can get its own value, like the
    sensors of several renderer configurations on one shared engine.
    """
    renderer.cfg.max_envs_per_render_product = max_envs
    cfg = _make_camera_cfg(["rgb"])
    cfg.width = cfg.height = width
    paths = tuple(f"/World/envs/env_{i}/{name}" for i in range(num_envs))
    return renderer.create_render_data(CameraRenderSpec(cfg, "cpu", num_envs, paths, num_envs))


def _make_search_cameras(
    monkeypatch, use_ovstage: bool, num_envs: int, cameras: list[tuple[int | Literal["auto", "all"], int]]
):
    """Return a mocked renderer and one registered sensor per ``(max_envs_per_render_product, width)``."""
    renderer = _make_mocked_scene_renderer(monkeypatch, use_ovstage, [f"/World/envs/env_{i}" for i in range(num_envs)])
    renderer.scene.paths.intern_path.side_effect = lambda path: types.SimpleNamespace(raw_id=0)
    renderer.scene.paths.intern_token.side_effect = lambda token: types.SimpleNamespace(raw_id=0)
    monkeypatch.setattr(renderer, "_process_render_frame", MagicMock())
    build = ovrtx_renderer_module.build_render_product_as_string
    # The CPU device has the ordinal -1, and USD cannot parse -1 in the uint[] deviceIds of the layer.
    monkeypatch.setattr(
        ovrtx_renderer_module,
        "build_render_product_as_string",
        lambda *args, **kwargs: build(*args, **{**kwargs, "device_id": 0}),
    )
    return renderer, [
        _add_search_camera(renderer, f"cam{index}", num_envs, max_envs, width)
        for index, (max_envs, width) in enumerate(cameras)
    ]


def _install_step(renderer: OVRTXRenderer, fails: Callable[[set[str]], bool], error: str = _FAILED_FRAME) -> list:
    """Replace the native steps with fakes that raise ``error`` when ``fails`` returns True for their render products.

    Returns:
        The render product set of every native step, in submission order.
    """
    submissions = []

    def step(*, render_products, **kwargs):
        submissions.append(set(render_products))
        if fails(render_products):
            raise RuntimeError(error)
        return {path: types.SimpleNamespace(frames=[object()]) for path in render_products}

    def step_async(**kwargs):
        # Cameras keep operations in sets, so the fake must be hashable.
        operation = MagicMock()
        try:
            operation.wait.return_value.fetch.return_value = step(**kwargs)
        except RuntimeError as failure:
            operation.wait.side_effect = failure
        return operation

    renderer.backend.renderer.step = step
    renderer.backend.renderer.step_async = step_async
    return submissions


def _assert_no_product_stepped_twice(submissions: list[set[str]]) -> None:
    """Check that no render product is in two steps. OVRTX can abort when it renders a failed render product again."""
    assert sum(map(len, submissions)) == len(set().union(*submissions))


def _referenced_render_vars(add_usd_reference: MagicMock) -> dict[str, list[str]]:
    """Reference the layer of the last call to ``add_usd_reference`` at the path of that call, like the scene does.

    Returns:
        The RenderVar paths of each render product in the composed layer.
    """
    from pxr import Sdf, Usd

    *_, usd, path = add_usd_reference.call_args.args
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(usd)
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim(path).GetReferences().AddReference(layer.identifier)
    return {
        str(prim.GetPath()): [str(target) for target in prim.GetRelationship("orderedVars").GetTargets()]
        for prim in stage.Traverse()
        if prim.GetTypeName() == "RenderProduct"
    }


@pytest.mark.parametrize(
    ("use_ovstage", "asynchronous"), [(False, False), (True, False), (False, True)], ids=["legacy", "ovstage", "async"]
)
def test_ovrtx_render_batch_halves_envs_per_render_product_while_the_first_frame_fails(
    monkeypatch, caplog, use_ovstage, asynchronous
):
    """A sensor with "auto" starts with one product, and each failed first frame halves its environments per product."""
    renderer, (camera,) = _make_search_cameras(monkeypatch, use_ovstage, 5, [("auto", 64)])
    renderer.cfg.async_rendering = asynchronous
    # The legacy path fails in its first step with zero time. The asynchronous mode fails in the wait for the operation.
    renderer._needs_transform_initialization = not use_ovstage and not asynchronous
    fail_all = False
    submissions = _install_step(renderer, lambda products: fail_all or len(products) < 3)

    with caplog.at_level(logging.WARNING):
        renderer.render_batch([camera])
    renderer.read_output(camera)

    # 5 -> (3, 2) -> (2, 2, 1). Each layout has a scope of its own and no failed layout is stepped again.
    layouts = [
        {"/RenderCamera_0/RenderProduct"},
        {"/RenderCamera_1/RenderProduct_0", "/RenderCamera_1/RenderProduct_1"},
        {f"/RenderCamera_2/RenderProduct_{k}" for k in range(3)},
    ]
    # On the legacy path, the working layout is stepped two times: first with zero time, then for the frame.
    assert submissions == layouts + ([] if use_ovstage or asynchronous else [layouts[2]])
    assert [product.num_envs for product in camera.products] == [2, 2, 1]
    # The scene references the layer of the last layout at its scope. The layer defines the stepped render products,
    # and they render the RenderVar that the sensor reads its rgb output from.
    population = ovrtx_renderer_module.ovstage.population if use_ovstage else renderer.backend.renderer
    assert _referenced_render_vars(population.add_usd_reference_from_string) == {
        path: [camera.render_var_keys["LdrColor"]] for path in layouts[2]
    }
    assert renderer.backend.render_products == layouts[2]
    assert not renderer.backend.unrendered
    assert "OVRTXRendererCfg.max_envs_per_render_product=2 " in caplog.text
    assert renderer._process_render_frame.call_count == 3

    # A layout that rendered is final: a later failure reaches the caller.
    fail_all = True
    if asynchronous:
        renderer.render_batch([camera])
        # A camera sensor that registers now first waits for the pending frame. Its error raises before the scene
        # changes.
        with pytest.raises(ExceptionGroup, match="before render product registration") as group:
            _add_search_camera(renderer, "late", 5, "auto", 64)
        (failure,) = group.value.exceptions
        assert "failed to produce a frame" in str(failure)
        camera.pending = camera.ready = None
    else:
        with pytest.raises(RuntimeError, match="failed to produce a frame"):
            renderer.render_batch([camera])
    assert submissions[-1] == layouts[2]
    assert renderer.backend.render_products == layouts[2]

    renderer.cleanup(camera)
    assert population.remove_usd.call_count == 3
    assert renderer.backend.render_products == set()


@pytest.mark.parametrize(
    ("use_ovstage", "asynchronous", "num_envs", "products_per_step"),
    [
        # 64 environments stop after the limit of 4 smaller layouts, with 4 environments per render product.
        pytest.param(False, False, 64, [1, 2, 4, 8, 16], id="limit-legacy"),
        pytest.param(False, True, 64, [1, 2, 4, 8, 16], id="limit-async"),
        # 4 environments stop after 2 smaller layouts, because 1 environment per render product cannot be halved.
        pytest.param(True, False, 4, [1, 2, 4], id="one-env-ovstage"),
    ],
)
def test_ovrtx_render_batch_stops_the_split_search_after_the_last_smaller_layout(
    monkeypatch, caplog, use_ovstage, asynchronous, num_envs, products_per_step
):
    """The search tries each smaller layout one time, down to 1 environment per render product.

    It tries at most 4 smaller layouts. Then it raises the last failure.
    """
    renderer, (camera,) = _make_search_cameras(monkeypatch, use_ovstage, num_envs, [("auto", 64)])
    renderer.cfg.async_rendering = asynchronous
    submissions = _install_step(renderer, lambda products: True)

    with pytest.raises(RuntimeError, match="failed to produce a frame") as error:
        renderer.render_batch([camera])

    assert [len(products) for products in submissions] == products_per_step
    (note,) = error.value.__notes__
    assert f"also failed with {len(products_per_step) - 1} smaller render product layout(s)" in note
    assert "Smaller render products do not fix a lack of GPU memory" in note

    # The camera left the steps. A later render batch raises at once and steps nothing.
    assert camera not in renderer.backend.unrendered
    assert renderer.backend.render_products == set()
    with pytest.raises(RuntimeError, match="renders no frames.*max_envs_per_render_product") as later:
        renderer.render_batch([camera])
    assert later.value.__cause__ is error.value
    assert [len(products) for products in submissions] == products_per_step
    _assert_no_product_stepped_twice(submissions)
    assert not caplog.records


@pytest.mark.parametrize("max_envs, layout", [(4, [4, 4]), ("all", [8])], ids=["integer", "all"])
def test_ovrtx_render_batch_does_not_search_without_auto(monkeypatch, caplog, max_envs, layout):
    """A failed frame of a sensor with "all" or an integer reaches the caller after one step, with the same layout."""
    renderer, (camera,) = _make_search_cameras(monkeypatch, False, 8, [(max_envs, 64)])
    failed = camera.render_product_paths
    submissions = _install_step(renderer, lambda products: True)

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="failed to produce a frame") as error:
        renderer.render_batch([camera])

    assert submissions == [set(failed)]
    (text,) = error.value.__notes__
    assert f"{camera.camera_paths[0]!r} ({layout[0]} environments)" in text
    # The same layout moves to a new scope, so a later render batch does not step the failed products again.
    assert [product.num_envs for product in camera.products] == layout
    assert renderer.backend.render_products == set(camera.render_product_paths)
    assert not renderer.backend.render_products & set(failed)
    assert camera.first_frame_error is None
    assert not caplog.records


def test_ovrtx_split_search_names_an_auto_sensor_without_one_camera_path_per_environment(monkeypatch, caplog):
    """The search cannot split an "auto" sensor with only one camera path. A warning and the error note name it."""
    renderer, _ = _make_search_cameras(monkeypatch, False, 8, [])
    cfg = _make_camera_cfg(["rgb"])
    # The camera is not below an environment root, so the renderer cannot complete its camera paths.
    cfg.prim_path = "/World/Camera"
    with caplog.at_level(logging.WARNING):
        camera = renderer.create_render_data(CameraRenderSpec(cfg, "cpu", 8, ("/World/Camera",), 8))
    assert not camera.searches_split
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    for text in ("/World/Camera", "1 camera path", "8 environments"):
        assert text in caplog.messages[0]
    submissions = _install_step(renderer, lambda products: True)

    with pytest.raises(RuntimeError, match="failed to produce a frame") as error:
        renderer.render_batch([camera])

    assert len(submissions) == 1
    (text,) = error.value.__notes__
    assert "only with one camera path per environment" in text
    assert "'/World/Camera' (8 environments)" in text
    assert [product.num_envs for product in camera.products] == [8]


@pytest.mark.parametrize(
    "written, failing_write",
    [(0, "add_usd_reference_from_string"), (1, "add_usd_reference_from_string"), (1, "query_prims")],
    ids=["first-write-fails", "last-write-fails", "last-copy-fails"],
)
def test_ovrtx_split_search_retires_every_failed_product_before_a_scene_write(monkeypatch, written, failing_write):
    """A scene write that fails during the search leaves no failed render product in the submitted set.

    The sensors without new render products leave the search. Their later renders name the write error as the cause.
    """
    renderer, cameras = _make_search_cameras(monkeypatch, False, 8, [("auto", 64), ("auto", 64)])
    failed = set(renderer.backend.render_products)
    submissions = _install_step(renderer, lambda products: True)
    # The scene writes of the first ``written`` sensors succeed, and the write of the next sensor fails. query_prims
    # starts the copy of the render product attributes, after the new render products joined the steps.
    scene_writes = [MagicMock()] * written + [OSError("scene write")]
    getattr(renderer.backend.renderer, failing_write).side_effect = scene_writes

    with pytest.raises(OSError, match="scene write") as error:
        renderer.render_batch(cameras)

    assert "failed to produce a frame" in str(error.value.__context__)
    # Only the new render products of the written sensors remain. No sensor keeps a failed one.
    assert renderer.backend.render_products == set().union(*(c.render_product_paths for c in cameras[:written]))
    assert not renderer.backend.render_products & failed
    assert list(renderer.backend.unrendered) == cameras[:written]
    # A later render batch raises at once and steps nothing.
    steps = len(submissions)
    with pytest.raises(RuntimeError, match="renders no frames.*max_envs_per_render_product") as later:
        renderer.render_batch(cameras)
    assert later.value.__cause__ is error.value
    assert len(submissions) == steps


def test_ovrtx_split_search_retires_every_failed_product_before_an_interrupted_scene_write(monkeypatch):
    """An interrupt in the first scene write of the search leaves no failed render product in the submitted set.

    An interrupt, for example Ctrl+C, is no ``Exception``, so it skips the clean-up after a failed scene write. OVRTX
    can abort when a later step renders a failed render product again.
    """
    renderer, cameras = _make_search_cameras(monkeypatch, False, 8, [("auto", 64), ("auto", 64)])
    failed = set(renderer.backend.render_products)
    _install_step(renderer, lambda products: True)
    renderer.backend.renderer.add_usd_reference_from_string.side_effect = KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        renderer.render_batch(cameras)

    # The second sensor gets no scene write, but its failed render products also left the steps.
    assert not renderer.backend.render_products & failed


def test_ovrtx_split_search_leaves_out_a_sensor_removed_before_its_first_frame(monkeypatch):
    """A sensor removed before its first frame gets no new render products when the first frame fails."""
    renderer, (kept, removed) = _make_search_cameras(monkeypatch, False, 8, [("auto", 64), ("auto", 64)])
    renderer.cleanup(removed)
    failed = kept.render_product_paths
    add_usd_reference = renderer.backend.renderer.add_usd_reference_from_string
    add_usd_reference.reset_mock()
    submissions = _install_step(renderer, lambda products: len(products) < 2)

    renderer.render_batch([kept])

    # Only the kept sensor gets a new scope, and no step renders a render product of the removed sensor.
    assert [c.args[1] for c in add_usd_reference.call_args_list] == [f"/{kept.render_scope_name}"]
    assert submissions == [set(failed), set(kept.render_product_paths)]


def test_ovrtx_split_search_skips_a_sensor_removed_during_the_search(monkeypatch):
    """A sensor that a cleanup removes during the search gets no new render products. The search goes on."""
    renderer, (first, removed, last) = _make_search_cameras(monkeypatch, False, 8, [("auto", 64)] * 3)
    failed = set(renderer.backend.render_products)
    removed_paths = removed.render_product_paths
    add_usd_reference = renderer.backend.renderer.add_usd_reference_from_string
    add_usd_reference.reset_mock()

    def write_and_clean_up(usd, path):
        # The scene write of the first sensor cleans up the second sensor, for example through a callback.
        add_usd_reference.side_effect = None
        renderer.cleanup(removed)
        return MagicMock()

    add_usd_reference.side_effect = write_and_clean_up
    submissions = _install_step(renderer, lambda products: bool(products & failed))

    renderer.render_batch([first, last])

    # Of equal sensors, the first in the order of the engine is halved. The last sensor keeps its split.
    assert [[product.num_envs for product in camera.products] for camera in (first, last)] == [[4, 4], [8]]
    assert [c.args[1] for c in add_usd_reference.call_args_list] == [
        f"/{first.render_scope_name}",
        f"/{last.render_scope_name}",
    ]
    assert removed.render_product_paths == removed_paths
    assert not renderer.backend.render_products & set(removed_paths)
    assert submissions[-1] == {*first.render_product_paths, *last.render_product_paths}


def test_ovrtx_split_search_gives_up_with_a_sensor_removed_during_the_search(monkeypatch):
    """A search that gives up raises the failed frame, also if a cleanup removes a sensor during the search."""
    # A sensor with 1 environment cannot be halved, so the search gives up on the "auto" sensor. It still moves the
    # sensor with an integer split to new render products, and that scene write cleans up the "auto" sensor.
    renderer, (fixed, removed) = _make_search_cameras(monkeypatch, False, 1, [(1, 64), ("auto", 64)])
    failed = set(renderer.backend.render_products)
    add_usd_reference = renderer.backend.renderer.add_usd_reference_from_string

    def write_and_clean_up(usd, path):
        add_usd_reference.side_effect = None
        renderer.cleanup(removed)
        return MagicMock()

    add_usd_reference.side_effect = write_and_clean_up
    _install_step(renderer, lambda products: True)

    with pytest.raises(RuntimeError, match="failed to produce a frame"):
        renderer.render_batch([fixed])

    assert renderer.backend.render_products == set(fixed.render_product_paths)
    assert not renderer.backend.render_products & failed


def test_ovrtx_render_batch_does_not_search_after_another_error(monkeypatch):
    """An error other than a failed frame reaches the caller unchanged and leaves the render products alone."""
    renderer, (camera,) = _make_search_cameras(monkeypatch, False, 8, [("auto", 64)])
    layout = camera.render_product_paths
    submissions = _install_step(renderer, lambda products: True, "Operation 'step' failed: device lost")

    with pytest.raises(RuntimeError, match="device lost") as error:
        renderer.render_batch([camera])

    assert submissions == [set(layout)]
    assert not hasattr(error.value, "__notes__")
    assert camera.render_product_paths == layout
    assert renderer.backend.render_products == set(layout)


def _plan_sensor(*num_envs: int, searches_split: bool = True, spp: int = 1) -> types.SimpleNamespace:
    """Stand-in with the fields that the split plan reads.

    It has one render product for each number in ``num_envs``. Each atlas is one row of 64 x 64 tiles.
    """
    products = [types.SimpleNamespace(num_envs=count, num_cols=count, num_rows=1) for count in num_envs]
    cfg = types.SimpleNamespace(render_settings={"omni:rtx:rtpt:spp": ("Int", spp)})
    return types.SimpleNamespace(
        searches_split=searches_split, products=products, width=64, height=64, spec=types.SimpleNamespace(cfg=cfg)
    )


def _plan_ids(plan) -> list[tuple[int, int | None]]:
    """Each sensor of ``plan`` by its identity with its next number of environments per render product.

    Two stand-ins with the same fields are equal, so only the identity tells them apart.
    """
    return [(id(data), max_envs) for data, max_envs in plan.max_envs_per_render_product]


def test_ovrtx_split_plan_halves_the_auto_sensor_with_the_largest_buffer():
    """The plan halves the "auto" sensor with the most atlas pixels times samples per pixel. Others keep their split.

    The halved sensor is not the first, so the test also checks that the plan keeps the order of the engine.
    """
    small, large, fixed = _plan_sensor(8, spp=4), _plan_sensor(16), _plan_sensor(64, searches_split=False)
    error = RuntimeError("OVRTX failed to produce a frame")

    plan = ovrtx_renderer_module._plan_split([large, small, fixed], error, 0)

    assert plan.halved is small
    assert _plan_ids(plan) == [(id(large), 16), (id(small), 4), (id(fixed), 64)]


def test_ovrtx_split_plan_halves_the_first_and_largest_render_product():
    """The plan halves the environments of the first render product, which has the most environments.

    10 environments with a limit of 3 give render products of 3, 3, 3 and 1 environments. The next limit is 2, not 1.
    """
    sensor = _plan_sensor(3, 3, 3, 1)

    plan = ovrtx_renderer_module._plan_split([sensor], RuntimeError("failed to produce a frame"), 0)

    assert plan.halved is sensor
    assert _plan_ids(plan) == [(id(sensor), 2)]


def test_ovrtx_split_plan_halves_the_first_of_equal_auto_sensors():
    """Of "auto" sensors with the same buffer size, the plan halves the first in the order of the engine."""
    first, second = _plan_sensor(16), _plan_sensor(16)

    plan = ovrtx_renderer_module._plan_split([first, second], RuntimeError("failed to produce a frame"), 0)

    assert plan.halved is first
    assert _plan_ids(plan) == [(id(first), 8), (id(second), 16)]


# "limit" stands for _MAX_SMALLER_LAYOUTS. Without ovrtx, ovrtx_renderer_module is None when pytest collects the tests.
@pytest.mark.parametrize(
    ("num_envs", "num_smaller_layouts"), [(16, "limit"), (1, 0)], ids=["after-the-limit", "one-environment"]
)
def test_ovrtx_split_plan_gives_up_on_every_auto_sensor(num_envs, num_smaller_layouts):
    """Without a sensor to halve, the plan gives up on the "auto" sensors and keeps the split of the others."""
    if num_smaller_layouts == "limit":
        num_smaller_layouts = ovrtx_renderer_module._MAX_SMALLER_LAYOUTS
    auto, fixed = _plan_sensor(num_envs), _plan_sensor(16, searches_split=False)

    plan = ovrtx_renderer_module._plan_split(
        [auto, fixed], RuntimeError("failed to produce a frame"), num_smaller_layouts
    )

    assert plan.halved is None
    assert _plan_ids(plan) == [(id(auto), None), (id(fixed), 16)]


def test_ovrtx_split_plan_is_empty_for_another_error():
    """Only a failed frame starts the search, so another error gives an empty plan."""
    plan = ovrtx_renderer_module._plan_split([_plan_sensor(16)], RuntimeError("device lost"), 0)

    assert plan == ovrtx_renderer_module._SplitPlan()


@pytest.mark.parametrize(
    "sensors, samples, too_large, layouts",
    [
        # Only sensors with "auto" are halved, the largest atlas first. Larger "all" and integer atlases stay whole.
        ([(16, 128), ("all", 128), ("auto", 32), ("auto", 64)], {}, [3], [[16], [16], [16], [8, 8]]),
        # 32 samples per pixel on a quarter of the pixels outrank the larger atlas with one sample per pixel.
        ([("auto", 64), ("auto", 32)], {1: 32}, [1], [[16], [8, 8]]),
        # Two sensors above the limit are halved one after the other.
        ([("auto", 64), ("auto", 64)], {}, [0, 1], [[8, 8], [8, 8]]),
    ],
    ids=["largest-auto", "more-samples", "each-sensor"],
)
def test_ovrtx_split_search_halves_the_sensor_with_the_largest_buffer_first(
    monkeypatch, caplog, sensors, samples, too_large, layouts
):
    """Each failed step halves one sensor with "auto", the highest pixels times samples per pixel first."""
    renderer, cameras = _make_search_cameras(monkeypatch, False, 16, sensors)
    for index, spp in samples.items():
        cameras[index].spec.cfg.render_settings = {"omni:rtx:rtpt:spp": ("Int", spp)}
    # A step fails while a sensor of too_large has more than 8 environments in its first and largest render product.
    submissions = _install_step(renderer, lambda products: any(cameras[i].products[0].num_envs > 8 for i in too_large))

    with caplog.at_level(logging.WARNING):
        renderer.render_batch(cameras)

    assert len(submissions) == len(too_large) + 1
    _assert_no_product_stepped_twice(submissions)
    assert [[product.num_envs for product in camera.products] for camera in cameras] == layouts
    assert not renderer.backend.unrendered
    assert len(caplog.records) == len(too_large)


def test_ovrtx_split_search_covers_every_renderer_of_a_shared_engine(monkeypatch):
    """One step renders the products of every renderer on the engine, so the search can halve another one's sensor."""
    renderer, (small,) = _make_search_cameras(monkeypatch, True, 16, [("auto", 32)])
    other = OVRTXRenderer(OVRTXRendererCfg(enable_shadows=True), sim=SimulationContext.instance())
    other.backend = other.scene = renderer.backend
    other._device = "cpu"
    other._use_ovstage = other._initialized_scene = True
    large = _add_search_camera(other, "large", 16, "auto", 64)
    shadows = {}
    build = ovrtx_renderer_module.build_render_product_as_string

    def record_shadows(spec, render_data, **kwargs):
        shadows[render_data.render_scope_name] = kwargs["enable_shadows"]
        return build(spec, render_data, **kwargs)

    monkeypatch.setattr(ovrtx_renderer_module, "build_render_product_as_string", record_shadows)
    submissions = _install_step(renderer, lambda products: len(products) < 3)

    # The renderer of the small sensor renders first. Its step fails for the sensor of the other renderer.
    renderer.render_batch([small])

    assert len(submissions) == 2
    _assert_no_product_stepped_twice(submissions)
    assert [[product.num_envs for product in camera.products] for camera in (small, large)] == [[16], [8, 8]]
    assert not renderer.backend.unrendered
    # Each sensor keeps the product settings of its own renderer.
    assert shadows == {small.render_scope_name: False, large.render_scope_name: True}


def _install_attribute_store(renderer: OVRTXRenderer, use_ovstage: bool) -> dict:
    """Replace the attribute reads and writes of the scene with fakes over a dictionary.

    Like OVRTX, the legacy fake raises when it reads a tag or an array without elements. Like OVStage, the OVStage fake
    rejects a write of the prim path, and both a write and the release of a read group invalidate the data of the reads.
    A write of the OVStage fake reads its data only in its wait, because only the wait guarantees that OVStage read it.

    Returns:
        The attributes of each render product: ``{path: {name: (value, is_array, semantic)}}``.
    """
    import ovrtx
    import ovstage

    store = {}
    if use_ovstage:
        renderer.scene.stage.query_from_path_list.side_effect = contextlib.nullcontext
        renderer.scene.paths.intern_token.side_effect = lambda name: types.SimpleNamespace(raw_id=0, name=name)
        renderer.scene.paths.token_to_string.side_effect = lambda token: token.name
        # The data that the read groups borrow. In OVStage, a write that adds attributes to other prims can change the
        # data of an open read group. So a write invalidates this data like the release of a group.
        borrowed = []

        def invalidate_borrowed(*exc_info):
            # A write that reads invalid data copies the sentinel 99.
            for tensor in borrowed:
                tensor[...] = np.array(99).astype(tensor.dtype)

        def column(name, value, is_array, semantic):
            # OVStage can map the row of a prim in the query to another data row of the column. So the fake puts
            # the value of the one prim after a row of zeros.
            rows = [np.zeros_like(value), np.array(value)]
            # An array column has one tensor per data row. A fixed-size column has one tensor with all data rows.
            tensors = rows if is_array else [np.stack(rows)]
            borrowed.extend(tensors)
            return types.SimpleNamespace(
                attribute=types.SimpleNamespace(name=name),
                is_array=is_array,
                semantic=semantic,
                data_row_index=lambda local: local + 1,
                # Like OVStage, the NumPy view of a column gives a half as uint16.
                array=lambda index: tensors[index].view(np.uint16) if value.dtype == np.float16 else tensors[index],
                # Like the borrowed view of OVStage, a NumPy array supports DLPack and keeps a half as float16.
                dlpack=lambda index: tensors[index],
            )

        def array_values(tensor):
            # Like OVStage, an array write takes the components of an element only from the lanes, not from the
            # shape. So a float3[] in the shape (n, 3) without lanes arrives as a float[] of 3 * n values.
            if not isinstance(tensor, ovstage.DLTensor):
                return np.from_dlpack(tensor).reshape(-1)
            values = ovstage.dltensor_to_numpy(tensor)
            if (tensor.dtype.code, tensor.dtype.bits) == (ovstage.DLDataTypeCode.kDLFloat, 16):
                # dltensor_to_numpy gives a half as uint16.
                values = values.view(np.float16)
            return values.reshape(-1, tensor.dtype.lanes) if tensor.dtype.lanes > 1 else values

        def read_attributes(query, attributes, ordinal_range):
            assert [token.name for token in attributes] == ["*"]
            (path,) = query
            group = MagicMock(columns=[column(name, *entry) for name, entry in store[path].items()])
            group.__enter__.return_value = group
            group.__exit__.side_effect = invalidate_borrowed
            read = MagicMock()
            read.__enter__.return_value.groups.return_value = [group]
            return read

        def write_attributes(query, writes, ordinal):
            if any(write.attribute.name == "usd-path" for write in writes):
                raise ovstage.OvstageError(
                    ovstage.ErrorCode.NOT_SUPPORTED,
                    "built-in attribute 'usd-path' is derived on demand and cannot be written",
                )
            invalidate_borrowed()

            def wait():
                for write in writes:
                    # Target i reads the source data row index_map[i]. An array has one tensor per data row. A
                    # fixed-size attribute has all data rows in one tensor. Like OVStage, the write copies the data.
                    tensors = write.tensors if isinstance(write.tensors, (list, tuple)) else [write.tensors]
                    rows = [array_values(tensor) for tensor in tensors] if write.is_array else np.from_dlpack(*tensors)
                    for path, index in zip(query, write.index_map, strict=True):
                        store[path][write.attribute.name] = (np.array(rows[index]), write.is_array, write.semantic)

            return types.SimpleNamespace(wait=wait)

        renderer.scene.stage.read_attributes.side_effect = read_attributes
        renderer.scene.stage.write_attributes.side_effect = write_attributes
        return store
    native = renderer.backend.renderer

    def write_attribute(prim_paths, attribute_name, tensor, semantic=ovrtx.Semantic.NONE):
        for path, value in zip(prim_paths, tensor, strict=True):
            store.setdefault(path, {})[attribute_name] = (np.asarray(value), False, semantic)

    def write_array_attribute(prim_paths, attribute_name, tensors, semantic=ovrtx.Semantic.NONE):
        for path, value in zip(prim_paths, tensors, strict=True):
            store.setdefault(path, {})[attribute_name] = (np.asarray(value), True, semantic)

    def read(name, paths):
        entries = [store[path][name] for path in paths]
        if any(
            semantic == ovrtx.Semantic.TAG or (is_array and not value.size) for value, is_array, semantic in entries
        ):
            raise RuntimeError(f"Read returned no data for {name!r}")
        return [value for value, _, _ in entries]

    native.query_prims.side_effect = lambda **kwargs: {
        path: {
            name: ovrtx.AttributeInfo(name, None, is_array, semantic)
            for name, (_, is_array, semantic) in entries.items()
        }
        for path, entries in store.items()
    }
    native.read_attribute.side_effect = lambda name, paths: np.stack(read(name, paths))
    native.read_array_attribute.side_effect = lambda name, paths: dict(zip(paths, read(name, paths), strict=True))
    native.write_attribute.side_effect = write_attribute
    native.write_array_attribute.side_effect = write_array_attribute
    return store


@pytest.mark.parametrize("use_ovstage", [False, True], ids=["legacy", "ovstage"])
def test_ovrtx_split_search_copies_the_attributes_of_the_failed_render_product(monkeypatch, use_ovstage):
    """Attributes written onto the failed render product reach the new ones, which keep their own layout.

    The copy skips the attributes of the layout, tags and arrays without elements, and goes on after each of them.
    """
    import ovrtx
    import ovstage

    renderer, (camera,) = _make_search_cameras(monkeypatch, use_ovstage, 4, [("auto", 64)])
    store = _install_attribute_store(renderer, use_ovstage)
    token, path_id = (
        (ovstage.AttributeSemantic.TOKEN_ID, ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID)
        if use_ovstage
        else (ovrtx.Semantic.TOKEN_ID, ovrtx.Semantic.PATH_ID)
    )
    add_render_products = renderer._add_render_products

    def add_and_author(data, spec):
        # The fake scene does not read the layer, so author the attributes that the layout owns here.
        add_render_products(data, spec)
        for product in data.products:
            store[product.path] = {
                "resolution": (np.array([product.num_cols * data.width, product.num_rows * data.height]), False, 0),
                "camera": (np.arange(product.num_envs, dtype=np.uint64), True, path_id),
                "orderedVars": (np.array([f"/{data.render_scope_name}/Vars/LdrColor"]), True, path_id),
            }
            if use_ovstage:
                store[product.path]["usd-path"] = (np.array(product.path), False, 0)

    monkeypatch.setattr(renderer, "_add_render_products", add_and_author)
    (failed,) = camera.render_product_paths
    # The copy leaves out these attributes. The layout of 4 environments has 2 x 2 tiles of 64 pixels.
    left_out = {
        "resolution": (np.array([128, 128]), False, 0),
        "camera": (np.arange(4, dtype=np.uint64), True, path_id),
        "orderedVars": (np.array(["/RenderCamera_0/Vars/LdrColor"]), True, path_id),
    }
    if use_ovstage:
        # OVStage reads the path of a prim like an attribute, but rejects a write of it.
        left_out["usd-path"] = (np.array(failed), False, 0)
    else:
        # OVRTX lists an applied schema as a tag without a value. Like OVRTX, the fake raises when it reads the tag
        # or the array without elements.
        left_out["OmniRtxSettingsCommonAdvancedAPI_1"] = (None, False, ovrtx.Semantic.TAG)
        left_out["omni:rtx:debug:view:target"] = (np.array([], dtype=np.uint64), True, 0)
    # A script writes the samples per pixel and the antialiasing mode. The layer authors waitForEvents. These
    # attributes come after the left-out attributes, so the copy must go on after each skipped attribute.
    store[failed] = {
        **left_out,
        "omni:rtx:rtpt:spp": (np.int32(5), False, 0),
        "omni:rtx:post:aa:op": (np.uint64(42), False, token),
        "omni:rtx:waitForEvents": (np.array([7, 8], dtype=np.uint64), True, token),
    }
    if use_ovstage:
        # The NumPy view of an OVStage column gives a half as uint16, which OVStage does not write as a half.
        store[failed]["omni:test:half"] = (np.float16(0.5), False, 0)
        store[failed]["omni:test:half_array"] = (np.array([0.5, 1.5], dtype=np.float16), True, 0)
        # An OVStage read gives the 3 components of each element of a float3[] as an axis.
        store[failed]["omni:test:float3s"] = (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), True, 0)
    _install_step(renderer, lambda products: len(products) < 2)

    renderer.render_batch([camera])

    assert [product.num_envs for product in camera.products] == [2, 2]
    for path in camera.render_product_paths:
        attributes = {
            name: (value.tolist(), is_array, semantic) for name, (value, is_array, semantic) in store[path].items()
        }
        expected = {
            # Each new render product keeps its own resolution, cameras and RenderVars: 2 x 1 tiles of 64 pixels.
            "resolution": ([128, 64], False, 0),
            "camera": ([0, 1], True, path_id),
            "orderedVars": (["/RenderCamera_1/Vars/LdrColor"], True, path_id),
            "omni:rtx:rtpt:spp": (5, False, 0),
            "omni:rtx:post:aa:op": (42, False, token),
            "omni:rtx:waitForEvents": ([7, 8], True, token),
        }
        if use_ovstage:
            expected["usd-path"] = (path, False, 0)
            expected["omni:test:half"] = (0.5, False, 0)
            expected["omni:test:half_array"] = ([0.5, 1.5], True, 0)
            expected["omni:test:float3s"] = ([[1, 2, 3], [4, 5, 6]], True, 0)
        # No tag and no array without elements reach the new render products.
        assert attributes == expected
        if use_ovstage:
            assert store[path]["omni:test:half"][0].dtype == store[path]["omni:test:half_array"][0].dtype == np.float16


@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.parametrize(
    "use_ovstage, asynchronous, geometry",
    [
        (False, False, "rigid"),
        (True, False, "rigid"),
        *[(False, True, kind) for kind in ("mesh", "particles", "cable")],
    ],
    ids=["legacy", "ovstage", "async-mesh", "async-particles", "async-cable"],
)
def test_ovrtx_multiple_cameras_render_independent_views(monkeypatch, capfd, use_ovstage, asynchronous, geometry):
    """Cameras preserve independent captures, GPU input lifetimes, and reset/cleanup boundaries.

    In the asynchronous rows, the second camera is a split sensor with two render products.
    """
    from ovrtx import PrimMode

    from pxr import Gf, Usd, UsdGeom, UsdLux

    from isaaclab.cloner import make_clone_plan, replicate
    from isaaclab.renderers.camera_render_spec import CameraRenderSpec
    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
    from isaaclab.utils.math import convert_camera_frame_orientation_convention
    from isaaclab.utils.warp import ProxyArray

    if not torch.cuda.is_available():
        pytest.skip("OVRTX rendering requires CUDA")
    SimulationContext.instance().device = "cuda:0"
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", str(int(use_ovstage)))
    env_template = "/Lab/Cells/Cell{}" if use_ovstage else "/World/envs/env_{}"
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, env_template.format(0))
    if not use_ovstage:
        UsdGeom.Xform.Define(stage, env_template.format(1))
    if geometry == "rigid":
        light = UsdLux.DomeLight.Define(stage, env_template.format(0) + "/Dome")
        light.CreateIntensityAttr(300.0)
        light.CreateColorAttr((0.8, 0.8, 0.8))
    else:
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)
        # RTX clips deforming meshes to initial partition bounds; this off-camera wall encloses their motion.
        # https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#limitations
        wall = UsdGeom.Cube.Define(stage, env_template.format(0) + "/Wall")
        wall.CreateSizeAttr(8.0)
        wall.AddTranslateOp().Set(Gf.Vec3d(10, 0, 3))
    batches, initial_points = [], []
    for index, x in enumerate((0.0, 2.0)):
        camera = UsdGeom.Camera.Define(stage, env_template.format(0) + f"/cam{index}")
        camera.CreateProjectionAttr("orthographic")
        camera.CreateHorizontalApertureAttr(20.0)
        camera.CreateVerticalApertureAttr(20.0)
        camera.AddTranslateOp().Set(Gf.Vec3d(x, 0, 5))
        path = env_template.format(0) + f"/object{index}"
        if geometry == "rigid":
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([(0.8, 0.1, 0.1) if index == 0 else (0.1, 0.8, 0.1)])
            cube.AddTranslateOp().Set(Gf.Vec3d(x, 0, index))
            continue
        if geometry == "mesh":
            prim = UsdGeom.Mesh.Define(stage, path)
            prim.CreateFaceVertexCountsAttr([4])
            prim.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            vertices = [(x + dx, dy, index + 0.5) for dx, dy in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))]
        elif geometry == "particles":
            prim = UsdGeom.Points.Define(stage, path)
            prim.CreateWidthsAttr([1.0])
            vertices = [(x, 0, index)]
        else:
            prim = UsdGeom.BasisCurves.Define(stage, path)
            prim.CreateTypeAttr("linear")
            prim.CreateWrapAttr("nonperiodic")
            prim.CreateCurveVertexCountsAttr([3])
            prim.CreateWidthsAttr([1.0])
            prim.SetWidthsInterpolation("constant")
            vertices = [(x, y, index) for y in (-0.5, 0, 0.5)]
        prim.CreatePointsAttr(vertices)
        initial_points.append(np.asarray(vertices, dtype=np.float32))
        source = SceneDataFormat.Points()
        source.points = wp.array(vertices, dtype=wp.vec3f, device="cuda:0")
        batches.append(
            (
                source,
                {
                    path.replace(env_template.format(0), env_template.format(world)): (0, len(vertices))
                    for world in range(2)
                },
            )
        )

    renderer = SimulationContext.instance().get_or_create_backend(OVRTXRendererCfg(async_rendering=asynchronous))
    publication = types.SimpleNamespace(
        transform_paths=[], geometry_timestamp=0, get_geometry_batches=lambda _format: batches
    )
    renderer._sdp = SceneDataProvider(publication)
    plan = make_clone_plan(
        (AssetBaseCfg(prim_path=env_template.format("[^/]+"), cloning_contexts=renderer.cfg.cloning_contexts),),
        ((0,),),
        2,
        positions=np.zeros((2, 3), dtype=np.float32),
        env_template=env_template,
    )
    sim = SimulationContext.instance()
    sim.plan = plan
    replicate(plan)
    cameras = []

    def check_depth(rd, data, expected):
        depth = data.output["distance_to_image_plane"].torch
        assert depth.shape == (2, rd.height, rd.width, 1)
        torch.testing.assert_close(
            depth[:, rd.height // 2, rd.width // 2, 0],
            torch.full((2,), expected, device="cuda:0"),
            atol=0.02,
            rtol=0,
        )

    try:
        renderer.prepare_stage(stage, 2)
        assert renderer.visual_material_writer is not None
        sizes = ((96, 96), (384, 512)) if geometry == "rigid" else ((480, 640), (384, 512))
        for index, (height, width) in enumerate(sizes):
            cfg = _make_camera_cfg(
                ["rgb", "distance_to_image_plane"] if index == 0 else ["rgb", "distance_to_image_plane", "normals"]
            )
            cfg.height, cfg.width = height, width
            # One environment per render product splits the second camera in the asynchronous rows. The renderer
            # reads this value when it registers the camera.
            split = asynchronous and index == 1
            renderer.cfg.max_envs_per_render_product = 1 if split else "auto"
            spec = CameraRenderSpec(
                cfg=cfg,
                device="cuda:0",
                num_instances=2,
                view_count=2,
                camera_prim_paths=tuple(
                    env_template.format(i) + f"/cam{index}" for i in range(1 if use_ovstage else 2)
                ),
            )
            rd = renderer.create_render_data(spec)
            assert len(rd.products) == (2 if split else 1)
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
            renderer.update_geometries()
            renderer.render(rd)
            renderer.read_output(rd, data)
            check_depth(rd, data, 5.0 - index - 0.5)
            if index == 1:
                normals = data.output["normals"].torch[:, height // 2, width // 2, :3]
                torch.testing.assert_close(
                    torch.linalg.vector_norm(normals, dim=-1), torch.ones(2, device="cuda:0"), atol=0.02, rtol=0
                )

        if geometry == "rigid":

            def capture_lighting() -> np.ndarray:
                frames = []
                for _ in range(24):
                    renderer.render(cameras[0][0])
                    renderer.read_output(*cameras[0])
                    frames.append(cameras[0][1].output["rgb"].torch.cpu().numpy().copy())
                return np.asarray(frames, dtype=np.float32)

            baseline = capture_lighting()
            ordinary = capture_lighting()
            before = ordinary[-8:].mean(axis=0)
            noise = np.abs(ordinary - baseline[-8:].mean(axis=0)).mean(axis=(2, 3, 4)).max(axis=0)
            color = np.asarray([[0.9, 0.02, 0.02]], dtype=np.float32)
            dome_path = env_template.format(0) + "/Dome"
            if use_ovstage:
                with renderer.scene.query([dome_path]) as query:
                    renderer.scene.stage.write_attribute(
                        query,
                        "inputs:color",
                        ordinal=renderer.scene.ordinal,
                        tensors=color,
                        is_array=False,
                    ).wait()
                renderer.scene.commit()
            else:
                renderer.backend.renderer.write_attribute(
                    [dome_path], "inputs:color", color, prim_mode=PrimMode.MUST_EXIST
                )
            changed = capture_lighting()
            delta = np.abs(changed - before).mean(axis=(2, 3, 4))
            assert delta[-8:, 0].mean() > 15.0, f"Selected dome update was not rendered: {delta[-8:].mean(axis=0)}"
            assert np.all(delta[:, 1:] <= np.maximum(3.0, 4.0 * noise[1:])), (
                f"Dome update changed untouched views: peak={delta.max(axis=0)}, noise={noise}"
            )

        # Move only cam1 farther away. cam0 must retain its original depth.
        positions = ProxyArray(wp.array([[2.0, 0, 7]] * 2, dtype=wp.vec3f, device="cuda:0"))
        quats = convert_camera_frame_orientation_convention(
            torch.tensor([[0.0, 0, 0, 1.0]] * 2, device="cuda:0"), origin="opengl", target="world"
        )
        orientations = ProxyArray(wp.from_torch(quats, dtype=wp.quatf))
        with monkeypatch.context() as patches:
            patches.setattr(wp, "empty", MagicMock(side_effect=AssertionError("Camera updates must reuse GPU buffers")))
            renderer.update_camera(cameras[1][0], positions, orientations, cameras[1][1].intrinsic_matrices)
        renderer.render_batch([rd for rd, _ in cameras])
        for rd, data in cameras:
            renderer.read_output(rd, data)
        check_depth(*cameras[0], 4.5)
        check_depth(*cameras[1], 3.5 if asynchronous else 5.5)

        if asynchronous:
            renderer.render_batch([rd for rd, _ in cameras])
            for rd, data in cameras:
                renderer.read_output(rd, data)
            check_depth(*cameras[1], 5.5)
            # Reuse camera buffers across producer streams without advancing the other camera.
            for height in (6.0, 8.0, 9.0):
                with wp.ScopedStream(wp.Stream("cuda:0")):
                    positions = ProxyArray(wp.array([[0.0, 0, height]] * 2, dtype=wp.vec3f, device="cuda:0"))
                    renderer.update_camera(cameras[0][0], positions, orientations, cameras[0][1].intrinsic_matrices)
                    renderer.render(cameras[0][0])
                    renderer.read_output(*cameras[0])
                    wp.synchronize_stream()
                check_depth(*cameras[1], 5.5)
            check_depth(*cameras[0], 7.5)
            renderer.reset(cameras[0][0])
            renderer.render(cameras[0][0])
            renderer.read_output(*cameras[0])
            check_depth(*cameras[0], 8.5)
            for height in (1.0, 2.0, 3.0):
                with wp.ScopedStream(wp.Stream("cuda:0")):
                    for (source, _), vertices in zip(batches, initial_points, strict=True):
                        source.points.assign(vertices + (0, 0, height))
                    publication.geometry_timestamp += 1
                    renderer.update_geometries()
                    # Physics may overwrite its storage immediately after SDP snapshots it.
                    for source, _ in batches:
                        source.points.fill_(wp.vec3f(-100))
                    renderer.render_batch([rd for rd, _ in cameras])
                    for rd, data in cameras:
                        renderer.read_output(rd, data)
                    wp.synchronize_stream()
                check_depth(*cameras[0], 8.5 - (height - 1))
                check_depth(*cameras[1], 5.5 - (height - 1))
            renderer.reset(cameras[1][0], [0])
            renderer.render(cameras[1][0])
            renderer.read_output(*cameras[1])
            check_depth(*cameras[1], 2.5)
        assert all(_camera_scope_exists(renderer, rd.render_scope_name, use_ovstage) for rd, _ in cameras)
        renderer.cleanup(cameras[0][0])
        assert not _camera_scope_exists(renderer, cameras[0][0].render_scope_name, use_ovstage)
        renderer.render(cameras[1][0])
        renderer.read_output(*cameras[1])
        check_depth(*cameras[1], 2.5 if asynchronous else 5.5)
        renderer.cleanup(cameras[1][0])
        renderer.cleanup(cameras[1][0])
        assert not _camera_scope_exists(renderer, cameras[1][0].render_scope_name, use_ovstage)
    finally:
        renderer.close()
        SimulationContext.instance().close_backend(renderer.backend)
        if use_ovstage:
            SimulationContext.instance().close_backend(renderer.scene)
    # The default log level keeps native cloning quiet; lower levels print a warning per attribute of every cloned
    # environment, over a million lines at 8192 environments.
    assert "has no valid data" not in capfd.readouterr().out


def _assert_split_outputs_match(
    reference: dict[str, ProxyArray], outputs: dict[str, ProxyArray], names: list[str]
) -> None:
    """Check that the outputs ``names`` equal those of ``reference`` in each environment.

    Path tracing and DLSS depend on the pixel position in the atlas. So color outputs match only in the mean of
    each environment and channel. Other float outputs match per pixel, apart from rounding.
    """
    for name in names:
        expected, actual = reference[name].torch, outputs[name].torch
        if name in _SPLIT_COLOR_MEAN_TOLERANCES:
            expected, actual = expected.float().mean(dim=(1, 2)), actual.float().mean(dim=(1, 2))
            atol, rtol = _SPLIT_COLOR_MEAN_TOLERANCES[name]
        else:
            atol, rtol = 1e-4 if expected.is_floating_point() else 0, 0
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda detail: f"{name}: {detail}")


def _segmentation_key(pixel: torch.Tensor) -> int | tuple[int, ...]:
    """Return the label map key of a segmentation pixel: the raw ID, or the RGBA color of a colorized output."""
    return int(pixel[0]) if pixel.numel() == 1 else tuple(pixel.tolist())


@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.parametrize(
    "use_ovstage, colorize_semantic, colorize_instance, simple_shading_types",
    [
        pytest.param(
            False, True, False, ("simple_shading_constant_diffuse", "simple_shading_diffuse_mdl"), id="legacy"
        ),
        pytest.param(True, False, True, ("simple_shading_full_mdl",), id="ovstage"),
    ],
)
def test_ovrtx_split_sensor_outputs_match_single_product(
    monkeypatch, use_ovstage, colorize_semantic, colorize_instance, simple_shading_types
):
    """A sensor split into render products gives the outputs of one render product, in environment order.

    Each camera kind has two sensors: one with ``"all"``, so one render product, and one with 2 environments per
    render product. One more sensor with ``"auto"`` gets its split from the search after a failed first frame. The
    cameras of environments 0 to 3 see different parts of the near cube, so a swap of two environments changes every
    output, also the mean colors. Only the camera of environment 4 sees the far cube, so only the last render product
    of the split sees it.

    The colorize settings apply to every sensor of a renderer, and each renderer start and stop costs seconds. So
    the two rows swap the colorize values. Simple shading is a data type of a camera, so the rows spread the three
    simple shading types over their camera kinds. An asynchronous row would not test the wait for a failed step,
    because the injected failure raises before that wait.
    """
    import ovrtx
    import ovstage

    from pxr import Gf, Usd, UsdGeom, UsdLux

    from isaaclab.cloner import make_clone_plan, replicate
    from isaaclab.scene_data import SceneDataProvider
    from isaaclab.sim import add_labels
    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    if not torch.cuda.is_available():
        pytest.skip("OVRTX rendering requires CUDA")
    num_envs = len(_SPLIT_CAMERA_POSITIONS)
    SimulationContext.instance().device = "cuda:0"
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", str(int(use_ovstage)))
    env_template = "/Lab/Cells/Cell{}" if use_ovstage else "/World/envs/env_{}"
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)
    for index in range(1 if use_ovstage else num_envs):
        UsdGeom.Xform.Define(stage, env_template.format(index))
    # A render product has at most one simple shading type, and simple shading excludes rgb. So each simple shading
    # type needs a camera kind of its own.
    camera_kinds = [
        [
            "rgb",
            "rgb_hdr",
            "albedo",
            "normals",
            "motion_vectors",
            "distance_to_image_plane",
            "depth",
            "distance_to_camera",
            "semantic_segmentation",
            "instance_segmentation",
        ],
        *([data_type] for data_type in simple_shading_types),
    ]
    # One camera prim per sensor, so each sensor owns the pose binding of the prims it renders. Each kind has two
    # sensors, and the searched sensor has one more prim.
    for index in range(2 * len(camera_kinds) + 1):
        camera = UsdGeom.Camera.Define(stage, env_template.format(0) + f"/cam{index}")
        camera.CreateProjectionAttr("orthographic")
        camera.CreateHorizontalApertureAttr(12.0)
        camera.CreateVerticalApertureAttr(12.0)
        camera.AddTranslateOp().Set(Gf.Vec3d(*_SPLIT_CAMERA_POSITIONS[0]))
    for name, x, label, color in _SPLIT_CUBES:
        cube = UsdGeom.Cube.Define(stage, env_template.format(0) + f"/{name}")
        cube.CreateSizeAttr(_SPLIT_CUBE_SIZE)
        cube.CreateDisplayColorAttr([color])
        cube.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
        add_labels(cube.GetPrim(), [label])

    submissions = []
    failing = set()

    def record_products(native_step):
        def step(renderer, render_products, *args, **kwargs):
            submissions.append(set(render_products))
            if failing & submissions[-1]:
                raise RuntimeError(_FAILED_FRAME)
            return native_step(renderer, render_products, *args, **kwargs)

        return step

    # Both the legacy and the ovstage path render each frame with step.
    monkeypatch.setattr(ovrtx.Renderer, "step", record_products(ovrtx.Renderer.step))
    cfg = OVRTXRendererCfg(
        colorize_semantic_segmentation=colorize_semantic, colorize_instance_segmentation=colorize_instance
    )
    renderer = SimulationContext.instance().get_or_create_backend(cfg)
    renderer._sdp = SceneDataProvider(
        types.SimpleNamespace(transform_paths=[], geometry_timestamp=0, get_geometry_batches=lambda _format: [])
    )
    plan = make_clone_plan(
        (AssetBaseCfg(prim_path=env_template.format("[^/]+"), cloning_contexts=renderer.cfg.cloning_contexts),),
        ((0,),),
        num_envs,
        positions=np.zeros((num_envs, 3), dtype=np.float32),
        env_template=env_template,
    )
    SimulationContext.instance().plan = plan
    replicate(plan)

    positions = torch.tensor(_SPLIT_CAMERA_POSITIONS, device="cuda:0")
    # The identity orientation of the OpenGL convention looks down the z axis, at the cubes.
    quats = convert_camera_frame_orientation_convention(
        torch.tensor([[0.0, 0, 0, 1.0]] * num_envs, device="cuda:0"), origin="opengl", target="world"
    )
    # The center pixel of each camera sees the top of a cube on the ground.
    center = _SPLIT_IMAGE_SIZE // 2
    expected_depth = positions[:, 2] - _SPLIT_CUBE_SIZE / 2
    cameras = []

    def add_camera(data_types, max_envs):
        # The renderer reads its split value when it registers a sensor. One renderer start costs seconds, so the
        # sensors share one renderer and each sensor gets its own value.
        renderer.cfg.max_envs_per_render_product = max_envs
        camera_cfg = _make_camera_cfg(data_types)
        camera_cfg.height = camera_cfg.width = _SPLIT_IMAGE_SIZE
        spec = CameraRenderSpec(
            cfg=camera_cfg,
            device="cuda:0",
            num_instances=num_envs,
            view_count=num_envs,
            camera_prim_paths=tuple(
                env_template.format(i) + f"/cam{len(cameras)}" for i in range(1 if use_ovstage else num_envs)
            ),
        )
        rd = renderer.create_render_data(spec)
        data = CameraData.allocate(
            data_types=data_types,
            height=_SPLIT_IMAGE_SIZE,
            width=_SPLIT_IMAGE_SIZE,
            num_views=num_envs,
            device="cuda:0",
            supported_specs=renderer.supported_output_types(),
        )
        renderer.set_outputs(rd, data.output)
        renderer.update_camera(
            rd,
            ProxyArray(wp.from_torch(positions, dtype=wp.vec3f)),
            ProxyArray(wp.from_torch(quats, dtype=wp.quatf)),
            None,
        )
        cameras.append((rd, data))
        return rd, data

    def render(sensors):
        renderer.render_batch([rd for rd, _ in sensors])
        for rd, data in sensors:
            renderer.read_output(rd, data)

    def check_depth(data):
        depth = data.output["distance_to_image_plane"].torch[:, center, center, 0]
        torch.testing.assert_close(depth, expected_depth, atol=0.02, rtol=0)

    def check_labels(reference, split):
        # The split sensor takes its label maps from its first render product. That product does not see the far
        # cube, but a label map lists the labels of the whole scene.
        (near_name, _, near_label, _), (far_name, _, far_label, _) = _SPLIT_CUBES
        # Semantic IDs map to the class label, instance IDs to the prim path of the instance.
        for output, (near_text, far_text) in (
            ("semantic_segmentation", (near_label, far_label)),
            ("instance_segmentation", (f"/{near_name}", f"/{far_name}")),
        ):
            assert split.info[output] == reference.info[output], output
            id_to_labels = split.info[output]["idToLabels"]
            labels = [
                str(id_to_labels.get(_segmentation_key(pixel)))
                for pixel in split.output[output].torch[:, center, center]
            ]
            expected = [near_text] * (num_envs - 1) + [far_text]
            assert all(text in label for text, label in zip(expected, labels)), (output, labels)

    # Two settings of a background texture. The scene has no background texture, so they do not change the images.
    texture_settings = {
        "omni:rtx:background:source:texture:luminanceScale": 0.5,
        "omni:rtx:background:source:texture:colorSpace": "sRGB",
    }

    def write_texture_settings(path):
        # A script writes the settings with the runtime API, directly onto a render product.
        (scale_name, token_name), (scale, token) = texture_settings, texture_settings.values()
        if not use_ovstage:
            renderer.backend.renderer.write_attribute([path], scale_name, np.array([scale], np.float32))
            renderer.backend.renderer.write_attribute([path], token_name, [token], semantic=ovrtx.Semantic.TOKEN_STRING)
            return
        token_id = np.array([renderer.scene.paths.intern_token(token).raw_id], np.uint64)
        with renderer.scene.query([path]) as query:
            for name, tensors, semantic in (
                (scale_name, np.array([scale], np.float32), ovstage.AttributeSemantic.NONE),
                (token_name, token_id, ovstage.AttributeSemantic.TOKEN_ID),
            ):
                renderer.scene.stage.write_attribute(
                    query, name, ordinal=renderer.scene.ordinal, tensors=tensors, is_array=False, semantic=semantic
                ).wait()

    def read_texture_settings(path):
        # A token reads back as its ID. Render products with the same token have the same ID.
        if not use_ovstage:
            return [
                np.from_dlpack(renderer.backend.renderer.read_attribute(name, [path])).item()
                for name in texture_settings
            ]
        names = [renderer.scene.paths.intern_token(name) for name in texture_settings]
        current = ovstage.OrdinalRange.latest(ovstage.ORDINAL_MAX)
        with renderer.scene.query([path]) as query, renderer.scene.stage.read_attributes(query, names, current) as read:
            (group,) = read.groups()
            with group:
                return [column.array(0)[column.data_row_index(0)].item() for column in group.columns]

    def check_outputs(pairs):
        for (_, reference), (split_rd, split) in pairs:
            _assert_split_outputs_match(reference.output, split.output, split_rd.data_types)
        (_, reference), (_, split) = pairs[0]
        check_depth(reference)
        check_depth(split)
        check_labels(reference, split)

    try:
        renderer.prepare_stage(stage, num_envs)
        pairs = [tuple(add_camera(data_types, max_envs) for max_envs in ("all", 2)) for data_types in camera_kinds]
        for (single, _), (split, _) in pairs:
            assert single.render_product_paths == (f"/{single.render_scope_name}/RenderProduct",)
            assert [product.num_envs for product in split.products] == [2, 2, 1]
        assert renderer.backend.render_products == {path for rd, _ in cameras for path in rd.render_product_paths}

        render(cameras)
        assert submissions and all(submitted == renderer.backend.render_products for submitted in submissions)
        check_outputs(pairs)

        # Clear every buffer that the split sensors write. A render product that the next frame does not write then
        # keeps zeros.
        for _, (split_rd, _) in pairs:
            for buffer in split_rd.warp_buffers.values():
                buffer.zero_()
        render(cameras)
        check_outputs(pairs)

        # A sensor with "auto" joins. Its one render product fails the first frame, which the step simulates
        # here. The renderer halves the product in a new scope, and that layout renders for real.
        searched, searched_data = add_camera(["rgb", "distance_to_image_plane"], "auto")
        failed_scope = searched.render_scope_name
        (failed,) = searched.render_product_paths
        failing.add(failed)
        write_texture_settings(failed)
        render([(searched, searched_data)])
        assert [product.num_envs for product in searched.products] == [3, 2]
        # The new render products have the settings that the script wrote onto the failed one. Another render
        # product has its default settings, so the comparison sees a missing copy.
        (single, _), _ = pairs[0]
        assert read_texture_settings(failed) != read_texture_settings(single.render_product_paths[0])
        for path in searched.render_product_paths:
            assert read_texture_settings(path) == read_texture_settings(failed)
        assert not failing & renderer.backend.render_products
        assert submissions[-1] == renderer.backend.render_products
        check_depth(searched_data)
        (_, reference), _ = pairs[0]
        _assert_split_outputs_match(reference.output, searched_data.output, searched.spec.cfg.data_types)

        # Cleanup discards every render product of the sensor. The failed layout stays in the scene until then.
        # The remaining sensors do not render again here. The multi-camera test checks rendering after a cleanup.
        assert _camera_scope_exists(renderer, failed_scope, use_ovstage)
        while cameras:
            rd, _ = cameras.pop()
            renderer.cleanup(rd)
            assert not _camera_scope_exists(renderer, rd.render_scope_name, use_ovstage)
            assert renderer.backend.render_products == {
                path for other, _ in cameras for path in other.render_product_paths
            }
        assert not _camera_scope_exists(renderer, failed_scope, use_ovstage)
    finally:
        renderer.close()
        SimulationContext.instance().close_backend(renderer.backend)
        if use_ovstage:
            SimulationContext.instance().close_backend(renderer.scene)


@pytest.mark.parametrize("batch", [False, True], ids=["lazy", "batched"])
def test_async_cameras_publish_independently_with_capture_metadata_and_reset(monkeypatch, batch):
    """Real captures match delayed metadata; generated images and other cameras cannot replace them."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg.async_rendering = True
    renderer._initialized_scene = True
    renderer._visual_material_writer_ref = None
    cameras = [_make_ovrtx_camera_render_data(f"RenderCamera_{index}") for index in range(2)]
    renderer._camera_render_data = cameras.copy()
    camera_data = []
    for camera in cameras:
        data = CameraData.allocate(["rgb"], 2, 2, 2, "cpu", renderer.supported_output_types())
        data.create_buffers(2, "cpu")
        renderer.set_outputs(camera, data.output)
        camera_data.append(data)
        renderer.backend.render_products.update(camera.render_product_paths)
    frame = ProxyArray(wp.zeros(2, dtype=wp.int64, device="cpu"))
    operations = []
    ordinal = 0

    def submit(render_products, delta_time):
        operation = MagicMock()
        frames = [types.SimpleNamespace(end_time=float(ordinal), value=ordinal)]
        if batch:
            # SDK capture times identify the real image independently of the returned list order.
            frames.insert(0, types.SimpleNamespace(end_time=ordinal - 0.5, value=-ordinal))
            frames.append(types.SimpleNamespace(end_time=ordinal - 0.25, value=-ordinal))
        operation.wait.return_value.fetch.return_value = {
            path: types.SimpleNamespace(frames=frames) for path in render_products
        }
        operations.append(operation)
        return operation

    def consume(camera, frame, buffers):
        buffers["rgba"].fill_(frame.value)

    renderer.backend.renderer = types.SimpleNamespace(step_async=submit)
    consume = MagicMock(side_effect=consume)
    monkeypatch.setattr(renderer, "_process_render_frame", consume)

    def capture(value, indices):
        nonlocal ordinal
        ordinal = value
        frame.warp.fill_(value)
        for index in indices:
            data = camera_data[index]
            data.pos_w.warp.fill_(wp.vec3f(value))
            data.intrinsic_matrices.warp.fill_(wp.mat33f(value))
            renderer.prepare_capture(cameras[index], data, frame)
        if batch:
            renderer.render_batch([cameras[index] for index in indices])
        else:
            for index in indices:
                renderer.render(cameras[index])
        for index in indices:
            renderer.read_output(cameras[index], camera_data[index])

    capture(1, (0, 1))
    capture(2, (0, 1))
    capture(3, (0,))
    assert consume.call_count == 3  # Priming images are not extracted again on the next capture.
    for index, expected in enumerate((2, 1)):
        data = camera_data[index]
        np.testing.assert_array_equal(data.output["rgb"].warp.numpy(), expected)
        for name in ("frame", "pos_w", "intrinsic_matrices"):
            np.testing.assert_array_equal(data.info["rgb"]["capture"][name].warp.numpy(), expected)
    np.testing.assert_array_equal(camera_data[0].pos_w.warp.numpy(), 3)
    saved_capture = camera_data[0].info["rgb"]["capture"]
    renderer.read_output(cameras[0], camera_data[0])
    assert camera_data[0].info["rgb"]["capture"] is saved_capture
    assert renderer.drain_pending_renders() == []
    np.testing.assert_array_equal(camera_data[1].output["rgb"].warp.numpy(), 1)

    renderer.reset(cameras[0])
    capture(4, (0,))
    np.testing.assert_array_equal(camera_data[0].output["rgb"].warp.numpy(), 4)
    np.testing.assert_array_equal(saved_capture["pos_w"].warp.numpy(), 2)
    capture(5, (0, 1))
    assert consume.call_count == 5
    np.testing.assert_array_equal(camera_data[0].output["rgb"].warp.numpy(), 4)
    np.testing.assert_array_equal(camera_data[1].output["rgb"].warp.numpy(), 2)
    renderer.cleanup(cameras[0])

    monkeypatch.setattr(renderer, "_process_render_frame", MagicMock(side_effect=RuntimeError("output extraction")))
    with pytest.raises(RuntimeError, match="output extraction"):
        capture(6, (1,))
    renderer.cleanup(cameras[1])
    assert all(operation.wait.called for operation in operations)
    assert renderer._camera_render_data == []


def test_ovrtx_set_outputs_wraps_caller_torch_zero_copy():
    """OVRTXRenderer.set_outputs publishes warp views over the caller's warp storage."""
    renderer = _make_ovrtx_renderer_without_backend()

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    layouts = {
        "rgba": RenderBufferSpec(4, wp.uint8),
        "rgb_hdr": RenderBufferSpec(3, wp.float32),
        "depth": RenderBufferSpec(1, wp.float32),
        "motion_vectors": RenderBufferSpec(2, wp.float32),
    }
    cfg = _make_camera_cfg(["rgb", *layouts])
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

    for name, spec in layouts.items():
        assert render_data.warp_buffers[name].ptr == data.output[name].warp.ptr
        assert render_data.warp_buffers[name].shape == (2, 8, 16, spec.channels)
        assert render_data.warp_buffers[name].dtype is spec.dtype
    assert "rgb" not in render_data.warp_buffers


@pytest.mark.parametrize("source_device", ["cpu", "cuda:1"])
def test_ovrtx_process_frame_routes_hdr_to_output_device(monkeypatch, source_device):
    """HDR tiles are extracted on the output device, including after an OVRTX device fallback."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_camera_render_data()
    hdr = wp.full((8, 32, 4), 0.5, dtype=wp.float32, device="cpu")
    source = hdr if source_device == "cpu" else types.SimpleNamespace(device=source_device, dtype=wp.float32)
    clone = MagicMock(return_value=hdr)
    monkeypatch.setattr(wp, "clone", clone)
    extract = renderer._launch_extract_all_tiles

    def extract_hdr(data, tiled, output):
        assert str(tiled.device) == str(output.device)
        extract(data, tiled, output)

    monkeypatch.setattr(renderer, "_launch_extract_all_tiles", extract_hdr)

    @contextlib.contextmanager
    def map_hdr(render_var):
        assert render_var is source, "An rgb_hdr request must not read OVRTX LdrColor"
        yield source

    monkeypatch.setattr(renderer, "_map_render_var_to_dlpack", map_hdr)
    frame = types.SimpleNamespace(
        render_vars={
            render_data.render_var_keys["LdrColor"]: object(),
            render_data.render_var_keys["HdrColor"]: source,
        }
    )
    output = wp.zeros((2, 8, 16, 3), dtype=wp.float32, device="cpu")
    renderer._process_render_frame(render_data, frame, {"rgb_hdr": output})
    np.testing.assert_array_equal(output.numpy(), 0.5)
    if source_device == "cpu":
        clone.assert_not_called()
    else:
        clone.assert_called_once_with(source, device="cpu")


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_ovrtx_process_frame_reads_authored_camera_render_vars(monkeypatch, use_ovstage):
    """Both OVRTX APIs extract each camera's outputs using keys from its authored USD.

    The 5 environments fill an atlas of 3 x 2 tiles, and each tile has its own value. So a tile that is read
    from the wrong position gives a wrong value.
    """
    from pxr import Usd

    num_envs = 5
    renderer = _make_mocked_scene_renderer(monkeypatch, use_ovstage, [f"/World/envs/env_{i}" for i in range(num_envs)])

    stages = {}
    build_render_product = ovrtx_renderer_module.build_render_product_as_string

    def capture_render_product(spec, render_data, **kwargs):
        # USD requires a CUDA ordinal even though this test extracts buffers on the CPU.
        kwargs["device_id"] = 0
        usd = build_render_product(spec, render_data, **kwargs)
        stage = Usd.Stage.CreateInMemory()
        assert stage.GetRootLayer().ImportFromString(usd)
        product = next(prim for prim in stage.Traverse() if prim.GetTypeName() == "RenderProduct")
        stages[str(product.GetPath())] = stage
        return usd

    monkeypatch.setattr(ovrtx_renderer_module, "build_render_product_as_string", capture_render_product)

    @contextlib.contextmanager
    def fake_map(self, render_var):
        yield render_var

    monkeypatch.setattr(OVRTXRenderer, "_map_render_var_to_dlpack", fake_map)
    outputs = {
        "rgba": ("LdrColor", 4, wp.uint8),
        "albedo": ("DiffuseAlbedoSD", 4, wp.uint8),
        "depth": ("DistanceToImagePlaneSD", 1, wp.float32),
    }

    def tiled(values, channels, dtype):
        # Environment i is the tile in row i // 3 and column i % 3 of the 3 x 2 tile atlas.
        atlas = np.zeros((2 * 8, 3 * 16, channels), dtype=wp.dtype_to_numpy(dtype))
        for env, value in enumerate(values):
            row, col = divmod(env, 3)
            atlas[row * 8 : (row + 1) * 8, col * 16 : (col + 1) * 16] = value
        return wp.array(atlas, dtype=dtype, device="cpu")

    for camera_id in range(2):
        cfg = _make_camera_cfg(list(outputs))
        render_data = renderer.create_render_data(
            CameraRenderSpec(
                cfg=cfg,
                device="cpu",
                num_instances=num_envs,
                view_count=num_envs,
                camera_prim_paths=[f"/World/envs/env_{i}/cam{camera_id}" for i in range(num_envs)],
            )
        )
        stage = stages[_single_render_product(render_data).path]
        keys = {
            prim.GetAttribute("sourceName").Get(): str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetTypeName() == "RenderVar"
        }
        frame = types.SimpleNamespace(end_time=1.0, render_vars={})
        buffers = {}
        expected = {}
        for index, (output, (source, channels, dtype)) in enumerate(outputs.items(), start=1):
            expected[output] = [10 * camera_id + index + 20 * env for env in range(num_envs)]
            frame.render_vars[keys[source]] = tiled(expected[output], channels, dtype)
            buffers[output] = wp.zeros((num_envs, 8, 16, channels), dtype=dtype, device="cpu")
        frames = [frame]
        if camera_id == 1:
            generated = types.SimpleNamespace(
                end_time=0.5, render_vars={keys["LdrColor"]: tiled([99] * num_envs, 4, wp.uint8)}
            )
            frames.insert(0, generated)
        render_data.warp_buffers = buffers
        renderer._process_render_products(
            (render_data,), {_single_render_product(render_data).path: types.SimpleNamespace(frames=frames)}
        )
        for output, values in expected.items():
            for env, value in enumerate(values):
                np.testing.assert_array_equal(buffers[output].numpy()[env], value)
        render_data.cleanup()


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_ovrtx_process_frame_stitches_split_frames_in_env_order(monkeypatch, use_ovstage):
    """A split sensor registers one product per environment range and writes each product's frame into that range."""
    from .test_ovrtx_annotator_utils import _encode_identifier_map

    num_envs = 5
    camera_paths = [f"/World/envs/env_{i}/cam" for i in range(num_envs)]
    renderer = _make_mocked_scene_renderer(monkeypatch, use_ovstage, [f"/World/envs/env_{i}" for i in range(num_envs)])
    renderer.cfg.colorize_semantic_segmentation = False
    renderer.scene.paths.intern_path.side_effect = lambda path: types.SimpleNamespace(raw_id=camera_paths.index(path))
    renderer.scene.paths.intern_token.side_effect = lambda token: types.SimpleNamespace(
        raw_id=int(token.removeprefix("env_"))
    )
    monkeypatch.setattr(renderer, "_map_render_var_to_dlpack", contextlib.nullcontext)

    renderer.cfg.max_envs_per_render_product = 2
    cfg = _make_camera_cfg(["rgba", "depth", "semantic_segmentation"])
    render_data = renderer.create_render_data(CameraRenderSpec(cfg, "cpu", num_envs, tuple(camera_paths), num_envs))
    paths = render_data.render_product_paths
    assert paths == tuple(f"/RenderCamera_0/RenderProduct_{k}" for k in range(3))
    assert renderer.backend.render_products == set(paths)
    slices = [camera_paths[0:2], camera_paths[2:4], camera_paths[4:5]]
    if use_ovstage:
        queried = [c.args[0] for c in renderer.scene.paths.create_path_list_from_strings.call_args_list]
        assert queried[:3] == [[path] for path in paths]
        writes = [c for c in renderer.scene.stage.write_attribute.call_args_list if c.args[1] == "camera"]
        assert [c.kwargs["tensors"].tolist() for c in writes] == [[0, 1], [2, 3], [4]]
    else:
        assert renderer.backend.renderer.write_array_attribute.call_args_list == [
            call(prim_paths=[path], attribute_name="camera", tensors=[cameras])
            for path, cameras in zip(paths, slices, strict=True)
        ]

    # Products 0 and 1 hold two environments side by side, the last product holds one. Every product
    # carries the scene-wide label map, which is decoded once from the first product.
    def semantic_map(entries):
        render_var = MagicMock()
        encoded = _encode_identifier_map(entries)
        render_var.map.side_effect = lambda **kwargs: contextlib.nullcontext(encoded)
        return render_var

    entries = [((2, 0, 0, 0), "class: label2;"), ((3, 0, 0, 0), "class: label3;")]
    frames = []
    for value, shape in zip((2, 3, 4), ((8, 32), (8, 32), (8, 16)), strict=True):
        render_vars = {
            render_data.render_var_keys["LdrColor"]: wp.full((*shape, 4), value, dtype=wp.uint8, device="cpu"),
            render_data.render_var_keys["DistanceToImagePlaneSD"]: wp.full(
                (*shape, 1), float(value), dtype=wp.float32, device="cpu"
            ),
            render_data.render_var_keys["SemanticSegmentation"]: wp.full(
                (*shape, 1), value, dtype=wp.uint32, device="cpu"
            ),
            render_data.render_var_keys["SemanticIdMap"]: semantic_map(entries),
        }
        frames.append(types.SimpleNamespace(render_vars=render_vars))
    buffers = {
        "rgba": wp.zeros((num_envs, 8, 16, 4), dtype=wp.uint8, device="cpu"),
        "depth": wp.zeros((num_envs, 8, 16, 1), dtype=wp.float32, device="cpu"),
        "semantic_segmentation": wp.zeros((num_envs, 8, 16, 1), dtype=wp.int32, device="cpu"),
    }
    renderer._process_render_frames(render_data, frames, buffers)
    # The first product gives its labels and its raw tensors from one mapping of each label map.
    id_map_key = render_data.render_var_keys["SemanticIdMap"]
    assert [frame.render_vars[id_map_key].map.call_count for frame in frames] == [1, 1, 1]
    expected = np.array([2, 2, 3, 3, 4]).reshape(num_envs, 1, 1, 1)
    for output in buffers.values():
        np.testing.assert_array_equal(output.numpy(), np.broadcast_to(expected, output.shape))
    labels = render_data.renderer_info["semantic_segmentation"]["idToLabels"]
    assert {key: value for key, value in labels.items() if key >= 2} == {2: {"class": "label2"}, 3: {"class": "label3"}}
    # The same entries in another order give other raw bytes but the same labels, so the sensor does not raise.
    frames[1].render_vars[render_data.render_var_keys["SemanticIdMap"]] = semantic_map(entries[::-1])
    renderer._process_render_frames(render_data, frames, buffers)
    # A product whose label map differs would give wrong labels for its IDs, so the sensor raises.
    frames[2].render_vars[render_data.render_var_keys["SemanticIdMap"]] = semantic_map(entries[:1])
    with pytest.raises(
        RuntimeError, match="different SemanticIdMap for render product '/RenderCamera_0/RenderProduct_2'"
    ) as error:
        renderer._process_render_frames(render_data, frames, buffers)
    assert 'max_envs_per_render_product to "all"' in str(error.value)
    # A map that only a later product returns would leave the IDs of the sensor without labels, so the sensor raises.
    del frames[0].render_vars[id_map_key]
    with pytest.raises(
        RuntimeError, match="different SemanticIdMap for render product '/RenderCamera_0/RenderProduct_1'"
    ):
        renderer._process_render_frames(render_data, frames, buffers)
    # No product returns a map, so the sensor gives no labels, as one render product does.
    for frame in frames[1:]:
        del frame.render_vars[id_map_key]
    renderer._process_render_frames(render_data, frames, buffers)
    assert "semantic_segmentation" not in render_data.renderer_info
    render_data.cleanup()
    assert renderer.backend.render_products == set()


def test_launch_extract_all_tiles_rejects_wider_output_channels():
    """An output wider than the tiled input would read out of bounds, so it must raise before launching."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_camera_render_data()

    with pytest.raises(ValueError, match="out of bounds"):
        renderer._launch_extract_all_tiles(
            render_data, types.SimpleNamespace(shape=(8, 16, 3)), types.SimpleNamespace(shape=(2, 8, 16, 4))
        )


def test_ovrtx_process_frame_writes_each_render_product_into_its_environment_range(monkeypatch):
    """Each render product writes its frame only into the outputs of its own environment range.

    5 environments over render products of 3 and 2 environments give atlases of 2 x 2 and 2 x 1 tiles. Each tile
    has the value of its environment, so a tile in a wrong environment gives a wrong value. A missing frame or
    label map of the second render product raises before the first render product writes its outputs.
    """
    num_envs, untouched = 5, 255
    camera_paths = [f"/World/envs/env_{i}/Camera" for i in range(num_envs)]
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    monkeypatch.setattr(renderer, "_map_render_var_to_dlpack", contextlib.nullcontext)
    spec = CameraRenderSpec(_make_camera_cfg(["rgba"]), "cpu", num_envs, tuple(camera_paths), num_envs)
    render_data = OVRTXCameraRenderData(spec, "RenderCamera_0", camera_paths, "auto")
    render_data.products = [
        OVRTXRenderProduct("/RenderCamera_0/RenderProduct_0", camera_paths[:3], 0, 3, 2, 2),
        OVRTXRenderProduct("/RenderCamera_0/RenderProduct_1", camera_paths[3:], 3, 2, 2, 1),
    ]
    frames = []
    for product in render_data.products:
        atlas = np.zeros((product.num_rows * 8, product.num_cols * 16, 4), dtype=np.uint8)
        for index in range(product.num_envs):
            row, col = divmod(index, product.num_cols)
            atlas[row * 8 : (row + 1) * 8, col * 16 : (col + 1) * 16] = 10 + product.env_offset + index
        color = wp.array(atlas, dtype=wp.uint8, device="cpu")
        frames.append(types.SimpleNamespace(render_vars={render_data.render_var_keys["LdrColor"]: color}))
    render_data.warp_buffers = {"rgba": wp.full((num_envs, 8, 16, 4), untouched, dtype=wp.uint8, device="cpu")}

    def process(product_frames):
        renderer._process_render_products(
            (render_data,),
            {
                path: types.SimpleNamespace(frames=found)
                for path, found in zip(render_data.render_product_paths, product_frames, strict=True)
            },
        )

    with pytest.raises(RuntimeError, match="RenderProduct_1"):
        process([[frames[0]], []])
    np.testing.assert_array_equal(render_data.warp_buffers["rgba"].numpy(), untouched)

    # The label maps are only checked here, so placeholders stand in for them.
    for key in ("SemanticIdMap", "StableIdMap", "StableIdSemanticIdMap"):
        for frame in frames:
            frame.render_vars[render_data.render_var_keys[key]] = object()
    del frames[1].render_vars[render_data.render_var_keys["StableIdMap"]]
    render_data.warp_buffers["instance_segmentation"] = wp.zeros((num_envs, 8, 16, 1), dtype=wp.int32, device="cpu")
    with pytest.raises(RuntimeError, match="StableIdMap"):
        process([[frame] for frame in frames])
    np.testing.assert_array_equal(render_data.warp_buffers["rgba"].numpy(), untouched)

    del render_data.warp_buffers["instance_segmentation"]
    process([[frame] for frame in frames])
    expected = np.arange(10, 10 + num_envs, dtype=np.uint8).reshape(num_envs, 1, 1, 1)
    np.testing.assert_array_equal(
        render_data.warp_buffers["rgba"].numpy(), np.broadcast_to(expected, (num_envs, 8, 16, 4))
    )


@pytest.mark.parametrize(
    ("colorized", "palette"),
    [(False, {}), (True, {}), (True, {"class:cone": (255, 36, 66, 255)})],
    ids=["ids", "generated-colors", "class-palette"],
)
def test_segmentation_outputs_share_metadata_and_clear_stale_labels(monkeypatch, colorized, palette):
    """Pixel types, label keys and missing-map handling follow the requested output contract."""
    from isaaclab.renderers.segmentation_colors import random_color_from_id

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    renderer.cfg.semantic_segmentation_mapping = palette
    renderer.cfg.colorize_semantic_segmentation = renderer.cfg.colorize_instance_segmentation = colorized
    render_data = _make_ovrtx_camera_render_data()
    kinds = ["semantic_segmentation", "instance_segmentation"]
    camera_data = CameraData.allocate(
        data_types=["rgb", *kinds],
        height=8,
        width=16,
        num_views=2,
        device="cpu",
        supported_specs=renderer.supported_output_types(),
    )
    renderer.set_outputs(render_data, camera_data.output)
    seeded_keys = tuple(camera_data.info)
    stable_id = (42, 0, 0, 0)
    maps = {
        "SemanticIdMap": {
            "id": np.array([[2, 0, 0, 0]], dtype=np.uint32),
            "label_offsets": np.array([0, 12], dtype=np.uint64),
            "label_data": np.frombuffer(b"class: cone;", dtype=np.uint8),
        },
        "StableIdMap": {
            "id": np.array([stable_id], dtype=np.uint32),
            "label_offsets": np.array([0, 11], dtype=np.uint64),
            "label_data": np.frombuffer(b"/World/Cone", dtype=np.uint8),
        },
        "StableIdSemanticIdMap": {
            "stable_id": np.array([stable_id], dtype=np.uint32),
            "semantic_id": np.array([[2, 0, 0, 0]], dtype=np.uint32),
        },
    }
    frame = types.SimpleNamespace(render_vars={})
    for key, value in maps.items():
        render_var = MagicMock()
        render_var.map.side_effect = lambda value=value, **kwargs: contextlib.nullcontext(value)
        frame.render_vars[render_data.render_var_keys[key]] = render_var
    for key in ("SemanticSegmentation", "NonStableInstanceSegmentation"):
        frame.render_vars[render_data.render_var_keys[key]] = wp.full((8, 32, 1), 2, dtype=wp.uint32, device="cpu")
    monkeypatch.setattr(renderer, "_map_render_var_to_dlpack", contextlib.nullcontext)
    renderer._process_render_frames(render_data, [frame], render_data.warp_buffers)
    renderer.read_output(render_data, camera_data)
    pixel = random_color_from_id(2) if colorized else 2
    semantic_pixel = palette.get("class:cone", pixel) if colorized else pixel
    for kind in kinds:
        output = camera_data.output[kind].warp
        assert output.dtype == (wp.uint8 if colorized else wp.int32)
        assert output.shape == (2, 8, 16, 4 if colorized else 1)
        expected_pixel = semantic_pixel if kind == "semantic_segmentation" else pixel
        np.testing.assert_array_equal(output.numpy(), np.broadcast_to(expected_pixel, output.shape))
    assert camera_data.info[kinds[0]]["idToLabels"][semantic_pixel] == {"class": "cone"}
    assert camera_data.info[kinds[1]]["idToLabels"][pixel] == "/World/Cone"
    assert camera_data.info[kinds[1]]["idToSemantics"][pixel] == {"class": "cone"}
    frame.render_vars[render_data.render_var_keys["SemanticIdMap"]].map.assert_called_once()
    frame.render_vars.clear()
    with pytest.raises(RuntimeError, match="SemanticIdMap"):
        renderer._process_render_frames(render_data, [frame], render_data.warp_buffers)
    renderer._process_render_frames(render_data, [frame], {kinds[0]: render_data.warp_buffers[kinds[0]]})
    renderer.read_output(render_data, camera_data)
    assert camera_data.info == dict.fromkeys(seeded_keys)


@pytest.mark.parametrize(
    "setting, value, error",
    [
        ("ISAAC_LAB_OVRTX_USE_OVSTAGE", "true", "Expected 0 or 1"),
        ("ISAAC_LAB_SHARE_OVSTAGE", "1", "requires OVPhysX"),
    ],
)
def test_ovrtx_clone_preparation_rejects_invalid_selection(monkeypatch, setting, value, error):
    """Reject malformed flags and unsupported sharing before native allocation."""
    from isaaclab_ov.cloner import OvrtxReplicateContext

    sim = SimulationContext.instance()
    renderer = sim.get_or_create_backend(OVRTXRendererCfg())
    monkeypatch.setenv(setting, value)

    with pytest.raises(ValueError, match=error):
        OvrtxReplicateContext.prepare(sim, {})
    assert renderer.backend is None


@pytest.mark.parametrize(
    ("platform", "setting", "gpu_side"),
    [
        ("win32", None, True),
        ("linux", None, False),
        ("linux", "0", False),
        ("linux", "1", True),
    ],
)
def test_render_var_sync_respects_platform_and_override(monkeypatch, platform, setting, gpu_side):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, raising=False)
    if setting is not None:
        monkeypatch.setenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, setting)
    assert _gpu_side_render_var_sync_enabled() is gpu_side


@pytest.mark.parametrize("value", ["", "true", "yes", "2"])
def test_ovrtx_render_var_sync_rejects_non_boolean_values(monkeypatch, value):
    """Values other than 0/1 are a configuration error, not a silent fallback to the host wait."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, value)
    with pytest.raises(ValueError, match="Expected 0 or 1"):
        _gpu_side_render_var_sync_enabled()


@pytest.mark.parametrize(
    ("gpu_side", "cuda_stream", "expected_stream", "consumer_fails"),
    [(True, 0, 1, False), (True, 99, 99, True), (False, 0, 1, False)],
)
def test_ovrtx_map_render_var_orders_the_read_against_render_completion(
    monkeypatch, gpu_side, cuda_stream, expected_stream, consumer_fails
):
    """Order reads after rendering and release after reads, including when the consumer raises."""
    events = []
    mapping = types.SimpleNamespace(
        wait=lambda: events.append("wait"), unmap=lambda *, stream: events.append(("release", stream))
    )
    render_var = types.SimpleNamespace(map=MagicMock(return_value=mapping))
    sentinel = object()
    monkeypatch.setattr(ovrtx_renderer_module, "_gpu_side_render_var_sync_enabled", lambda: gpu_side)
    monkeypatch.setattr(ovrtx_renderer_module.wp, "from_dlpack", lambda mapping: sentinel)

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._warp_device = types.SimpleNamespace(stream=types.SimpleNamespace(cuda_stream=cuda_stream))
    with pytest.raises(ValueError, match="consumer failed") if consumer_fails else contextlib.nullcontext():
        with renderer._map_render_var_to_dlpack(render_var) as array:
            assert array is sentinel
            events.append("consume")
            if consumer_fails:
                raise ValueError("consumer failed")

    assert render_var.map.call_args.kwargs["sync_stream"] == (expected_stream if gpu_side else 0)
    assert events == ([] if gpu_side else ["wait"]) + ["consume", ("release", expected_stream)]


@pytest.mark.parametrize(
    "use_ovstage, cleanup_directly, failure",
    [(stage, direct, None) for stage in (False, True) for direct in (False, True)]
    + [(False, False, "render"), (False, False, "write")],
)
def test_ovrtx_cleanup_releases_only_the_given_render_data(monkeypatch, cleanup_directly, use_ovstage, failure):
    """Release only the given camera, even if its pending native operations fail."""
    events = []
    renderer = _make_renderer_with_backend(events, use_ovstage)
    other_camera = renderer._camera_render_data[0]
    render_data = _make_ovrtx_camera_render_data("RenderCamera_to_remove")
    renderer._camera_render_data.append(render_data)
    render_data.warp_buffers = {"rgba": wp.zeros((8, 16, 4), dtype=wp.uint8, device="cpu")}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": {}}}
    if use_ovstage:
        render_data.camera_xform_binding = "to_remove"
        render_data.resources.callback(renderer.scene.paths.destroy_path_list, "to_remove")
        render_data.resources.enter_context(renderer.scene.stage.query_from_path_list("to_remove"))
    else:
        render_data.camera_xform_binding = _RecordingBinding(events, "pose")
        render_data.resources.callback(render_data.camera_xform_binding.unbind)
        render_data.intrinsic_bindings = [_RecordingBinding(events, "intrinsics")]
        render_data.resources.callback(render_data.intrinsic_bindings[0].unbind)

    operations = [MagicMock(), MagicMock()]
    operations[-1].wait.side_effect = RuntimeError("device lost")
    if failure == "render":
        render_data.pending = (operations[-1], {})
    elif failure == "write":
        render_data.camera_writes = _AsyncWriteBuffers((object(), object()))
        binding = MagicMock()
        binding.write_async.side_effect = operations
        stream = types.SimpleNamespace(cuda_stream=0)
        for _ in operations:
            render_data.camera_writes.submit(binding, object(), stream)
        monkeypatch.setattr(wp, "synchronize_stream", lambda _stream: None)
    if failure:
        with pytest.raises(ExceptionGroup if failure == "render" else RuntimeError):
            renderer.cleanup(render_data)
        if failure == "write":
            operations[0].wait.assert_called_once_with()

    if cleanup_directly:
        render_data.cleanup()
    renderer.cleanup(None)
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
    assert render_data.intrinsic_bindings == []
    assert render_data.warp_buffers == {}
    assert render_data.renderer_info == {}
    assert render_data.pending is render_data.ready is None
    assert renderer._initialized_scene is True


@pytest.mark.parametrize("stage_mode", ["legacy", "typed_handles"])
def test_pose_and_intrinsic_updates_target_the_given_camera(monkeypatch, stage_mode):
    """Distinct cameras, including a prototype-only wrist camera, bind every environment independently."""
    use_ovstage = stage_mode != "legacy"

    class Handle:
        def __init__(self, raw_id):
            self.raw_id = raw_id

        def __int__(self):
            raise TypeError("use .raw_id for explicit numeric conversion")

    renderer = _make_mocked_scene_renderer(monkeypatch, use_ovstage, ["/Lab/Cell0", "/Lab/Cell1"])
    renderer.backend.renderer.bind_attribute.side_effect = lambda **kwargs: MagicMock()
    paths = [[f"/Lab/Cell{i}/{name}" for i in range(2)] for name in ("CameraA", "Robot/ee_link/palm_link/CameraB")]
    paths[0].reverse()
    paths[0][0] = "/Lab/Cell1/Robot/CameraA"
    paths.append(["/World/PerspectiveCamera"])
    paths.append(paths[0][:1])
    renderer.scene.paths.intern_token.side_effect = {"env_0": Handle(10), "env_1": Handle(11)}.__getitem__
    path_ids = {path: 2**63 + i for i, path in enumerate(dict.fromkeys(path for group in paths for path in group))}
    renderer.scene.paths.intern_path.side_effect = lambda path: Handle(path_ids[path])
    if use_ovstage:
        renderer.scene.render_copies_pending = False
        monkeypatch.setattr(renderer, "_setup_xform_bindings", lambda stage: None)
        monkeypatch.setattr(renderer, "_setup_geometry_bindings", lambda: None)
        renderer.scene.resources = contextlib.ExitStack()
        renderer.prepare_stage(SimulationContext.instance().stage, 2)
        partition_write = next(
            call
            for call in renderer.scene.stage.write_attribute.call_args_list
            if call.args[1] == "primvars:omni:scenePartition"
        )
        np.testing.assert_array_equal(partition_write.kwargs["tensors"], [10, 11])
        assert partition_write.kwargs["tensors"].dtype == np.uint64
        renderer.scene.paths.create_path_list_from_strings.reset_mock()
    cameras = [
        renderer.create_render_data(
            CameraRenderSpec(
                cfg=_make_camera_cfg(["depth"]),
                device="cpu",
                num_instances=len(paths[index]),
                view_count=len(paths[index]),
                camera_prim_paths=camera_paths,
            )
        )
        for index, camera_paths in enumerate((paths[0], paths[1][1:], paths[2], paths[3]))
    ]
    if use_ovstage:
        relationships = [
            call.kwargs["tensors"]
            for call in renderer.scene.stage.write_attribute.call_args_list
            if call.args[1] == "camera"
        ]
        for relationship, expected_paths in zip(relationships, paths, strict=True):
            np.testing.assert_array_equal(
                relationship, np.array([path_ids[path] for path in expected_paths], np.uint64)
            )
            assert relationship.dtype == np.uint64
        partitions = [
            call.kwargs["tensors"].tolist()
            for call in renderer.scene.stage.write_attribute.call_args_list
            if call.args[1] == "omni:scenePartition"
        ]
        assert partitions == [[11, 10], [10, 11], [11]]
    else:
        partitions = [
            call.args[2]
            for call in renderer.backend.renderer.write_attribute.call_args_list
            if len(call.args) > 1 and call.args[1] == "omni:scenePartition"
        ]
        assert partitions == [["env_1", "env_0"], ["env_0", "env_1"], ["env_1"]]
    assert not vars(renderer).keys() & {"_camera_xform_binding", "_camera_xform_query", "_camera_paths_list"}
    monkeypatch.setattr(wp, "get_stream", lambda device: types.SimpleNamespace(cuda_stream=99))
    monkeypatch.setattr(wp, "synchronize_stream", lambda stream: None)
    renderer._warp_device = types.SimpleNamespace(stream=types.SimpleNamespace(cuda_stream=99))
    monkeypatch.setattr(ovrtx_renderer_module, "xform_tensor_from_warp", lambda value: value)
    positions = wp.array([[1, 2, 3], [4, 5, 6]], dtype=wp.vec3f, device="cpu")
    rotations = np.array([[1, 2, 3, 4], [0, 0, 0, 1]], dtype=np.float32)
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    renderer.update_camera(cameras[1], positions, wp.array(rotations, dtype=wp.quatf, device="cpu"), None)
    matrices = (
        renderer.scene.stage.write_attribute.call_args.kwargs["tensors"]
        if use_ovstage
        else cameras[1].camera_xform_binding.write_async.call_args.args[0]
    )
    from pxr import Gf

    # OpenGL's right/up/back axes are -Y/+Z/-X in the world camera convention.
    basis = np.eye(4)
    basis[:3, :3] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    world_rotations = [Gf.Quatd(float(q[3]), Gf.Vec3d(*map(float, q[:3]))) for q in rotations]
    expected = np.array([basis @ np.array(Gf.Matrix4d().SetRotate(q)) for q in world_rotations])
    expected[:, 3, :3] = positions.numpy()
    np.testing.assert_allclose(matrices.numpy(), expected, atol=1e-6, rtol=0)
    parameters = wp.zeros((5, 2), dtype=wp.float32, device="cpu")
    renderer.update_camera_intrinsics(cameras[1], wp.zeros(2, dtype=wp.mat33f, device="cpu"), parameters)
    if use_ovstage:
        assert renderer.scene.stage.write_attributes.call_args.args[0] is cameras[1].camera_xform_binding
        bound_paths = [call.args[0] for call in renderer.scene.paths.create_path_list_from_strings.call_args_list]
        assert bound_paths == [
            entry
            for camera, targets in zip(cameras, paths, strict=True)
            for entry in ([_single_render_product(camera).path], targets)
        ]
    else:
        bound_paths = [call.kwargs["prim_paths"] for call in renderer.backend.renderer.bind_attribute.call_args_list]
        binding_count = 1 + len(ovrtx_renderer_module._CAMERA_INTRINSIC_ATTRIBUTES)
        assert bound_paths == [targets for targets in paths for _ in range(binding_count)]
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


def _make_renderer_with_backend(events: list[str], use_ovstage: bool) -> OVRTXRenderer:
    """Exercise real binding setup with recorded native resource lifetimes."""
    from isaaclab_ov.stage import OvstageBackend, OvstageBackendCfg

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    renderer._use_ovstage = use_ovstage
    renderer._sdp = types.SimpleNamespace(
        backend=types.SimpleNamespace(transform_paths=["object"]),
        get_geometry_points=lambda: {"geometry": wp.zeros(1, dtype=wp.vec3f, device="cpu")},
    )
    renderer.backend.renderer = types.SimpleNamespace(
        bind_attribute=lambda **kwargs: _RecordingBinding(events, "object"),
        bind_array_attribute=lambda **kwargs: _RecordingBinding(events, "geometry"),
        write_attribute=lambda **kwargs: None,
        destroy=lambda: events.append("destroy_renderer"),
    )
    render_data = _make_ovrtx_camera_render_data("RenderCamera_camera")
    if use_ovstage:

        @contextlib.contextmanager
        def query(paths):
            try:
                yield paths
            finally:
                events.append(f"release_query:{paths}")

        renderer.scene = OvstageBackend.__new__(OvstageBackend)
        SimulationContext.instance()._backend_registry.insert(
            0, (OvstageBackendCfg(scene_key=renderer.cfg), renderer.scene)
        )
        renderer.scene.stage = types.SimpleNamespace(
            query_from_path_list=query,
            write_attribute=lambda *args, **kwargs: types.SimpleNamespace(wait=lambda: None),
        )
        renderer.scene.paths = types.SimpleNamespace(
            create_path_list_from_strings=lambda paths: paths[0],
            destroy_path_list=lambda paths: events.append(f"destroy_path_list:{paths}"),
        )
        renderer.scene.resources = contextlib.ExitStack()
        renderer.scene.resources.callback(events.append, "close_stage")
        renderer.scene.ordinal = 7
        render_data.resources.callback(renderer.scene.paths.destroy_path_list, "camera")
        render_data.camera_xform_binding = render_data.resources.enter_context(query("camera"))
    else:
        render_data.camera_xform_binding = _RecordingBinding(events, "camera")
        render_data.resources.callback(render_data.camera_xform_binding.unbind)
    renderer.scene.clone_copies = []
    renderer._setup_xform_bindings(SimulationContext.instance().stage)
    renderer._setup_geometry_bindings()
    render_data.renderer_info = {"rgb": object()}
    renderer._camera_render_data.append(render_data)
    renderer._output_id_color_buffers = {"semantic_segmentation": object()}
    renderer._initialized_scene = True
    return renderer


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_ovrtx_close_releases_bindings_before_registry_resources(use_ovstage):
    """Release every binding before its native owners, propagating failures after cleanup."""
    events: list[str] = []
    renderer = _make_renderer_with_backend(events, use_ovstage)
    if use_ovstage:
        renderer.close()
    else:

        def fail_release():
            raise RuntimeError("binding release failed")

        renderer._bindings.callback(fail_release)
        with pytest.raises(RuntimeError, match="binding release failed"):
            renderer.close()
    renderer.close()
    if use_ovstage:
        assert len(events) == 6
        for name in ("camera", "object", "geometry"):
            assert events.index(f"release_query:{name}") < events.index(f"destroy_path_list:{name}")
    else:
        assert sorted(events) == ["unbind:camera", "unbind:geometry", "unbind:object"]
    assert renderer._object_xform_binding is renderer._geometry_points_binding is None
    SimulationContext.instance().close_backend(renderer.backend)
    assert events[-1] == "destroy_renderer"
    if use_ovstage:
        assert "close_stage" not in events
        SimulationContext.instance().close_backend(renderer.scene)
        assert events[-1] == "close_stage"
    events.clear()
    renderer.close()
    assert events == []


@pytest.mark.parametrize("stage_mode", ["legacy", "independent", "independent-physics", "shared", "shared-debug"])
def test_scene_preparation_respects_physics_population_and_camera_transform_mode(monkeypatch, tmp_path, stage_mode):
    """Physics population requires composed geometry; camera GPU reads are disabled before attachment."""
    from pxr import Sdf, Usd, UsdGeom, UsdRender

    attribute_name = "omni:rtx:scene:hydra:readGPUCameraTransformsFromFabricInRenderDelegate"
    settings_path = "/Render/CustomSettings"
    stage = Usd.Stage.CreateInMemory()
    settings = UsdRender.Settings.Define(stage, settings_path)
    settings.GetPrim().CreateAttribute(attribute_name, Sdf.ValueTypeNames.Bool).Set(True)
    stage.SetMetadata("renderSettingsPrimPath", settings_path)
    asset_path = tmp_path / "asset.usda"
    asset = Usd.Stage.CreateNew(str(asset_path))
    asset.SetDefaultPrim(UsdGeom.Xform.Define(asset, "/Asset").GetPrim())
    UsdGeom.Mesh.Define(asset, "/Asset/Geometry")
    asset.GetRootLayer().Save()
    UsdGeom.Xform.Define(stage, "/World/Asset").GetPrim().GetReferences().AddReference(str(asset_path))
    shared = stage_mode.startswith("shared")
    populate_physics = shared or stage_mode == "independent-physics"
    events = []
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = stage_mode != "legacy"
    if stage_mode == "shared-debug":
        renderer.cfg.temp_usd_dir = str(tmp_path / "debug")
    monkeypatch.setattr(ovrtx_renderer_module.ovstage, "PopulationDomain", types.SimpleNamespace(NONE=0, RENDERING=1))
    monkeypatch.setattr(renderer, "_setup_xform_bindings", lambda stage: None)
    monkeypatch.setattr(renderer, "_setup_geometry_bindings", lambda: None)
    monkeypatch.setattr(ovrtx_renderer_module, "ovrtx_replicate", lambda *args: None)
    export_stage = ovrtx_renderer_module.export_stage_to_string

    def export(*args, **kwargs):
        events.append("export")
        assert stage_mode != "shared"
        return export_stage(*args, **kwargs)

    monkeypatch.setattr(ovrtx_renderer_module, "export_stage_to_string", export)

    def load(text):
        layer = Sdf.Layer.CreateAnonymous()
        assert layer.ImportFromString(text)
        result = Usd.Stage.Open(layer)
        assert result.GetPrimAtPath("/World/Asset/Geometry").IsA(UsdGeom.Mesh)
        assert result.GetPrimAtPath("/World/Asset").HasAuthoredReferences() is not populate_physics
        assert (layer.GetPrimAtPath("/World/Asset/Geometry") is not None) is populate_physics
        value = layer.GetAttributeAtPath(Sdf.Path(settings_path).AppendProperty(attribute_name)).default
        assert value is (stage_mode != "legacy")
        events.append("load")

    def write(query, name, *, ordinal, tensors, is_array):
        assert query == [settings_path]
        assert name == attribute_name and (not is_array)
        np.testing.assert_array_equal(tensors, np.array([False], dtype=np.bool_))
        events.append("disable-camera-reads")
        return types.SimpleNamespace(wait=lambda: None)

    def attach(native_stage):
        assert "disable-camera-reads" in events
        events.append("attach")

    renderer.backend.renderer = types.SimpleNamespace(open_usd_from_string=load, attach_ovstage=attach)
    renderer.scene = types.SimpleNamespace(
        clone_env_paths=[],
        clone_copies=[],
        clone_positions=np.zeros((0, 3)),
        population_env_paths=[],
        ordinal=1 if shared else 0,
        cfg=types.SimpleNamespace(
            scene_key=None if shared else renderer.backend.cfg, populate_physics=populate_physics
        ),
        populate=load,
        stage=types.SimpleNamespace(write_attribute=write),
        query=lambda paths: contextlib.nullcontext(paths),
        commit=lambda: events.append("commit"),
        resources=contextlib.ExitStack(),
    )
    if renderer._use_ovstage:
        renderer.scene.render_copies_pending = False
    renderer.prepare_stage(stage, 0)
    assert renderer.backend.cfg.read_gpu_transforms is True
    assert events.count("export") == (stage_mode != "shared")
    assert events.count("disable-camera-reads") == (stage_mode != "legacy")
    if stage_mode != "legacy":
        assert events[-2:] == ["commit", "attach"]


@pytest.mark.parametrize(
    ("gpu_side", "cuda_stream", "expected_stream", "consumer_fails"),
    [(True, 0, 1, False), (True, 99, 99, True), (False, 0, 1, False)],
)
def test_ovrtx_lidar_map_orders_the_copy_against_map_release(
    monkeypatch, gpu_side, cuda_stream, expected_stream, consumer_fails
):
    """Order CUDA lidar copies after rendering and release after the copy, including when it raises."""
    events = []

    class _Mapped:
        wait_event = object()

        def wait(self):
            events.append("wait")

        def unmap(self, *, stream):
            events.append(("release", stream))

        def __contains__(self, key):
            return key == "Coordinates"

        def __getitem__(self, key):
            return types.SimpleNamespace(shape=(4, 3))

    mapped = _Mapped()
    point_cloud = types.SimpleNamespace(map=MagicMock(return_value=mapped))
    monkeypatch.setattr(ovrtx_renderer_module, "_gpu_side_render_var_sync_enabled", lambda: gpu_side)

    def consume(*_args, **_kwargs):
        events.append("consume")
        if consumer_fails:
            raise ValueError("consumer failed")

    monkeypatch.setattr(ovrtx_renderer_module, "copy_tiled_point_cloud", consume)

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._warp_device = types.SimpleNamespace(stream=types.SimpleNamespace(cuda_stream=cuda_stream))
    lidar_data = types.SimpleNamespace(
        points=types.SimpleNamespace(torch=object()),
        intensity=types.SimpleNamespace(torch=object()),
        valid=types.SimpleNamespace(torch=object()),
    )
    with pytest.raises(ValueError, match="consumer failed") if consumer_fails else contextlib.nullcontext():
        renderer._copy_mapped_lidar_point_cloud(point_cloud, lidar_data, ovrtx_renderer_module.Device.CUDA)

    assert point_cloud.map.call_args.kwargs["sync_stream"] == (expected_stream if gpu_side else 0)
    assert events == ([] if gpu_side else ["wait"]) + ["consume", ("release", expected_stream)]
