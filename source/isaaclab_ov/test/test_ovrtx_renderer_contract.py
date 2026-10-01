# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX renderer output contract."""

import contextlib
import ctypes
import importlib.util
import sys
import types
from builtins import ExceptionGroup
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg, SimulationContext
from isaaclab.utils import replace
from isaaclab.utils.warp import ProxyArray

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
    from isaaclab_ov.renderers.ovrtx_renderer import (  # noqa: E402
        _DISABLE_LINUX_CUDA_CPU_SYNC_ENV,
        OVRTXBackend,
        OVRTXCameraRenderData,
        OVRTXRenderer,
        _AsyncWriteBuffers,
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
    spec = types.SimpleNamespace(cfg=_make_camera_cfg(["rgb"]), num_instances=2)
    return OVRTXCameraRenderData(spec, "cpu", render_scope_name="RenderCamera_0")


def _make_ovrtx_renderer_without_backend() -> OVRTXRenderer:
    renderer = OVRTXRenderer(OVRTXRendererCfg())
    renderer.backend = OVRTXBackend.__new__(OVRTXBackend)
    renderer.scene = renderer.backend
    from isaaclab_ov.stage import OvstageBackend

    renderer.scene.commit = OvstageBackend.commit.__get__(renderer.scene)
    renderer.scene.next_camera_id = 0
    renderer.backend.attached = False
    cfg = OVRTXBackendCfg(scene_key=renderer.cfg, use_ovstage=False, read_gpu_transforms=True)
    renderer.scene.stage = renderer.scene.paths = None
    SimulationContext.instance()._backend_registry.append((cfg, renderer.backend))
    return renderer


@pytest.fixture(autouse=True)
def _simulation_registry(monkeypatch):
    from pxr import Usd

    from isaaclab.renderers import RenderContext

    registry = []
    sim = types.SimpleNamespace(
        _backend_registry=registry,
        _render_context=RenderContext(registry),
        physics_manager=types.SimpleNamespace(clone_context_type=None, backend=None, _clone_recipes=[]),
        clone_contexts={},
        stage=Usd.Stage.CreateInMemory(),
        plan=None,
    )
    sim.render_context = sim._render_context
    sim.get_clone_plan = lambda: sim.plan
    sim.get_scene_data_provider = lambda: types.SimpleNamespace(
        backend=types.SimpleNamespace(transform_paths=[]), get_geometry_points=lambda: {}
    )
    sim.get_or_create_backend = SimulationContext.get_or_create_backend.__get__(sim)
    sim.close_backend = SimulationContext.close_backend.__get__(sim)
    monkeypatch.setattr(SimulationContext, "_instance", sim)


@pytest.mark.parametrize("use_ovstage, with_physics", [(False, False), (True, False), (False, True), (True, True)])
def test_ovrtx_renderer_config_enables_supported_runtime_options(monkeypatch, tmp_path, use_ovstage, with_physics):
    """Preparation resolves routes before native allocation; registry ownership outlives borrowers."""
    from isaaclab_ov.cloner import OvPhysxReplicateContext, OvrtxReplicateContext, OvstageReplicateContext

    from pxr import UsdGeom

    from isaaclab.cloner import make_clone_plan, replicate

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
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", str(int(use_ovstage)))

    @contextlib.contextmanager
    def stage_resource(label):
        yield object()
        stage_releases.append(label)

    monkeypatch.setattr("isaaclab_ov.stage.create_ovstage", lambda _: stage_resource("stage"))
    monkeypatch.setattr(ovrtx_renderer_module.ovstage, "PathDictionary", lambda _: stage_resource("paths"))

    sim = SimulationContext.instance()
    if with_physics:
        sim.physics_manager.clone_context_type = OvPhysxReplicateContext
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
    other = sim.get_or_create_backend(replace(cfg, enable_shadows=True))
    assert other.backend is not None
    shared_physics = use_ovstage and with_physics
    assert (other.backend is renderer.backend) is shared_physics
    assert (other.scene is renderer.scene) is shared_physics
    assert loaded == [str(dependency)] * (1 if shared_physics else 2)
    assert len(redirected) == (1 if shared_physics else 2)
    assert renderer.backend.renderer is not None
    assert config_kwargs["suppress_deprecation_warnings"] is True
    assert config_kwargs["texture_streaming_mode"] is ovrtx_renderer_module.TextureStreamingMode.SYNCHRONOUS
    expected_contexts = {OvstageReplicateContext if use_ovstage else OvrtxReplicateContext}
    if with_physics and not shared_physics:
        expected_contexts.add(OvPhysxReplicateContext)
    assert set(sim.clone_contexts) == expected_contexts
    assert {source for source, _ in renderer.scene.clone_copies} == {
        "/World/envs/env_0/Visual",
        *(["/World/envs/env_0/Body"] if shared_physics else []),
    }
    assert bool(sim.physics_manager._clone_recipes) is (with_physics and not shared_physics)
    # Once prepared, hard/soft reset must retain the selected resource identity.
    from isaaclab.cloner.replicate_session import _prepare_clone_contexts

    resources = tuple(sim._backend_registry)
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", str(int(not use_ovstage)))
    _prepare_clone_contexts(sim)
    assert tuple(sim._backend_registry) == resources
    if shared_physics:
        assert renderer.scene.cfg.population_domains == ovrtx_renderer_module.ovstage.PopulationDomain.ALL
        with pytest.raises(ValueError, match="one OVRTX engine"):
            sim.get_or_create_backend(replace(renderer.cfg, log_level="error"))
        conflict = next(
            resource
            for cfg, resource in sim._backend_registry
            if isinstance(cfg, OVRTXRendererCfg) and cfg.log_level == "error"
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
    cameras = [_make_ovrtx_camera_render_data() for _ in range(2 if batch else 1)]
    products = {}
    processed = []
    postprocessed = []
    submissions = []
    published_ordinals = []
    for index, camera in enumerate(cameras):
        camera.render_product_path = f"/Render/Camera{index}"
        camera.warp_buffers = {str(RenderBufferKind.RGB_HDR): object(), str(RenderBufferKind.RGBA): object()}
        camera.ppisp_pipeline = types.SimpleNamespace(apply=lambda *buffers: postprocessed.append(buffers))
        products[camera.render_product_path] = types.SimpleNamespace(frames=[object()])
    renderer._camera_render_data = [*cameras, types.SimpleNamespace(render_product_path="/Render/UnrequestedCamera")]
    if missing_output == "product":
        del products[cameras[-1].render_product_path]
    elif missing_output == "frame":
        products[cameras[-1].render_product_path].frames.clear()

    def step(**kwargs):
        submissions.append(kwargs)
        return products

    def advance_write_floor(*, ordinal):
        published_ordinals.append(ordinal)
        return types.SimpleNamespace(wait=lambda: None)

    renderer.backend.renderer = types.SimpleNamespace(step=step)
    renderer.scene.stage = types.SimpleNamespace(advance_write_floor=advance_write_floor)
    monkeypatch.setattr(renderer, "_process_render_frame", lambda *args: processed.append(args))

    render = renderer.render_batch if batch else renderer.render
    request = cameras if batch else cameras[0]
    if missing_output is None:
        render(request)
        assert processed == [
            (camera, products[camera.render_product_path].frames[0], camera.warp_buffers) for camera in cameras
        ]
        assert postprocessed == [
            (camera.warp_buffers[str(RenderBufferKind.RGB_HDR)], camera.warp_buffers[str(RenderBufferKind.RGBA)])
            for camera in cameras
        ]
    else:
        with pytest.raises(RuntimeError, match=cameras[-1].render_product_path):
            render(request)
        assert not processed
        assert not postprocessed

    assert len(submissions) == 1
    # Unrequested products are submitted too, and only the requested cameras are read back.
    assert submissions[0]["render_products"] == {
        *[f"/Render/Camera{i}" for i in range(len(cameras))],
        "/Render/UnrequestedCamera",
    }
    if use_ovstage:
        assert submissions[0]["ordinal"] == 7
        assert published_ordinals == [7]
        assert renderer.scene.ordinal == 8
    else:
        assert "ordinal" not in submissions[0]
        assert not published_ordinals


def test_ovrtx_render_batch_empty_sequence_does_not_require_initialized_backend():
    """An empty render request has no backend work or initialization precondition."""
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.render_batch([])


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
def test_ovrtx_multiple_cameras_render_independent_views(monkeypatch, use_ovstage, asynchronous, geometry):
    """Cameras preserve independent captures, GPU input lifetimes, and reset/cleanup boundaries."""
    from pxr import Gf, Usd, UsdGeom, UsdLux

    from isaaclab.cloner import make_clone_plan, replicate
    from isaaclab.renderers.camera_render_spec import CameraRenderSpec
    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
    from isaaclab.utils.math import convert_camera_frame_orientation_convention
    from isaaclab.utils.warp import ProxyArray

    if not torch.cuda.is_available():
        pytest.skip("OVRTX rendering requires CUDA")
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", str(int(use_ovstage)))
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    if not use_ovstage:
        UsdGeom.Xform.Define(stage, "/World/envs/env_1")
    batches, initial_points = [], []
    for index, x in enumerate((0.0, 2.0)):
        camera = UsdGeom.Camera.Define(stage, f"/World/envs/env_0/cam{index}")
        camera.CreateProjectionAttr("orthographic")
        camera.CreateHorizontalApertureAttr(20.0)
        camera.CreateVerticalApertureAttr(20.0)
        camera.AddTranslateOp().Set(Gf.Vec3d(x, 0, 5))
        path = f"/World/envs/env_0/object{index}"
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
        batches.append((source, {path.replace("env_0", f"env_{world}"): (0, len(vertices)) for world in range(2)}))

    renderer = SimulationContext.instance().get_or_create_backend(OVRTXRendererCfg(async_rendering=asynchronous))
    publication = types.SimpleNamespace(
        transform_paths=[], geometry_timestamp=0, get_geometry_batches=lambda _format: batches
    )
    renderer._sdp = SceneDataProvider(publication)
    renderer._exported_usd_string = stage.ExportToString()
    plan = make_clone_plan(
        (AssetBaseCfg(prim_path="/World/envs/env_[^/]+", cloning_contexts=renderer.cfg.cloning_contexts),),
        ((0,),),
        2,
        positions=np.zeros((2, 3), dtype=np.float32),
    )
    sim = SimulationContext.instance()
    sim.plan = plan
    replicate(plan)
    cameras = []

    def camera_scope_exists(rd):
        scope = f"/{rd.render_scope_name}/"
        if use_ovstage:
            import ovstage

            camera_filter = ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.PREFIX, [scope])])
            with renderer.scene.stage.query(filter=camera_filter) as query:
                return query.result().total_prim_count > 0
        return any(path.startswith(scope) for path in renderer.backend.renderer.query_prims())

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
            renderer.update_geometries()
            renderer.render(rd)
            renderer.read_output(rd, data)
            check_depth(rd, data, 5.0 - index - 0.5)
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
        assert all(camera_scope_exists(rd) for rd, _ in cameras)
        renderer.cleanup(cameras[0][0])
        assert not camera_scope_exists(cameras[0][0])
        renderer.render(cameras[1][0])
        renderer.read_output(*cameras[1])
        check_depth(*cameras[1], 2.5 if asynchronous else 5.5)
        renderer.cleanup(cameras[1][0])
        renderer.cleanup(cameras[1][0])
        assert not camera_scope_exists(cameras[1][0])
    finally:
        renderer.close()
        SimulationContext.instance().close_backend(renderer.backend)
        if use_ovstage:
            SimulationContext.instance().close_backend(renderer.scene)


@pytest.mark.parametrize("batch", [False, True], ids=["lazy", "batched"])
def test_async_cameras_publish_independently_with_capture_metadata_and_reset(monkeypatch, batch):
    """Completed batches do not update another camera; reset retires only that camera's history."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg.async_rendering = True
    renderer._initialized_scene = True
    renderer._visual_material_writer_ref = None
    cameras = [_make_ovrtx_camera_render_data() for _ in range(2)]
    renderer._camera_render_data = cameras.copy()
    camera_data = []
    for index, camera in enumerate(cameras):
        camera.render_product_path = f"/Render/Camera{index}"
        data = CameraData.allocate(["rgb"], 2, 2, 2, "cpu", renderer.supported_output_types())
        data.create_buffers(2, "cpu")
        renderer.set_outputs(camera, data.output)
        camera_data.append(data)
    frame = ProxyArray(wp.zeros(2, dtype=wp.int64, device="cpu"))
    operations = []
    ordinal = 0

    def submit(render_products, delta_time):
        operation = MagicMock()
        operation.wait.return_value.fetch.return_value = {
            path: types.SimpleNamespace(frames=[ordinal]) for path in render_products
        }
        operations.append(operation)
        return operation

    def consume(camera, value, buffers):
        buffers["rgba"].fill_(value)

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


@pytest.mark.parametrize("source_device", ["cpu", "cuda:1"])
def test_ovrtx_process_frame_routes_ppisp_hdr_to_output_device(monkeypatch, source_device):
    """PPISP reads HDR on the output device and leaves RGBA for its own pipeline."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_camera_render_data()
    render_data.ppisp_pipeline = object()
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
        assert render_var is source, "PPISP RGBA output must not read OVRTX LdrColor"
        yield source

    monkeypatch.setattr(renderer, "_map_render_var_to_dlpack", map_hdr)
    frame = types.SimpleNamespace(
        render_vars={
            render_data.render_var_keys["LdrColor"]: object(),
            render_data.render_var_keys["HdrColor"]: source,
        }
    )
    output = wp.zeros((2, 8, 16, 3), dtype=wp.float32, device="cpu")
    renderer._process_render_frame(render_data, frame, {"rgba": object(), "rgb_hdr": output})
    np.testing.assert_array_equal(output.numpy(), 0.5)
    if source_device == "cpu":
        clone.assert_not_called()
    else:
        clone.assert_called_once_with(source, device="cpu")


# Render-var keys depend only on the OVRTX version; ``use_ovstage`` only selects the registration path.
@pytest.mark.parametrize(("use_ovstage", "version"), [(False, "0.4"), (True, "0.5")])
def test_ovrtx_process_frame_reads_authored_camera_render_vars(monkeypatch, use_ovstage, version):
    """Both OVRTX APIs extract each camera's outputs using keys from its authored USD."""
    from packaging.version import Version

    from pxr import Usd

    monkeypatch.setattr(ovrtx_renderer_module, "OVRTX_VERSION", Version(version))
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._initialized_scene = True
    renderer._device = "cpu"
    renderer._exported_usd_string = None
    renderer.scene.next_camera_id = 0
    renderer._use_ovstage = use_ovstage
    renderer.scene.ordinal = 1
    renderer.backend.renderer = MagicMock()
    renderer.scene.stage = MagicMock()
    renderer.scene.paths = MagicMock()
    renderer.scene.paths.create_path_list_from_strings.side_effect = tuple
    renderer.scene.stage.query_from_path_list.side_effect = lambda paths: contextlib.nullcontext(object())
    for name in ("add_usd_reference_from_string", "apply_usd_changes", "remove_usd"):
        monkeypatch.setattr(ovrtx_renderer_module.ovstage.population, name, MagicMock())

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
    for camera_id in range(2):
        cfg = _make_camera_cfg(list(outputs))
        render_data = renderer.create_render_data(
            types.SimpleNamespace(
                cfg=cfg,
                device="cpu",
                num_instances=2,
                camera_prim_paths=[f"/World/envs/env_{i}/cam{camera_id}" for i in range(2)],
            )
        )
        stage = stages[render_data.render_product_path]
        keys = {
            prim.GetAttribute("sourceName").Get(): (
                str(prim.GetPath()) if version == "0.5" else prim.GetAttribute("sourceName").Get()
            )
            for prim in stage.Traverse()
            if prim.GetTypeName() == "RenderVar"
        }
        frame = types.SimpleNamespace(render_vars={})
        buffers = {}
        for index, (output, (source, channels, dtype)) in enumerate(outputs.items(), start=1):
            value = 10 * camera_id + index
            frame.render_vars[keys[source]] = wp.full((8, 32, channels), value, dtype=dtype, device="cpu")
            buffers[output] = wp.zeros((2, 8, 16, channels), dtype=dtype, device="cpu")
        renderer._process_render_frame(render_data, frame, buffers)
        for index, output in enumerate(outputs, start=1):
            np.testing.assert_array_equal(buffers[output].numpy(), 10 * camera_id + index)
        render_data.cleanup()


def test_launch_extract_all_tiles_rejects_wider_output_channels():
    """An output wider than the tiled input would read out of bounds, so it must raise before launching."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_camera_render_data()

    with pytest.raises(ValueError, match="out of bounds"):
        renderer._launch_extract_all_tiles(
            render_data, types.SimpleNamespace(shape=(8, 16, 3)), types.SimpleNamespace(shape=(2, 8, 16, 4))
        )


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


@pytest.mark.parametrize("kind", [RenderBufferKind.SEMANTIC_SEGMENTATION, RenderBufferKind.INSTANCE_SEGMENTATION])
@pytest.mark.parametrize("colorized", [False, True])
def test_segmentation_spec_follows_colorize_flag(kind, colorized):
    renderer = _make_ovrtx_renderer_without_backend()
    setattr(renderer.cfg, f"colorize_{kind.value}", colorized)
    expected = RenderBufferSpec(4, wp.uint8) if colorized else RenderBufferSpec(1, wp.int32)
    assert renderer.supported_output_types()[kind] == expected


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
    render_data = _make_ovrtx_camera_render_data()
    render_data.render_product_path = "/RenderCamera_0/RenderProduct_to_remove"
    renderer._camera_render_data.append(render_data)
    render_data.warp_buffers = {"rgba": wp.zeros((8, 16, 4), dtype=wp.uint8, device="cpu")}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": {}}}
    render_data.ppisp_pipeline = object()
    if use_ovstage:
        render_data.camera_xform_query = "to_remove"
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
    assert render_data.camera_xform_query is None
    assert render_data.intrinsic_bindings == []
    assert render_data.warp_buffers == {}
    assert render_data.renderer_info == {}
    assert render_data.ppisp_pipeline is None
    assert render_data.pending is render_data.ready is None
    assert renderer._initialized_scene is True


@pytest.mark.parametrize(
    "camera_path",
    [
        "/World/Camera",
        "/World/envs/env_1/Camera",
        "/World/envs/env_00/Camera",
        "/World/envs/env_0",
        "/World/envs/env_0/",
    ],
)
def test_create_render_data_rejects_cameras_outside_source_environment(camera_path):
    """Camera registration requires a source camera beneath env_0 before touching the backend."""
    from isaaclab.renderers.camera_render_spec import CameraRenderSpec

    renderer = _make_ovrtx_renderer_without_backend()
    renderer.backend.renderer = MagicMock()
    spec = CameraRenderSpec(
        cfg=_make_camera_cfg(["depth"]),
        device="cpu",
        num_instances=2,
        camera_prim_paths=(camera_path,),
        view_count=2,
    )

    with pytest.raises(ValueError, match="/World/envs/env_0/"):
        renderer.create_render_data(spec)

    assert not renderer.backend.renderer.mock_calls


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_intrinsic_updates_target_the_given_camera(monkeypatch, use_ovstage):
    """Distinct cameras, including a prototype-only wrist camera, bind every environment independently."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._initialized_scene = True
    renderer._device = "cpu"
    renderer.scene.next_camera_id = 0
    renderer._use_ovstage = use_ovstage
    renderer.scene.ordinal = 1
    renderer.backend.renderer = MagicMock()
    renderer.backend.renderer.bind_attribute.side_effect = lambda **kwargs: MagicMock()
    renderer.scene.stage = MagicMock()
    renderer.scene.paths = MagicMock()
    renderer.scene.paths.create_path_list_from_strings.side_effect = tuple
    renderer.scene.stage.query_from_path_list.side_effect = lambda paths: contextlib.nullcontext(object())
    for name in ("add_usd_reference_from_string", "apply_usd_changes", "remove_usd"):
        monkeypatch.setattr(ovrtx_renderer_module.ovstage.population, name, MagicMock())
    paths = [
        [f"/World/envs/env_{i}/{name}" for i in range(2)] for name in ("CameraA", "Robot/ee_link/palm_link/CameraB")
    ]
    cameras = [
        renderer.create_render_data(
            types.SimpleNamespace(
                cfg=_make_camera_cfg(["depth"]),
                device="cpu",
                num_instances=2,
                camera_prim_paths=camera_paths,
            )
        )
        for camera_paths in (paths[0], paths[1][:1])
    ]
    monkeypatch.setattr(wp, "get_stream", lambda device: types.SimpleNamespace(cuda_stream=99))
    parameters = wp.zeros((5, 2), dtype=wp.float32, device="cpu")
    renderer.update_camera_intrinsics(cameras[1], wp.zeros(2, dtype=wp.mat33f, device="cpu"), parameters)
    if use_ovstage:
        assert renderer.scene.stage.write_attributes.call_args.args[0] is cameras[1].camera_xform_query
        bound_paths = [call.args[0] for call in renderer.scene.paths.create_path_list_from_strings.call_args_list]
        assert bound_paths == [[cameras[0].render_product_path], paths[0], [cameras[1].render_product_path], paths[1]]
    else:
        bound_paths = [call.kwargs["prim_paths"] for call in renderer.backend.renderer.bind_attribute.call_args_list]
        binding_count = 1 + len(ovrtx_renderer_module._CAMERA_INTRINSIC_ATTRIBUTES)
        assert bound_paths == [paths[0]] * binding_count + [paths[1]] * binding_count
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
    render_data = _make_ovrtx_camera_render_data()
    render_data.render_product_path = "/RenderCamera_0/RenderProduct_camera"
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
        renderer.scene._resources = contextlib.ExitStack()
        renderer.scene._resources.callback(events.append, "close_stage")
        renderer.scene.ordinal = 7
        render_data.resources.callback(renderer.scene.paths.destroy_path_list, "camera")
        render_data.camera_xform_query = render_data.resources.enter_context(query("camera"))
    else:
        render_data.camera_xform_binding = _RecordingBinding(events, "camera")
        render_data.resources.callback(render_data.camera_xform_binding.unbind)
    renderer._setup_xform_bindings()
    renderer._setup_geometry_bindings()
    render_data.renderer_info = {"rgb": object()}
    renderer._camera_render_data.append(render_data)
    renderer._output_id_color_buffers = {"semantic_segmentation": object()}
    renderer._initialized_scene = True
    return renderer


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_ovrtx_close_releases_bindings_before_registry_resources(use_ovstage):
    """Bindings release once, before engines and stages owned by the simulation registry."""
    events: list[str] = []
    renderer = _make_renderer_with_backend(events, use_ovstage)
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
