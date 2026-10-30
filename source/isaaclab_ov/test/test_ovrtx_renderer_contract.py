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
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
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


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/envs/env_0/Camera",
        spawn=_SPAWN,
        data_types=data_types,
    )


def _make_ovrtx_camera_render_data() -> OVRTXCameraRenderData:
    spec = CameraRenderSpec(_make_camera_cfg(["rgb"]), "cpu", 2, ("/World/envs/env_0/Camera",), 2)
    return OVRTXCameraRenderData(spec, "RenderCamera_0", [f"/World/envs/env_{i}/Camera" for i in range(2)])


def _make_ovrtx_renderer_without_backend() -> OVRTXRenderer:
    renderer = OVRTXRenderer(OVRTXRendererCfg(), sim=SimulationContext.instance())
    renderer.backend = OVRTXBackend.__new__(OVRTXBackend)
    renderer.backend._render_lease_owner = None
    renderer.scene = renderer.backend
    from isaaclab_ov.stage import OvstageBackend

    renderer.scene.commit = OvstageBackend.commit.__get__(renderer.scene)
    renderer.scene.query = OvstageBackend.query.__get__(renderer.scene)
    renderer.scene.next_camera_id = 0
    renderer.scene.clone_env_paths = ["/World/envs/env_0", "/World/envs/env_1"]
    renderer.backend.attached = False
    renderer.backend.render_products = set()
    cfg = OVRTXBackendCfg(scene_key=renderer.cfg, use_ovstage=False, read_gpu_transforms=True, device="cpu")
    renderer.scene.stage = renderer.scene.paths = None
    renderer.backend.cfg = cfg
    SimulationContext.instance()._backend_registry.append((cfg, renderer.backend))
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
    if with_physics and not shared_physics:
        expected_contexts.add(OvPhysxReplicateContext)
    assert set(sim.clone_contexts) == expected_contexts
    assert {source for source, _ in renderer.scene.clone_copies} == {
        "/World/envs/env_0/Visual",
        *(["/World/envs/env_0/Body"] if shared_physics else []),
    }
    assert bool(sim.backend_manager._clone_recipes) is (with_physics and not shared_physics)
    # Repeated preparation retains the selected resource identity.
    resources = tuple(sim._backend_registry)
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", str(int(not use_ovstage)))
    OvrtxReplicateContext.prepare(sim, {OvrtxReplicateContext: set()})
    assert tuple(sim._backend_registry) == resources
    if use_ovstage:
        domains = ovrtx_renderer_module.ovstage.PopulationDomain
        assert renderer.scene.cfg.population_domains == (domains.ALL if shared_physics else domains.RENDERING)
    if shared_physics:
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
    renderer._camera_render_data = cameras
    other = OVRTXRenderer(OVRTXRendererCfg(enable_shadows=True), sim=SimulationContext.instance())
    other.backend = renderer.backend
    other._camera_render_data = [types.SimpleNamespace(render_product_path="/Render/UnrequestedCamera")]
    renderer.backend.render_products.update(
        camera.render_product_path for camera in (*cameras, *other._camera_render_data)
    )
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
    SimulationContext.instance().device = "cuda:0"
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", str(int(use_ovstage)))
    env_template = "/Lab/Cells/Cell{}" if use_ovstage else "/World/envs/env_{}"
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)
    UsdGeom.Xform.Define(stage, env_template.format(0))
    if not use_ovstage:
        UsdGeom.Xform.Define(stage, env_template.format(1))
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
        renderer.prepare_stage(stage, 2)
        assert renderer.visual_material_writer is not None
        for index, (height, width) in enumerate(((480, 640), (384, 512))):
            cfg = _make_camera_cfg(
                ["rgb", "distance_to_image_plane"] if index == 0 else ["rgb", "distance_to_image_plane", "normals"]
            )
            cfg.height, cfg.width = height, width
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
    """Real captures match delayed metadata; generated images and other cameras cannot replace them."""
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
        renderer.backend.render_products.add(camera.render_product_path)
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


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_ovrtx_process_frame_reads_authored_camera_render_vars(monkeypatch, use_ovstage):
    """Both OVRTX APIs extract each camera's outputs using keys from its authored USD."""
    from pxr import Usd

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._initialized_scene = True
    renderer._device = "cpu"
    renderer.scene.next_camera_id = 0
    renderer.scene.clone_env_paths = ["/World/envs/env_0", "/World/envs/env_1"]
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
            CameraRenderSpec(
                cfg=cfg,
                device="cpu",
                num_instances=2,
                view_count=2,
                camera_prim_paths=[f"/World/envs/env_{i}/cam{camera_id}" for i in range(2)],
            )
        )
        stage = stages[render_data.render_product_path]
        keys = {
            prim.GetAttribute("sourceName").Get(): str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetTypeName() == "RenderVar"
        }
        frame = types.SimpleNamespace(end_time=1.0, render_vars={})
        buffers = {}
        for index, (output, (source, channels, dtype)) in enumerate(outputs.items(), start=1):
            value = 10 * camera_id + index
            frame.render_vars[keys[source]] = wp.full((8, 32, channels), value, dtype=dtype, device="cpu")
            buffers[output] = wp.zeros((2, 8, 16, channels), dtype=dtype, device="cpu")
        frames = [frame]
        if camera_id == 1:
            generated = types.SimpleNamespace(
                end_time=0.5,
                render_vars={keys["LdrColor"]: wp.full((8, 32, 4), 99, dtype=wp.uint8, device="cpu")},
            )
            frames.insert(0, generated)
        render_data.warp_buffers = buffers
        renderer._process_render_products(
            (render_data,), {render_data.render_product_path: types.SimpleNamespace(frames=frames)}
        )
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


@pytest.mark.parametrize("colorized, named_tensors", [(False, False), (True, True)])
def test_segmentation_outputs_share_metadata_and_clear_stale_labels(monkeypatch, colorized, named_tensors):
    """Pixel types, label keys and missing-map handling follow the requested output contract."""
    from isaaclab_ov.renderers import ovrtx_annotator_utils as annotator_utils

    from isaaclab.renderers.segmentation_colors import random_color_from_id

    # Exercise each wire format with its import-selected implementation, independently
    # of the native wheel installed in this unit-test process.
    monkeypatch.setattr(
        annotator_utils,
        "_decode_identifier_map",
        (
            annotator_utils._decode_identifier_map_named
            if named_tensors
            else annotator_utils._decode_identifier_map_packed
        ),
    )
    monkeypatch.setattr(
        annotator_utils,
        "_decode_stable_id_semantic_id_map",
        (
            annotator_utils._decode_stable_id_semantic_id_map_named
            if named_tensors
            else annotator_utils._decode_stable_id_semantic_id_map_packed
        ),
    )
    monkeypatch.setattr(
        ovrtx_renderer_module,
        "_decode_segmentation_map",
        (
            ovrtx_renderer_module._decode_segmentation_map_named
            if named_tensors
            else ovrtx_renderer_module._decode_segmentation_map_packed
        ),
    )

    from .test_ovrtx_annotator_utils import _encode_identifier_map, _encode_stable_id_semantic_id_map

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
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
        "SemanticIdMap": _encode_identifier_map([((2, 0, 0, 0), "class: cone;")], named_tensors=False),
        "StableIdMap": _encode_identifier_map([(stable_id, "/World/Cone")], named_tensors=False),
        "StableIdSemanticIdMap": _encode_stable_id_semantic_id_map([(stable_id, 2)], named_tensors=False),
    }
    if named_tensors:
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
    renderer._process_render_frame(render_data, frame, render_data.warp_buffers)
    renderer.read_output(render_data, camera_data)
    pixel = random_color_from_id(2) if colorized else 2
    for kind in kinds:
        output = camera_data.output[kind].warp
        assert output.dtype == (wp.uint8 if colorized else wp.int32)
        assert output.shape == (2, 8, 16, 4 if colorized else 1)
        np.testing.assert_array_equal(output.numpy(), np.broadcast_to(pixel, output.shape))
    assert camera_data.info[kinds[0]]["idToLabels"][pixel] == {"class": "cone"}
    assert camera_data.info[kinds[1]]["idToLabels"][pixel] == "/World/Cone"
    assert camera_data.info[kinds[1]]["idToSemantics"][pixel] == {"class": "cone"}
    frame.render_vars[render_data.render_var_keys["SemanticIdMap"]].map.assert_called_once()

    frame.render_vars.clear()
    with pytest.raises(RuntimeError, match="SemanticIdMap"):
        renderer._process_render_frame(render_data, frame, render_data.warp_buffers)
    renderer._process_render_frame(render_data, frame, {kinds[0]: render_data.warp_buffers[kinds[0]]})
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
    """Reject malformed flags and unsupported sharing before allocating native resources."""
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
    [(True, 0, 1, False), (True, 99, 99, True), (False, 0, 1, False), (False, None, None, True)],
)
def test_ovrtx_map_render_var_orders_the_read_against_render_completion(
    monkeypatch, gpu_side, cuda_stream, expected_stream, consumer_fails
):
    """Order reads after rendering and release after reads, including when the consumer raises."""
    events = []
    mapping = types.SimpleNamespace(
        wait=lambda: events.append("wait"), unmap=lambda *, stream=None: events.append(("release", stream))
    )
    render_var = types.SimpleNamespace(map=MagicMock(return_value=mapping))
    sentinel = object()
    monkeypatch.setattr(ovrtx_renderer_module, "_gpu_side_render_var_sync_enabled", lambda: gpu_side)
    monkeypatch.setattr(ovrtx_renderer_module.wp, "from_dlpack", lambda mapping: events.append("wrap") or sentinel)

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._warp_device = types.SimpleNamespace(
        is_cuda=cuda_stream is not None, stream=types.SimpleNamespace(cuda_stream=cuda_stream)
    )
    with pytest.raises(ValueError, match="consumer failed") if consumer_fails else contextlib.nullcontext():
        with renderer._map_render_var_to_dlpack(render_var) as array:
            assert array is sentinel
            events.append("consume")
            if consumer_fails:
                raise ValueError("consumer failed")

    if cuda_stream is None:
        assert render_var.map.call_args.kwargs == {"device": ovrtx_renderer_module.Device.CPU}
    else:
        assert render_var.map.call_args.kwargs["sync_stream"] == (expected_stream if gpu_side else 0)
    assert events == ([] if gpu_side else ["wait"]) + ["wrap", "consume", ("release", expected_stream)]


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
    assert render_data.ppisp_pipeline is None
    assert render_data.pending is render_data.ready is None
    assert renderer._initialized_scene is True


@pytest.mark.parametrize("use_ovstage", [False, True])
def test_pose_and_intrinsic_updates_target_the_given_camera(monkeypatch, use_ovstage):
    """Distinct cameras, including a prototype-only wrist camera, bind every environment independently."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._initialized_scene = True
    renderer._device = "cpu"
    renderer.scene.next_camera_id = 0
    renderer.scene.clone_env_paths = ["/Lab/Cell0", "/Lab/Cell1"]
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
    paths = [[f"/Lab/Cell{i}/{name}" for i in range(2)] for name in ("CameraA", "Robot/ee_link/palm_link/CameraB")]
    paths[0].reverse()
    paths[0][0] = "/Lab/Cell1/Robot/CameraA"
    paths.append(["/World/PerspectiveCamera"])
    paths.append(paths[0][:1])
    renderer.scene.paths.intern_token.side_effect = {"env_0": 10, "env_1": 11}.__getitem__
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
    renderer._warp_device = types.SimpleNamespace(is_cuda=True, stream=types.SimpleNamespace(cuda_stream=99))
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
        assert all(desc.cuda_stream is None for desc in renderer.scene.stage.write_attributes.call_args.args[1])
        bound_paths = [call.args[0] for call in renderer.scene.paths.create_path_list_from_strings.call_args_list]
        assert bound_paths == [
            entry
            for camera, targets in zip(cameras, paths, strict=True)
            for entry in ([camera.render_product_path], targets)
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
        renderer.scene._render_lease_owner = None
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


@pytest.mark.parametrize(
    "runtime_version, stage_mode",
    [
        ("0.6.0", "legacy"),
        ("0.7.0.dev1", "legacy"),
        ("0.6.0", "independent"),
        ("0.7.0", "independent"),
        ("0.7.0.dev1", "shared"),
        ("0.8.0", "shared"),
    ],
)
def test_camera_transform_mode_is_set_before_scene_attachment(monkeypatch, runtime_version, stage_mode):
    """Only 0.7 disables camera GPU reads, before consumption, with one owner per stage path."""
    from packaging.version import Version

    from pxr import Sdf, Usd, UsdRender

    attribute_name = "omni:rtx:scene:hydra:readGPUCameraTransformsFromFabricInRenderDelegate"
    settings_path = "/Render/CustomSettings"
    stage = Usd.Stage.CreateInMemory()
    settings = UsdRender.Settings.Define(stage, settings_path)
    settings.GetPrim().CreateAttribute(attribute_name, Sdf.ValueTypeNames.Bool).Set(True)
    stage.SetMetadata("renderSettingsPrimPath", settings_path)
    exported = stage.GetRootLayer().ExportToString()
    needs_fallback = runtime_version in ("0.7.0.dev1", "0.7.0")
    events = []
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = stage_mode != "legacy"
    monkeypatch.setattr(ovrtx_renderer_module, "_OVRTX_VERSION", Version(runtime_version))
    monkeypatch.setattr(renderer, "_setup_xform_bindings", lambda stage: None)
    monkeypatch.setattr(renderer, "_setup_geometry_bindings", lambda: None)
    monkeypatch.setattr(ovrtx_renderer_module, "ovrtx_replicate", lambda *args: None)

    def export(*args, **kwargs):
        events.append("export")
        assert stage_mode != "shared"
        return exported

    monkeypatch.setattr(ovrtx_renderer_module, "export_stage_to_string", export)

    def load(text):
        layer = Sdf.Layer.CreateAnonymous()
        layer.ImportFromString(text)
        value = layer.GetAttributeAtPath(Sdf.Path(settings_path).AppendProperty(attribute_name)).default
        # OVStage population preserves authored input; only the legacy path rewrites USD.
        assert value is not (needs_fallback and stage_mode == "legacy")
        events.append("load")

    def write(query, name, *, ordinal, tensors, is_array):
        assert query == [settings_path]
        assert name == attribute_name and not is_array
        np.testing.assert_array_equal(tensors, np.array([False], dtype=np.bool_))
        events.append("disable-camera-reads")
        return types.SimpleNamespace(wait=lambda: None)

    def attach(native_stage):
        assert ("disable-camera-reads" in events) is needs_fallback
        events.append("attach")

    renderer.backend.renderer = types.SimpleNamespace(open_usd_from_string=load, attach_ovstage=attach)
    renderer.scene = types.SimpleNamespace(
        clone_env_paths=[],
        clone_copies=[],
        clone_positions=np.zeros((0, 3)),
        population_env_paths=[],
        ordinal=1 if stage_mode == "shared" else 0,
        cfg=types.SimpleNamespace(population_domains=ovrtx_renderer_module.ovstage.PopulationDomain.RENDERING),
        populate=load,
        stage=types.SimpleNamespace(write_attribute=write),
        query=lambda paths: contextlib.nullcontext(paths),
        commit=lambda: events.append("commit"),
        resources=contextlib.ExitStack(),
    )
    renderer.prepare_stage(stage, 0)
    assert renderer.backend.cfg.read_gpu_transforms is True
    assert events.count("export") == (stage_mode != "shared")
    assert events.count("disable-camera-reads") == (needs_fallback and stage_mode != "legacy")
    if stage_mode != "legacy":
        assert events[-2:] == ["commit", "attach"]


@pytest.fixture
def leased_renderer(monkeypatch):
    """Exercise real deferred resource acquisition with native SDK endpoints doubled."""
    import isaaclab_ov.stage as stage_module

    monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
    native, stage, paths = MagicMock(), MagicMock(), MagicMock()
    floor = stage.get_attribute_write_floor.return_value.__enter__.return_value
    floor.fetch.return_value = 0

    def advance(*, ordinal, **kwargs):
        return types.SimpleNamespace(wait=lambda: setattr(floor.fetch, "return_value", ordinal))

    stage.advance_write_floor.side_effect = advance
    monkeypatch.setattr(ovrtx_renderer_module, "Renderer", lambda _cfg: native)
    monkeypatch.setattr(ovrtx_renderer_module, "redirect_shader_cache", lambda _cfg: None)
    monkeypatch.setattr(stage_module, "create_ovstage", lambda _name: contextlib.nullcontext(stage))
    monkeypatch.setattr(stage_module.ovstage, "PathDictionary", lambda _: contextlib.nullcontext(paths))
    monkeypatch.setattr(stage_module.ovstage.population, "open_usd_from_string", lambda *args, **kwargs: None)
    renderer = SimulationContext.instance().get_or_create_backend(OVRTXRendererCfg())
    assert renderer.backend is renderer.scene is None
    return renderer


def _prepare_leased_renderer(renderer, *, bind=True):
    from isaaclab_ov.cloner import OvrtxReplicateContext

    OvrtxReplicateContext.prepare(renderer._sim, {})
    if bind:
        renderer.prepare_stage(renderer._sim.stage, 1)


def test_render_lease_lease_borrows_registry_pair_and_blocks_shared_users(leased_renderer, monkeypatch):
    renderer = leased_renderer
    shared = SimulationContext.instance().get_or_create_backend(renderer.cfg)
    assert shared is renderer
    lease = renderer.acquire_ov_libraries_render_lease()
    with pytest.raises(RuntimeError, match="already active"):
        shared.acquire_ov_libraries_render_lease()
    _prepare_leased_renderer(renderer)
    with pytest.raises(RuntimeError, match="Lease"):
        shared.render_batch((object(),))
    shared.render_batch(())
    with monkeypatch.context() as patch:
        patch.setattr(ovrtx_renderer_module, "OVRTXMesh", lambda *_: pytest.fail("constructed a mesh while leased"))
        with pytest.raises(RuntimeError, match="log runtime meshes"):
            shared.log_mesh("blocked")
    assert "blocked" not in shared.meshes
    with pytest.raises(RuntimeError, match="active render lease"):
        SimulationContext.instance().close_backend(renderer.backend)
    with pytest.raises(RuntimeError, match="active render lease"):
        renderer.scene.reset()
    native = lease.ovrtx_renderer
    assert native is shared.backend.renderer
    assert lease.ovstage is shared.scene.stage
    lease.release()
    lease.release()
    assert renderer.backend._render_lease_owner is None
    assert renderer.scene._render_lease_owner is None
    native.destroy.assert_not_called()
    SimulationContext.instance().close_backend(renderer.backend)
    native.destroy.assert_called_once_with()


def test_render_lease_rejects_acquisition_after_clone_preparation(leased_renderer):
    renderer = leased_renderer
    _prepare_leased_renderer(renderer, bind=False)
    with pytest.raises(RuntimeError, match="before clone preparation"):
        renderer.acquire_ov_libraries_render_lease()
    assert renderer.backend._render_lease_owner is None


def test_render_lease_lease_rejects_calibration_before_scene_access(leased_renderer, monkeypatch):
    """An ordinary calibration write cannot bypass the external owner's publication lane."""
    renderer = leased_renderer
    lease = renderer.acquire_ov_libraries_render_lease()
    camera = _make_ovrtx_camera_render_data()
    intrinsics = wp.zeros(2, dtype=wp.mat33f, device="cpu")
    parameters = wp.zeros((5, 2), dtype=wp.float32, device="cpu")
    drain = MagicMock(return_value=[])
    monkeypatch.setattr(renderer, "drain_pending_renders", drain)
    monkeypatch.setattr(wp, "get_stream", lambda device: types.SimpleNamespace(cuda_stream=99))
    try:
        with pytest.raises(RuntimeError, match="Lease"):
            renderer.update_camera_intrinsics(camera, intrinsics, parameters)
        drain.assert_not_called()
        assert renderer.backend is renderer.scene is None
    finally:
        lease.release()


def test_render_lease_owner_transfer_is_single_use(leased_renderer):
    import threading

    renderer = leased_renderer
    lease = renderer.acquire_ov_libraries_render_lease()
    with pytest.raises(RuntimeError, match="capability is invalid"):
        lease.claim_owner_transfer(None)
    token = lease.offer_owner_transfer()
    errors = []

    def owner():
        try:
            lease.claim_owner_transfer(token)
            with pytest.raises(RuntimeError, match="capability is invalid"):
                lease.claim_owner_transfer(token)
            lease.release()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=owner)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    assert renderer._render_lease._owner_thread is thread
    assert renderer.backend is renderer.scene is None


@pytest.fixture
def complete_scene_publication(leased_renderer, monkeypatch):
    """Exercise the application publisher with native component endpoints doubled."""
    renderer = leased_renderer
    lease = renderer.acquire_ov_libraries_render_lease()
    publisher = lease.capture_scene_publisher()
    _prepare_leased_renderer(renderer)
    lease._external_camera_scene_partitions = (("/Camera/left", "env"), ("/Camera/right", "env"))
    lease._external_camera_scene_partitions_pending = True
    stage = renderer.scene.stage
    stage.get_attribute_write_floor.reset_mock()
    stage.advance_write_floor.reset_mock()
    state = types.SimpleNamespace(
        renderer=renderer,
        lease=lease,
        publisher=publisher,
        ordinal=23,
        events=[],
        failure=None,
        cause=ValueError("component endpoint failed"),
        initial_id=renderer._render_lease._last_step_id,
    )

    def component(name):
        state.events.append((name, renderer.scene.ordinal))
        if state.failure == name:
            raise state.cause

    writer = types.SimpleNamespace(
        has_pending_updates=False,
        publish=lambda: component("material"),
        drain=lambda: component("material-drain"),
    )
    monkeypatch.setattr(renderer, "_visual_material_writer_ref", lambda: writer)
    monkeypatch.setattr(renderer, "_update_transforms", lambda: component("transform"))
    monkeypatch.setattr(renderer, "_update_geometries", lambda: component("geometry"))
    monkeypatch.setattr(renderer._render_lease, "_update_cameras", lambda: component("auxiliary"))

    def partitions(selected, ordinal):
        assert selected is lease
        assert selected._external_camera_scene_partitions == (("/Camera/left", "env"), ("/Camera/right", "env"))
        assert ordinal == renderer.scene.ordinal
        component("partition")

    monkeypatch.setattr(renderer._render_lease, "_write_camera_partitions", partitions)
    state.publish = lambda step_id="step": publisher.write_complete_scene(state.ordinal, step_id)
    try:
        yield state
    finally:
        lease.release()


@pytest.mark.parametrize("pending", [False, True])
def test_render_lease_complete_scene_components_share_the_supplied_ordinal(complete_scene_publication, pending):
    state = complete_scene_publication
    renderer, lease = state.renderer, state.lease
    lease._external_camera_scene_partitions_pending = pending
    assert state.publisher.publication_required("step")
    assert state.publish() is None
    names = ["material", "transform", "geometry", "auxiliary"] + (["partition"] if pending else []) + ["material-drain"]
    assert state.events == [(name, state.ordinal) for name in names]
    assert renderer._render_lease._last_step_id == "step"
    assert not lease._external_camera_scene_partitions_pending
    assert renderer.scene.ordinal == state.ordinal
    assert not state.publisher.publication_required("step")
    with pytest.raises(RuntimeError, match="publication is not required"):
        state.publish()
    assert state.events == [(name, state.ordinal) for name in names]
    renderer.scene.stage.get_attribute_write_floor.assert_not_called()
    renderer.scene.stage.advance_write_floor.assert_not_called()
    assert not renderer._render_lease._poisoned


@pytest.mark.parametrize("failure", ["material", "transform", "geometry", "auxiliary", "partition", "material-drain"])
def test_render_lease_complete_scene_failure_preserves_commit_boundary(complete_scene_publication, failure):
    state = complete_scene_publication
    renderer, lease = state.renderer, state.lease
    state.failure = failure
    with pytest.raises(RuntimeError, match="Producer scene publication failed") as caught:
        state.publish()
    assert caught.value.__cause__ is state.cause
    order = ["material", "transform", "geometry", "auxiliary", "partition", "material-drain"]
    expected = order[: order.index(failure) + 1]
    if failure != "material-drain":
        expected.append("material-drain")
    assert state.events == [(name, state.ordinal) for name in expected]
    assert renderer._render_lease._last_step_id == state.initial_id
    assert lease._external_camera_scene_partitions_pending
    assert renderer.scene.ordinal == state.ordinal
    assert renderer._render_lease._poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        _ = lease.ovstage
    previous_events = list(state.events)
    with pytest.raises(RuntimeError, match="unavailable|poisoned"):
        state.publish("next-step")
    assert state.events == previous_events
    renderer.scene.stage.advance_write_floor.assert_not_called()
    renderer.scene.stage.get_attribute_write_floor.assert_not_called()


def test_render_lease_complete_scene_public_authority_precedes_components(complete_scene_publication):
    import threading

    state = complete_scene_publication
    errors = []

    def foreign_thread():
        try:
            state.publish()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=foreign_thread)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert "thread" in str(errors[0])
    with pytest.raises(TypeError, match="physics_step_id must be hashable"):
        state.publish([])
    with pytest.raises(ValueError, match="ordinal must be nonzero"):
        state.publisher.write_complete_scene(0, "step")
    state.publisher.close()
    with pytest.raises(RuntimeError, match="publisher is closed"):
        state.publish()
    assert not state.events
    state.renderer.scene.stage.get_attribute_write_floor.assert_not_called()
    state.renderer.scene.stage.advance_write_floor.assert_not_called()
    assert not state.renderer._render_lease._poisoned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA material buffers")
@pytest.mark.parametrize("deferred", [False, True])
def test_leased_material_updates_reach_the_reserved_scene(leased_renderer, monkeypatch, deferred):
    """Real material composition and the lease publisher must reach native writes, including before physics changes."""
    renderer = leased_renderer
    lease = renderer.acquire_ov_libraries_render_lease()
    publisher = lease.capture_scene_publisher()
    context = renderer._sim.render_context
    if deferred:
        cfg = _make_camera_cfg(["rgb"])
        cfg.prim_path = "/World/Camera"
        lease.configure_auxiliary_camera(cfg)
    _prepare_leased_renderer(renderer)
    material = types.SimpleNamespace(
        channels=("color",),
        is_per_env=False,
        num_instances=1,
        _material_paths=("/Looks/Feedback",),
        _shader_paths=("/Looks/Feedback/Shader",),
        _input_names={"color": "diffuseColor"},
        _offsets={},
        _values={"color": torch.tensor([[1.0, 0.0, 0.0]], device="cuda")},
    )
    context.register_visual_material(material)
    context.finalize_consumers([])
    if deferred:
        spec = CameraRenderSpec(cfg, "cpu", 1, (cfg.prim_path,), 1)
        data = renderer.create_render_data(spec)
        monkeypatch.setattr(ovrtx_renderer_module.ovstage.population, "add_usd_reference_from_string", lambda *a: 1)
        monkeypatch.setattr(ovrtx_renderer_module.ovstage.population, "apply_usd_changes", lambda *a, **kw: None)
        token = lease.offer_owner_transfer()
        lease.claim_owner_transfer(token)
        assert lease.materialize_deferred_auxiliary_camera_stage(renderer._sim.stage, 1)
        renderer.update_camera(
            data,
            wp.zeros(1, dtype=wp.vec3f, device="cpu"),
            wp.array([[0.0, 0.0, 0.0, 1.0]], dtype=wp.quatf, device="cpu"),
            None,
        )
    stage = renderer.scene.stage
    try:
        for ordinal, color in ((23, [0.0, 1.0, 0.0]), (24, [0.0, 0.0, 1.0])):
            context.write_visual_materials([material], {"color": torch.tensor([color], device="cuda")})
            assert publisher.publication_required("unchanged-physics")
            stage.write_attribute.reset_mock()
            completion = MagicMock()
            stage.write_attribute.return_value = completion
            publisher.write_complete_scene(ordinal, "unchanged-physics")
            writes = [c for c in stage.write_attribute.call_args_list if c.args[1] == "inputs:diffuseColor"]
            assert len(writes) == 1
            assert writes[0].kwargs["ordinal"] == ordinal
            torch.testing.assert_close(writes[0].kwargs["tensors"].cpu(), torch.tensor([color]))
            assert completion.wait.called
            assert not publisher.publication_required("unchanged-physics")
    finally:
        lease.release()
        for writer in context._visual_material_writers:
            writer.close()


@pytest.mark.parametrize("use_ovstage, asynchronous", [(True, False), (False, False), (False, True)])
def test_cpu_scene_publication_uses_host_tensors_without_a_cuda_stream(use_ovstage, asynchronous):
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = use_ovstage
    renderer.cfg = OVRTXRendererCfg(async_rendering=asynchronous)
    renderer._warp_device = wp.get_device("cpu")
    renderer._device = "cpu"
    renderer._object_xform_binding = MagicMock()
    renderer._geometry_points_binding = MagicMock()
    renderer._transforms_timestamp = renderer._geometry_timestamp = None
    renderer._object_scales = None
    renderer._geometry_paths = ["/World/Object"]
    matrices = wp.array(np.eye(4, dtype=np.float64)[None], dtype=wp.mat44d, device="cpu")
    points = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3f, device="cpu")
    renderer._transform_writes = _AsyncWriteBuffers([types.SimpleNamespace(matrices=matrices)])
    renderer._geometry_writes = _AsyncWriteBuffers([wp.empty_like(points)])
    renderer._geometry_offsets = {"/World/Object": 0}

    def get_points(*, output=None, offsets=None):
        if output is None:
            return {"/World/Object": points}
        wp.copy(output, points)
        return None

    renderer._sdp = types.SimpleNamespace(
        backend=types.SimpleNamespace(transforms_timestamp=1, geometry_timestamp=1),
        get_transforms=lambda *_args, **_kwargs: True,
        get_geometry_points=get_points,
    )
    stage = MagicMock()
    renderer.scene = types.SimpleNamespace(stage=stage, ordinal=11)
    renderer.update_transforms()
    renderer.update_geometries()
    bindings = (renderer._object_xform_binding, renderer._geometry_points_binding)
    writes = [getattr(binding, "write_async" if asynchronous else "write") for binding in bindings]
    calls = stage.write_attribute.call_args_list if use_ovstage else [write.call_args for write in writes]
    assert len(calls) == 2 and all(c.kwargs["cuda_stream"] is None for c in calls)
    if use_ovstage:
        assert [c.args[1] for c in calls] == ["omni:xform", "points"]
        assert all(c.kwargs["ordinal"] == 11 for c in calls)
    else:
        np.testing.assert_array_equal(calls[0].args[0].numpy(), matrices.numpy())
        np.testing.assert_array_equal(calls[1].args[0][0].numpy(), points.numpy())
    # A repeated publication of unchanged state must not duplicate either write.
    renderer.update_transforms()
    renderer.update_geometries()
    assert stage.write_attribute.call_count == 2 if use_ovstage else all(write.call_count == 1 for write in writes)
    camera = _make_ovrtx_camera_render_data()
    camera.camera_xform_binding = MagicMock()
    camera.camera_writes = _AsyncWriteBuffers([wp.empty(2, dtype=wp.mat44d, device="cpu")])
    positions = wp.array([[1.0, 2.0, 3.0]] * 2, dtype=wp.vec3f, device="cpu")
    orientations = wp.array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=wp.quatf, device="cpu")
    update = renderer.update_camera
    update(camera, positions, orientations, None)
    call = stage.write_attribute.call_args if use_ovstage else camera.camera_xform_binding.write_async.call_args
    assert call.kwargs["cuda_stream"] is None
    if use_ovstage:
        assert call.args == (camera.camera_xform_binding, "omni:xform")
        assert call.kwargs["ordinal"] == 11
    else:
        # Reusing the camera ring waits for previous writes without a CUDA stream.
        update(camera, positions, orientations, None)
        update(camera, positions, orientations, None)
        camera.camera_writes.close()
        renderer._transform_writes.close()
        renderer._geometry_writes.close()


@pytest.mark.parametrize("cuda_ordinal", [None, 2])
def test_camera_registration_authors_valid_render_device_selection(monkeypatch, cuda_ordinal):
    """CPU camera consumers leave GPU selection automatic; CUDA consumers retain their device."""
    from pxr import Usd

    class AuthoredProduct(Exception):
        pass

    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg = OVRTXRendererCfg()

    class Device:
        is_cuda = cuda_ordinal is not None
        ordinal = cuda_ordinal

        def __str__(self):
            return f"cuda:{cuda_ordinal}" if self.is_cuda else "cpu"

    renderer._warp_device = Device()
    renderer._device = str(renderer._warp_device)
    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_device", lambda _device: renderer._warp_device)
    renderer.scene.clone_env_paths = []
    renderer.drain_pending_renders = lambda: []
    spec = types.SimpleNamespace(
        cfg=_make_camera_cfg(["rgb"]),
        device=renderer._device,
        num_instances=1,
        camera_prim_paths=["/World/envs/env_0/Camera"],
        render_settings={},
    )
    build_product = ovrtx_renderer_module.build_render_product_as_string

    def consume_product(spec, render_data, **kwargs):
        stage = Usd.Stage.CreateInMemory()
        assert stage.GetRootLayer().ImportFromString(build_product(spec, render_data, **kwargs))
        product = stage.GetPrimAtPath(render_data.render_product_path)
        devices = product.GetAttribute("deviceIds")
        if cuda_ordinal is None:
            assert not devices.HasAuthoredValueOpinion()
        else:
            assert list(devices.Get()) == [cuda_ordinal]
        # Native population is outside this parser/selection regression.
        raise AuthoredProduct

    monkeypatch.setattr(ovrtx_renderer_module, "build_render_product_as_string", consume_product)
    with pytest.raises(AuthoredProduct):
        renderer.create_render_data(spec)


def test_external_camera_native_rate_is_authored_only_on_export(leased_renderer):
    from pxr import Usd, UsdGeom

    renderer = leased_renderer
    lease = renderer.acquire_ov_libraries_render_lease()
    cfg = _make_camera_cfg(["rgb"])
    cfg.prim_path = "/World/envs/env_0/Camera"
    lease.configure_auxiliary_camera(cfg, max_fps=20.0)
    lease._auxiliary_cameras[0].spec = types.SimpleNamespace(camera_prim_paths=[cfg.prim_path])
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, cfg.prim_path).GetPrim()
    exported = Usd.Stage.CreateInMemory()
    assert exported.GetRootLayer().ImportFromString(
        renderer._render_lease._author_camera_schedules(stage.ExportToString(), lease)
    )
    assert not camera.HasAttribute("omni:sensor:frameRate")
    assert not camera.HasAttribute("omni:sensor:Core:partialOutputs")
    prim = exported.GetPrimAtPath(cfg.prim_path)
    assert tuple(prim.GetAttribute("omni:sensor:frameRate").Get()) == (20.0, 1.0)
    assert prim.GetAttribute("omni:sensor:Core:partialOutputs").Get() is False
    lease.release()


def test_selected_external_camera_order_is_independent_of_sensor_preparation(leased_renderer):
    renderer = leased_renderer
    lease = renderer.acquire_ov_libraries_render_lease()
    specs = []
    for name, hz in zip(("Head", "Left", "Right", "Overview"), (20, 10, 10, 5), strict=True):
        cfg = types.SimpleNamespace(prim_path=f"/World/{name}", width=640, height=480, data_types=("rgb",))
        lease.configure_auxiliary_camera(cfg, max_fps=hz)
        specs.append(types.SimpleNamespace(cfg=cfg, num_instances=1, view_count=1, camera_prim_paths=(cfg.prim_path,)))
    for spec in reversed(specs):
        assert renderer._render_lease.accept_camera_spec(spec)
    assert [camera.spec for camera in lease._auxiliary_cameras] == specs
    assert [camera.max_fps for camera in lease._auxiliary_cameras] == [20, 10, 10, 5]
    lease.release()


def test_external_camera_partition_validation_is_not_stereo_specific(leased_renderer, monkeypatch):
    renderer = leased_renderer
    lease = renderer.acquire_ov_libraries_render_lease()
    _prepare_leased_renderer(renderer)
    cameras = ("/Camera/front", "/Camera/side", "/Camera/rear")
    tokens = {
        ("/World/envs/env_0", "primvars:omni:scenePartition"): "env_0",
        **{(path, "omni:scenePartition"): "env_0" for path in cameras},
    }
    monkeypatch.setattr(renderer._render_lease, "_read_partition_tokens", lambda *_: tokens)
    try:
        with lease.ovstage_owner_lane():
            lease.configure_external_camera_scene_partitions("/World/envs/env_0", cameras)
            assert lease.validate_external_scene_partitions(1) == {
                "/World/envs/env_0": "env_0",
                **dict.fromkeys(cameras, "env_0"),
            }
            tokens[(cameras[-1], "omni:scenePartition")] = "other"
            with pytest.raises(RuntimeError, match="scene-partition proof failed"):
                lease.validate_external_scene_partitions(1)
    finally:
        lease.release()


def test_external_camera_schedules_reach_native_stage(monkeypatch):
    """The renderer's real population path retains only selected camera scheduling properties."""
    import ovrtx
    import ovstage

    ovrtx.register_schema_paths()
    camera_paths = [f"/World/envs/env_0/{name}" for name in ("Head", "Wrist", "Unselected")]
    source = """#usda 1.0
def Xform "World" {
 def Xform "envs" {
  def Xform "env_0" {
   def Camera "Head" {
    double2 omni:sensor:frameRate = (20, 1)
    bool omni:sensor:Core:partialOutputs = false
   }
   def Camera "Wrist" {
    double2 omni:sensor:frameRate = (10, 1)
    bool omni:sensor:Core:partialOutputs = false
   }
   def Camera "Unselected" {
    double2 omni:sensor:frameRate = (999, 1)
   }
  }
 }
}
"""
    cfg = _make_camera_cfg(["rgb"])
    cfg.prim_path = camera_paths[0]
    spec = types.SimpleNamespace(cfg=cfg, num_instances=1, camera_prim_paths=[camera_paths[0]], render_settings={})
    data = OVRTXCameraRenderData(spec, "RenderCamera_0", list(spec.camera_prim_paths))
    from isaaclab_ov.stage import OvstageBackend, OvstageBackendCfg

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._render_lease.stage_deferred = False
    renderer._render_lease.active = None
    renderer.backend.renderer = MagicMock()
    with contextlib.ExitStack() as resources:
        renderer.scene = OvstageBackend(OvstageBackendCfg())
        resources.callback(renderer.scene.close)
        stage, paths = renderer.scene.stage, renderer.scene.paths
        population = ovstage.population
        description = population.Desc(
            domains=ovstage.PopulationDomain.RENDERING,
            selectors=[
                population.Selector(
                    population.PrimPredicate.has_path(*camera_paths[:2]),
                    population.PropertyPredicate.has_name("omni:sensor:frameRate", "omni:sensor:Core:partialOutputs"),
                )
            ],
        )
        renderer.scene.populate(source, population_desc=description)
        try:
            renderer._register_camera(spec, data)
            renderer.scene.commit()
            resolved = {}
            plist = paths.create_path_list_from_strings(camera_paths)
            try:
                with stage.query_from_path_list(plist) as query:
                    query.wait()
                    names = ("omni:sensor:frameRate", "omni:sensor:Core:partialOutputs")
                    with stage.read_attributes(
                        query,
                        [paths.intern_token(name) for name in names],
                        ovstage.OrdinalRange.latest(renderer.scene.ordinal - 1),
                    ) as read:
                        read.wait()
                        for group in read.groups():
                            try:
                                values = np.asarray(group.array(0).copy()).reshape(group.data_count, -1)
                                group_paths = paths.get_path_strings(group.prim_list)
                                attribute = paths.token_to_string(group.attribute)
                                for index in range(min(group.prim_count, group.data_count)):
                                    path = group_paths[group.prim_index(index)]
                                    resolved[path, attribute] = values[group.data_row_index(index)].tolist()
                            finally:
                                stage.release_group(group)
            finally:
                paths.destroy_path_list(plist)
            assert resolved.get((camera_paths[0], names[0])) == [20.0, 1.0]
            assert resolved.get((camera_paths[1], names[0])) == [10.0, 1.0]
            assert resolved.get((camera_paths[0], names[1])) == [False]
            assert resolved.get((camera_paths[1], names[1])) == [False]
            assert (camera_paths[2], names[0]) not in resolved
        finally:
            data.resources.close()
