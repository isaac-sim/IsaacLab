# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVXR renderer: renderer contract, lease, owner transfer and scene publication.

The native OVRTX and OVStage endpoints are doubled; the renderer, its lease and the clone
preparation are real.
"""

import ast
import contextlib
import importlib.util
import inspect
import types
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock

import numpy as np
import ovstage
import pytest
import torch
import warp as wp

from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.renderers import BaseRenderer, RenderBufferKind
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import PinholeCameraCfg, SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

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
    import isaaclab_teleop.ovxr_gpu_publication as ovxr_gpu_publication  # noqa: E402
    import isaaclab_teleop.ovxr_renderer as ovxr_renderer_module  # noqa: E402
    import isaaclab_teleop.ovxr_scene as ovxr_scene_module  # noqa: E402
    from isaaclab_teleop.ovxr_gpu_publication import _GeometryPublicationBuffers  # noqa: E402
    from isaaclab_teleop.ovxr_renderer import OvxrRenderer, OvxrRendererCfg, OvxrScenePublisher  # noqa: E402

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


@pytest.fixture(autouse=True)
def _simulation_registry(monkeypatch):
    from pxr import Usd

    from isaaclab.renderers import RenderContext

    registry = []
    sim = types.SimpleNamespace(
        device="cpu",
        has_gui=False,
        has_offscreen_render=False,
        visualizers=[],
        get_setting=lambda _key: False,
        vis_marker_registry=VisMarkerRegistry(),
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
    # Isolate mocked engine startup from the process's retained rendering GPU selection.
    if not _MISSING_MODULES:
        monkeypatch.setattr(ovxr_scene_module, "_initial_gpu_selection", None)


@pytest.fixture
def leased_renderer(monkeypatch):
    """Exercise real deferred resource acquisition with native SDK endpoints doubled."""
    import isaaclab_ov.stage as stage_module

    native, stage, paths = MagicMock(), MagicMock(), MagicMock()
    floor = stage.get_attribute_write_floor.return_value.__enter__.return_value
    floor.fetch.return_value = ovstage.ORDINAL_MIN  # A new OVStage is sealed through the minimum ordinal.

    def advance(*, ordinal, **kwargs):
        return types.SimpleNamespace(wait=lambda: setattr(floor.fetch, "return_value", ordinal))

    stage.advance_write_floor.side_effect = advance
    monkeypatch.setattr(ovxr_scene_module, "Renderer", lambda _cfg: native)
    monkeypatch.setattr(ovxr_scene_module, "redirect_shader_cache", lambda _cfg: None)
    monkeypatch.setattr(stage_module, "create_ovstage", lambda _name: contextlib.nullcontext(stage))
    monkeypatch.setattr(stage_module.ovstage, "PathDictionary", lambda _: contextlib.nullcontext(paths))
    monkeypatch.setattr(stage_module.ovstage.population, "open_usd_from_string", lambda *args, **kwargs: None)
    renderer = SimulationContext.instance().get_or_create_backend(OvxrRendererCfg())
    assert renderer.backend is renderer.scene is None
    return renderer


def _prepare_leased_renderer(renderer, *, bind=True):
    """Create the lease's pair and, with ``bind``, prepare the Stage on the claimed render owner."""
    lease = renderer._render_lease.active
    lease.create_native_pair()
    if bind:
        lease.claim_owner_transfer(lease.offer_owner_transfer())
        renderer.prepare_stage(renderer._sim.stage, 1)


def test_ovxr_renderer_is_a_base_renderer_independent_of_the_ovrtx_renderer():
    """The OVXR renderer implements the core renderer interface without extending ``isaaclab_ov``'s classes."""
    import isaaclab_teleop
    import isaaclab_teleop.ovxr_session as ovxr_session_module
    from isaaclab_ov.renderers import OVRTXRenderer, OVRTXRendererCfg

    assert issubclass(OvxrRenderer, BaseRenderer) and not issubclass(OvxrRenderer, OVRTXRenderer)
    assert issubclass(OvxrRendererCfg, RendererCfg) and not issubclass(OvxrRendererCfg, OVRTXRendererCfg)
    for module in (ovxr_renderer_module, ovxr_scene_module, ovxr_session_module):
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__:
                inherited = [base for base in cls.__mro__ if base.__module__.split(".")[0] == "isaaclab_ov"]
                assert not inherited, f"{cls.__qualname__} inherits from {inherited}"
    # No teleop module imports the OVRTX renderer's implementation or its material writer.
    forbidden_modules = {"isaaclab_ov.renderers.ovrtx_renderer", "isaaclab_ov.renderers.visual_materials"}
    forbidden_names = {"OVRTXRenderer", "OVRTXBackend", "OVRTXVisualMaterialWriter", "ovrtx_renderer"}
    for path in Path(isaaclab_teleop.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert not {alias.name for alias in node.names} & forbidden_modules, path
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("isaaclab_ov"):
                assert node.module not in forbidden_modules, path
                assert not {alias.name for alias in node.names} & forbidden_names, path


def test_renderer_configuration_declares_its_own_ovstage_clone_route():
    cfg = OvxrRendererCfg()
    assert cfg.renderer_type == "ovxr"
    assert cfg.cloning_contexts == ("isaaclab_ov.cloner:OvstageReplicateContext",)
    assert type(cfg).launcher_type == "isaaclab_ov.app:OvrtxLauncher"
    # Camera panels request colour, or depth for a direct check of the rendered panel.
    assert set(cfg.supported_output_types()) == {RenderBufferKind.RGBA, RenderBufferKind.RGB, RenderBufferKind.DEPTH}


def test_base_renderer_contract_under_the_session_lease(leased_renderer, monkeypatch):
    """Each ``BaseRenderer`` operation has the behavior the OVXR session requires, before and after the lease."""
    renderer = leased_renderer
    stage = renderer._sim.stage
    spec = CameraRenderSpec(_make_camera_cfg(["rgb"]), "cpu", 1, ("/World/envs/env_0/Camera",), 1)
    camera = object()
    frame = wp.zeros(1, dtype=wp.int64, device="cpu")

    # Without the session's lease the renderer neither prepares a Stage nor writes materials.
    with pytest.raises(RuntimeError, match="only under the OVXR session"):
        renderer.prepare_stage(stage, 1)
    assert renderer.visual_material_writer is None
    lease = renderer.acquire_render_lease()
    assert renderer.visual_material_writer == renderer._create_visual_material_writer
    lease.create_native_pair()

    # Before the render worker claims the pair, preparation is only recorded; the claimed owner completes it.
    renderer.prepare_stage(stage, 1)
    assert renderer._render_lease.stage_deferred and not renderer._initialized_scene
    lease.claim_owner_transfer(lease.offer_owner_transfer())
    assert lease.materialize_deferred_stage(stage, 1)
    assert renderer._initialized_scene and renderer.backend.attached
    lease.validate_live()

    # Ordinary scene updates write nothing: the session's publisher writes every scene.
    monkeypatch.setattr(renderer, "_publish_transforms", lambda: pytest.fail("eager transform write"))
    monkeypatch.setattr(renderer, "_publish_geometries", lambda: pytest.fail("eager geometry write"))
    renderer.update_transforms()
    renderer.update_geometries()

    # Cameras other than the session's camera panels are rejected, and nothing produces camera images.
    with pytest.raises(RuntimeError, match="only the OVXR session's camera panels"):
        renderer.create_render_data(spec)
    with pytest.raises(RuntimeError, match="no such camera panel"):
        renderer.update_camera(camera, None, None, None)
    for name, operation in {
        "render": lambda: renderer.render(camera),
        "render_batch": lambda: renderer.render_batch((camera,)),
        "read_output": lambda: renderer.read_output(camera, None),
        "update_camera_intrinsics": lambda: renderer.update_camera_intrinsics(camera, None, None),
    }.items():
        with pytest.raises(RuntimeError, match="produces no camera images"):
            operation()
    renderer.render_batch(())
    assert renderer.set_outputs(camera, {}) is None
    assert renderer.prepare_capture(camera, None, frame) is None
    assert renderer.reset(camera) is None

    # Resources stay alive while the session borrows them.
    renderer.cleanup(None)
    with pytest.raises(RuntimeError, match="while an OvxrRenderLease is active"):
        renderer.cleanup(camera)
    with pytest.raises(RuntimeError, match="while an OvxrRenderLease is active"):
        renderer.close()
    assert renderer._initialized_scene and not renderer._closed

    lease.release()
    assert renderer.visual_material_writer is None
    renderer.update_transforms()
    renderer.update_geometries()
    renderer.cleanup(camera)
    renderer.close()
    assert renderer._closed and not renderer._initialized_scene
    renderer.close()


def test_render_lease_borrows_registry_pair(leased_renderer, monkeypatch):
    renderer = leased_renderer
    shared = SimulationContext.instance().get_or_create_backend(renderer.cfg)
    assert shared is renderer
    lease = renderer.acquire_render_lease()
    with pytest.raises(RuntimeError, match="already active"):
        shared.acquire_render_lease()
    _prepare_leased_renderer(renderer)
    native = lease.ovrtx_renderer
    assert native is shared.backend.renderer
    assert lease.ovstage is shared.scene.stage
    # A failed transfer drain keeps the lease, so the renderer still refuses to close.
    publication = _GeometryPublicationBuffers.__new__(_GeometryPublicationBuffers)
    source, stream = object(), object()
    publication._sources = [source]
    publication._source_streams = [stream]
    publication._stream = None
    publication._buffers = {}
    renderer._geometry_publication = publication
    with monkeypatch.context() as patch:
        patch.setattr(wp, "synchronize_stream", MagicMock(side_effect=RuntimeError("producer drain failed")))
        with pytest.raises(RuntimeError, match="producer drain failed"):
            lease.release()
        assert publication._sources == [source]
        assert renderer._sim.vis_marker_registry.has_consumers
        with pytest.raises(RuntimeError, match="while an OvxrRenderLease is active"):
            SimulationContext.instance().close_backend(renderer)
        native.destroy.assert_not_called()
    monkeypatch.setattr(wp, "synchronize_stream", lambda _stream: None)
    lease.release()
    lease.release()
    assert not renderer._sim.vis_marker_registry.has_consumers
    native.destroy.assert_not_called()
    SimulationContext.instance().close_backend(renderer.backend)
    native.destroy.assert_called_once_with()


def test_render_lease_rejects_acquisition_after_stage_preparation(leased_renderer):
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    _prepare_leased_renderer(renderer)
    lease.release()
    with pytest.raises(RuntimeError, match="before Stage preparation"):
        renderer.acquire_render_lease()
    assert renderer._render_lease.active is None


def test_lease_creates_its_own_stage_and_engine(leased_renderer, monkeypatch):
    """The lease registers the renderer's own OVStage, then the engine that attaches to it, on the simulation."""
    from isaaclab_ov.stage import OvstageBackend

    renderer = leased_renderer
    sim = renderer._sim
    configs = []
    monkeypatch.setattr(ovxr_scene_module, "RendererConfig", lambda **kwargs: configs.append(kwargs) or kwargs)
    lease = renderer.acquire_render_lease()
    assert renderer.backend is renderer.scene is None
    lease.create_native_pair()
    backend, scene = renderer.backend, renderer.scene
    assert isinstance(backend, ovxr_scene_module._OvxrEngine) and isinstance(scene, OvstageBackend)
    assert backend.cfg.scene_key == renderer.cfg and scene.cfg.scene_key == renderer.cfg
    assert not backend.attached and backend.render_products == set()
    resources = [resource for _cfg, resource in sim._backend_registry]
    assert resources.index(scene) < resources.index(backend)
    (native_cfg,) = configs
    assert native_cfg["read_gpu_transforms"] is True
    assert native_cfg["active_cuda_gpus"] is None  # The CPU simulation device leaves GPU selection to OVRTX.
    assert native_cfg["log_file_path"] == renderer.cfg.log_file_path
    lease.create_native_pair()
    assert renderer.backend is backend and renderer.scene is scene
    lease.release()


def test_ordinary_ovrtx_clone_preparation_leaves_the_session_pair(leased_renderer, monkeypatch):
    """An ordinary OVRTX camera renderer in the same process builds its own engine; the session's pair is untouched."""
    from isaaclab_ov.cloner import OvrtxReplicateContext
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module

    monkeypatch.setattr(ovrtx_renderer_module, "Renderer", lambda _cfg: MagicMock())
    monkeypatch.setattr(ovrtx_renderer_module, "redirect_shader_cache", lambda _cfg: None)
    renderer = leased_renderer
    sim = renderer._sim
    lease = renderer.acquire_render_lease()
    lease.create_native_pair()
    backend, scene = renderer.backend, renderer.scene
    ordinary = sim.get_or_create_backend(OVRTXRendererCfg(log_file_path="ordinary.log"))
    OvrtxReplicateContext.prepare(sim, {OvrtxReplicateContext: {0}})
    assert ordinary.backend is not None and ordinary.backend is not backend
    assert renderer.backend is backend and renderer.scene is scene
    lease.release()


def test_clone_plan_reaches_the_session_stage_through_the_real_cloner(leased_renderer):
    """Clone dispatch routes spawned assets to the renderer's declared OVStage context and its own Stage."""
    from isaaclab_ov.cloner import OvstageReplicateContext

    from isaaclab.assets import AssetBaseCfg
    from isaaclab.cloner import make_clone_plan, replicate
    from isaaclab.sim import SpawnerCfg

    renderer = leased_renderer
    sim = renderer._sim
    sim.cfg = types.SimpleNamespace(physics=types.SimpleNamespace(mesh_variant_source_root=None))
    lease = renderer.acquire_render_lease()
    lease.create_native_pair()
    cfgs = (AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Robot", spawn=SpawnerCfg(spawn_path="/Sources/Robot")),)
    sim.plan = make_clone_plan(cfgs, ((0,),), 2, positions=np.zeros((2, 3), dtype=np.float32))
    replicate(sim.plan)
    assert isinstance(sim.clone_contexts[OvstageReplicateContext], OvstageReplicateContext)
    scene = renderer.scene
    assert scene.clone_copies == [("/Sources/Robot", ["/World/envs/env_0/Robot", "/World/envs/env_1/Robot"])]
    assert scene.clone_env_paths == ["/World/envs/env_0", "/World/envs/env_1"]
    assert scene.population_env_paths == scene.clone_env_paths
    np.testing.assert_array_equal(scene.clone_positions, np.zeros((2, 3)))
    lease.release()


def test_native_pair_creation_failure_leaves_the_lease_retryable(leased_renderer, monkeypatch):
    """A failed engine start assigns nothing; the same lease can create the pair once the cause is gone."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    construct = ovxr_scene_module.Renderer
    monkeypatch.setattr(ovxr_scene_module, "Renderer", MagicMock(side_effect=RuntimeError("native start failed")))
    with pytest.raises(RuntimeError, match="native start failed"):
        lease.create_native_pair()
    assert renderer.backend is renderer.scene is None
    monkeypatch.setattr(ovxr_scene_module, "Renderer", construct)
    lease.create_native_pair()
    assert renderer.backend is not None and renderer.scene is not None
    lease.release()


def test_released_unprepared_lease_can_be_acquired_again(leased_renderer):
    """A pair an earlier lease created and never prepared serves the next lease on the same renderer."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    lease.create_native_pair()
    backend, scene = renderer.backend, renderer.scene
    lease.release()
    lease = renderer.acquire_render_lease()
    lease.create_native_pair()
    assert renderer.backend is backend and renderer.scene is scene
    lease.release()


def test_leased_stage_preparation_requires_the_pair_the_lease_creates(leased_renderer):
    """Without the session's pair the claimed render owner has no Stage to prepare."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    lease.claim_owner_transfer(lease.offer_owner_transfer())
    with pytest.raises(RuntimeError, match="create_native_pair"):
        renderer.prepare_stage(renderer._sim.stage, 1)
    assert not renderer._initialized_scene
    lease.release()


def test_leased_stage_preparation_requires_an_unused_stage(leased_renderer):
    """The claimed render owner prepares only the fresh OVStage that the lease created."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    _prepare_leased_renderer(renderer, bind=False)
    lease.claim_owner_transfer(lease.offer_owner_transfer())
    floor = renderer.scene.stage.get_attribute_write_floor.return_value.__enter__.return_value
    floor.fetch.return_value = ovstage.ORDINAL_MIN + 1
    with pytest.raises(RuntimeError, match="unused OVStage"):
        renderer.prepare_stage(renderer._sim.stage, 1)
    assert not renderer._initialized_scene
    floor.fetch.return_value = ovstage.ORDINAL_MIN
    renderer.prepare_stage(renderer._sim.stage, 1)
    lease.validate_live()
    lease.release()


def test_render_lease_owner_transfer_is_single_use(leased_renderer):
    import threading

    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
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


class _Ledger:
    """Stage ordinal ledger double that reserves one fixed ordinal and records seals and abandons."""

    def __init__(self, ordinal: int):
        self.ordinal = ordinal
        self.reserved, self.published, self.abandoned = [], [], []

    def reserve(self) -> int:
        self.reserved.append(self.ordinal)
        return self.ordinal

    def publish(self, ordinal: int) -> None:
        self.published.append(ordinal)

    def abandon(self, ordinal: int) -> None:
        self.abandoned.append(ordinal)


@pytest.fixture
def complete_scene_publication(leased_renderer, monkeypatch):
    """Exercise the application publisher with native component endpoints doubled."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    publisher = lease.capture_scene_publisher()
    _prepare_leased_renderer(renderer)
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
    monkeypatch.setattr(renderer, "_publish_transforms", lambda: component("transform"))
    monkeypatch.setattr(renderer, "_publish_geometries", lambda: component("geometry"))
    monkeypatch.setattr(renderer._panels, "publish_poses", lambda: component("auxiliary"))
    state.ledger = _Ledger(state.ordinal)
    state.publish = lambda step_id="step", force=False: publisher.publish(step_id, state.ledger, force=force)
    try:
        yield state
    finally:
        lease.release()


def test_render_lease_complete_scene_components_share_the_supplied_ordinal(complete_scene_publication):
    state = complete_scene_publication
    renderer = state.renderer
    assert state.publisher.publication_required("step")
    assert state.publish() == state.ordinal
    assert state.ledger.published == [state.ordinal]
    names = ["material", "transform", "geometry", "auxiliary", "material-drain"]
    assert state.events == [(name, state.ordinal) for name in names]
    assert renderer._render_lease._last_step_id == "step"
    assert renderer.scene.ordinal == state.ordinal
    assert not state.publisher.publication_required("step")
    assert state.publish() is None
    assert state.ledger.reserved == [state.ordinal]
    assert state.events == [(name, state.ordinal) for name in names]
    renderer.scene.stage.get_attribute_write_floor.assert_not_called()
    renderer.scene.stage.advance_write_floor.assert_not_called()
    assert not renderer._render_lease._poisoned

    # An independently changed XR anchor still needs this same scene identity.
    state.events.clear()
    assert state.publisher.publish("step", _Ledger(state.ordinal + 1), force=True) == state.ordinal + 1
    assert state.events == [(name, state.ordinal + 1) for name in names]


def test_publication_prepares_only_a_scene_that_needs_publication(complete_scene_publication):
    state = complete_scene_publication
    prepared = []
    assert state.publisher.publish("step", state.ledger, prepare=lambda: prepared.append(1)) == state.ordinal
    assert state.publisher.publish("step", state.ledger, prepare=lambda: prepared.append(2)) is None
    assert prepared == [1]


def test_failed_seal_poisons_the_pair_and_abandons_its_ordinal(complete_scene_publication):
    """A failed seal is a failed publication, after every component was written."""
    state = complete_scene_publication

    def seal(_ordinal):
        raise RuntimeError("seal failed")

    with pytest.raises(RuntimeError, match="seal failed"):
        state.publisher.publish("step", state.ledger, seal=seal)
    assert [name for name, _ in state.events][-1] == "material-drain"
    assert state.ledger.abandoned == [state.ordinal] and not state.ledger.published
    assert state.renderer._render_lease._poisoned
    assert state.publisher.closed


def test_markers_require_publication_only_when_their_registry_revision_changes(complete_scene_publication):
    """Unchanged markers do not repeat a scene write; added, updated, hidden or removed markers do."""
    state = complete_scene_publication
    renderer = state.renderer
    registry = renderer._sim.vis_marker_registry
    registry.revision = 4
    renderer.marker_groups = (object(),)
    assert state.publisher.publication_required("step")
    state.publish()
    assert not state.publisher.publication_required("step")
    registry.revision += 1
    assert state.publisher.publication_required("step")
    state.publish()
    assert not state.publisher.publication_required("step")
    # Removing the last group still publishes once, so that the renderer retires its markers.
    renderer.marker_groups = ()
    renderer.markers = {"retired": object()}
    registry.revision += 1
    assert state.publisher.publication_required("step")


def test_missing_camera_sample_is_rejected_before_any_component_and_can_recover(complete_scene_publication):
    """A missing second camera cannot leave the first camera or geometry partially committed."""
    state = complete_scene_publication
    ready = types.SimpleNamespace(state=("sample",), render_data=types.SimpleNamespace(camera_xform_binding=object()))
    missing = types.SimpleNamespace(state=None, render_data=types.SimpleNamespace(camera_xform_binding=object()))
    state.renderer._panels.cameras = [ready, missing]
    original_ordinal = state.renderer.scene.ordinal
    with pytest.raises(RuntimeError, match="no live world-pose sample"):
        state.publish()
    assert not state.events
    assert not state.ledger.reserved
    assert state.renderer.scene.ordinal == original_ordinal
    assert not state.renderer._render_lease._poisoned
    assert not state.publisher.closed
    assert state.renderer._render_lease._last_step_id == state.initial_id
    missing.state = ("fresh sample",)
    assert state.publish() == state.ordinal
    assert state.renderer._render_lease._last_step_id == "step"
    assert state.events


@pytest.mark.parametrize("failure", ["material", "transform", "geometry", "auxiliary", "material-drain"])
def test_render_lease_complete_scene_failure_preserves_commit_boundary(complete_scene_publication, failure):
    state = complete_scene_publication
    renderer, lease = state.renderer, state.lease
    state.failure = failure
    with pytest.raises(RuntimeError, match="Producer scene publication failed") as caught:
        state.publish()
    assert caught.value.__cause__ is state.cause
    order = ["material", "transform", "geometry", "auxiliary", "material-drain"]
    expected = order[: order.index(failure) + 1]
    if failure != "material-drain":
        expected.append("material-drain")
    assert state.events == [(name, state.ordinal) for name in expected]
    assert state.ledger.abandoned == [state.ordinal] and not state.ledger.published
    assert renderer._render_lease._last_step_id == state.initial_id
    assert renderer.scene.ordinal == state.ordinal
    assert renderer._render_lease._poisoned
    assert state.publisher.closed
    with pytest.raises(RuntimeError, match="poisoned"):
        _ = lease.ovstage
    previous_events = list(state.events)
    with pytest.raises(RuntimeError, match="unavailable|poisoned|closed"):
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
    state.publisher.close()
    with pytest.raises(RuntimeError, match="publisher is closed"):
        state.publish()
    assert not state.events
    state.renderer.scene.stage.get_attribute_write_floor.assert_not_called()
    state.renderer.scene.stage.advance_write_floor.assert_not_called()
    assert not state.renderer._render_lease._poisoned


def test_render_lease_closed_publisher_reports_no_pending_publication(leased_renderer, monkeypatch):
    """A closed publisher must not report a write that :meth:`~OvxrScenePublisher.publish` would then reject."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    publisher = lease.capture_scene_publisher()
    assert isinstance(publisher, OvxrScenePublisher)
    _prepare_leased_renderer(renderer)
    monkeypatch.setattr(renderer, "_publish_transforms", lambda: None)
    monkeypatch.setattr(renderer, "_publish_geometries", lambda: None)
    assert publisher.publication_required("step-1")
    publisher.close()
    publisher.close()
    with pytest.raises(RuntimeError, match="publisher is closed"):
        publisher.publication_required("step-1")
    with pytest.raises(RuntimeError, match="publisher is closed"):
        publisher.publish("step-1", _Ledger(18))
    lease.release()


@pytest.mark.parametrize("failure_owner", ["producer", "borrower"])
def test_render_lease_poison_blocks_publication_but_not_retirement(leased_renderer, monkeypatch, failure_owner):
    """A failed producer write must not prevent the borrower from retiring the pair (Capture's close order)."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    publisher = lease.capture_scene_publisher()
    _prepare_leased_renderer(renderer)
    monkeypatch.setattr(renderer, "_publish_transforms", MagicMock(side_effect=ValueError("write failed")))
    monkeypatch.setattr(renderer, "_publish_geometries", lambda: None)
    if failure_owner == "producer":
        ledger = _Ledger(19)
        with pytest.raises(RuntimeError, match="scene publication failed"):
            publisher.publish("step-1", ledger)
        assert ledger.abandoned == [19] and not ledger.published
    else:
        publisher.invalidate()
    with pytest.raises(RuntimeError, match="unavailable|closed"):
        publisher.publication_required("step-2")
    with pytest.raises(RuntimeError, match="poisoned"):
        lease.validate_live()
    # OvxrFramePump.close: close publication, destroy the OVXR context inside the owner lane, release the lease.
    publisher.close()
    with lease.ovstage_owner_lane():
        pass
    ordinal = renderer.scene.ordinal
    lease.release()
    assert renderer.scene.ordinal == ordinal  # The abandoned ordinal is not rebased onto a sealed floor.
    # The pair cannot serve another lease; the renderer can still be closed and recreated.
    with pytest.raises(RuntimeError, match="before Stage preparation"):
        renderer.acquire_render_lease()
    renderer.close()


def test_render_lease_release_rebases_ordinal(leased_renderer):
    """After release, scene removals while closing the renderer write above the last sealed ordinal."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    _prepare_leased_renderer(renderer)
    floor = renderer.scene.stage.get_attribute_write_floor.return_value.__enter__.return_value
    floor.fetch.return_value = 41
    lease.release()
    assert renderer.scene.ordinal == 42
    renderer.close()


def test_render_lease_defers_application_thread_stage_until_owner_transfer(leased_renderer):
    """Camera initialization prepares every renderer on the application thread before XR transfers ownership."""
    import threading

    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    _prepare_leased_renderer(renderer, bind=False)
    context = renderer._sim.render_context
    context.ensure_prepare_stage(renderer._sim.stage, 1)
    assert not renderer._initialized_scene
    assert renderer._render_lease.stage_deferred
    # The deferred call must leave the pair eligible for the XR worker handoff.
    publisher = lease.capture_scene_publisher()
    token = lease.offer_owner_transfer()
    errors = []

    def owner():
        try:
            lease.claim_owner_transfer(token)
            assert lease.materialize_deferred_stage(renderer._sim.stage, 1)
            assert not lease.materialize_deferred_stage(renderer._sim.stage, 1)
            lease.validate_live()
            publisher.close()
            lease.release()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=owner)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    assert renderer._initialized_scene


def test_render_lease_eager_scene_updates_defer_to_the_publisher(leased_renderer, monkeypatch):
    """``lazy_sensor_update=False`` refreshes every renderer; the OVXR renderer's publisher owns those writes."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    _prepare_leased_renderer(renderer)
    transforms, geometries = MagicMock(), MagicMock()
    monkeypatch.setattr(renderer, "_publish_transforms", transforms)
    monkeypatch.setattr(renderer, "_publish_geometries", geometries)
    renderer._sim.render_context.update_scene_state(1)
    lease.release()
    renderer._sim.render_context.update_scene_state(2)
    transforms.assert_not_called()
    geometries.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA material buffers")
@pytest.mark.parametrize("deferred", [False, True])
def test_leased_material_updates_reach_the_reserved_scene(leased_renderer, monkeypatch, deferred):
    """Real material composition and the lease publisher must reach native writes, including before physics changes."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    publisher = lease.capture_scene_publisher()
    context = renderer._sim.render_context
    if deferred:
        cfg = _make_camera_cfg(["rgb"])
        cfg.prim_path = "/World/Camera"
        lease.configure_auxiliary_camera(cfg)
    # Real lifecycle: clone preparation, then environment reset finalizes material consumers,
    # then the XR borrower binds the Stage.
    _prepare_leased_renderer(renderer, bind=False)
    if deferred:
        # Camera initialization requests the Stage on the application thread; the lease defers it.
        renderer.prepare_stage(renderer._sim.stage, 1)
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
    assert renderer._visual_material_writer_ref is not None
    if deferred:
        spec = CameraRenderSpec(cfg, "cpu", 1, (cfg.prim_path,), 1)
        data = renderer.create_render_data(spec)
        monkeypatch.setattr(ovstage.population, "add_usd_reference_from_string", lambda *a: 1)
        monkeypatch.setattr(ovstage.population, "apply_usd_changes", lambda *a, **kw: None)
        token = lease.offer_owner_transfer()
        lease.claim_owner_transfer(token)
        assert lease.materialize_deferred_stage(renderer._sim.stage, 1)
        renderer.update_camera(
            data,
            wp.zeros(1, dtype=wp.vec3f, device="cpu"),
            wp.array([[0.0, 0.0, 0.0, 1.0]], dtype=wp.quatf, device="cpu"),
            None,
        )
    else:
        # The render owner claims the pair first, so the Stage is prepared without deferral.
        lease.claim_owner_transfer(lease.offer_owner_transfer())
        renderer.prepare_stage(renderer._sim.stage, 1)
    stage = renderer.scene.stage
    try:
        for ordinal, color in ((23, [0.0, 1.0, 0.0]), (24, [0.0, 0.0, 1.0])):
            context.write_visual_materials([material], {"color": torch.tensor([color], device="cuda")})
            assert publisher.publication_required("unchanged-physics")
            stage.write_attribute.reset_mock()
            completion = MagicMock()
            stage.write_attribute.return_value = completion
            assert publisher.publish("unchanged-physics", _Ledger(ordinal)) == ordinal
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


def test_public_markers_reach_leased_renderer_without_an_ordinary_visualizer(leased_renderer, monkeypatch):
    """Public markers appear, move without a physics step, hide and retire through the leased publisher."""
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sim import SphereCfg

    renderer = leased_renderer
    registry = renderer._sim.vis_marker_registry
    markers = VisualizationMarkers(
        VisualizationMarkersCfg(prim_path="/Visuals/HandJoints", markers={"joint": SphereCfg(radius=0.005)})
    )
    markers.visualize(translations=[[0.0, 0.0, 1.0]])
    assert not registry.get_groups()  # No renderer or visualizer requested marker state yet.
    observed = []

    class NativeMarkers:
        def __init__(self, _renderer, _cfg):
            pass

        def update(self, translations, orientations, scales, indices, *, visible):
            observed.append((translations.tolist(), visible))

        def close(self):
            observed.append("retired")

    monkeypatch.setattr(ovxr_renderer_module, "_OvxrMarkers", NativeMarkers)
    lease = renderer.acquire_render_lease()
    try:
        publisher = lease.capture_scene_publisher()
        _prepare_leased_renderer(renderer)
        ledger = _Ledger(23)
        markers.visualize(translations=[[0.0, 0.0, 1.0]])
        publisher.publish("unchanged-physics", ledger)
        assert observed[-1] == ([[0.0, 0.0, 1.0]], True)
        assert not publisher.publication_required("unchanged-physics")
        markers.visualize(translations=[[0.5, 0.0, 1.0]])
        assert publisher.publication_required("unchanged-physics")
        publisher.publish("unchanged-physics", ledger)
        assert observed[-1] == ([[0.5, 0.0, 1.0]], True)
        markers.set_visibility(False)
        publisher.publish("unchanged-physics", ledger)
        assert observed[-1] == ([[0.5, 0.0, 1.0]], False)
        del markers
        publisher.publish("unchanged-physics", ledger)
        assert observed[-1] == "retired"
    finally:
        lease.release()
    assert not registry.has_consumers
    unused = VisualizationMarkers(
        VisualizationMarkersCfg(prim_path="/Visuals/NoConsumer", markers={"joint": SphereCfg(radius=0.005)})
    )
    unused.visualize(translations=[[0.0, 0.0, 1.0]])
    assert not registry.get_groups()


def test_publication_runs_debug_marker_callbacks_without_an_ordinary_visualizer(leased_renderer, monkeypatch):
    """Debug-vis callbacks move their markers before each required write, as a visualizer tick would."""
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sim import SphereCfg

    renderer = leased_renderer
    registry = renderer._sim.vis_marker_registry
    assert not renderer._sim.visualizers
    observed = []

    class NativeMarkers:
        def __init__(self, _renderer, _cfg):
            pass

        def update(self, translations, orientations, scales, indices, *, visible):
            observed.append(translations.tolist())

        def close(self):
            pass

    class Goal:
        """A command term's debug visualization: markers follow state the environment step changes."""

        def __init__(self):
            self.position = [0.0, 0.0, 1.0]
            self.calls = 0
            self.markers = VisualizationMarkers(
                VisualizationMarkersCfg(prim_path="/Visuals/Goal", markers={"goal": SphereCfg(radius=0.01)})
            )

        def _debug_vis_callback(self, _event):
            self.calls += 1
            self.markers.visualize(translations=[self.position])

    monkeypatch.setattr(ovxr_renderer_module, "_OvxrMarkers", NativeMarkers)
    goal = Goal()
    registry.add_debug_vis_callback(goal)
    lease = renderer.acquire_render_lease()
    try:
        publisher = lease.capture_scene_publisher()
        _prepare_leased_renderer(renderer)
        ledger = _Ledger(31)
        assert publisher.publish("step-1", ledger) is not None
        assert observed[-1] == [[0.0, 0.0, 1.0]]
        goal.position = [0.25, 0.0, 1.0]
        assert publisher.publish("step-2", ledger) is not None
        assert observed[-1] == [[0.25, 0.0, 1.0]]
        calls = goal.calls
        # A state that needs no write runs no callbacks, so the callbacks cannot force publications.
        assert publisher.publish("step-2", ledger) is None
        assert goal.calls == calls
    finally:
        lease.release()


def test_publication_on_a_cpu_simulation_device_writes_host_memory_without_a_cuda_stream(leased_renderer, monkeypatch):
    """Poses, points and markers of a CPU simulation reach the reserved ordinal without a CUDA stream."""
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sim import SphereCfg

    renderer = leased_renderer
    assert not renderer._warp_device.is_cuda
    pose = np.eye(4)
    pose[3, :3] = (1.0, 2.0, 3.0)  # Row-vector transform: translation in the last row.
    matrices = wp.array([pose], dtype=wp.mat44d, device="cpu")
    points = wp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=wp.vec3f, device="cpu")

    def get_transforms(output, *, allow_passthrough, scales):
        assert allow_passthrough and scales is None
        output.matrices = matrices
        return True

    renderer._sdp = types.SimpleNamespace(
        backend=types.SimpleNamespace(
            transform_paths=["/World/envs/env_0/Body"], transforms_timestamp=1, geometry_timestamp=1
        ),
        get_transforms=get_transforms,
        get_geometry_points=lambda: {"/World/envs/env_0/Cloth": points},
    )
    # Inspect the written arrays instead of the native tensor descriptions.
    for module, name in (
        (ovxr_renderer_module, "xform_tensor_from_warp"),
        (ovxr_renderer_module, "points_tensor_from_warp"),
        (ovxr_scene_module, "xform_tensor_from_warp"),
    ):
        monkeypatch.setattr(module, name, lambda array: array)
    for name in ("add_usd_reference_from_string", "apply_usd_changes", "remove_usd"):
        monkeypatch.setattr(ovstage.population, name, lambda *args, **kwargs: None)
    markers = VisualizationMarkers(
        VisualizationMarkersCfg(prim_path="/Visuals/HandJoints", markers={"joint": SphereCfg(radius=0.005)})
    )
    lease = renderer.acquire_render_lease()
    try:
        publisher = lease.capture_scene_publisher()
        _prepare_leased_renderer(renderer)
        markers.visualize(translations=torch.tensor([[0.5, 0.0, 1.0]], device="cpu"))
        stage = renderer.scene.stage
        stage.write_attribute.reset_mock()
        assert publisher.publish("step", _Ledger(23)) == 23
        writes = stage.write_attribute.call_args_list
        assert writes and all(c.kwargs["ordinal"] == 23 and c.kwargs.get("cuda_stream") is None for c in writes)
        poses = [c.kwargs["tensors"].numpy()[0] for c in writes if c.args[1] == "omni:xform"]
        np.testing.assert_allclose(sorted(tuple(p[3, :3]) for p in poses), [(0.5, 0.0, 1.0), (1.0, 2.0, 3.0)])
        (point_write,) = [c for c in writes if c.args[1] == "points"]
        np.testing.assert_array_equal(point_write.kwargs["tensors"][0].numpy(), points.numpy())
    finally:
        lease.release()
        renderer.close()


def test_leased_camera_cleanup_waits_for_lease_release(leased_renderer, monkeypatch):
    """Releasing a leased camera would remove its product from the borrowed Stage; it waits for release."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    cfg = _make_camera_cfg(["rgb"])
    cfg.prim_path = "/World/Camera"
    lease.configure_auxiliary_camera(cfg)
    _prepare_leased_renderer(renderer, bind=False)
    renderer.prepare_stage(renderer._sim.stage, 1)
    data = renderer.create_render_data(CameraRenderSpec(cfg, "cpu", 1, (cfg.prim_path,), 1))
    population = ovstage.population
    removed = []
    monkeypatch.setattr(population, "add_usd_reference_from_string", lambda *args: 7)
    monkeypatch.setattr(population, "apply_usd_changes", lambda *args, **kwargs: None)
    monkeypatch.setattr(population, "remove_usd", lambda _stage, reference: removed.append(reference))
    lease.claim_owner_transfer(lease.offer_owner_transfer())
    assert lease.materialize_deferred_stage(renderer._sim.stage, 1)

    with pytest.raises(RuntimeError, match="release camera render data while an OvxrRenderLease is active"):
        renderer.cleanup(data)
    assert removed == []
    assert data in renderer._camera_render_data
    assert data.camera_xform_binding is not None

    lease.release()
    renderer.close()
    assert removed == [7]
    assert data not in renderer._camera_render_data


def test_only_the_session_camera_panel_is_accepted_and_its_pose_waits_for_publication(leased_renderer, monkeypatch):
    """The panel's product is registered on the render owner; its latest pose is written with the next scene."""
    renderer = leased_renderer
    monkeypatch.setattr(ovxr_renderer_module, "xform_tensor_from_warp", lambda array: array)
    for name in ("add_usd_reference_from_string", "apply_usd_changes", "remove_usd"):
        monkeypatch.setattr(ovstage.population, name, lambda *args, **kwargs: None)
    cfg = _make_camera_cfg(["rgb"])
    cfg.prim_path = "/World/Camera"
    panel = CameraRenderSpec(cfg, "cpu", 1, (cfg.prim_path,), 1)
    other = CameraRenderSpec(_make_camera_cfg(["rgb"]), "cpu", 1, ("/World/envs/env_0/Camera",), 1)
    lease = renderer.acquire_render_lease()
    try:
        lease.configure_auxiliary_camera(cfg)
        publisher = lease.capture_scene_publisher()
        _prepare_leased_renderer(renderer, bind=False)
        renderer.prepare_stage(renderer._sim.stage, 1)
        with pytest.raises(RuntimeError, match="only the OVXR session's camera panels"):
            renderer.create_render_data(other)
        data = renderer.create_render_data(panel)
        with pytest.raises(RuntimeError, match="render data already exists"):
            renderer.create_render_data(panel)
        # The application thread only reserves the product; the render owner registers it.
        assert data.camera_xform_binding is None and not renderer.backend.render_products
        lease.claim_owner_transfer(lease.offer_owner_transfer())
        assert lease.materialize_deferred_stage(renderer._sim.stage, 1)
        assert lease.auxiliary_render_product_paths == (data.render_product_path,)
        assert renderer.backend.render_products == {data.render_product_path}
        assert data.camera_xform_binding is not None

        stage = renderer.scene.stage
        stage.write_attribute.reset_mock()
        position = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3f, device="cpu")
        renderer.update_camera(data, position, wp.array([[0.0, 0.0, 0.0, 1.0]], dtype=wp.quatf, device="cpu"), None)
        with pytest.raises(RuntimeError, match="no such camera panel"):
            renderer.update_camera(object(), position, None, None)
        stage.write_attribute.assert_not_called()
        assert publisher.publish("step", _Ledger(23)) == 23
        (pose,) = [c for c in stage.write_attribute.call_args_list if c.args[1] == "omni:xform"]
        assert pose.args[0] is data.camera_xform_binding
        assert pose.kwargs["ordinal"] == 23 and pose.kwargs["cuda_stream"] is None
        # Row-vector transform: translation in the last row.
        np.testing.assert_allclose(pose.kwargs["tensors"].numpy()[0][3, :3], (1.0, 2.0, 3.0))
    finally:
        lease.release()
        renderer.close()
    assert renderer.backend.render_products == set()
    assert not renderer._camera_render_data


@pytest.mark.parametrize("cuda_ordinal", [None, 1])
@pytest.mark.parametrize("render_settings", [{}, {"omni:rtx:quality": ("Int", 0)}])
def test_panel_product_is_pinned_to_the_simulation_gpu_or_unpinned_on_a_cpu_device(
    leased_renderer, cuda_ordinal, render_settings
):
    """A CPU simulation device has no GPU to pin, so OVRTX chooses the product's GPU."""
    from pxr import Sdf

    renderer = leased_renderer
    if cuda_ordinal is not None:
        renderer._warp_device = types.SimpleNamespace(is_cuda=True, ordinal=cuda_ordinal)
    cfg = _make_camera_cfg(["rgb"])
    cfg.render_settings = render_settings
    spec = CameraRenderSpec(cfg, "cpu", 1, ("/World/Camera",), 1)
    data = ovxr_scene_module._OvxrPanelRenderData(spec, "RenderCamera_3", "cpu")
    layer = Sdf.Layer.CreateAnonymous()
    assert layer.ImportFromString(renderer._panel_product_usd(spec, data))
    product = layer.GetPrimAtPath(data.render_product_path)
    assert product.relationships["camera"].targetPathList.explicitItems == [Sdf.Path("/World/Camera")]
    device_ids = product.attributes.get("deviceIds")
    if cuda_ordinal is None:
        assert device_ids is None
    else:
        assert list(device_ids.default) == [cuda_ordinal]
    for name, (_, value) in render_settings.items():
        assert product.attributes[name].default == value


@pytest.mark.parametrize("max_fps", [None, 20.0])
def test_panel_population_selects_frame_rates_only_for_rate_limited_panels(leased_renderer, monkeypatch, max_fps):
    """A rate-limited panel adds its rate properties to the population; otherwise the stage holder populates."""
    import isaaclab_ov.stage as stage_module

    renderer = leased_renderer
    population = ovstage.population
    calls = []
    monkeypatch.setattr(population, "open_usd_from_string", lambda _stage, usd, **kw: calls.append(("plain", usd, kw)))
    monkeypatch.setattr(
        population, "open_usd_from_string_with_desc", lambda _stage, usd, **kw: calls.append(("desc", usd, kw))
    )
    monkeypatch.setattr(population, "Desc", lambda **kw: kw)
    monkeypatch.setattr(population, "Selector", lambda *args: args)
    monkeypatch.setattr(population, "PrimPredicate", types.SimpleNamespace(has_path=lambda *paths: paths))
    monkeypatch.setattr(population, "PropertyPredicate", types.SimpleNamespace(has_name=lambda *names: names))
    for module in (ovxr_renderer_module, stage_module):
        monkeypatch.setattr(module, "ovstage_replicate", lambda *args, ordinal: calls.append(("replicate", ordinal)))
    cfg = _make_camera_cfg(["rgb"])
    cfg.prim_path = "/World/Camera"
    usd = '#usda 1.0\ndef Xform "World" {\n def Camera "Camera" {\n }\n}\n'
    lease = renderer.acquire_render_lease()
    try:
        lease.configure_auxiliary_camera(cfg, max_fps=max_fps)
        lease.create_native_pair()
        renderer._panels.cameras[0].spec = CameraRenderSpec(cfg, "cpu", 1, (cfg.prim_path,), 1)
        renderer._panels.populate(usd)
        scene = renderer.scene
        domains = scene.cfg.population_domains
        kind, populated, kwargs = calls[0]
        if max_fps is None:
            assert (kind, populated, kwargs) == ("plain", usd, {"ordinal": 1, "domains": domains})
        else:
            assert kind == "desc" and "omni:sensor:frameRate" in populated
            assert kwargs["ordinal"] == 1 and kwargs["time"] is population.DEFAULT_TIME
            rates = ((cfg.prim_path,), ("omni:sensor:frameRate", "omni:sensor:Core:partialOutputs"))
            assert kwargs["descs"] == ({"domains": domains}, {"domains": domains, "selectors": [rates]})
        assert calls[1:] == [("replicate", 1)]
        # Population and cloning are sealed at the first ordinal, as the stage holder seals them.
        scene.stage.advance_write_floor.assert_called_once_with(ordinal=1)
        assert scene.ordinal == 2
    finally:
        lease.release()


def test_external_camera_matches_the_actual_custom_clone_root(leased_renderer):
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    renderer.scene = types.SimpleNamespace(clone_env_paths=["/Lab/Cell0"])
    cfg = types.SimpleNamespace(prim_path="{ENV_REGEX_NS}/Robot/Head", width=640, height=480, data_types=("rgb",))
    lease.configure_auxiliary_camera(cfg)
    spec = types.SimpleNamespace(
        cfg=cfg, num_instances=1, view_count=1, camera_prim_paths=("/Lab/Cell0/Robot/Head/camera",)
    )
    assert renderer._panels.accept(spec)
    assert renderer._panels.cameras[0].spec is spec
    lease.release()


def test_only_a_matched_multi_view_camera_is_rejected_as_external(leased_renderer):
    """An unrelated tiled camera takes the ordinary path; a selected camera must have one view."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    renderer.scene = types.SimpleNamespace(clone_env_paths=["/World/envs/env_0"])
    cfg = types.SimpleNamespace(prim_path="{ENV_REGEX_NS}/Robot/Head", width=640, height=480, data_types=("rgb",))
    lease.configure_auxiliary_camera(cfg)

    def tiled(path):
        return types.SimpleNamespace(cfg=cfg, num_instances=2, view_count=2, camera_prim_paths=(path, path))

    try:
        assert renderer._panels.accept(tiled("/World/envs/env_0/Other")) is None
        with pytest.raises(RuntimeError, match="exactly one camera view"):
            renderer._panels.accept(tiled("/World/envs/env_0/Robot/Head"))
    finally:
        lease.release()


def test_selected_external_camera_order_is_independent_of_sensor_preparation(leased_renderer):
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    specs = []
    for name, hz in zip(("Head", "Left", "Right", "Overview"), (20, 10, 10, 5), strict=True):
        cfg = types.SimpleNamespace(prim_path=f"/World/{name}", width=640, height=480, data_types=("rgb",))
        lease.configure_auxiliary_camera(cfg, max_fps=hz)
        specs.append(types.SimpleNamespace(cfg=cfg, num_instances=1, view_count=1, camera_prim_paths=(cfg.prim_path,)))
    for spec in reversed(specs):
        assert renderer._panels.accept(spec)
    assert [camera.spec for camera in renderer._panels.cameras] == specs
    assert [camera.max_fps for camera in renderer._panels.cameras] == [20, 10, 10, 5]
    lease.release()


@pytest.mark.parametrize("max_fps", [None, 20.0])
def test_external_camera_schedules_preserve_export_composition(leased_renderer, tmp_path, max_fps):
    """Rates are authored as layer opinions; preserved references are not flattened into the export."""
    from pxr import Sdf, Usd

    asset = tmp_path / "robot.usda"
    asset.write_text('#usda 1.0\n(defaultPrim = "Robot")\ndef Xform "Robot" {\n def Camera "Head" {\n }\n}\n')
    usd = f'#usda 1.0\ndef Xform "World" {{\n def "Robot" (prepend references = @{asset}@) {{\n }}\n}}\n'
    camera = types.SimpleNamespace(max_fps=max_fps, spec=types.SimpleNamespace(camera_prim_paths=["/World/Robot/Head"]))
    leased_renderer._panels.cameras = [camera]
    authored = leased_renderer._panels._author_schedules(usd)
    if max_fps is None:
        assert authored is usd
        return
    layer = Sdf.Layer.CreateAnonymous()
    layer.ImportFromString(authored)
    assert layer.GetPrimAtPath("/World/Robot").referenceList.prependedItems[0].assetPath == str(asset)
    stage = Usd.Stage.Open(layer)
    head = stage.GetPrimAtPath("/World/Robot/Head")
    assert head.GetAttribute("omni:sensor:frameRate").Get() == (20.0, 1.0)
    assert head.GetAttribute("omni:sensor:Core:partialOutputs").Get() is False
    camera.spec.camera_prim_paths = ["/World/Robot/Missing"]
    with pytest.raises(RuntimeError, match="absent from the exported Stage"):
        leased_renderer._panels._author_schedules(usd)


def test_external_camera_schedules_reach_native_stage():
    """Leased population authors the selected cameras' native rates and retains only those properties."""
    import ovrtx
    import ovstage
    from isaaclab_ov.stage import OvstageBackend, OvstageBackendCfg

    ovrtx.register_schema_paths()
    camera_paths = [f"/World/envs/env_0/{name}" for name in ("Head", "Wrist", "Unselected")]
    # The application export has no camera schedules; the unselected camera's own rate must not populate.
    source = """#usda 1.0
def Xform "World" {
 def Xform "envs" {
  def Xform "env_0" {
   def Camera "Head" {
   }
   def Camera "Wrist" {
   }
   def Camera "Unselected" {
    double2 omni:sensor:frameRate = (999, 1)
   }
  }
 }
}
"""
    renderer = OvxrRenderer(OvxrRendererCfg(), sim=SimulationContext.instance())
    lease = renderer.acquire_render_lease()
    with contextlib.ExitStack() as resources:
        resources.callback(lease.release)
        for path, max_fps in zip(camera_paths[:2], (20.0, 10.0), strict=True):
            cfg = _make_camera_cfg(["rgb"])
            cfg.prim_path = path
            lease.configure_auxiliary_camera(cfg, max_fps=max_fps)
        for camera in renderer._panels.cameras:
            camera.spec = types.SimpleNamespace(camera_prim_paths=[camera.cfg.prim_path])
        renderer.scene = OvstageBackend(OvstageBackendCfg())
        resources.callback(renderer.scene.close)
        renderer._panels.populate(source)

        stage, paths = renderer.scene.stage, renderer.scene.paths
        names = ("omni:sensor:frameRate", "omni:sensor:Core:partialOutputs")
        resolved = {}
        plist = paths.create_path_list_from_strings(camera_paths)
        try:
            with stage.query_from_path_list(plist) as query:
                query.wait()
                with stage.read_attributes(
                    query,
                    [paths.intern_token(name) for name in names],
                    ovstage.OrdinalRange.latest(renderer.scene.ordinal - 1),
                ) as read:
                    read.wait()
                    for group in read.groups():
                        try:
                            group_paths = paths.get_path_strings(group.prim_list)
                            for column in group.columns:
                                values = np.asarray(column.array(0).copy()).reshape(column.data_count, -1)
                                attribute = paths.token_to_string(column.attribute)
                                for index in range(min(group.prim_count, column.data_count)):
                                    path = group_paths[group.prim_index(index)]
                                    resolved[path, attribute] = values[column.data_row_index(index)].tolist()
                        finally:
                            stage.release_group(group)
        finally:
            paths.destroy_path_list(plist)
    assert resolved.get((camera_paths[0], names[0])) == [20.0, 1.0]
    assert resolved.get((camera_paths[1], names[0])) == [10.0, 1.0]
    assert resolved.get((camera_paths[0], names[1])) == [False]
    assert resolved.get((camera_paths[1], names[1])) == [False]
    assert (camera_paths[2], names[0]) not in resolved


# Split-GPU rendering


def _engine_cfg(active_cuda_gpus=None, device="cpu"):
    return ovxr_scene_module._OvxrEngineCfg(
        scene_key=OvxrRendererCfg(),
        log_file_path="ovxr.log",
        log_level="error",
        device=device,
        active_cuda_gpus=active_cuda_gpus,
    )


def _register_ordinary_ovrtx(kind):
    """Register an ordinary OVRTX camera renderer, or the native engine its clone preparation creates."""
    from isaaclab_ov.renderers import OVRTXBackendCfg, OVRTXRendererCfg

    sim = SimulationContext.instance()
    if kind == "renderer":
        sim.get_or_create_backend(OVRTXRendererCfg(log_file_path="ordinary.log"))
    else:
        cfg = OVRTXBackendCfg(use_ovstage=False, read_gpu_transforms=True, device=sim.device)
        sim._backend_registry.append((cfg, object()))


def test_lease_applies_the_rendering_gpu_selection_to_the_engine_it_creates(leased_renderer, monkeypatch):
    """The session passes its selection to the engine it creates, which hands it to native OVRTX."""
    monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 2)
    configs = []
    monkeypatch.setattr(ovxr_scene_module, "RendererConfig", lambda **kwargs: configs.append(kwargs) or kwargs)
    renderer = SimulationContext.instance().get_or_create_backend(OvxrRendererCfg(active_cuda_gpus="1"))
    lease = renderer.acquire_render_lease()
    try:
        lease.create_native_pair()
        assert renderer.backend.cfg.active_cuda_gpus == "1"
        (native_cfg,) = configs
        assert native_cfg["active_cuda_gpus"] == "1"
        assert ovxr_scene_module._initial_gpu_selection == (1,)
    finally:
        lease.release()


@pytest.mark.parametrize(
    ("device", "active_cuda_gpus", "requested"), [("cpu", None, None), ("cpu", "1", "1"), ("cuda:0", None, "0")]
)
def test_engine_renders_on_the_simulation_gpu_unless_a_selection_is_given(
    monkeypatch, device, active_cuda_gpus, requested
):
    """Without a selection the engine renders on the CUDA simulation GPU; a CPU simulation uses native selection."""
    monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 2)
    monkeypatch.setattr(
        ovxr_scene_module.wp, "get_device", lambda name: types.SimpleNamespace(is_cuda=name != "cpu", ordinal=0)
    )
    configs = []
    monkeypatch.setattr(ovxr_scene_module, "RendererConfig", lambda **kwargs: configs.append(kwargs) or kwargs)
    monkeypatch.setattr(ovxr_scene_module, "redirect_shader_cache", lambda _config: None)
    monkeypatch.setattr(ovxr_scene_module, "Renderer", lambda _config: types.SimpleNamespace(destroy=lambda: None))
    ovxr_scene_module._OvxrEngine(_engine_cfg(active_cuda_gpus, device)).close()
    assert configs[0]["active_cuda_gpus"] == requested


def test_unavailable_rendering_gpu_is_rejected_before_the_selection_is_reserved(monkeypatch):
    """A mistyped GPU index must not fix the process to a selection the native renderer cannot use."""
    monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 2)
    monkeypatch.setattr(ovxr_scene_module, "redirect_shader_cache", lambda _config: pytest.fail("redirected"))
    monkeypatch.setattr(ovxr_scene_module, "Renderer", lambda _config: pytest.fail("constructed a renderer"))
    with pytest.raises(ValueError, match="cannot see"):
        ovxr_scene_module._OvxrEngine(_engine_cfg("7"))
    assert ovxr_scene_module._initial_gpu_selection is None


@pytest.mark.parametrize(("initial_selection", "failure_point"), [(None, None), ("0", "redirect"), ("0", "renderer")])
def test_native_gpu_selection_survives_close_and_partial_startup(monkeypatch, initial_selection, failure_point):
    """Default selection is repeatable; partial native startup cannot authorize a different GPU."""
    monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 2)
    calls = []

    def redirect(config):
        calls.append("redirect")
        if failure_point == "redirect":
            raise RuntimeError("native startup failed")

    def construct(config):
        calls.append("renderer")
        if failure_point == "renderer":
            raise RuntimeError("native startup failed")
        return types.SimpleNamespace(destroy=lambda: None)

    monkeypatch.setattr(ovxr_scene_module, "redirect_shader_cache", redirect)
    monkeypatch.setattr(ovxr_scene_module, "Renderer", construct)
    cfg = _engine_cfg(initial_selection)
    if failure_point is None:
        ovxr_scene_module._OvxrEngine(cfg).close()
    else:
        with pytest.raises(RuntimeError, match="native startup failed"):
            ovxr_scene_module._OvxrEngine(cfg)
    before = calls.copy()
    failure_point = None
    with pytest.raises(ValueError, match="new process"):
        ovxr_scene_module._OvxrEngine(cfg.replace(active_cuda_gpus="1"))
    assert calls == before, "Reject conflicting selections before any native initialization."
    ovxr_scene_module._OvxrEngine(cfg).close()
    assert calls[len(before) :] == ["redirect", "renderer"]


def test_unset_selection_conflict_names_the_simulation_device(monkeypatch):
    """An unset selection resolves from the simulation device; the rejection says so before native startup."""
    monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 2)
    monkeypatch.setattr(ovxr_scene_module, "redirect_shader_cache", lambda _config: None)
    monkeypatch.setattr(ovxr_scene_module, "Renderer", lambda _config: types.SimpleNamespace(destroy=lambda: None))
    ovxr_scene_module._OvxrEngine(_engine_cfg("1")).close()
    monkeypatch.setattr(ovxr_scene_module, "Renderer", lambda _config: pytest.fail("constructed a renderer"))
    with pytest.raises(ValueError, match="simulation device 'cpu', since active_cuda_gpus is unset") as raised:
        ovxr_scene_module._OvxrEngine(_engine_cfg(None))
    assert "already uses (1,)" in str(raised.value)
    assert "must resolve to the same GPUs" in str(raised.value)


@pytest.mark.parametrize(
    ("device", "active_cuda_gpus", "rejected"),
    [
        ("cuda:0", None, False),
        ("cuda:0", "0", False),
        ("cuda:0", "0,1", True),
        ("cuda:0", "1", True),
        ("cuda:0", "1,2", True),
        ("cpu", None, False),
        ("cpu", "0", True),
    ],
)
@pytest.mark.parametrize("kind", ["renderer", "engine"])
def test_only_a_selection_other_than_the_simulation_gpu_rejects_ordinary_ovrtx_cameras(
    device, active_cuda_gpus, rejected, kind
):
    """Ordinary OVRTX renderers render on the simulation GPU; only another selection cannot share their simulation."""
    renderer = OvxrRenderer(OvxrRendererCfg(active_cuda_gpus=active_cuda_gpus), sim=SimulationContext.instance())
    renderer._device = device
    # Warp reports ordinal -1 for the CPU.
    renderer._warp_device = types.SimpleNamespace(is_cuda=device != "cpu", ordinal=0 if device != "cpu" else -1)
    renderer._reject_ordinary_ovrtx_renderers()
    _register_ordinary_ovrtx(kind)
    if rejected:
        with pytest.raises(ValueError, match="ordinary OVRTX camera cannot share a simulation"):
            renderer._reject_ordinary_ovrtx_renderers()
    else:
        renderer._reject_ordinary_ovrtx_renderers()


@pytest.mark.parametrize("kind", ["renderer", "engine"])
@pytest.mark.parametrize("check", ["engine creation", "deferred preparation", "claimed preparation"])
def test_split_gpu_session_rejects_an_ordinary_ovrtx_camera_in_its_simulation(
    leased_renderer, monkeypatch, kind, check
):
    """The conflict is checked when the session creates its engine and again when the Stage is prepared."""
    monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 2)
    # On the CPU simulation device, ordinary OVRTX renderers use native GPU selection, not GPU 0.
    renderer = SimulationContext.instance().get_or_create_backend(OvxrRendererCfg(active_cuda_gpus="0"))
    lease = renderer.acquire_render_lease()
    try:
        if check == "engine creation":
            _register_ordinary_ovrtx(kind)
            with pytest.raises(ValueError, match="ordinary OVRTX camera cannot share a simulation"):
                lease.create_native_pair()
            assert renderer.backend is renderer.scene is None
            assert ovxr_scene_module._initial_gpu_selection is None
            return
        lease.create_native_pair()
        _register_ordinary_ovrtx(kind)
        if check == "claimed preparation":
            lease.claim_owner_transfer(lease.offer_owner_transfer())
        with pytest.raises(ValueError, match="ordinary OVRTX camera cannot share a simulation"):
            renderer.prepare_stage(renderer._sim.stage, 1)
        assert not renderer._render_lease.stage_deferred and not renderer._initialized_scene
        assert renderer._geometry_publication is None
    finally:
        lease.release()


@pytest.mark.parametrize(
    ("simulation", "active_cuda_gpus", "requested"),
    [("cpu", None, None), ("cuda:2", None, 2), ("cuda:2", "0,2", 2), ("cuda:2", "0", None), ("cuda:2", "0,1", None)],
)
def test_panel_products_request_the_simulation_gpu_only_when_it_renders(simulation, active_cuda_gpus, requested):
    """OVRTX skips a product request outside the rendering set, so such products leave GPU choice to OVRTX."""
    from pxr import Sdf

    renderer = OvxrRenderer(OvxrRendererCfg(active_cuda_gpus=active_cuda_gpus), sim=SimulationContext.instance())
    renderer._device = simulation
    renderer._warp_device = types.SimpleNamespace(is_cuda=simulation != "cpu", ordinal=2 if simulation != "cpu" else -1)
    spec = CameraRenderSpec(_make_camera_cfg(["rgb"]), "cpu", 1, ("/World/Camera",), 1)
    data = ovxr_scene_module._OvxrPanelRenderData(spec, "RenderCamera_0", "cpu")
    layer = Sdf.Layer.CreateAnonymous()
    assert layer.ImportFromString(renderer._panel_product_usd(spec, data))
    device_ids = layer.GetPrimAtPath(data.render_product_path).attributes.get("deviceIds")
    if requested is None:
        assert device_ids is None
    else:
        assert list(device_ids.default) == [requested]


def test_geometry_publication_does_not_wait_on_the_host_before_stage_consumes_it(monkeypatch):
    """A batch without foreign sources does not block on the transfer stream of an earlier copy."""
    devices = test_devices(DeviceScope.CUDA)
    if not devices:
        pytest.skip("Geometry publication requires a CUDA device")
    device = devices[0]
    publisher = _GeometryPublicationBuffers(device)
    # The transfer stream exists once a foreign batch has been copied.
    publisher._stream = wp.Stream(device)
    syncs = []
    monkeypatch.setattr(wp, "synchronize_stream", syncs.append)
    writes = []
    stage = types.SimpleNamespace(
        write_attribute=lambda *_args, **kwargs: writes.append(kwargs) or types.SimpleNamespace(wait=lambda: None)
    )
    publisher.write(stage, object(), ["/World/Mesh"], [wp.zeros(3, dtype=wp.vec3, device=device)], ordinal=1)
    assert syncs == []
    # Stage still orders its copy-in after same-device producer work.
    assert writes[0]["cuda_stream"] == publisher._stream.cuda_stream


@pytest.mark.parametrize("family", ["transforms", "points"])
@pytest.mark.parametrize("producer_named", [True, False])
def test_split_gpu_publication_orders_the_transfer_after_the_producing_stream(monkeypatch, family, producer_named):
    """The renderer's scene writes go through the publication buffers, which wait on the producing stream.

    Streams, arrays and the Stage are doubles, so the recorded order does not depend on the host's GPUs:
    producer, copy, rendering device, then the Stage write on the transfer stream and its completion wait.
    The renderer names the simulation device's stream at the write; a caller that names none falls back
    to the source device's current stream.
    """
    events = []

    class Stream:
        def __init__(self, name):
            self.name = self.cuda_stream = name

        def wait_stream(self, other):
            events.append((f"{self.name} waits for", other.name))

    producer, current, transfer, rendering = map(Stream, ("producer", "current", "transfer", "rendering"))
    # The source array's device reports another current stream than the one that produced the data.
    simulation_gpu = types.SimpleNamespace(is_cuda=True, stream=current)
    rendering_gpu = types.SimpleNamespace(is_cuda=True, stream=rendering)
    source = types.SimpleNamespace(device=simulation_gpu, shape=(1,), dtype=wp.mat44d)
    staged = types.SimpleNamespace(device=rendering_gpu, shape=(1,), dtype=wp.mat44d)
    publisher = _GeometryPublicationBuffers.__new__(_GeometryPublicationBuffers)
    publisher._device = rendering_gpu
    publisher._stream = transfer
    publisher._buffers = {"@transforms": staged, "/World/Mesh": staged}
    publisher._sources, publisher._source_streams, publisher._failed = [], [], False

    def copy(target, origin, *, stream):
        assert target is staged and origin is source
        events.append(("copy on", stream.name))

    def write_attribute(_binding, attribute, *, tensors, cuda_stream, **_kwargs):
        assert attribute == ("points" if family == "points" else "omni:xform")
        assert (tensors[0] if family == "points" else tensors) is staged
        events.append(("stage reads on", cuda_stream))
        return types.SimpleNamespace(wait=lambda: events.append("stage wait"))

    monkeypatch.setattr(wp, "copy", copy)
    for convert in ("points_tensor_from_warp", "xform_tensor_from_warp"):
        monkeypatch.setattr(ovxr_gpu_publication, convert, lambda array: array)
    stage = types.SimpleNamespace(write_attribute=write_attribute)
    if producer_named:
        renderer = OvxrRenderer(OvxrRendererCfg(active_cuda_gpus="1"), sim=SimulationContext.instance())
        renderer._device = "cuda:0"
        renderer._warp_device = types.SimpleNamespace(is_cuda=True, ordinal=0, stream=producer)
        renderer._geometry_publication = publisher
        renderer.scene = types.SimpleNamespace(stage=stage, ordinal=3)
        # The Warp struct accepts only Warp arrays; the source here is a double.
        renderer._transforms = types.SimpleNamespace(matrices=None)

        def get_transforms(output, **_kwargs):
            output.matrices = source
            return True

        renderer._sdp = types.SimpleNamespace(
            backend=types.SimpleNamespace(transforms_timestamp=1, geometry_timestamp=1),
            get_transforms=get_transforms,
            get_geometry_points=lambda: {"/World/Mesh": source},
        )
        if family == "transforms":
            renderer._object_xform_binding = object()
            renderer._publish_transforms()
            assert renderer._transforms_timestamp == 1
        else:
            renderer._geometry_points_binding = object()
            renderer._geometry_paths = ["/World/Mesh"]
            renderer._publish_geometries()
            assert renderer._geometry_timestamp == 1
    elif family == "transforms":
        publisher.write_transforms(stage, object(), source, 3)
    else:
        publisher.write(stage, object(), ["/World/Mesh"], [source], 3)
    assert events == [
        ("transfer waits for", "producer" if producer_named else "current"),
        ("copy on", "transfer"),
        ("transfer waits for", "rendering"),
        ("stage reads on", "transfer"),
        "stage wait",
    ]


def _complete_handed_stream(publisher, synchronize=None):
    """Side effect for a Stage double: finish the work on the stream it is handed, as the native copy-in does.

    The transfer stream is non-blocking and ``numpy()`` does not wait for it, so a staged array is only
    safe to read once the Stage's write has completed the transfer.
    """
    synchronize = synchronize or wp.synchronize_stream

    def write_attribute(*_args, cuda_stream=None, **_kwargs):
        assert cuda_stream == publisher._stream.cuda_stream
        synchronize(publisher._stream)
        return DEFAULT

    return write_attribute


def test_geometry_publication_cuda_devices_resize_and_failed_consumption(monkeypatch):
    """Only foreign CUDA points move; replacement sources and failed consumption cannot corrupt a snapshot."""
    with pytest.raises(ValueError, match="explicit CUDA device"):
        _GeometryPublicationBuffers("cuda")
    devices = test_devices(DeviceScope.CUDA)
    if len(devices) < 2:
        pytest.skip("Cross-device geometry publication requires two selected CUDA devices")
    source_device, target_device = devices[0], devices[-1]
    publisher = _GeometryPublicationBuffers(target_device)
    inputs = [
        wp.array([[4, 5, 6]], dtype=wp.vec3, device=target_device),
        wp.array([[7, 8, 9], [10, 11, 12]], dtype=wp.vec3, device=source_device),
    ]
    observed = []
    wrap_tensor = ovxr_gpu_publication.points_tensor_from_warp
    empty_like = wp.empty_like
    allocation_streams = []

    def allocate(source, *, device):
        # Stream-ordered allocations must precede their first copy on the consuming stream.
        allocation_streams.append(wp.get_stream(device).cuda_stream)
        assert allocation_streams[-1] == publisher._stream.cuda_stream
        return empty_like(source, device=device)

    monkeypatch.setattr(wp, "empty_like", allocate)

    def observe(array):
        observed.append(array)
        return wrap_tensor(array)

    monkeypatch.setattr(ovxr_gpu_publication, "points_tensor_from_warp", observe)
    stage = MagicMock()
    paths = ["/Local", "/Foreign"]
    syncs = []
    synchronize = wp.synchronize_stream
    monkeypatch.setattr(wp, "synchronize_stream", lambda stream: syncs.append(stream) or synchronize(stream))
    stage.write_attribute.side_effect = _complete_handed_stream(publisher, synchronize)
    try:
        publisher.write(stage, object(), paths, inputs, 1)
        # Stage reads after the copy on the transfer stream; no host wait on that stream precedes it.
        assert publisher._stream not in syncs
        assert observed[0] is inputs[0]
        assert str(observed[1].device) == target_device
        np.testing.assert_array_equal(observed[1].numpy(), [[7, 8, 9], [10, 11, 12]])
        observed.clear()
        # The provider may replace and resize its point storage while preserving the prim path.
        with wp.ScopedStream(wp.Stream(source_device)):
            inputs[1] = wp.array([[13, 14, 15]], dtype=wp.vec3, device=source_device)
            publisher.write(stage, object(), paths, inputs, 2)
            inputs[1].fill_(wp.vec3(-100))
        snapshot = observed[1]
        assert len(allocation_streams) == 2  # First use and replacement after resize.
        np.testing.assert_array_equal(snapshot.numpy(), [[13, 14, 15]])
        observed.clear()
        operation = stage.write_attribute.return_value
        operation.wait.side_effect = RuntimeError("Stage rejected the write")
        with pytest.raises(RuntimeError, match="Stage rejected"):
            publisher.write(stage, object(), paths, inputs, 3)
        waits = operation.wait.call_count
        with pytest.raises(RuntimeError, match="close and recreate"):
            publisher.write(stage, object(), paths, inputs, 4)
        publisher.close()
        assert operation.wait.call_count == waits  # Native wait already consumed the failed operation.
    finally:
        publisher.close()


@pytest.mark.parametrize(
    ("cuda_ordinal", "active_cuda_gpus", "expected"),
    [
        (0, None, None),
        (0, "0,1", None),
        (None, "0", None),
        (1, "0", "cuda:0"),
        (2, "1,0", "cuda:1"),
    ],
)
def test_geometry_is_staged_on_the_first_rendering_gpu_when_rendering_excludes_the_simulation_gpu(
    monkeypatch, cuda_ordinal, active_cuda_gpus, expected
):
    """Only rendering away from the CUDA simulation GPU stages geometry, before population; others write in place."""

    class Reached(Exception):
        pass

    renderer = OvxrRenderer(OvxrRendererCfg(active_cuda_gpus=active_cuda_gpus), sim=SimulationContext.instance())
    renderer._device = "cpu" if cuda_ordinal is None else f"cuda:{cuda_ordinal}"
    # Warp reports ordinal -1 for the CPU.
    renderer._warp_device = types.SimpleNamespace(
        is_cuda=cuda_ordinal is not None, ordinal=-1 if cuda_ordinal is None else cuda_ordinal
    )
    assert renderer._geometry_publication_device() == expected
    if expected is None:
        return

    def reach(device):
        assert device == expected
        raise Reached

    # Stand in for a claimed lease; Stage preparation creates the buffers before it populates the Stage.
    monkeypatch.setattr(renderer._render_lease, "prepare_stage", lambda num_envs: True)
    monkeypatch.setattr(renderer, "_populate_scene", lambda stage: pytest.fail("populated before staging"))
    monkeypatch.setattr(ovxr_renderer_module, "_GeometryPublicationBuffers", reach)
    with pytest.raises(Reached):
        renderer.prepare_stage(renderer._sim.stage, 1)
    assert renderer._geometry_publication is None


def test_rigid_publication_uses_target_device_and_retains_failed_producer(monkeypatch):
    """Rigid poses follow the selected GPU; failed drains keep the exact producer and scene alive."""
    devices = test_devices(DeviceScope.CUDA)
    if len(devices) < 2:
        pytest.skip("Cross-device rigid publication requires two selected CUDA devices")
    source_device, target_device = devices[0], devices[-1]
    renderer = OvxrRenderer(
        OvxrRendererCfg(active_cuda_gpus=target_device.removeprefix("cuda:")), sim=SimulationContext.instance()
    )
    renderer._warp_device = wp.get_device(source_device)
    renderer._device = source_device
    renderer._object_xform_binding = MagicMock()
    publisher = renderer._geometry_publication = _GeometryPublicationBuffers(target_device)
    expected = np.array([[[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [2, 3, 4, 1]]], dtype=np.float64)
    producer = wp.Stream(source_device)
    with wp.ScopedStream(producer):
        matrices = wp.array(expected, dtype=wp.mat44d, device=source_device)

    def get_transforms(output, **_kwargs):
        output.matrices = matrices
        return True

    renderer._sdp = types.SimpleNamespace(
        backend=types.SimpleNamespace(transforms_timestamp=1), get_transforms=get_transforms
    )
    stage = MagicMock()
    renderer.scene = types.SimpleNamespace(stage=stage, ordinal=11)
    stage.write_attribute.side_effect = _complete_handed_stream(publisher)
    observed = []
    convert = ovxr_gpu_publication.xform_tensor_from_warp

    def observe(array):
        observed.append(array)
        return convert(array)

    monkeypatch.setattr(ovxr_gpu_publication, "xform_tensor_from_warp", observe)
    try:
        with wp.ScopedStream(producer):
            renderer._publish_transforms()
        assert str(observed[-1].device) == target_device
        np.testing.assert_array_equal(observed[-1].numpy(), expected)
        assert stage.write_attribute.call_args.args[1] == "omni:xform"
        assert stage.write_attribute.call_args.kwargs["is_array"] is False
        assert stage.write_attribute.call_args.kwargs["ordinal"] == 11
        # Reset/changed state must reuse the target allocation while publishing fresh values.
        snapshot = observed[-1]
        renderer._sdp.backend.transforms_timestamp = 2
        with wp.ScopedStream(producer):
            matrices.fill_(wp.mat44d(0.0))
            renderer._publish_transforms()
        assert observed[-1] is snapshot
        np.testing.assert_array_equal(snapshot.numpy(), np.zeros_like(expected))
        renderer._sdp.backend.transforms_timestamp = 3
        stage.write_attribute.return_value.wait.side_effect = RuntimeError("Rejected consumption")
        with wp.ScopedStream(producer), pytest.raises(RuntimeError, match="Rejected consumption"):
            renderer._publish_transforms()
        # The device's current stream has changed. Drain the captured producer, not its replacement.
        original_sync = wp.synchronize_stream
        drained = []

        def failed_drain(stream):
            drained.append(stream)
            raise RuntimeError("Drain failed")

        with monkeypatch.context() as patch:
            patch.setattr(wp, "synchronize_stream", failed_drain)
            with pytest.raises(RuntimeError, match="Drain failed"):
                renderer.close()
        assert drained == [producer]
        assert renderer._geometry_publication is publisher
        assert publisher._sources == [matrices]
        assert renderer._object_xform_binding is not None
        original_sync(producer)
    finally:
        renderer.close()
    assert renderer._geometry_publication is None and renderer._object_xform_binding is None


@pytest.mark.parametrize(
    ("value", "message"), [("cuda:0", "must list CUDA indices"), ("0,,1", "must list"), ("0,0", "more than once")]
)
def test_renderer_cfg_rejects_malformed_gpu_selection(value, message):
    with pytest.raises(ValueError, match=message):
        OvxrRendererCfg(active_cuda_gpus=value)
