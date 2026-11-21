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
from unittest.mock import MagicMock

import numpy as np
import ovstage
import pytest
import torch
import warp as wp

from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.renderers import BaseRenderer
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import PinholeCameraCfg, SimulationContext

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
    import isaaclab_teleop.ovxr_renderer as ovxr_renderer_module  # noqa: E402
    import isaaclab_teleop.ovxr_scene as ovxr_scene_module  # noqa: E402
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
    assert cfg.supported_output_types() == {}


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

    # Camera sensors are rejected, and nothing produces camera images.
    with pytest.raises(RuntimeError, match="renders no camera sensors"):
        renderer.create_render_data(spec)
    with pytest.raises(RuntimeError, match="no camera to update"):
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


def test_render_lease_borrows_registry_pair(leased_renderer):
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
    names = ["material", "transform", "geometry", "material-drain"]
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


@pytest.mark.parametrize("failure", ["material", "transform", "geometry", "material-drain"])
def test_render_lease_complete_scene_failure_preserves_commit_boundary(complete_scene_publication, failure):
    state = complete_scene_publication
    renderer, lease = state.renderer, state.lease
    state.failure = failure
    with pytest.raises(RuntimeError, match="Producer scene publication failed") as caught:
        state.publish()
    assert caught.value.__cause__ is state.cause
    order = ["material", "transform", "geometry", "material-drain"]
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
def test_leased_material_updates_reach_the_reserved_scene(leased_renderer, monkeypatch):
    """Real material composition and the lease publisher must reach native writes, including before physics changes."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    publisher = lease.capture_scene_publisher()
    context = renderer._sim.render_context
    # Real lifecycle: clone preparation, then environment reset finalizes material consumers,
    # then the XR borrower binds the Stage.
    _prepare_leased_renderer(renderer, bind=False)
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
