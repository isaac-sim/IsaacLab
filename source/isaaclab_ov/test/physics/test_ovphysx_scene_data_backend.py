# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OvPhysxSceneDataBackend (new SceneDataBackend interface, post-#5128) and OvPhysxManager."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from types import ModuleType, SimpleNamespace

import pytest

# The OVPhysX runtime wheel is optional. Skip gracefully when it is not installed;
# CI jobs that need OVPhysX coverage install it explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

_LEGACY_LIFECYCLE_ENTRY_POINTS = {"warmup": "warmup_gpu", "destroy": "release"}
_CURRENT_LIFECYCLE_ENTRY_POINTS = {"warmup": "warmup", "destroy": "destroy"}


@pytest.fixture(autouse=True)
def manager(monkeypatch):
    from isaaclab_ov.physics import OvPhysxCfg, ovphysx_compat
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxBackend, OvPhysxManager, OvPhysxSceneDataBackend

    # These fixtures model the legacy API, regardless of the locally installed wheel.
    monkeypatch.setattr(ovphysx_compat, "OVPHYSX_USES_TASKS", False)
    backend = OvPhysxBackend.__new__(OvPhysxBackend)
    backend.physx = None
    backend.stage = backend.scene = None
    backend.owns_scene = True
    backend.rigid_body_view = None
    manager = OvPhysxManager(OvPhysxCfg())
    manager.backend = backend
    manager._scene_data_backend = OvPhysxSceneDataBackend(manager)
    return manager


@pytest.fixture(autouse=True)
def _close_test_views():
    from isaaclab_ov.sim.views import OvPhysxView

    existing_views = set(OvPhysxView._live_views)
    yield
    for view in OvPhysxView._live_views - existing_views:
        view.close()


@pytest.fixture(scope="module", autouse=True)
def _register_ovphysx_schemas_before_test_stages():
    """Register OvPhysX schemas before this module creates any USD stage."""
    from isaaclab_ov.physics import OvPhysxManager

    OvPhysxManager._prepare_stage_creation()


def _make_two_environment_stage():
    """Create an in-memory USD stage with one cube in each of two environments."""
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/envs/env_0/Cube", "Cube")
    stage.DefinePrim("/World/envs/env_1/Cube", "Cube")
    return stage


def test_manager_forced_rewarm_invalidates_bindings_before_loading(manager, monkeypatch):
    """A forced re-warm invalidates views before replacing their attached stage."""

    from isaaclab.physics import PhysicsEvent

    calls = []
    monkeypatch.setattr(manager, "_warmup_done", False)
    manager.backend.stage = object()
    monkeypatch.setattr(manager, "_warmup_and_load", lambda: calls.append("warmup"))
    monkeypatch.setattr(manager, "dispatch_event", lambda event, payload=None: calls.append(event))

    version = manager._scene_data_backend.transforms_timestamp
    manager.reset()

    assert calls == [PhysicsEvent.STOP, "warmup", PhysicsEvent.PHYSICS_READY]
    assert manager._scene_data_backend.transforms_timestamp > version
    assert manager.kinematics_dirty


@pytest.mark.parametrize(("device", "expected_active_cuda_gpus"), [("cpu", None), ("cuda:2", "2")])
def test_manager_supports_pinned_runtime_api(manager, monkeypatch, tmp_path, device, expected_active_cuda_gpus):
    """The pinned OVPhysX wheel keeps its constructor and step API.

    The runtime never enters the wheel's sticky process-wide CPU-only mode, which would block later CUDA scenes.
    """
    import isaaclab_ov.physics.ovphysx_manager as module
    from isaaclab_ov.physics import OvPhysxBackendCfg

    cache_dir = str(tmp_path / "cooked_colliders")

    class PinnedPhysX:
        cpu_mode = None

        @classmethod
        def set_cpu_mode(cls, enabled):
            cls.cpu_mode = enabled

        def __init__(self, *, active_cuda_gpus=None, config=None):
            self.constructor = {"active_cuda_gpus": active_cuda_gpus, "config": config}
            self.calls = []

        def step_sync(self, *, dt):
            self.calls.append(("step_sync", dt))

        def update_articulations_kinematic(self):
            self.calls.append(("update_articulations_kinematic",))

    # Strict signature: a keyword the real wheel would reject fails here.
    def pinned_config(*, num_threads=None, cooked_collider_cache_dir=None, carbonite_overrides=None):
        return SimpleNamespace(
            num_threads=num_threads,
            cooked_collider_cache_dir=cooked_collider_cache_dir,
            carbonite_overrides=carbonite_overrides,
        )

    runtime = SimpleNamespace(PhysX=PinnedPhysX, PhysXConfig=pinned_config, bootstrap=lambda: None)
    monkeypatch.setattr(module, "ovphysx", runtime)

    backend = module.OvPhysxBackend(OvPhysxBackendCfg(device=device, cooked_collider_cache_dir=cache_dir))
    physx = backend.physx
    manager.backend.physx = physx
    monkeypatch.setattr(manager, "get_physics_dt", lambda: 0.02)
    monkeypatch.setattr(manager, "_sim_time", 0.0)
    version = manager._scene_data_backend.transforms_timestamp
    manager.step()

    assert PinnedPhysX.cpu_mode is None
    assert physx.constructor["active_cuda_gpus"] == expected_active_cuda_gpus
    assert physx.constructor["config"].num_threads == 8
    assert physx.constructor["config"].cooked_collider_cache_dir == cache_dir
    assert physx.constructor["config"].carbonite_overrides["/ovphysx/clone/useEnvIds"] is device.startswith("cuda")
    assert physx.calls == [("step_sync", 0.02), ("update_articulations_kinematic",)]
    assert manager._sim_time == 0.02
    assert manager._scene_data_backend.transforms_timestamp > version
    assert not manager.kinematics_dirty


def test_transforms_finish_dirty_kinematics_before_native_reads(manager, monkeypatch):
    """Direct SDP consumers refresh pending FK once, before reading native poses."""
    import warp as wp

    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    calls = []
    manager.backend.physx = SimpleNamespace(update_articulations_kinematic=lambda: calls.append("fk"))
    backend = manager._scene_data_backend
    poses = wp.zeros(1, dtype=wp.transformf, device="cpu")
    backend._transforms.data.transforms = poses
    backend.backend = manager.backend
    backend.backend.rigid_body_view = SimpleNamespace(read=lambda _: calls.append("read"))
    sdp = SceneDataProvider(backend)
    monkeypatch.setattr(manager, "kinematics_dirty", True)
    sdp.get_transforms(SceneDataFormat.Transform())
    sdp.get_transforms(SceneDataFormat.Transform())
    assert calls == ["fk", "read"]
    assert not manager.kinematics_dirty

    version = backend.transforms_timestamp
    manager.update_kinematics()
    manager.kinematics_dirty = True
    manager.forward()
    assert backend.transforms_timestamp > version
    sdp.get_transforms(SceneDataFormat.Transform())
    sdp.get_transforms(SceneDataFormat.Transform())
    assert calls == ["fk", "read", "fk", "read"]


@pytest.mark.parametrize(("modern", "failure"), [(False, None), (True, None), (True, "attach")])
def test_manager_attaches_and_releases_owned_ovstage(manager, monkeypatch, modern, failure):
    """The registered resource releases the attached OVStage once, after PhysX."""
    import isaaclab_ov.physics.ovphysx_manager as om_mod

    events = []
    stages = []
    attach_task, detach_task = object(), object()
    monkeypatch.setattr(om_mod.ovphysx_compat, "OVPHYSX_USES_TASKS", modern)
    monkeypatch.setattr(
        om_mod,
        "OVPHYSX_LIFECYCLE_ENTRY_POINTS",
        _CURRENT_LIFECYCLE_ENTRY_POINTS if modern else _LEGACY_LIFECYCLE_ENTRY_POINTS,
    )

    class FakeStage:
        def __init__(self, name):
            stages.append(self)
            events.append(("stage", name))

        def __enter__(self):
            return self

        def advance_write_floor(self, ordinal):
            return SimpleNamespace(wait=lambda: events.append(("seal", ordinal)))

        def __exit__(self, *exc):
            events.append(("destroy",))

    class FakePhysX:
        def attach_ovstage(self, stage, read_ordinal):
            events.append(("attach", stage, read_ordinal))
            return attach_task if modern else None

        def reset_stage(self):
            events.append(("reset",))
            return 17

        def wait_op(self, op):
            events.append(("wait", op))

        def detach_ovstage(self):
            events.append(("detach",))
            return detach_task

        def wait_task(self, task):
            assert task in (attach_task, detach_task)
            events.append(("wait_task", task))
            if failure == "attach" and task is attach_task:
                raise RuntimeError(f"{failure} failed")

        def destroy(self):
            events.append(("destroy_physx",))

        def release(self):
            events.append(("release",))

    fake_ovstage = ModuleType("ovstage")
    fake_ovstage.population = SimpleNamespace(
        open_usd_from_string=lambda stage, usda, ordinal, domains: events.append(
            ("populate", stage, usda, ordinal, domains)
        )
    )
    from isaaclab_ov import stage as stage_module

    fake_ovstage.PathDictionary = lambda stage: nullcontext(object())
    monkeypatch.setattr(stage_module, "ovstage", fake_ovstage)
    monkeypatch.setattr(stage_module, "create_ovstage", FakeStage)

    physx = FakePhysX()
    manager.backend.physx = physx
    manager.backend.rigid_body_view = SimpleNamespace(destroy=lambda: events.append(("destroy_view",)))
    monkeypatch.setattr(
        om_mod.OvPhysxView,
        "_close_all_for",
        lambda value: events.append(("close_views", value)),
    )
    if failure == "attach":
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            manager.backend.load("#usda 1.0")
    else:
        manager.backend.load("#usda 1.0")
    stage = stages[0]
    assert manager.backend.stage is stage
    manager.backend.close()
    manager.backend.close()

    # The seal must land between population and attach: ovphysx reads sealed data
    # only, so attaching at an unsealed ordinal silently yields an empty scene.
    assert events == [
        ("stage", "isaaclab.scene"),
        ("populate", stage, "#usda 1.0", 1, om_mod.ovstage.PopulationDomain.ALL),
        ("seal", 1),
        ("attach", stage, 1),
    ] + ([("wait_task", attach_task)] if modern else []) + [
        ("destroy_view",),
        ("close_views", physx),
    ] + ([("detach",), ("wait_task", detach_task)] if modern else [("reset",), ("wait", 17)]) + [
        ("destroy_physx",) if modern else ("release",),
        ("destroy",),
    ]
    assert manager.backend.rigid_body_view is None


@pytest.mark.parametrize(
    ("entry_points", "retryable", "modern"),
    [
        pytest.param(_LEGACY_LIFECYCLE_ENTRY_POINTS, False, False, id="legacy-release-error"),
        pytest.param(_CURRENT_LIFECYCLE_ENTRY_POINTS, False, False, id="terminal-destroy-error"),
        pytest.param(_CURRENT_LIFECYCLE_ENTRY_POINTS, True, False, id="retryable-destroy-error"),
        pytest.param(_CURRENT_LIFECYCLE_ENTRY_POINTS, True, True, id="task-detach-and-retryable-destroy-error"),
    ],
)
def test_manager_close_preserves_only_retryable_native_owners(manager, monkeypatch, entry_points, retryable, modern):
    """Terminal errors free native owners; pre-teardown errors retain them for the next close."""
    from isaaclab_ov.physics import OvPhysxBackendCfg
    from isaaclab_ov.physics import ovphysx_manager as om_mod

    from isaaclab.sim import SimulationContext

    events = []
    monkeypatch.setattr(om_mod.ovphysx_compat, "OVPHYSX_USES_TASKS", modern)
    monkeypatch.setattr(om_mod, "OVPHYSX_LIFECYCLE_ENTRY_POINTS", entry_points)

    class FakePhysX:
        fail_destroy = True
        terminal = False

        @property
        def handle(self):
            if self.terminal:
                raise RuntimeError("PhysX instance has been destroyed")
            return 17

        def reset_stage(self):
            events.append("reset")
            return 23

        def wait_op(self, operation):
            events.append(("wait", operation))

        def destroy(self):
            events.append("destroy")
            if self.fail_destroy:
                self.terminal = not retryable
                raise RuntimeError("native teardown failed")

        def release(self):
            events.append("release")
            raise RuntimeError("native teardown failed")

    class TaskPhysX(FakePhysX):
        fail_detach = True

        def detach_ovstage(self):
            events.append("reset")
            return receipt

        def wait_task(self, task):
            assert task is receipt
            events.append(("wait", receipt))
            if self.fail_detach:
                self.fail_detach = False
                raise RuntimeError("detach task failed")

    receipt = object() if modern else 23
    physx = TaskPhysX() if modern else FakePhysX()
    stage = SimpleNamespace(destroy=lambda: events.append("destroy_stage"))
    manager.backend.physx = physx
    manager.backend.stage = stage
    resources = ExitStack()
    resources.callback(stage.destroy)
    manager.backend.scene = SimpleNamespace(close=resources.close)
    backend = manager.backend
    cfg = OvPhysxBackendCfg(device="cpu")
    sim = SimpleNamespace(
        _backend_registry=[(cfg, backend)],
        backend_manager=manager,
    )
    sim.close_backend = SimulationContext.close_backend.__get__(sim)
    monkeypatch.setattr(SimulationContext, "_instance", sim)
    monkeypatch.setattr(manager, "_sim", sim)
    monkeypatch.setattr(om_mod.OvPhysxView, "_close_all_for", lambda value: events.append("close_views"))
    with pytest.raises(RuntimeError, match="native teardown failed") as error:
        manager.close()
    if modern:
        assert isinstance(error.value.__context__, RuntimeError)
        assert str(error.value.__context__) == "detach task failed"

    # A failed close keeps the registry owner so the next close can retry.
    assert manager._sim is sim
    assert sim._backend_registry == [(cfg, backend)]
    assert backend.physx is (physx if retryable else None)
    assert backend.stage is (stage if retryable else None)

    teardown = ["close_views", "reset", ("wait", receipt), entry_points["destroy"]]
    assert events == teardown + ([] if retryable else ["destroy_stage"])

    physx.fail_destroy = False
    manager.close()

    assert manager.backend is None
    assert not sim._backend_registry
    assert backend.physx is None and backend.stage is None
    assert events == teardown * (2 if retryable else 1) + ["destroy_stage"]


def test_manager_destroys_ovstage_when_population_fails(manager, monkeypatch):
    """A failed in-memory population does not leak its OVStage allocation."""
    from isaaclab_ov import stage as stage_module

    destroyed = []

    class FakeStage:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            destroyed.append(self.name)

    def fail_population(*args, **kwargs):
        raise RuntimeError("population failed")

    fake_ovstage = ModuleType("ovstage")
    fake_ovstage.population = SimpleNamespace(open_usd_from_string=fail_population)
    fake_ovstage.PathDictionary = lambda stage: nullcontext(object())
    monkeypatch.setattr(stage_module, "ovstage", fake_ovstage)
    monkeypatch.setattr(stage_module, "create_ovstage", FakeStage)

    with pytest.raises(RuntimeError, match="population failed"):
        manager.backend.load("#usda 1.0")
    assert manager.backend.stage is None

    assert destroyed == ["isaaclab.scene"]


def test_ovphysx_cfg_does_not_register_unselected_backend_schemas(monkeypatch):
    """Creating an eager preset alternative leaves global USD plugins unchanged."""
    from isaaclab_ov.physics import OvPhysxCfg, OvPhysxManager

    calls = []
    monkeypatch.setattr(
        OvPhysxManager,
        "_ensure_physx_schemas_registered",
        classmethod(lambda cls: calls.append(cls)),
    )

    OvPhysxCfg()

    assert calls == []


def test_automatic_physx_selection_prepares_ovphysx_before_stage_creation(monkeypatch):
    """Automatic kitless PhysX selection prepares OvPhysX before creating the USD stage."""
    from isaaclab_ov.physics import OvPhysxCfg, OvPhysxManager
    from isaaclab_physx.physics import PhysxCfg

    import isaaclab.sim.simulation_context as simulation_context_module
    from isaaclab.physics import PhysxAutoCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

    class StageCreationReached(Exception):
        """Signal that initialization reached stage creation."""

    class StubPhysxManager:
        """Stand in for the Kit-only manager while recording its pre-stage hook."""

        @classmethod
        def _prepare_stage_creation(cls):
            events.append("physx")

    events = []
    monkeypatch.setattr(simulation_context_module, "has_kit", lambda: False)
    # Record schema registration so the real pre-stage hook must reach it.
    monkeypatch.setattr(
        OvPhysxManager,
        "_ensure_physx_schemas_registered",
        classmethod(lambda cls: events.append("ovphysx")),
    )

    def _stop_at_stage_creation():
        events.append("stage")
        raise StageCreationReached

    monkeypatch.setattr(simulation_context_module, "create_new_stage", _stop_at_stage_creation)
    cfg = SimulationCfg(create_stage_in_memory=True)
    physics_cfg = PhysxAutoCfg(isaacsim_physx=PhysxCfg(), ovphysx=OvPhysxCfg())
    physics_cfg.class_type = StubPhysxManager
    cfg.physics = physics_cfg

    with pytest.raises(StageCreationReached):
        SimulationContext(cfg)

    assert events == ["ovphysx", "stage"]
    assert SimulationContext.instance() is None


@pytest.mark.parametrize("modern", [False, True])
def test_scene_data_binding_is_deferred_until_requested(manager, monkeypatch, modern):
    """Resolve nonempty rigid/link sessions in native order, once, before their dictionaries expire."""
    from contextlib import contextmanager, nullcontext

    import ovstage
    from isaaclab_ov.physics import ovphysx_compat
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend
    from ovphysx.types import SimObjectType

    monkeypatch.setattr(ovphysx_compat, "OVPHYSX_USES_TASKS", modern)
    calls = []
    active = False
    expected_paths = ["/World/Body", "/World/Robot/Link"]

    def get_paths(kind):
        assert active, "path dictionary used after session exit"
        return [expected_paths[0 if kind == SimObjectType.RIGID_BODY else 1]]

    dictionary = SimpleNamespace(get_path_strings=get_paths)
    monkeypatch.setattr(ovstage, "PathDictionary", lambda: nullcontext(dictionary))

    @contextmanager
    def read(kind, fields):
        nonlocal active
        calls.append((kind, fields))
        active = True
        try:
            yield SimpleNamespace(groups=[SimpleNamespace(prim_list=kind)], dictionary=dictionary)
        finally:
            active = False

    def bind(*, prim_paths):
        assert prim_paths == expected_paths
        return SimpleNamespace(count=2, prim_paths=prim_paths)

    runtime = SimpleNamespace(create_tensor_binding=bind, **{"read_sync" if modern else "read": read})
    native = SimpleNamespace(physx=runtime, rigid_body_view=None)
    backend = OvPhysxSceneDataBackend(manager)
    backend._defer_setup(native, "cpu", ())
    assert calls == []
    assert backend.transform_count == 2
    assert backend.transform_paths == expected_paths
    assert len(calls) == 2
    assert backend.transform_count == 2
    assert len(calls) == 2


def test_transforms_read_native_buffer_only_when_dirty(manager):
    """One fused native binding fills the pose buffer directly and skips clean publications."""
    import isaaclab_ov.physics.ovphysx_manager as module
    import numpy as np

    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    expected = np.array([[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 1], [7, 8, 9, 0, 0, 0, 1]], dtype=np.float32)
    paths = ["/World/envs/env_0/Cart", "/World/envs/env_1/Cart", "/World/envs/env_0/Pole"]
    reads = []

    def read(dst):
        reads.append(dst.ptr)
        dst.assign(expected)

    native = SimpleNamespace(physx=None, rigid_body_view=SimpleNamespace(count=len(paths), prim_paths=paths, read=read))

    backend = module.OvPhysxSceneDataBackend(manager)
    publication = backend.transforms
    backend.setup(native, "cpu")
    assert backend.transforms is publication
    sdp = SceneDataProvider(backend)

    native = SceneDataFormat.Transform()
    assert sdp.get_transforms(native)
    assert backend.transform_count == len(paths)
    assert backend.transform_paths == paths
    assert reads == [native.transforms.ptr]
    np.testing.assert_array_equal(native.transforms.numpy(), expected)
    second_output = SceneDataFormat.Transform()
    assert sdp.get_transforms(second_output)
    assert second_output.transforms is native.transforms
    assert len(reads) == 1

    expected[:, 0] += 10
    backend.transforms_timestamp += 1
    assert sdp.get_transforms(second_output)
    assert second_output.transforms is native.transforms
    assert len(reads) == 2
    np.testing.assert_array_equal(native.transforms.numpy(), expected)

    backend.setup(SimpleNamespace(rigid_body_view=None), "cpu")
    assert backend.transforms is publication
    assert publication.transforms is None
    assert backend.transform_count == 0 and backend.transform_paths == []


def test_failed_rigid_read_is_retried(manager):
    """A read failure propagates rather than caching a partial or stale publication."""
    import warp as wp
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend

    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    def fail_read(dst):
        raise RuntimeError("simulated read failure")

    backend = OvPhysxSceneDataBackend(manager)
    backend._transforms.data.transforms = wp.empty(1, dtype=wp.transformf, device="cpu")
    backend.backend = SimpleNamespace(rigid_body_view=SimpleNamespace(read=fail_read))
    sdp = SceneDataProvider(backend)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="simulated read failure"):
            sdp.get_transforms(SceneDataFormat.Transform())


@pytest.mark.parametrize("declared", [False, True])
def test_geometry_publication_distinguishes_undeclared_and_empty_scenes(manager, declared):
    """Native setup without a geometry declaration cannot publish an empty scene."""
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend

    backend = OvPhysxSceneDataBackend(manager)
    native = SimpleNamespace(physx=None, rigid_body_view=SimpleNamespace(count=0))
    backend.setup(native, "cpu", () if declared else None)
    if declared:
        assert backend.get_geometry_batches() == []
        assert backend.native_geometry_formats == ()
        backend.setup(native, "cpu")
    with pytest.raises(RuntimeError, match="ClonePlan"):
        backend.get_geometry_batches()
    with pytest.raises(RuntimeError, match="ClonePlan"):
        _ = backend.native_geometry_formats


@pytest.mark.parametrize("node_padding", [0, 1])
def test_deformable_only_setup_publishes_declared_geometry_in_native_order(manager, node_padding):
    """Mixed native views fill declared geometry slices without a packing pass."""
    import isaaclab_ov.physics.ovphysx_manager as module
    import numpy as np
    import warp as wp
    from isaaclab_ov import tensor_types as TT

    from isaaclab.scene_data import SceneDataProvider
    from isaaclab.scene_data.deformable_discovery import DeformableStageEntry

    values = {"/Clones/slot_2/Asset": 2.0, "/Clones/slot_9/Asset": 9.0, "/Shared": 100.0}
    entries = [
        DeformableStageEntry(path, path + "/sim", path + "/vis", "volume" if path == "/Shared" else "surface", 4, 4)
        for path in values
    ]
    entries[-1].vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    entries[-1].indices = np.array([0, 1, 2, 3], dtype=np.int32)
    entries[-1].vis_vertices = np.array([[0.5, 0, 0], [0, 0.5, 0]], dtype=np.float32)
    entries[-1].vis_vertex_count = 2
    reads = []
    bindings = []

    class NativePhysX:
        def create_tensor_binding(self, *, prim_paths, tensor_type):
            bindings.append((prim_paths, tensor_type))
            nodes = 4 + node_padding

            def read(dst):
                data = np.array(
                    [np.arange(nodes * 3).reshape(nodes, 3) + values[path] for path in reversed(prim_paths)],
                    dtype=np.float32,
                )
                wp.copy(dst, wp.array(data, dtype=wp.float32, device="cpu"))
                reads.append((dst.ptr, dst.size * wp.types.type_size_in_bytes(dst.dtype)))

            return SimpleNamespace(
                shape=(len(prim_paths), nodes, 3),
                count=len(prim_paths),
                dtype=SimpleNamespace(code=2, bits=32, lanes=1),
                prim_paths=[path + "/sim" for path in reversed(prim_paths)],
                read=read,
                destroy=lambda: None,
            )

    backend = module.OvPhysxSceneDataBackend(manager)
    native = SimpleNamespace(physx=NativePhysX(), rigid_body_view=SimpleNamespace(count=0))
    if node_padding:
        with pytest.raises(RuntimeError, match="node counts"):
            backend.setup(native, "cpu", entries)
        return
    backend.setup(native, "cpu", entries)

    assert {kind for _, kind in bindings} == {TT.SURFACE_DEFORMABLE_SIM_POSITION, TT.DEFORMABLE_SIM_NODAL_POSITION}
    provider = SceneDataProvider(backend)
    visual = provider.get_geometry_points()
    assert set(visual) == {path + "/vis" for path in values}
    assert reads[0][0] == visual["/Clones/slot_9/Asset/vis"].ptr
    assert reads[1][0] == reads[0][0] + reads[0][1]
    assert visual["/Clones/slot_2/Asset/vis"].ptr == reads[0][0] + 4 * wp.types.type_size_in_bytes(wp.vec3f)
    nodes = np.arange(12).reshape(4, 3)
    for path in ("/Clones/slot_2/Asset", "/Clones/slot_9/Asset"):
        np.testing.assert_array_equal(visual[path + "/vis"].numpy(), nodes + values[path])
    expected = (nodes[0] + nodes[[1, 2]]) / 2 + values["/Shared"]
    np.testing.assert_array_equal(visual["/Shared/vis"].numpy(), expected)
    assert provider.get_geometry_points() is visual
    assert len(reads) == len(bindings)
    values["/Shared"] += 1
    backend.geometry_timestamp += 1
    updated = provider.get_geometry_points()
    assert len(reads) == 2 * len(bindings)
    assert updated["/Shared/vis"] is visual["/Shared/vis"]
    np.testing.assert_array_equal(updated["/Shared/vis"].numpy(), expected + 1)
