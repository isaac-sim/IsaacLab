# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OvPhysxSceneDataBackend (new SceneDataBackend interface, post-#5128) and OvPhysxManager."""

from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

# The OVPhysX runtime wheel is optional. Skip gracefully when it is not installed;
# CI jobs that need OVPhysX coverage install it explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

_LEGACY_LIFECYCLE_ENTRY_POINTS = {"warmup": "warmup_gpu", "destroy": "release"}
_CURRENT_LIFECYCLE_ENTRY_POINTS = {"warmup": "warmup", "destroy": "destroy"}


@pytest.fixture(autouse=True)
def _native_backend(monkeypatch):
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxBackend, OvPhysxManager, OvPhysxSceneDataBackend

    backend = OvPhysxBackend.__new__(OvPhysxBackend)
    backend.physx = None
    backend.stage = None
    monkeypatch.setattr(OvPhysxManager, "backend", backend)
    monkeypatch.setattr(OvPhysxManager, "_scene_data_backend", OvPhysxSceneDataBackend())
    monkeypatch.setattr(OvPhysxManager, "_kinematics_dirty", False)


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


def _serialize_full_stage_with_pending_clones(stage) -> str:
    """Serialize ``stage`` with pending clones materialized via the production path."""
    from isaaclab_ov.physics import OvPhysxManager

    previous = OvPhysxManager._requires_full_stage
    try:
        OvPhysxManager._requires_full_stage = True
        return OvPhysxManager._serialize_selected_stage(stage)
    finally:
        OvPhysxManager._requires_full_stage = previous


def _fake_rigid_body_prim(path: str):
    """Build a traversal stub with RigidBodyAPI and no deformable schemas."""
    return SimpleNamespace(
        HasAPI=lambda api: True,
        GetPath=lambda p=path: SimpleNamespace(pathString=p),
        GetAppliedSchemas=lambda: [],
        GetMetadata=lambda key: None,
    )


def test_manager_full_stage_materializes_only_missing_heterogeneous_targets():
    """A full-stage export copies missing heterogeneous targets without replacing authored ones."""
    from isaaclab_ov.physics import OvPhysxManager

    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_1", "Xform")
    stage.DefinePrim("/World/envs/env_2", "Xform")
    source = stage.DefinePrim("/World/envs/env_0/Object", "Xform")
    source.CreateAttribute("test:variant", Sdf.ValueTypeNames.String).Set("source")
    existing = stage.DefinePrim("/World/envs/env_1/Object", "Xform")
    existing.CreateAttribute("test:variant", Sdf.ValueTypeNames.String).Set("authored")
    previous = OvPhysxManager._pending_clones
    try:
        OvPhysxManager._pending_clones = [
            (
                "/World/envs/env_0/Object",
                ["/World/envs/env_1/Object", "/World/envs/env_2/Object"],
                [(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0), (4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0)],
                [1, 2],
            )
        ]
        materialized_usda = _serialize_full_stage_with_pending_clones(stage)
        layer = Sdf.Layer.CreateAnonymous("materialized.usda")
        assert layer.ImportFromString(materialized_usda)
        exported = Usd.Stage.Open(layer)
        assert exported.GetPrimAtPath("/World/envs/env_1/Object").GetAttribute("test:variant").Get() == "authored"
        assert exported.GetPrimAtPath("/World/envs/env_2/Object").GetAttribute("test:variant").Get() == "source"
        assert OvPhysxManager._pending_clones == []
    finally:
        OvPhysxManager._pending_clones = previous


def test_manager_full_stage_materializes_nested_targets_parent_before_child():
    """Nested clone targets materialize shallow-to-deep regardless of queue order."""
    from isaaclab_ov.physics import OvPhysxManager

    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_1", "Xform")
    groceries = stage.DefinePrim("/World/envs/env_0/Groceries", "Xform")
    groceries.CreateAttribute("test:collection", Sdf.ValueTypeNames.String).Set("source")
    object_prim = stage.DefinePrim("/World/envs/env_0/Groceries/Object", "Xform")
    object_prim.CreateAttribute("physics:enabled", Sdf.ValueTypeNames.Bool).Set(True)

    previous = OvPhysxManager._pending_clones
    try:
        OvPhysxManager._pending_clones = [
            (
                "/World/envs/env_0/Groceries/Object",
                ["/World/envs/env_1/Groceries/Object"],
                [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
                [1],
            ),
            (
                "/World/envs/env_0/Groceries",
                ["/World/envs/env_1/Groceries"],
                [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
                [1],
            ),
        ]
        materialized_usda = _serialize_full_stage_with_pending_clones(stage)
        layer = Sdf.Layer.CreateAnonymous("materialized.usda")
        assert layer.ImportFromString(materialized_usda)
        exported = Usd.Stage.Open(layer)
        target_parent = exported.GetPrimAtPath("/World/envs/env_1/Groceries")
        target_child = exported.GetPrimAtPath("/World/envs/env_1/Groceries/Object")
        assert target_parent.GetAttribute("test:collection").Get() == "source"
        assert target_child.GetAttribute("physics:enabled").Get() is True
        assert OvPhysxManager._pending_clones == []
    finally:
        OvPhysxManager._pending_clones = previous


def test_manager_full_stage_promotes_generated_nested_ancestors_to_def():
    """A child-only nested target composes beneath a generated defined parent."""
    from isaaclab_ov.physics import OvPhysxManager

    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_1", "Xform")
    source = stage.DefinePrim("/World/envs/env_0/Groceries/Object", "Xform")
    source.CreateAttribute("physics:enabled", Sdf.ValueTypeNames.Bool).Set(True)

    previous = OvPhysxManager._pending_clones
    try:
        OvPhysxManager._pending_clones = [
            (
                "/World/envs/env_0/Groceries/Object",
                ["/World/envs/env_1/Groceries/Object"],
                [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
                [1],
            )
        ]
        materialized_usda = _serialize_full_stage_with_pending_clones(stage)
        layer = Sdf.Layer.CreateAnonymous("materialized.usda")
        assert layer.ImportFromString(materialized_usda)
        exported = Usd.Stage.Open(layer)
        target_parent = exported.GetPrimAtPath("/World/envs/env_1/Groceries")
        target_child = exported.GetPrimAtPath("/World/envs/env_1/Groceries/Object")
        assert target_parent.IsDefined()
        assert target_child.IsDefined()
        assert target_child.GetAttribute("physics:enabled").Get() is True
        assert OvPhysxManager._pending_clones == []
    finally:
        OvPhysxManager._pending_clones = previous


def test_manager_full_stage_overlays_existing_ancestor_without_removing_descendants():
    """An ancestor created for another asset gains source physics while retaining descendants."""
    from isaaclab_ov.physics import OvPhysxManager

    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_1", "Xform")
    source = stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
    source.CreateAttribute("physics:enabled", Sdf.ValueTypeNames.Bool).Set(True)
    physics = stage.DefinePrim("/World/envs/env_0/Robot/Physics", "Xform")
    physics.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(3.0)
    camera = stage.DefinePrim("/World/envs/env_1/Robot/Camera", "Xform")
    camera.CreateAttribute("test:keep", Sdf.ValueTypeNames.Bool).Set(True)

    previous = OvPhysxManager._pending_clones
    try:
        OvPhysxManager._pending_clones = [
            ("/World/envs/env_0/Robot", ["/World/envs/env_1/Robot"], [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)], [1])
        ]
        materialized_usda = _serialize_full_stage_with_pending_clones(stage)
        layer = Sdf.Layer.CreateAnonymous("materialized.usda")
        assert layer.ImportFromString(materialized_usda)
        exported = Usd.Stage.Open(layer)
        robot = exported.GetPrimAtPath("/World/envs/env_1/Robot")
        assert robot.GetAttribute("physics:enabled").Get() is True
        assert exported.GetPrimAtPath("/World/envs/env_1/Robot/Physics").GetAttribute("physics:mass").Get() == 3.0
        assert exported.GetPrimAtPath("/World/envs/env_1/Robot/Camera").GetAttribute("test:keep").Get() is True
    finally:
        OvPhysxManager._pending_clones = previous


def test_manager_retains_clone_recipes_across_full_stage_serializations():
    """A second full-stage serialization rematerializes targets from active recipes."""
    from isaaclab_ov.physics import OvPhysxManager

    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_1", "Xform")
    source = stage.DefinePrim("/World/envs/env_0/Object", "Xform")
    source.CreateAttribute("physics:enabled", Sdf.ValueTypeNames.Bool).Set(True)
    previous_pending = OvPhysxManager._pending_clones
    previous_active = OvPhysxManager._active_clone_recipes
    try:
        OvPhysxManager._pending_clones = []
        OvPhysxManager._active_clone_recipes = []
        OvPhysxManager.register_clone("/World/envs/env_0/Object", ["/World/envs/env_1/Object"], [(1.0, 0.0, 0.0)])
        for _ in range(2):
            OvPhysxManager._rearm_pending_clones()
            materialized_usda = _serialize_full_stage_with_pending_clones(stage)
            layer = Sdf.Layer.CreateAnonymous("materialized.usda")
            assert layer.ImportFromString(materialized_usda)
            exported = Usd.Stage.Open(layer)
            assert exported.GetPrimAtPath("/World/envs/env_1/Object").GetAttribute("physics:enabled").Get() is True
            assert OvPhysxManager._pending_clones == []
        assert len(OvPhysxManager._active_clone_recipes) == 1
    finally:
        OvPhysxManager._pending_clones = previous_pending
        OvPhysxManager._active_clone_recipes = previous_active


def test_manager_full_stage_materialization_is_atomic_on_invalid_target():
    """A validation failure clears the queue without partially modifying the export."""
    from isaaclab_ov.physics import OvPhysxManager

    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_1", "Xform")
    stage.DefinePrim("/World/envs/env_0/Object", "Xform")
    previous = OvPhysxManager._pending_clones
    try:
        OvPhysxManager._pending_clones = [
            (
                "/World/envs/env_0/Object",
                ["/World/envs/env_1/Object", "/World/envs/env_2/Object"],
                [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
                [1, 2],
            )
        ]
        with pytest.raises(RuntimeError, match="clone target parent is absent"):
            _serialize_full_stage_with_pending_clones(stage)
        assert OvPhysxManager._pending_clones == []
        assert not stage.GetPrimAtPath("/World/envs/env_1/Object").IsValid()
    finally:
        OvPhysxManager._pending_clones = previous


@pytest.mark.parametrize("requires_full_stage", [False, True])
def test_manager_replays_pending_runtime_clones_without_full_stage_requirement(requires_full_stage):
    """Only the default replay path forwards final world transforms; a full-stage load never clones."""
    from isaaclab_ov.physics import OvPhysxManager

    class FakePhysX:
        def __init__(self):
            self.calls = []

        def clone(self, source, targets, transforms, env_ids):
            self.calls.append(("clone", source, targets, transforms, env_ids))
            return 19

        def wait_op(self, operation):
            self.calls.append(("wait_op", operation))

    fake = FakePhysX()
    previous = OvPhysxManager._pending_clones
    try:
        OvPhysxManager._pending_clones = [("/env_0", ["/env_1"], [(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)], [1])]
        OvPhysxManager._replay_pending_clones(fake, requires_full_stage=requires_full_stage)
        if requires_full_stage:
            assert fake.calls == []
        else:
            assert fake.calls == [
                ("clone", "/env_0", ["/env_1"], [(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)], [1]),
                ("wait_op", 19),
            ]
        assert OvPhysxManager._pending_clones == []
    finally:
        OvPhysxManager._pending_clones = previous


def test_manager_forced_rewarm_invalidates_bindings_before_loading(monkeypatch):
    """A forced re-warm invalidates views before replacing their attached stage."""
    from isaaclab_ov.physics import OvPhysxManager

    from isaaclab.physics import PhysicsEvent

    calls = []
    monkeypatch.setattr(OvPhysxManager, "_warmup_done", False)
    OvPhysxManager.backend.stage = object()
    monkeypatch.setattr(OvPhysxManager, "_warmup_and_load", lambda: calls.append("warmup"))
    monkeypatch.setattr(
        OvPhysxManager,
        "dispatch_event",
        lambda event, payload=None: calls.append(event),
    )

    version = OvPhysxManager._scene_data_backend.transforms_version
    OvPhysxManager.reset()

    assert calls == [PhysicsEvent.STOP, "warmup", PhysicsEvent.PHYSICS_READY]
    assert OvPhysxManager._scene_data_backend.transforms_version > version
    assert OvPhysxManager._kinematics_dirty


@pytest.mark.parametrize(
    ("device", "expected_cpu_mode", "expected_active_cuda_gpus"),
    [("cpu", True, None), ("cuda:2", False, "2")],
)
def test_manager_supports_pinned_runtime_api(
    monkeypatch, tmp_path, device, expected_cpu_mode, expected_active_cuda_gpus
):
    """The pinned OVPhysX wheel keeps its constructor, step, and reset API."""
    import isaaclab_ov.physics.ovphysx_manager as module
    from isaaclab_ov.physics import OvPhysxBackendCfg, OvPhysxManager

    from isaaclab.physics import PhysicsManager

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

        def reset_stage(self):
            self.calls.append(("reset_stage",))
            return 23

        def wait_op(self, operation):
            self.calls.append(("wait_op", operation))

    # Strict signature: a keyword the real wheel would reject fails here.
    def pinned_config(*, num_threads=None, cooked_collider_cache_dir=None, carbonite_overrides=None):
        return SimpleNamespace(
            num_threads=num_threads,
            cooked_collider_cache_dir=cooked_collider_cache_dir,
            carbonite_overrides=carbonite_overrides,
        )

    runtime = SimpleNamespace(PhysX=PinnedPhysX, PhysXConfig=pinned_config, bootstrap=lambda: None)
    monkeypatch.setattr(module, "import_ovphysx", lambda: runtime)

    backend = module.OvPhysxBackend(OvPhysxBackendCfg(device=device, cooked_collider_cache_dir=cache_dir))
    physx = backend.physx
    OvPhysxManager.backend.physx = physx
    monkeypatch.setattr(OvPhysxManager, "get_physics_dt", lambda: 0.02)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 0.0)
    version = OvPhysxManager._scene_data_backend.transforms_version
    OvPhysxManager.step()
    OvPhysxManager._prepare_physx_for_stage_reuse()

    assert PinnedPhysX.cpu_mode is expected_cpu_mode
    assert physx.constructor["active_cuda_gpus"] == expected_active_cuda_gpus
    assert physx.constructor["config"].num_threads == 8
    assert physx.constructor["config"].cooked_collider_cache_dir == cache_dir
    assert physx.calls == [("step_sync", 0.02), ("update_articulations_kinematic",), ("reset_stage",), ("wait_op", 23)]
    assert PhysicsManager._sim_time == 0.02
    assert OvPhysxManager._scene_data_backend.transforms_version > version
    assert not OvPhysxManager._kinematics_dirty


def test_transforms_finish_dirty_kinematics_before_native_reads(monkeypatch):
    """Direct SDP consumers refresh pending FK once, before reading native poses."""
    import warp as wp
    from isaaclab_ov.physics import OvPhysxManager

    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    calls = []
    OvPhysxManager.backend.physx = SimpleNamespace(update_articulations_kinematic=lambda: calls.append("fk"))
    backend = OvPhysxManager._scene_data_backend
    poses = wp.zeros(1, dtype=wp.transformf, device="cpu")
    backend._transforms.transforms = poses
    backend._rigid_bindings = [(SimpleNamespace(read_into=lambda *args: calls.append("read")), poses)]
    sdp = SceneDataProvider(backend)
    monkeypatch.setattr(OvPhysxManager, "_kinematics_dirty", True)
    sdp.get_transforms(SceneDataFormat.Transform())
    sdp.get_transforms(SceneDataFormat.Transform())
    assert calls == ["fk", "read"]
    assert not OvPhysxManager._kinematics_dirty

    version = backend.transforms_version
    OvPhysxManager.forward()
    assert backend.transforms_version > version
    sdp.get_transforms(SceneDataFormat.Transform())
    sdp.get_transforms(SceneDataFormat.Transform())
    assert calls == ["fk", "read", "fk", "read"]


def test_manager_serializes_env0_only_stage_in_memory(caplog):
    """The OVPhysX input keeps globals and env 0 without writing cloned envs."""
    from isaaclab_ov.physics import OvPhysxManager

    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    for path in ("/World/Ground", "/World/envs/env_0/Cube", "/World/envs/env_1/Cube"):
        UsdGeom.Xform.Define(stage, path)

    previous = OvPhysxManager._requires_full_stage
    try:
        OvPhysxManager._requires_full_stage = False
        with caplog.at_level(logging.INFO, logger=OvPhysxManager.__module__):
            usda = OvPhysxManager._serialize_selected_stage(stage)
    finally:
        OvPhysxManager._requires_full_stage = previous
    layer = Sdf.Layer.CreateAnonymous("filtered.usda")
    assert layer.ImportFromString(usda)
    filtered = Usd.Stage.Open(layer)

    assert filtered.GetPrimAtPath("/World/Ground").IsValid()
    assert filtered.GetPrimAtPath("/World/envs/env_0/Cube").IsValid()
    assert not filtered.GetPrimAtPath("/World/envs/env_1").IsValid()
    assert "stripped 1 env_<i!=0> subtrees from in-memory USD" in caplog.text


def test_manager_serializes_stage_without_envs_as_is():
    """The in-memory serializer keeps stages without the standard env namespace intact."""
    from isaaclab_ov.physics import OvPhysxManager

    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Ground")

    previous = OvPhysxManager._requires_full_stage
    try:
        OvPhysxManager._requires_full_stage = False
        usda = OvPhysxManager._serialize_selected_stage(stage)
    finally:
        OvPhysxManager._requires_full_stage = previous

    layer = Sdf.Layer.CreateAnonymous("no_envs.usda")
    assert layer.ImportFromString(usda)
    assert Usd.Stage.Open(layer).GetPrimAtPath("/World/Ground").IsValid()


def test_manager_attaches_and_releases_owned_ovstage(monkeypatch):
    """The registered resource releases the attached OVStage once, after PhysX."""
    import isaaclab_ov.physics.ovphysx_manager as om_mod
    from isaaclab_ov.physics import OvPhysxManager

    events = []
    monkeypatch.setattr(om_mod, "OVPHYSX_LIFECYCLE_ENTRY_POINTS", _LEGACY_LIFECYCLE_ENTRY_POINTS)

    class FakeWriteFloorOp:
        def __init__(self, ordinal):
            self._ordinal = ordinal

        def wait(self):
            events.append(("seal", self._ordinal))

    class FakeStage:
        def __init__(self, name):
            events.append(("stage", name))

        def advance_write_floor(self, ordinal):
            return FakeWriteFloorOp(ordinal)

        def destroy(self):
            events.append(("destroy",))

    class FakePhysX:
        def attach_ovstage(self, stage, read_ordinal):
            events.append(("attach", stage, read_ordinal))

        def reset_stage(self):
            events.append(("reset",))
            return 17

        def wait_op(self, op):
            events.append(("wait", op))

        def release(self):
            events.append(("release",))

    fake_ovstage = ModuleType("ovstage")
    fake_ovstage.PopulationDomain = SimpleNamespace(ALL="all")
    fake_ovstage.population = SimpleNamespace(
        open_usd_from_string=lambda stage, usda, ordinal, domains: events.append(
            ("populate", stage, usda, ordinal, domains)
        )
    )
    monkeypatch.setitem(sys.modules, "ovstage", fake_ovstage)
    # The manager builds its stage through the shared helper so every stage in the process gets
    # the same ovstage configuration; that is the seam to fake, not ``ovstage.Stage``.
    monkeypatch.setattr(om_mod, "create_ovstage", FakeStage)

    physx = FakePhysX()
    OvPhysxManager.backend.physx = physx
    monkeypatch.setattr(
        om_mod.OvPhysxView,
        "_close_all_for",
        lambda value: events.append(("close_views", value)),
    )
    OvPhysxManager._attach_ovstage("#usda 1.0")
    stage = OvPhysxManager.backend.stage
    OvPhysxManager.backend.close()
    OvPhysxManager.backend.close()

    # The seal must land between population and attach: ovphysx reads sealed data
    # only, so attaching at an unsealed ordinal silently yields an empty scene.
    assert events == [
        ("stage", "isaaclab"),
        ("populate", stage, "#usda 1.0", 1, "all"),
        ("seal", 1),
        ("attach", stage, 1),
        ("close_views", physx),
        ("reset",),
        ("wait", 17),
        ("release",),
        ("destroy",),
    ]


@pytest.mark.parametrize(
    ("entry_points", "expected_calls"),
    [
        (_LEGACY_LIFECYCLE_ENTRY_POINTS, ["warmup_gpu", "release"]),
        (_CURRENT_LIFECYCLE_ENTRY_POINTS, ["warmup", "destroy"]),
    ],
)
def test_manager_uses_version_selected_lifecycle_apis(monkeypatch, entry_points, expected_calls):
    """The selected lifecycle generation controls both entry points."""
    from isaaclab_ov.physics import OvPhysxManager
    from isaaclab_ov.physics import ovphysx_manager as om_mod

    calls = []
    physx = SimpleNamespace(
        warmup=lambda: calls.append("warmup"),
        warmup_gpu=lambda: calls.append("warmup_gpu"),
        destroy=lambda: calls.append("destroy"),
        release=lambda: calls.append("release"),
        reset_stage=lambda: None,
        wait_op=lambda op: None,
    )
    monkeypatch.setattr(om_mod, "OVPHYSX_LIFECYCLE_ENTRY_POINTS", entry_points)

    OvPhysxManager._warmup_physx(physx)
    OvPhysxManager.backend.physx = physx
    OvPhysxManager.backend.close()

    assert calls == expected_calls


@pytest.mark.parametrize("operation", ["warmup", "destroy"])
def test_manager_rejects_missing_lifecycle_api(monkeypatch, operation):
    """A runtime that lacks its selected lifecycle entry point reports it."""
    from isaaclab_ov.physics import OvPhysxManager
    from isaaclab_ov.physics import ovphysx_manager as om_mod

    monkeypatch.setattr(om_mod, "OVPHYSX_LIFECYCLE_ENTRY_POINTS", _CURRENT_LIFECYCLE_ENTRY_POINTS)
    entry_point = _CURRENT_LIFECYCLE_ENTRY_POINTS[operation]
    with pytest.raises(AttributeError, match=rf"selected {entry_point}\(\) lifecycle entry point"):
        if operation == "warmup":
            OvPhysxManager._warmup_physx(SimpleNamespace())
        else:
            OvPhysxManager.backend.physx = SimpleNamespace(reset_stage=lambda: None, wait_op=lambda op: None)
            OvPhysxManager.backend.close()


@pytest.mark.parametrize(
    ("entry_points", "retryable"),
    [
        pytest.param(_LEGACY_LIFECYCLE_ENTRY_POINTS, False, id="legacy-release-error"),
        pytest.param(_CURRENT_LIFECYCLE_ENTRY_POINTS, False, id="terminal-destroy-error"),
        pytest.param(_CURRENT_LIFECYCLE_ENTRY_POINTS, True, id="retryable-destroy-error"),
    ],
)
def test_manager_close_preserves_only_retryable_native_owners(monkeypatch, entry_points, retryable):
    """Terminal errors free native owners; pre-teardown errors retain them for the next close."""
    from isaaclab_ov.physics import OvPhysxBackendCfg, OvPhysxManager
    from isaaclab_ov.physics import ovphysx_manager as om_mod

    from isaaclab.physics import PhysicsManager
    from isaaclab.sim import SimulationContext

    events = []
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

    physx = FakePhysX()
    stage = SimpleNamespace(destroy=lambda: events.append("destroy_stage"))
    OvPhysxManager.backend.physx = physx
    OvPhysxManager.backend.stage = stage
    backend = OvPhysxManager.backend
    cfg = OvPhysxBackendCfg(device="cpu")
    sim = SimpleNamespace(
        _backend_registry=[(cfg, backend)],
        physics_manager=OvPhysxManager,
    )
    sim.close_backend = SimulationContext.close_backend.__get__(sim)
    monkeypatch.setattr(SimulationContext, "_instance", sim)
    for name in ("_cfg", "_sim_time"):
        monkeypatch.setattr(PhysicsManager, name, getattr(PhysicsManager, name))
    monkeypatch.setattr(PhysicsManager, "_sim", sim)
    monkeypatch.setattr(PhysicsManager, "_callbacks", {})
    monkeypatch.setattr(PhysicsManager, "views", {})
    monkeypatch.setattr(om_mod.OvPhysxView, "_close_all_for", lambda value: events.append("close_views"))
    with pytest.raises(RuntimeError, match="native teardown failed"):
        OvPhysxManager.close()

    assert PhysicsManager._sim is None
    assert sim._backend_registry == [(cfg, backend)]
    assert backend.physx is (physx if retryable else None)
    assert backend.stage is (stage if retryable else None)

    teardown = ["close_views", "reset", ("wait", 23), entry_points["destroy"]]
    assert events == teardown + ([] if retryable else ["destroy_stage"])

    physx.fail_destroy = False
    OvPhysxManager.close()

    assert OvPhysxManager.backend is None
    assert not sim._backend_registry
    assert backend.physx is None and backend.stage is None
    assert events == teardown * (2 if retryable else 1) + ["destroy_stage"]


def test_manager_destroys_ovstage_when_population_fails(monkeypatch):
    """A failed in-memory population does not leak its OVStage allocation."""
    import isaaclab_ov.physics.ovphysx_manager as om_mod
    from isaaclab_ov.physics import OvPhysxManager

    destroyed = []

    class FakeStage:
        def __init__(self, name):
            self.name = name

        def destroy(self):
            destroyed.append(self.name)

    def fail_population(*args, **kwargs):
        raise RuntimeError("population failed")

    fake_ovstage = ModuleType("ovstage")
    fake_ovstage.PopulationDomain = SimpleNamespace(ALL="all")
    fake_ovstage.population = SimpleNamespace(open_usd_from_string=fail_population)
    monkeypatch.setitem(sys.modules, "ovstage", fake_ovstage)
    monkeypatch.setattr(om_mod, "create_ovstage", FakeStage)

    with pytest.raises(RuntimeError, match="population failed"):
        OvPhysxManager._attach_ovstage("#usda 1.0")
    assert OvPhysxManager.backend.stage is None

    assert destroyed == ["isaaclab"]


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
    from isaaclab_ov.physics import OvPhysxManager

    import isaaclab.sim.simulation_context as simulation_context_module
    from isaaclab.app.sim_launcher import make_physics_cfg
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
    physics_cfg = make_physics_cfg("physx")
    physics_cfg.class_type = StubPhysxManager
    cfg.physics = physics_cfg

    with pytest.raises(StageCreationReached):
        SimulationContext(cfg)

    assert events == ["ovphysx", "stage"]
    assert SimulationContext.instance() is None


def test_transforms_read_native_slices_only_when_dirty(monkeypatch):
    """Native bindings fill one shared pose buffer directly and skip clean publications."""
    import isaaclab_ov.physics.ovphysx_manager as module
    import numpy as np
    import warp as wp

    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    expected = np.array([[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 1], [7, 8, 9, 0, 0, 0, 1]], dtype=np.float32)
    paths = ["/World/envs/env_0/Cart", "/World/envs/env_1/Cart", "/World/envs/env_0/Pole"]
    reads = []

    class FakePhysX:
        def create_tensor_binding(self, pattern, tensor_type):
            start, end = (0, 2) if pattern.endswith("/Cart") else (2, 3)

            def read(dst):
                reads.append((start, dst.ptr))
                wp.copy(dst, wp.array(expected[start:end], dtype=wp.float32, device="cpu"))

            return SimpleNamespace(
                shape=(end - start, 7),
                count=end - start,
                dtype=SimpleNamespace(code=2, bits=32, lanes=1),
                prim_paths=paths[start:end],
                read=read,
                destroy=lambda: None,
            )

    monkeypatch.setattr(module, "UsdPhysics", SimpleNamespace(RigidBodyAPI=object()))
    stage = SimpleNamespace(Traverse=lambda: (_fake_rigid_body_prim(path) for path in paths))
    backend = module.OvPhysxSceneDataBackend()
    backend.setup(FakePhysX(), stage, "cpu")
    sdp = SceneDataProvider(backend)

    native = SceneDataFormat.Transform()
    assert sdp.get_transforms(native)
    assert backend.transform_count == len(paths)
    assert backend.transform_paths == paths
    assert reads == [(0, native.transforms.ptr), (2, native.transforms.ptr + 2 * 7 * 4)]
    np.testing.assert_array_equal(native.transforms.numpy(), expected)
    second_output = SceneDataFormat.Transform()
    assert sdp.get_transforms(second_output)
    assert second_output.transforms is native.transforms
    assert len(reads) == 2

    expected[:, 0] += 10
    backend.transforms_version += 1
    assert sdp.get_transforms(second_output)
    assert second_output.transforms is native.transforms
    assert len(reads) == 4
    np.testing.assert_array_equal(native.transforms.numpy(), expected)


def test_setup_propagates_failed_rigid_binding(monkeypatch):
    """A failed binding cannot silently remove a body from the publication."""
    import isaaclab_ov.physics.ovphysx_manager as module

    class FailingPhysX:
        def create_tensor_binding(self, pattern, tensor_type):
            raise RuntimeError("simulated binding failure")

    monkeypatch.setattr(module, "UsdPhysics", SimpleNamespace(RigidBodyAPI=object()))
    stage = SimpleNamespace(Traverse=lambda: iter([_fake_rigid_body_prim("/World/Object")]))
    backend = module.OvPhysxSceneDataBackend()
    with pytest.raises(RuntimeError, match="simulated binding failure"):
        backend.setup(FailingPhysX(), stage, "cpu")


def test_failed_rigid_read_is_retried():
    """A read failure propagates rather than caching a partial or stale publication."""
    import warp as wp
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend

    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    def fail_read(name, dst):
        raise RuntimeError("simulated read failure")

    backend = OvPhysxSceneDataBackend()
    backend._transforms.transforms = wp.empty(1, dtype=wp.transformf, device="cpu")
    backend._rigid_bindings = [(SimpleNamespace(read_into=fail_read), backend._transforms.transforms)]
    sdp = SceneDataProvider(backend)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="simulated read failure"):
            sdp.get_transforms(SceneDataFormat.Transform())


@pytest.mark.parametrize("declared", [False, True])
def test_geometry_publication_distinguishes_undeclared_and_empty_scenes(declared):
    """Native setup without a geometry declaration cannot publish an empty scene."""
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend

    backend = OvPhysxSceneDataBackend()
    stage = SimpleNamespace(Traverse=lambda: iter(()))
    backend.setup(None, stage, "cpu", () if declared else None)
    if declared:
        assert backend.get_geometry_batches() == []
        assert backend.native_geometry_formats == ()
        backend.setup(None, stage, "cpu")
    with pytest.raises(RuntimeError, match="ClonePlan"):
        backend.get_geometry_batches()
    with pytest.raises(RuntimeError, match="ClonePlan"):
        _ = backend.native_geometry_formats


@pytest.mark.parametrize("node_padding", [0, 1])
def test_deformable_only_setup_publishes_declared_geometry_in_native_order(node_padding):
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

    backend = module.OvPhysxSceneDataBackend()
    stage = SimpleNamespace(Traverse=lambda: iter(()))
    if node_padding:
        with pytest.raises(RuntimeError, match="node counts"):
            backend.setup(NativePhysX(), stage, "cpu", entries)
        return
    backend.setup(NativePhysX(), stage, "cpu", entries)

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
