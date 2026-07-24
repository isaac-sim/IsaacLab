# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``NewtonManager.update_visualization_state`` and shadow-model build.

When the active sim backend is PhysX and a Newton-native visualizer/renderer is in
use, :meth:`NewtonManager._ensure_visualization_model` must build the manager's
``_model`` / ``_state_0`` directly from the USD stage, and
:meth:`NewtonManager.update_visualization_state` must copy fresh transforms into
``_state_0.body_q`` via the new
:class:`~isaaclab.scene_data.SceneDataProvider`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration

_DEFAULT = object()


def _reset_newton_manager_state():
    from isaaclab_newton.physics import NewtonManager

    NewtonManager._builder = None
    NewtonManager._model = None
    NewtonManager._state_0 = None
    NewtonManager._num_envs = None
    NewtonManager._scene_data = None
    NewtonManager._scene_data_mapping = None


def _make_env_stage(num_envs: int = 1):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    for env_id in range(num_envs):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
    return stage


def _make_standalone_stage():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Robot")
    return stage


def _set_sim_context(monkeypatch, nm, clone_plan=_DEFAULT, scene_data_provider=_DEFAULT):
    clone_plan = SimpleNamespace() if clone_plan is _DEFAULT else clone_plan
    scene_data_provider = SimpleNamespace() if scene_data_provider is _DEFAULT else scene_data_provider
    sim = SimpleNamespace(
        get_clone_plan=lambda: clone_plan,
        get_scene_data_provider=lambda: scene_data_provider,
    )
    monkeypatch.setattr(nm.SimulationContext, "instance", classmethod(lambda cls: sim))
    return sim


def test_physics_manager_close_only_clears_active_manager_binding(monkeypatch):
    """Only the active physics manager can clear shared SimulationContext state."""
    from isaaclab.physics import PhysicsManager

    class _ActiveManager(PhysicsManager):
        _callbacks = {}

    class _InactiveManager(PhysicsManager):
        pass

    _ActiveManager.close()
    assert PhysicsManager._sim is None

    active_sim = SimpleNamespace(physics_manager=_ActiveManager)
    monkeypatch.setattr(PhysicsManager, "_sim", active_sim, raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", "active-cfg", raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 1.25, raising=False)

    monkeypatch.setattr(PhysicsManager, "_callbacks", {1: (None, lambda _: None, 0, "stale", None)}, raising=False)
    _InactiveManager.close()
    assert PhysicsManager._callbacks == {}
    assert (PhysicsManager._sim, PhysicsManager._cfg, PhysicsManager._sim_time) == (active_sim, "active-cfg", 1.25)

    _ActiveManager.close()
    assert (PhysicsManager._sim, PhysicsManager._cfg, PhysicsManager._sim_time) == (None, None, 0.0)


def test_ensure_visualization_model_noop_when_backend_is_newton(monkeypatch):
    """When sim backend is Newton, the manager keeps its own model/state untouched."""
    from isaaclab_newton.physics import NewtonManager

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: True))
    NewtonManager._ensure_visualization_model()
    assert NewtonManager._model is None
    assert NewtonManager._state_0 is None


def test_ensure_visualization_model_builds_from_stage_when_backend_is_physx(monkeypatch):
    """With a PhysX sim backend, the shadow Newton model is built directly from the stage."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: _make_env_stage())
    monkeypatch.setattr(nm.PhysicsManager, "_sim", None, raising=False)
    _set_sim_context(monkeypatch, nm)
    monkeypatch.setattr(nm.PhysicsManager, "_device", "cpu", raising=False)

    finalize_calls: list[str] = []

    class _FakeBuilder:
        body_count = 3

        def finalize(self, device):
            finalize_calls.append(device)
            return SimpleNamespace(state=lambda: SimpleNamespace(body_q=None))

    monkeypatch.setattr(nm, "build_visualization_builder_from_stage_envs", lambda *args, **kwargs: _FakeBuilder())

    NewtonManager._ensure_visualization_model()

    assert finalize_calls == ["cpu"]
    assert NewtonManager._model is not None
    assert NewtonManager._state_0 is not None


def test_ensure_visualization_model_empty_builder_supports_marker_only_scene(monkeypatch, caplog):
    """An empty shadow model supports marker-only and geometry-only scenes."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: _make_env_stage())
    monkeypatch.setattr(nm.PhysicsManager, "_sim", None, raising=False)
    _set_sim_context(monkeypatch, nm)

    class _EmptyBuilder:
        body_count = 0

        def finalize(self, device):
            return SimpleNamespace(state=lambda: SimpleNamespace(body_q=None))

    monkeypatch.setattr(nm, "build_visualization_builder_from_stage_envs", lambda *args, **kwargs: _EmptyBuilder())

    with caplog.at_level("INFO"):
        NewtonManager._ensure_visualization_model()

    assert NewtonManager._model is not None
    assert NewtonManager._state_0 is not None
    assert any("no Newton bodies" in r.message for r in caplog.records)


def test_ensure_visualization_model_populates_num_envs_when_backend_is_physx(monkeypatch):
    """Shadow-model build must populate ``_num_envs`` so ``get_num_envs`` is correct under PhysX."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: _make_env_stage(num_envs=4))
    monkeypatch.setattr(nm.PhysicsManager, "_sim", None, raising=False)
    _set_sim_context(monkeypatch, nm)
    monkeypatch.setattr(nm.PhysicsManager, "_device", "cpu", raising=False)

    class _FakeBuilder:
        body_count = 3

        def finalize(self, device):
            return SimpleNamespace(state=lambda: SimpleNamespace(body_q=None))

    monkeypatch.setattr(nm, "build_visualization_builder_from_stage_envs", lambda *args, **kwargs: _FakeBuilder())

    NewtonManager._ensure_visualization_model()

    assert NewtonManager.get_num_envs() == 4
    assert NewtonManager._model.num_envs == 4


def test_ensure_visualization_model_builds_single_world_for_standalone_scene(monkeypatch):
    """A scene outside ``/World/envs`` is imported as one visualization world."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: _make_standalone_stage())
    monkeypatch.setattr(nm.PhysicsManager, "_sim", None, raising=False)
    _set_sim_context(monkeypatch, nm, clone_plan=None)
    monkeypatch.setattr(nm.PhysicsManager, "_device", "cpu", raising=False)

    build_calls = []

    class _FakeBuilder:
        body_count = 1

        def finalize(self, device):
            return SimpleNamespace(state=lambda: SimpleNamespace(body_q=None))

    def _build(stage, env_paths, clone_plan, *, up_axis):
        build_calls.append((stage, env_paths, clone_plan, up_axis))
        return _FakeBuilder()

    monkeypatch.setattr(nm, "build_visualization_builder_from_stage_envs", _build)

    NewtonManager._ensure_visualization_model()

    assert build_calls[0][1:3] == ([], None)
    assert NewtonManager.get_num_envs() == 1
    assert NewtonManager._model.num_envs == 1


def test_ensure_visualization_model_missing_stage_leaves_state_unset(monkeypatch, caplog):
    """When no USD stage is available, model/state stay unset and an error is logged."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: None)

    with caplog.at_level("ERROR"):
        NewtonManager._ensure_visualization_model()

    assert NewtonManager._model is None
    assert NewtonManager._state_0 is None
    assert any("No USD stage available" in r.message for r in caplog.records)


def test_update_visualization_state_noop_when_backend_is_newton(monkeypatch):
    """When sim backend is Newton, update_visualization_state is a no-op."""
    from isaaclab_newton.physics import NewtonManager

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: True))
    monkeypatch.setattr(NewtonManager, "get_scene_data_provider", classmethod(lambda cls: SimpleNamespace()))

    # Pre-set sentinel values to ensure update doesn't touch them.
    NewtonManager._model = "live-model"
    NewtonManager._state_0 = "live-state"
    NewtonManager.update_visualization_state()
    assert NewtonManager._model == "live-model"
    assert NewtonManager._state_0 == "live-state"


@pytest.mark.parametrize("newton_active", [True, False])
def test_get_state_forwards_only_for_live_newton_state(monkeypatch, newton_active):
    """PhysX shadow state keeps its visualization update without entering Newton FK."""
    from isaaclab_newton.physics import NewtonManager

    events: list[str] = []
    state = object()
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", object(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_backend_is_newton",
        classmethod(lambda cls, provider=None: newton_active),
    )
    monkeypatch.setattr(NewtonManager, "forward", classmethod(lambda cls: events.append("forward")))
    monkeypatch.setattr(
        NewtonManager,
        "update_visualization_state",
        classmethod(lambda cls, provider=None: events.append("visualization")),
    )
    monkeypatch.setattr(NewtonManager, "get_state_0", classmethod(lambda cls: state))

    assert NewtonManager.get_state() is state
    expected = ["forward", "visualization"] if newton_active else ["visualization"]
    assert events == expected


def test_scene_data_reads_through_public_state_boundary(monkeypatch):
    """SceneData does not bypass the coherent Newton state accessor."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    events: list[str] = []
    body_q = wp.zeros(1, dtype=wp.transformf, device="cpu")
    state = SimpleNamespace(body_q=body_q)
    backend = nm.NewtonSceneDataBackend()
    monkeypatch.setattr(
        NewtonManager,
        "get_state",
        classmethod(lambda cls, provider=None: events.append("state") or state),
    )

    transforms = backend.transforms

    assert events == ["state"]
    assert transforms.transforms is body_q


def test_resolve_scene_data_body_paths_uses_joint_body_targets():
    """PhysX visualization sync maps Newton joint labels to the actual body prim path."""
    pytest.importorskip("pxr")
    from isaaclab_newton.physics import NewtonManager

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    body_prim = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/robot0_forearm").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/envs/env_0/Robot/joints/robot0_forearm")
    joint.GetBody1Rel().SetTargets([body_prim.GetPath()])

    body_paths = ["/World/envs/env_0/Robot/joints/robot0_forearm"]
    resolved_paths = NewtonManager._resolve_scene_data_body_paths(body_paths, stage)

    assert resolved_paths == ["/World/envs/env_0/Robot/robot0_forearm"]
