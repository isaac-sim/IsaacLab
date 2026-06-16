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


def _set_sim_context(monkeypatch, nm, clone_plan=None, scene_data_provider=None):
    clone_plan = clone_plan if clone_plan is not None else SimpleNamespace()
    scene_data_provider = scene_data_provider if scene_data_provider is not None else SimpleNamespace()
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
    monkeypatch.setattr(nm, "replace_newton_shape_colors", lambda model, *a, **kw: 0)

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


def test_ensure_visualization_model_empty_builder_logs_and_skips(monkeypatch, caplog):
    """When the stage walk produces no bodies, model/state stay unset and an error is logged."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: _make_env_stage())
    monkeypatch.setattr(nm.PhysicsManager, "_sim", None, raising=False)
    _set_sim_context(monkeypatch, nm)

    class _EmptyBuilder:
        body_count = 0

    monkeypatch.setattr(nm, "build_visualization_builder_from_stage_envs", lambda *args, **kwargs: _EmptyBuilder())

    with caplog.at_level("ERROR"):
        NewtonManager._ensure_visualization_model()

    assert NewtonManager._model is None
    assert NewtonManager._state_0 is None
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
    monkeypatch.setattr(nm, "replace_newton_shape_colors", lambda model, *a, **kw: 0)

    class _FakeBuilder:
        body_count = 3

        def finalize(self, device):
            return SimpleNamespace(state=lambda: SimpleNamespace(body_q=None))

    monkeypatch.setattr(nm, "build_visualization_builder_from_stage_envs", lambda *args, **kwargs: _FakeBuilder())

    NewtonManager._ensure_visualization_model()

    assert NewtonManager.get_num_envs() == 4
    assert NewtonManager._model.num_envs == 4


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


def test_resolve_scene_data_body_paths_uses_joint_body_targets():
    """PhysX visualization sync maps Newton joint labels to the actual body prim path."""
    import pytest

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
