# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``NewtonManager.update_visualization_state`` and shadow-model build.

When the active sim backend is PhysX and a Newton-native visualizer/renderer is in
use, :meth:`NewtonManager._ensure_visualization_model` must build the manager's
``_model`` / ``_state_0`` directly from the USD stage (via
:meth:`NewtonManager._build_visualization_model_from_stage`), and
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
    NewtonManager._visualization_scene_data = None
    NewtonManager._visualization_mapping = None


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
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(nm, "replace_newton_shape_colors", lambda model, *a, **kw: 0)

    finalize_calls: list[str] = []

    class _FakeBuilder:
        body_count = 3

        def finalize(self, device):
            finalize_calls.append(device)
            return SimpleNamespace(state=lambda: SimpleNamespace(body_q=None))

    monkeypatch.setattr(
        NewtonManager,
        "_build_visualization_model_from_stage",
        classmethod(lambda cls, stage: _FakeBuilder()),
    )
    monkeypatch.setattr(nm.PhysicsManager, "_device", "cpu", raising=False)

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
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: SimpleNamespace())

    class _EmptyBuilder:
        body_count = 0

    monkeypatch.setattr(
        NewtonManager,
        "_build_visualization_model_from_stage",
        classmethod(lambda cls, stage: _EmptyBuilder()),
    )

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
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(nm, "replace_newton_shape_colors", lambda model, *a, **kw: 0)

    class _FakeBuilder:
        body_count = 3

        def finalize(self, device):
            return SimpleNamespace(state=lambda: SimpleNamespace(body_q=None))

    def _fake_build(cls, stage):
        # Mirror the real shadow-build behaviour: writes the env count discovered on the stage.
        NewtonManager._num_envs = 4
        return _FakeBuilder()

    monkeypatch.setattr(NewtonManager, "_build_visualization_model_from_stage", classmethod(_fake_build))
    monkeypatch.setattr(nm.PhysicsManager, "_device", "cpu", raising=False)

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

    # Pre-set sentinel values to ensure update doesn't touch them.
    NewtonManager._model = "live-model"
    NewtonManager._state_0 = "live-state"
    NewtonManager.update_visualization_state()
    assert NewtonManager._model == "live-model"
    assert NewtonManager._state_0 == "live-state"
