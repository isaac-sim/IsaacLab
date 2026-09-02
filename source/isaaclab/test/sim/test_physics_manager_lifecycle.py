# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared physics-manager lifecycle behavior."""

import gc
import weakref
from types import SimpleNamespace

import pytest

from isaaclab.physics import PhysicsCfg, PhysicsEvent, PhysicsManager
from isaaclab.renderers import RendererCfg
from isaaclab.visualizers import VisualizerCfg


def test_backend_registry_uses_only_backend_type():
    """Backend type is both the public and stored identity of a native resource."""
    import inspect

    from isaaclab.sim import SimulationContext

    class Backend:
        def __init__(self, value):
            created.append(value)

    class OtherBackend(Backend):
        pass

    context = object.__new__(SimulationContext)
    context._backend_registry = {}
    context._backend_clone_roles = {}
    created = []

    first = context.get_or_create_backend(Backend, 1, clone_role="physics")
    same = context.get_or_create_backend(Backend, 2, clone_role="model")
    other_type = context.get_or_create_backend(OtherBackend, 4, clone_role="scene")
    same_other_type = context.get_or_create_backend(OtherBackend, 5, clone_role="model")

    assert same is first
    assert same_other_type is other_type is not first
    assert created == [1, 4]
    assert set(context._backend_registry) == {Backend, OtherBackend}
    assert context._backend_clone_roles == {Backend: {"physics", "model"}, OtherBackend: {"scene", "model"}}
    assert "resource_key" not in inspect.signature(SimulationContext.get_or_create_backend).parameters
    cfg_types = (PhysicsCfg, RendererCfg, VisualizerCfg)
    assert all("resource_key" not in cfg_type.__dataclass_fields__ for cfg_type in cfg_types)

    with pytest.raises(ValueError, match="Unknown clone role"):
        context.get_or_create_backend(Backend, clone_role="late")


def test_service_locator_abstraction_is_removed():
    """Backend ownership stays directly on SimulationContext."""
    from pathlib import Path

    from isaaclab.sim import SimulationContext

    sim_package = Path(__file__).parents[2] / "isaaclab" / "sim"
    assert not (sim_package / "service_locator.py").exists()
    assert not hasattr(SimulationContext, "services")


def test_close_runs_all_live_stop_listeners_and_aggregates_failures(monkeypatch):
    """STOP fan-out and shared-state cleanup survive an individual listener failure."""

    class TestManager(PhysicsManager):
        pass

    events = []
    monkeypatch.setattr(TestManager, "_callbacks", {})
    monkeypatch.setattr(TestManager, "_callback_id", 0)
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=TestManager))
    monkeypatch.setattr(PhysicsManager, "_cfg", object())
    monkeypatch.setattr(PhysicsManager, "_sim_time", 1.0)
    monkeypatch.setattr(PhysicsManager, "views", {(TestManager, "/World/Robot"): object()})

    TestManager.register_callback(
        lambda _payload: events.append("first"),
        PhysicsEvent.STOP,
        order=0,
        wrap_weak_ref=False,
    )

    class CollectedListener:
        def callback(self, _payload):
            events.append("collected")

    collected_listener = CollectedListener()
    listener_ref = weakref.ref(collected_listener)
    TestManager.register_callback(collected_listener.callback, PhysicsEvent.STOP, order=1)
    del collected_listener
    gc.collect()
    assert listener_ref() is None

    def failing_listener(_payload):
        events.append("failed")
        raise ReferenceError("listener failure")

    TestManager.register_callback(
        failing_listener,
        PhysicsEvent.STOP,
        order=2,
        wrap_weak_ref=False,
    )
    TestManager.register_callback(
        lambda _payload: events.append("last"),
        PhysicsEvent.STOP,
        order=3,
        wrap_weak_ref=False,
    )

    with pytest.raises(RuntimeError, match=r"1 callback\(s\) failed") as exc_info:
        TestManager.close()

    assert isinstance(exc_info.value.__cause__, ReferenceError)
    assert events == ["first", "failed", "last"]
    assert TestManager._callbacks == {}
    assert PhysicsManager._sim is None
    assert PhysicsManager._cfg is None
    assert PhysicsManager._sim_time == 0.0
    assert PhysicsManager.views == {}


def test_close_surfaces_stop_errors_stored_by_safe_callback_invoke(monkeypatch):
    """STOP failures stored for an external event bus are drained during close."""

    class TestManager(PhysicsManager):
        _callback_exception = None

        @classmethod
        def store_callback_exception(cls, exception):
            cls._callback_exception = exception

        @classmethod
        def raise_callback_exception_if_any(cls):
            if cls._callback_exception is not None:
                exception = cls._callback_exception
                cls._callback_exception = None
                raise exception

    events = []
    monkeypatch.setattr(TestManager, "_callbacks", {})
    monkeypatch.setattr(TestManager, "_callback_id", 0)
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=TestManager))

    def fail_stop(_payload):
        events.append("failed")
        raise ValueError("stored STOP failure")

    TestManager.register_callback(
        lambda payload: PhysicsManager.safe_callback_invoke(
            fail_stop,
            payload,
            physics_manager=TestManager,
        ),
        PhysicsEvent.STOP,
        order=0,
        wrap_weak_ref=False,
    )
    TestManager.register_callback(
        lambda _payload: events.append("last"),
        PhysicsEvent.STOP,
        order=1,
        wrap_weak_ref=False,
    )

    with pytest.raises(RuntimeError, match=r"1 callback\(s\) failed") as exc_info:
        TestManager.close()

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert events == ["failed", "last"]
    assert TestManager._callback_exception is None
    assert TestManager._callbacks == {}
    assert PhysicsManager._sim is None


def test_clear_instance_finishes_teardown_after_physics_close_failure(monkeypatch):
    """A STOP failure is re-raised only after the remaining context teardown."""
    import isaaclab.sim.simulation_context as context_module
    from isaaclab.sim import SimulationContext

    events = []

    class FailingManager:
        @classmethod
        def close(cls):
            events.append("physics")
            raise RuntimeError("STOP failed")

    class Visualizer:
        def __init__(self, name, error=None):
            self.name = name
            self.error = error

        def close(self):
            events.append(self.name)
            if self.error is not None:
                raise self.error

    class Backend:
        def __init__(self, name, error=None):
            self.name = name
            self.error = error

        def clear(self):
            events.append(self.name)
            if self.error is not None:
                raise self.error

    class OtherBackend(Backend):
        pass

    class InvalidBackend:
        pass

    class RenderContext:
        def close(self):
            events.append("renderers")

    context = SimpleNamespace(
        physics_manager=FailingManager,
        _render_context=RenderContext(),
        _visualizers=[
            Visualizer("visualizer_failed", ValueError("visualizer failed")),
            Visualizer("visualizer_last"),
        ],
        _backend_registry={
            Backend: Backend("backend_failed", LookupError("backend failed")),
            OtherBackend: OtherBackend("backend_last"),
            InvalidBackend: InvalidBackend(),
        },
        _backend_clone_roles={Backend: {"physics"}, OtherBackend: {"scene"}},
    )
    monkeypatch.setattr(SimulationContext, "_instance", context)
    monkeypatch.setattr(context_module.stage_utils, "close_stage", lambda: events.append("stage"))
    monkeypatch.setattr(context_module, "clear_resolve_matching_names_cache", lambda: events.append("cache"))
    monkeypatch.setattr(context_module.gc, "collect", lambda: events.append("gc"))

    with pytest.raises(RuntimeError, match=r"3 error\(s\) occurred during teardown") as exc_info:
        SimulationContext.clear_instance()

    assert str(exc_info.value) == (
        "SimulationContext.clear_instance(): 3 error(s) occurred during teardown: "
        "RuntimeError: STOP failed; ValueError: visualizer failed; LookupError: backend failed"
    )
    assert str(exc_info.value.__cause__) == "STOP failed"
    assert events == [
        "physics",
        "renderers",
        "visualizer_failed",
        "visualizer_last",
        "backend_failed",
        "backend_last",
        "stage",
        "cache",
        "gc",
    ]
    assert context._visualizers == []
    assert context._backend_registry == {}
    assert context._backend_clone_roles == {}
    assert SimulationContext.instance() is None


def test_clear_instance_drops_owned_context_references_before_garbage_collection(monkeypatch):
    """The singleton and method-local context references are gone before garbage collection."""
    import isaaclab.sim.simulation_context as context_module
    from isaaclab.sim import SimulationContext

    class Manager:
        @classmethod
        def close(cls):
            pass

    class RenderContext:
        def close(self):
            pass

    class Context:
        pass

    context = Context()
    context.physics_manager = Manager
    context._render_context = RenderContext()
    context._visualizers = []
    context._backend_registry = {}
    context._backend_clone_roles = {}
    context_ref = weakref.ref(context)
    context_alive_during_gc = []

    monkeypatch.setattr(SimulationContext, "_instance", context)
    monkeypatch.setattr(context_module.stage_utils, "close_stage", lambda: None)
    monkeypatch.setattr(context_module, "clear_resolve_matching_names_cache", lambda: None)
    monkeypatch.setattr(
        context_module.gc,
        "collect",
        lambda: context_alive_during_gc.append(context_ref() is not None),
    )
    del context

    SimulationContext.clear_instance()

    assert context_alive_during_gc == [False]
    assert context_ref() is None
