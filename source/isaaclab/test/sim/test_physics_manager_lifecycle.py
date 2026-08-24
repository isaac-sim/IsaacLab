# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared physics-manager lifecycle behavior."""

import gc
import weakref
from types import SimpleNamespace

import pytest

from isaaclab.physics import PhysicsEvent, PhysicsManager


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

    class Services:
        def close_all(self, caught_exceptions):
            assert caught_exceptions == []
            events.append("services")

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
        _services=Services(),
    )
    monkeypatch.setattr(SimulationContext, "_instance", context)
    monkeypatch.setattr(context_module.stage_utils, "close_stage", lambda: events.append("stage"))
    monkeypatch.setattr(context_module, "clear_resolve_matching_names_cache", lambda: events.append("cache"))
    monkeypatch.setattr(context_module.gc, "collect", lambda: events.append("gc"))

    with pytest.raises(RuntimeError, match=r"2 error\(s\) occurred during teardown") as exc_info:
        SimulationContext.clear_instance()

    assert str(exc_info.value) == (
        "SimulationContext.clear_instance(): 2 error(s) occurred during teardown: "
        "RuntimeError: STOP failed; ValueError: visualizer failed"
    )
    assert str(exc_info.value.__cause__) == "STOP failed"
    assert events == [
        "physics",
        "renderers",
        "visualizer_failed",
        "visualizer_last",
        "services",
        "stage",
        "cache",
        "gc",
    ]
    assert context._visualizers == []
    assert SimulationContext.instance() is None


def test_clear_instance_drops_owned_context_references_before_garbage_collection(monkeypatch):
    """The singleton and method-local context references are gone before garbage collection."""
    import isaaclab.sim.simulation_context as context_module
    from isaaclab.sim import SimulationContext

    class Manager:
        @classmethod
        def close(cls):
            pass

    class Services:
        def close_all(self, caught_exceptions):
            assert caught_exceptions == []

    class RenderContext:
        def close(self):
            pass

    class Context:
        pass

    context = Context()
    context.physics_manager = Manager
    context._render_context = RenderContext()
    context._visualizers = []
    context._services = Services()
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
