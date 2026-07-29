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
        def close(self):
            events.append("visualizer")

    class Services:
        def close_all(self, caught_exceptions):
            assert caught_exceptions == []
            events.append("services")

    context = SimpleNamespace(
        physics_manager=FailingManager,
        _visualizers=[Visualizer()],
        _services=Services(),
    )
    monkeypatch.setattr(SimulationContext, "_instance", context)
    monkeypatch.setattr(context_module.stage_utils, "close_stage", lambda: events.append("stage"))
    monkeypatch.setattr(context_module, "clear_resolve_matching_names_cache", lambda: events.append("cache"))
    monkeypatch.setattr(context_module.gc, "collect", lambda: events.append("gc"))

    with pytest.raises(RuntimeError, match="STOP failed"):
        SimulationContext.clear_instance()

    assert events == ["physics", "visualizer", "services", "stage", "cache", "gc"]
    assert SimulationContext.instance() is None
