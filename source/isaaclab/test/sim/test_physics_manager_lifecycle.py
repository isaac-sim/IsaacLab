# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared physics-manager lifecycle behavior."""

import gc
import inspect
import weakref
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab.physics import PhysicsCfg, PhysicsEvent, PhysicsManager
from isaaclab.renderers import RenderContext, RendererCfg
from isaaclab.visualizers import VisualizerCfg


def test_backend_registry_identity_and_lifecycle():
    """Share by type/cfg, construct once, and release only the selected resource after successful cleanup."""
    from isaaclab.sim import BackendCfg, SimulationContext

    @dataclass(kw_only=True)
    class Cfg(BackendCfg):
        values: list[int]

        def __deepcopy__(self, memo):
            pytest.fail("Registering a finalized cfg must not copy it.")

    @dataclass(kw_only=True)
    class OtherCfg(Cfg):
        pass

    class Backend:
        def __init__(self, cfg):
            if not cfg.values:
                raise ValueError("construction failed")
            self.cfg = cfg
            self.close = Mock()

        def __eq__(self, other):
            return isinstance(other, Backend)

    class OtherBackend(Backend):
        pass

    context = object.__new__(SimulationContext)
    context._backend_registry = []
    cfg = Cfg(class_type=Backend, values=[1])
    with pytest.raises(ValueError, match="construction failed"):
        context.get_or_create_backend(replace(cfg, values=[]))
    assert not context._backend_registry

    first = context.get_or_create_backend(cfg)
    assert first.cfg is cfg
    different_cfg = replace(cfg, values=[1, 2])
    second = context.get_or_create_backend(different_cfg)
    other_cfg = context.get_or_create_backend(OtherCfg(class_type=Backend, values=[1]))
    other_type = context.get_or_create_backend(replace(cfg, class_type=OtherBackend))
    assert len({id(resource) for resource in (first, second, other_cfg, other_type)}) == 4
    assert context.get_or_create_backend(Cfg(class_type=Backend, values=[1])) is first

    context.close_backend(first)
    first.close.assert_called_once_with()
    assert all(resource.close.call_count == 0 for resource in (second, other_cfg, other_type))
    replacement = context.get_or_create_backend(cfg)
    assert replacement is not first
    with pytest.raises(KeyError):
        context.close_backend(first)
    replacement.close.assert_not_called()

    second.close.side_effect = RuntimeError("release failed")
    with pytest.raises(RuntimeError, match="release failed"):
        context.close_backend(second)
    assert context.get_or_create_backend(different_cfg) is second
    second.close.side_effect = None
    context.close_backend(second)
    assert second.close.call_count == 2


def test_backend_ownership_has_no_service_locator_or_resource_keys():
    """Backend ownership stays directly on SimulationContext and uses cfg identity, not custom keys."""
    from pathlib import Path

    from isaaclab.sim import BackendCfg, SimulationContext

    sim_package = Path(__file__).parents[2] / "isaaclab" / "sim"
    assert not (sim_package / "service_locator.py").exists()
    assert not hasattr(SimulationContext, "services")
    assert issubclass(RendererCfg, BackendCfg)
    assert not hasattr(RenderContext, "get_renderer")
    assert "_renderer_entries" not in RenderContext.__slots__
    assert tuple(inspect.signature(SimulationContext.get_or_create_backend).parameters) == ("self", "cfg")
    cfg_types = (PhysicsCfg, RendererCfg, VisualizerCfg)
    assert all("resource_key" not in cfg_type.__dataclass_fields__ for cfg_type in cfg_types)


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


@pytest.mark.parametrize("renderer_first", [False, True])
@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_clear_instance_closes_renderers_before_native_backends(monkeypatch, renderer_first, fail_cleanup):
    """Creation order and cleanup failures do not change ownership or teardown phases."""
    import isaaclab.sim.simulation_context as context_module
    from isaaclab.sim import BackendCfg, SimulationContext

    events = []

    class Manager:
        @classmethod
        def close(cls):
            events.append("physics")
            if fail_cleanup:
                raise RuntimeError("STOP failed")

    class Resource:
        def __init__(self, name, error=None):
            self.name = name
            self.error = error

        def close(self):
            events.append(self.name)
            if fail_cleanup and self.error is not None:
                raise self.error

    context = object.__new__(SimulationContext)
    context.physics_manager = Manager
    context._backend_registry = []
    context._render_context = RenderContext(context._backend_registry)
    context._render_context._visual_material_writers = (Resource("writers"),)
    context._visualizers = [Resource("visualizer_failed", ValueError("visualizer failed")), Resource("visualizer_last")]
    context._pending_visualizers = [Resource("visualizer_pending")]
    context.clone_contexts = {object: object()}
    groups = (
        (RendererCfg, (("renderer_failed", OSError("renderer failed")), ("renderer_last", None))),
        (BackendCfg, (("backend_failed", LookupError("backend failed")), ("backend_last", None))),
    )
    for cfg_type, resources in groups if renderer_first else reversed(groups):
        for name, error in resources:
            context.get_or_create_backend(
                cfg_type(class_type=lambda cfg, name=name, error=error: Resource(name, error))
            )
    monkeypatch.setattr(SimulationContext, "_instance", context)
    monkeypatch.setattr(context_module.stage_utils, "close_stage", lambda: events.append("stage"))
    monkeypatch.setattr(context_module, "clear_resolve_matching_names_cache", lambda: events.append("cache"))
    monkeypatch.setattr(context_module.gc, "collect", lambda: events.append("gc"))

    if fail_cleanup:
        with pytest.raises(RuntimeError, match=r"4 error\(s\) occurred during teardown") as exc_info:
            SimulationContext.clear_instance()
        assert str(exc_info.value) == (
            "SimulationContext.clear_instance(): 4 error(s) occurred during teardown: "
            "RuntimeError: STOP failed; OSError: renderer failed; ValueError: visualizer failed; "
            "LookupError: backend failed"
        )
        assert str(exc_info.value.__cause__) == "STOP failed"
    else:
        SimulationContext.clear_instance()

    SimulationContext.clear_instance()
    assert events == [
        "physics",
        "writers",
        "renderer_failed",
        "renderer_last",
        "visualizer_failed",
        "visualizer_last",
        "visualizer_pending",
        "backend_failed",
        "backend_last",
        "stage",
        "cache",
        "gc",
    ]
    assert context._visualizers == []
    assert context._pending_visualizers == []
    assert context._backend_registry == []
    assert context.clone_contexts == {}
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
    context._pending_visualizers = []
    context._backend_registry = []
    context.clone_contexts = {}
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


def _stub_context_for_step(recorder: list[str]):
    """Build the minimum ``SimulationContext`` surface :meth:`SimulationContext.step` touches."""
    from isaaclab.sim import SimulationContext

    context = object.__new__(SimulationContext)
    context._physics_step_count = 0
    context.physics_manager = SimpleNamespace(
        wait_for_playing=lambda: recorder.append("wait"),
        step=lambda: recorder.append("step"),
    )
    return context


def test_step_calls_physics_manager_in_order(monkeypatch):
    """``SimulationContext.step`` does not know about profiling -- that lives in ``PhysicsManager.step``.

    It must still wait for the timeline, step physics, and bump the step count in order.
    """
    from isaaclab.sim import SimulationContext

    calls: list[str] = []
    context = _stub_context_for_step(calls)

    SimulationContext.step(context, render=False)

    assert calls == ["wait", "step"]
    assert context._physics_step_count == 1


def test_physics_manager_step_prints_timing_line_when_profile_enabled(monkeypatch, capsys):
    """The printed line must match the format ``scripts/benchmarks/benchmark_renderer.py`` parses."""
    import re

    from isaaclab.physics import physics_manager as physics_manager_module

    class TestManager(PhysicsManager):
        @classmethod
        def _step(cls):
            pass

    monkeypatch.setattr(physics_manager_module, "_PHYSICS_PROFILE_ENABLED", True)

    TestManager.step()

    assert re.search(
        rf"{re.escape(physics_manager_module.PHYSICS_PROFILE_SCOPE)} took [\d.]+ ms", capsys.readouterr().out
    )


def test_physics_manager_step_prints_nothing_when_profile_disabled(monkeypatch, capsys):
    """Profiling is off by default, so an ordinary run pays neither the print nor the sync."""
    from isaaclab.physics import physics_manager as physics_manager_module

    class TestManager(PhysicsManager):
        @classmethod
        def _step(cls):
            pass

    monkeypatch.setattr(physics_manager_module, "_PHYSICS_PROFILE_ENABLED", False)

    TestManager.step()

    assert physics_manager_module.PHYSICS_PROFILE_SCOPE not in capsys.readouterr().out
