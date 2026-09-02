# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for clone-context dispatch without a simulator runtime."""

from types import SimpleNamespace

import pytest
import torch

import isaaclab.cloner.replicate_session as replicate_session
from isaaclab.cloner import ClonePlan, UsdReplicateContext, clone_plan_from_env_0, make_clone_plan
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import SimulationContext


def test_sensor_default_does_not_request_a_cloning_context():
    """Sensors rely on automatic Kit replication unless a user explicitly overrides it."""

    assert SensorBaseCfg().cloning_contexts == ()


def _plan(*context_types: type) -> ClonePlan:
    """Build one routed row for dispatch tests."""
    return ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/Ground"),
        destinations=("/World/envs/env_{}/Robot", "/World/Ground"),
        clone_mask=torch.tensor([[True, True], [False, False]]),
        env_ids=torch.arange(2),
        positions=torch.zeros((2, 3)),
        context_rows={context_type: (0,) for context_type in context_types},
    )


def test_replicate_dispatches_only_the_plan(monkeypatch):
    """Dispatch passes the immutable plan instead of rebuilding parallel backend arguments."""

    class Context:
        replicate_priority = 0

        def __init__(self):
            self.plans = []

        def replicate(self, plan):
            self.plans.append(plan)

    context = Context()
    simulation = SimpleNamespace(
        _backend_registry={Context: context}, _backend_clone_roles={Context: {"physics"}}, clone_plan=None
    )
    simulation.set_clone_plan = lambda plan: setattr(simulation, "clone_plan", plan)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    replicate_session.REPLICATION_QUEUE.append(object())
    plan = _plan(Context)

    replicate_session.replicate(plan)

    assert context.plans == [plan]
    assert simulation.clone_plan is plan
    assert replicate_session.REPLICATION_QUEUE == []


def test_replicate_orders_contexts_by_priority(monkeypatch):
    """USD-like scene work runs before contexts that consume the cloned stage."""
    calls = []

    class LateContext:
        replicate_priority = 10

        def replicate(self, plan):
            calls.append("late")

    class EarlyContext:
        replicate_priority = -10

        def replicate(self, plan):
            calls.append("early")

    simulation = SimpleNamespace(
        _backend_registry={LateContext: LateContext(), EarlyContext: EarlyContext()},
        _backend_clone_roles={LateContext: {"scene"}, EarlyContext: {"scene"}},
    )
    simulation.set_clone_plan = lambda plan: None
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    replicate_session.replicate(_plan(LateContext, EarlyContext))

    assert calls == ["early", "late"]


def test_explicit_usd_context_registers_without_kit(monkeypatch):
    """An explicit USD route does not depend on automatic Kit detection."""
    simulation = SimpleNamespace(_backend_registry={}, _backend_clone_roles={}, stage=object())
    simulation.get_or_create_backend = lambda backend_type, *args, **kwargs: SimulationContext.get_or_create_backend(
        simulation, backend_type, *args, **kwargs
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    monkeypatch.setattr("isaaclab.cloner.clone_plan.has_kit", lambda: False)
    cfg = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Robot",
        spawn=SimpleNamespace(spawn_path=None),
        cloning_contexts=(UsdReplicateContext,),
    )

    plan = make_clone_plan((cfg,), 2, 1.0, "cpu")

    assert plan.context_rows[UsdReplicateContext] == (0,)
    assert isinstance(simulation._backend_registry[UsdReplicateContext], UsdReplicateContext)


def test_clone_plan_rejects_non_class_contexts(monkeypatch):
    """Context routing fails during planning instead of later dispatch."""
    monkeypatch.setattr(SimulationContext, "instance", lambda: None)
    cfg = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Robot",
        spawn=SimpleNamespace(spawn_path=None),
        cloning_contexts=(lambda: None,),
    )

    with pytest.raises(TypeError, match="must contain only context classes"):
        make_clone_plan((cfg,), 2, 1.0, "cpu")


def test_cfg_plan_does_not_route_unowned_physics_context(monkeypatch):
    """Missing cfg ownership cannot silently broaden a backend to every plan row."""

    class PhysicsContext:
        pass

    simulation = SimpleNamespace(
        _backend_registry={PhysicsContext: PhysicsContext()},
        _backend_clone_roles={PhysicsContext: {"physics"}},
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    monkeypatch.setattr("isaaclab.cloner.clone_plan.has_kit", lambda: False)
    cfg = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Robot",
        spawn=SimpleNamespace(spawn_path=None),
        cloning_contexts=(),
    )

    plan = make_clone_plan((cfg,), 2, 1.0, "cpu")

    assert plan.context_rows[PhysicsContext] == ()


def test_env_0_plan_explicitly_routes_whole_environment_contexts(monkeypatch):
    """The post-construction whole-env builder owns its root row without a cfg fallback."""

    class PhysicsContext:
        pass

    simulation = SimpleNamespace(
        _backend_registry={PhysicsContext: PhysicsContext()},
        _backend_clone_roles={PhysicsContext: {"physics"}},
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    plan = clone_plan_from_env_0("/World/envs/env_0", "/World/envs/env_{}", 2, "cpu", torch.zeros((2, 3)))

    assert plan.context_rows[PhysicsContext] == (0,)


def test_replicate_physics_false_keeps_scene_roles(monkeypatch):
    """The outer lifecycle switch skips physics-only contexts without hiding scene consumers."""

    class PhysicsContext:
        replicate_priority = 0

        def replicate(self, plan):
            pytest.fail("physics-only context was dispatched")

    class SceneContext:
        replicate_priority = 1

        def __init__(self):
            self.plan = None

        def replicate(self, plan):
            self.plan = plan

    physics, scene = PhysicsContext(), SceneContext()
    simulation = SimpleNamespace(
        _backend_registry={PhysicsContext: physics, SceneContext: scene},
        _backend_clone_roles={PhysicsContext: {"physics"}, SceneContext: {"scene"}},
    )
    simulation.set_clone_plan = lambda plan: None
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    plan = _plan(PhysicsContext, SceneContext)

    replicate_session.replicate(plan, replicate_physics=False)

    assert scene.plan is plan


def test_replicate_rejects_unregistered_context(monkeypatch):
    """A routed context must be registered before atomic dispatch starts."""

    class Context:
        pass

    simulation = SimpleNamespace(_backend_registry={}, _backend_clone_roles={})
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with pytest.raises(RuntimeError, match="must be registered"):
        replicate_session.replicate(_plan(Context))
