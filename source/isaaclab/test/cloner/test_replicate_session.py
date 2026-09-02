# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for clone-plan routing and dispatch without a simulator runtime."""

from types import SimpleNamespace

import pytest
import torch

import isaaclab.cloner.clone_plan as clone_plan
import isaaclab.cloner.replicate_session as replicate_session
from isaaclab.cloner import ClonePlan, UsdReplicateContext, make_clone_plan
from isaaclab.sim import SimulationContext


class _Context:
    replicate_priority = 0

    def __init__(self, calls):
        self.calls = calls

    def replicate(self, plan):
        self.calls.append((type(self), plan))


def _plan(*context_types):
    return ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        env_ids=torch.arange(2),
        positions=torch.zeros((2, 3)),
        context_rows={context_type: (0,) for context_type in context_types},
    )


def test_make_clone_plan_routes_default_and_explicit_contexts(monkeypatch):
    """Planning records the rows consumed by default and explicit clone contexts."""
    calls = []

    class Unrelated(_Context):
        pass

    simulation = SimpleNamespace(
        physics_manager=SimpleNamespace(clone_context_type=_Context),
        _backend_registry={_Context: _Context(calls), Unrelated: Unrelated(calls)},
        stage=object(),
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    monkeypatch.setattr(clone_plan, "has_kit", lambda: False)
    cfg = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Robot", spawn=SimpleNamespace(spawn_path=None), cloning_contexts=None
    )

    plan = make_clone_plan((cfg,), 2, 1.0, "cpu")

    assert plan.context_rows == {_Context: (0,)}

    class Explicit(_Context):
        pass

    cfg.cloning_contexts = (Explicit,)
    assert make_clone_plan((cfg,), 2, 1.0, "cpu").context_rows == {Explicit: (0,)}


def test_replicate_dispatches_the_same_plan_in_priority_order(monkeypatch):
    """Registered contexts receive one shared plan in backend priority order."""
    calls = []

    class Late(_Context):
        replicate_priority = 1

    class Early(_Context):
        replicate_priority = -1

    plan = _plan(Late, Early)
    simulation = SimpleNamespace(
        physics_manager=SimpleNamespace(clone_context_type=Late),
        _backend_registry={Late: Late(calls), Early: Early(calls)},
        set_clone_plan=lambda value: calls.append(value),
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    replicate_session.replicate(plan)

    assert calls == [(Early, plan), (Late, plan), plan]


def test_replicate_physics_false_runs_only_usd(monkeypatch):
    """Disabling physics replication preserves only USD authoring."""
    calls = []

    class Physics(_Context):
        pass

    class Usd(_Context):
        pass

    plan = _plan(Physics, UsdReplicateContext)
    simulation = SimpleNamespace(
        physics_manager=SimpleNamespace(clone_context_type=Physics),
        _backend_registry={UsdReplicateContext: Usd(calls)},
        set_clone_plan=lambda value: None,
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    replicate_session.replicate(plan, replicate_physics=False)

    assert calls == [(Usd, plan)]


def test_replicate_rejects_unregistered_context(monkeypatch):
    """A routed context must register before dispatch rather than using a fallback."""
    plan = _plan(_Context)
    simulation = SimpleNamespace(
        physics_manager=SimpleNamespace(clone_context_type=_Context),
        _backend_registry={},
        set_clone_plan=lambda _: None,
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with pytest.raises(RuntimeError, match="must be registered"):
        replicate_session.replicate(plan)
