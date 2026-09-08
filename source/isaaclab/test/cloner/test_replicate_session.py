# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for clone-plan routing and dispatch without a simulator runtime."""

from types import SimpleNamespace

import numpy as np
import pytest

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
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2, dtype=np.int64),
        positions=np.zeros((2, 3), dtype=np.float32),
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

    plan = make_clone_plan((cfg,), 2, 1.0)

    assert plan.context_rows == {_Context: (0,)}

    class Explicit(_Context):
        pass

    cfg.cloning_contexts = (Explicit,)
    assert make_clone_plan((cfg,), 2, 1.0).context_rows == {Explicit: (0,)}


def test_queue_collects_only_before_plan_publication(monkeypatch):
    """Post-construction planning ignores cfgs built after a plan is active."""
    cfg = object()
    plan = _plan()
    published = None
    simulation = SimpleNamespace(get_clone_plan=lambda: published)
    replicate_session.REPLICATION_QUEUE.clear()
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    replicate_session.queue_replication(cfg)
    assert [cfg] == replicate_session.REPLICATION_QUEUE

    published = plan
    replicate_session.queue_replication(object())
    assert [cfg] == replicate_session.REPLICATION_QUEUE


@pytest.mark.parametrize("valid_set", [np.asarray([["0"]]), np.asarray([[0 + 1j]])])
def test_make_clone_plan_rejects_non_integer_combinations(valid_set):
    """Prototype indices must be integer data rather than values NumPy can coerce to integers."""
    cfg = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Robot", spawn=SimpleNamespace(spawn_path=None), cloning_contexts=None
    )

    with pytest.raises(ValueError, match="integer prototype indices"):
        make_clone_plan((cfg,), 2, 1.0, valid_set=valid_set)


def test_grid_transforms_always_returns_float32():
    """NumPy scalar inputs do not widen the public transform arrays."""
    positions, orientations = clone_plan.grid_transforms(2, np.float64(1.0))

    assert positions.dtype == orientations.dtype == np.float32


def test_replicate_dispatches_the_same_plan_in_priority_order(monkeypatch):
    """The manager applies filters after stage authoring and before physics replication."""
    calls = []

    class Late(_Context):
        replicate_priority = 1

    class Early(_Context):
        replicate_priority = -1

    plan = _plan(Late, Early)
    published = []
    manager = SimpleNamespace(
        clone_context_type=Late,
        apply_collision_filter=lambda received, *, isolate_environments, replicate_physics: calls.append(
            ("collision_filter", received, isolate_environments, replicate_physics)
        ),
    )
    simulation = SimpleNamespace(
        physics_manager=manager,
        _backend_registry={Late: Late(calls), Early: Early(calls)},
        get_clone_plan=lambda: None,
        set_clone_plan=published.append,
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    replicate_session.replicate(plan)

    assert calls == [(Early, plan), ("collision_filter", plan, True, True), (Late, plan)]
    assert published == [plan]


def test_replicate_physics_false_runs_only_usd(monkeypatch):
    """Disabling physics replication preserves only USD authoring."""
    calls = []

    class Physics(_Context):
        pass

    class Usd(_Context):
        pass

    plan = _plan(Physics, UsdReplicateContext)
    applied = []
    simulation = SimpleNamespace(
        physics_manager=SimpleNamespace(
            clone_context_type=Physics,
            apply_collision_filter=lambda received, *, isolate_environments, replicate_physics: applied.append(
                (received, isolate_environments, replicate_physics)
            ),
        ),
        _backend_registry={UsdReplicateContext: Usd(calls)},
        get_clone_plan=lambda: plan,
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    replicate_session.replicate(plan, replicate_physics=False)

    assert calls == [(Usd, plan)]
    assert applied == [(plan, True, False)]


def test_replicate_rejects_unregistered_context(monkeypatch):
    """A routed context must register before dispatch rather than using a fallback."""
    plan = _plan(_Context)
    simulation = SimpleNamespace(
        physics_manager=SimpleNamespace(clone_context_type=_Context),
        _backend_registry={},
        get_clone_plan=lambda: plan,
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with pytest.raises(RuntimeError, match="must be registered"):
        replicate_session.replicate(plan)


@pytest.mark.parametrize("kwargs", [{"replicate_physics": 1}, {"isolate_environments": 1}])
def test_replicate_validates_options_before_stage_dispatch(monkeypatch, kwargs):
    """Invalid barrier options cannot leave partially cloned USD topology."""
    calls = []

    class Early(_Context):
        replicate_priority = -1

    plan = _plan(Early)
    simulation = SimpleNamespace(
        physics_manager=SimpleNamespace(),
        _backend_registry={Early: Early(calls)},
        get_clone_plan=lambda: plan,
    )
    queued = [object()]
    monkeypatch.setattr(replicate_session, "REPLICATION_QUEUE", queued)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with pytest.raises(TypeError, match="must be a bool"):
        replicate_session.replicate(plan, **kwargs)

    assert calls == []
    assert queued == replicate_session.REPLICATION_QUEUE


def test_replicate_validates_plan_shape_before_stage_dispatch(monkeypatch):
    """A malformed plan cannot leave partially cloned USD topology."""
    calls = []

    class Early(_Context):
        replicate_priority = -1

    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 1), dtype=np.bool_),
        env_ids=np.arange(2, dtype=np.int64),
        context_rows={Early: (0,)},
    )
    simulation = SimpleNamespace(
        physics_manager=SimpleNamespace(),
        _backend_registry={Early: Early(calls)},
        get_clone_plan=lambda: plan,
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with pytest.raises(ValueError, match=r"clone_mask must have shape \(1, 2\)"):
        replicate_session.replicate(plan)

    assert calls == []
