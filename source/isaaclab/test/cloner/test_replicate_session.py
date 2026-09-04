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
    empty = make_clone_plan((), 2, 1.0, global_paths=("/World/Ground",))
    assert empty.context_rows == {_Context: ()}
    assert empty.global_paths == ("/World/Ground",)


def test_queue_accepts_only_cfgs_owned_by_published_plan(monkeypatch):
    """Cfg-first constructors cannot escape the published plan."""
    planned = SimpleNamespace(prim_path="/World/envs/env_[^/]+/Robot", spawn=object())
    unplanned = SimpleNamespace(prim_path="/World/envs/env_[^/]+/Object", spawn=object())
    covered_reference = SimpleNamespace(prim_path="/World/envs/env_[^/]+/Existing", spawn=None)
    unplanned_reference = SimpleNamespace(prim_path="/World/Outside", spawn=None)
    plan = _plan()
    plan.cfg_rows[id(planned)] = (0,)
    simulation = SimpleNamespace(get_clone_plan=lambda: plan, _clone_plan_consumed=False)
    replicate_session.REPLICATION_QUEUE.clear()
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    replicate_session.queue_replication(planned)
    with pytest.raises(RuntimeError, match="not owned"):
        replicate_session.queue_replication(unplanned)
    simulation._clone_plan_consumed = True
    replicate_session.queue_replication(covered_reference)
    with pytest.raises(RuntimeError, match="not owned"):
        replicate_session.queue_replication(unplanned_reference)

    assert replicate_session.REPLICATION_QUEUE == []


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
        _clone_plan=None,
        _clone_plan_consumed=False,
    )
    simulation.get_clone_plan = lambda: simulation._clone_plan
    simulation.set_clone_plan = lambda value: SimulationContext.set_clone_plan(simulation, value)
    simulation._consume_clone_plan = lambda value: SimulationContext._consume_clone_plan(simulation, value)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    replicate_session.replicate(plan)

    assert calls == [(Early, plan), (Late, plan)]
    assert simulation._clone_plan is plan


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
        _clone_plan=plan,
        get_clone_plan=lambda: plan,
        _clone_plan_consumed=False,
    )
    simulation.set_clone_plan = lambda value: SimulationContext.set_clone_plan(simulation, value)
    simulation._consume_clone_plan = lambda value: SimulationContext._consume_clone_plan(simulation, value)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    replicate_session.replicate(plan, replicate_physics=False)

    assert calls == [(Usd, plan)]


def test_replicate_rejects_unregistered_context(monkeypatch):
    """A routed context must register before dispatch rather than using a fallback."""
    plan = _plan(_Context)
    simulation = SimpleNamespace(
        physics_manager=SimpleNamespace(clone_context_type=_Context),
        _backend_registry={},
        _clone_plan=plan,
        get_clone_plan=lambda: plan,
        _clone_plan_consumed=False,
    )
    simulation.set_clone_plan = lambda value: SimulationContext.set_clone_plan(simulation, value)
    simulation._consume_clone_plan = lambda value: SimulationContext._consume_clone_plan(simulation, value)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with pytest.raises(RuntimeError, match="must be registered"):
        replicate_session.replicate(plan)
    assert simulation._clone_plan_consumed is False


def test_replicate_rejects_a_second_dispatch(monkeypatch):
    """One published plan has exactly one backend dispatch."""
    plan = _plan()
    simulation = SimpleNamespace(_clone_plan=plan, get_clone_plan=lambda: plan, _clone_plan_consumed=True)
    simulation.set_clone_plan = lambda value: SimulationContext.set_clone_plan(simulation, value)
    simulation._consume_clone_plan = lambda value: SimulationContext._consume_clone_plan(simulation, value)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    replicate_session.REPLICATION_QUEUE.append(object())

    with pytest.raises(RuntimeError, match="consumed"):
        replicate_session.replicate(plan)
    assert replicate_session.REPLICATION_QUEUE == []
    with pytest.raises(RuntimeError, match="consumed"):
        simulation.set_clone_plan(None)
