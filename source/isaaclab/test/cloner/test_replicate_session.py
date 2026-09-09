# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for clone-plan routing and dispatch without a simulator runtime."""

from dataclasses import replace
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

import isaaclab.cloner.clone_plan as clone_plan
import isaaclab.cloner.replicate_session as replicate_session
from isaaclab.cloner import CloneCfg, ClonePlan, UsdReplicateContext, make_clone_plan
from isaaclab.scene import InteractiveSceneCfg
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
        apply_collision_filter=lambda received: calls.append(
            ("collision_filter", received, received.isolate_environments, received.replicate_physics)
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


@pytest.mark.parametrize("legacy_override", [False, True])
def test_replicate_uses_plan_policy_and_legacy_override(monkeypatch, legacy_override):
    """Plan policy is canonical; the compatibility override publishes a copied plan."""
    calls = []

    class Physics(_Context):
        pass

    class Usd(_Context):
        pass

    plan = _plan(Physics, UsdReplicateContext)
    if not legacy_override:
        plan = replace(plan, replicate_physics=False)
    applied = []
    published = []
    simulation = SimpleNamespace(
        physics_manager=SimpleNamespace(
            clone_context_type=Physics,
            apply_collision_filter=lambda received: applied.append(received),
        ),
        _backend_registry={UsdReplicateContext: Usd(calls)},
        get_clone_plan=lambda: None if legacy_override else plan,
        set_clone_plan=published.append,
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    kwargs = {"replicate_physics": False} if legacy_override else {}
    replicate_session.replicate(plan, **kwargs)

    execution_plan = applied[0]
    assert execution_plan.replicate_physics is False
    assert (execution_plan is not plan) is legacy_override
    assert calls == [(Usd, execution_plan)]
    assert published == ([execution_plan] if legacy_override else [])


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


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"clone_cfg": CloneCfg(isolate_environments=False, replicate_physics=False)},
            (False, False, "/World/envs/env_{}"),
        ),
        ({"replicate_physics": False, "env_template": "/World/cells/cell_{}"}, (True, False, "/World/cells/cell_{}")),
    ],
)
def test_replicate_session_snapshots_canonical_and_legacy_policy(monkeypatch, kwargs, expected):
    published = []
    simulation = SimpleNamespace(
        get_clone_plan=lambda: published[-1] if published else None,
        set_clone_plan=published.append,
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    session = replicate_session.ReplicateSession((), num_clones=2, env_spacing=1.0, **kwargs)
    if clone_cfg := kwargs.get("clone_cfg"):
        clone_cfg.isolate_environments = clone_cfg.replicate_physics = True

    plan = session.__enter__().plan
    assert (plan.isolate_environments, plan.replicate_physics, plan.env_template) == expected


def test_cloner_policy_has_one_canonical_input_and_legacy_inputs_normalize():
    for public_api in (make_clone_plan, replicate_session.ReplicateSession, clone_plan.clone_plan_from_env_0):
        assert signature(public_api).parameters["clone_cfg"].default is None
    for public_api in (make_clone_plan, replicate_session.ReplicateSession, replicate_session.replicate):
        assert "isolate_environments" not in signature(public_api).parameters

    default_plan = make_clone_plan((), 2, 1.0)
    legacy_plan = make_clone_plan((), 2, 1.0, env_template="/World/cells/cell_{}")
    assert default_plan.env_template == "/World/envs/env_{}"
    assert default_plan.isolate_environments is default_plan.replicate_physics is True
    assert legacy_plan.env_template == "/World/cells/cell_{}"

    clone_cfg = CloneCfg(clone_template="/ignored/env_{}", isolate_environments=False, replicate_physics=False)
    plan = clone_plan.clone_plan_from_env_0("/World/cells/cell_0", "/World/cells/cell_{}", 2, clone_cfg=clone_cfg)
    assert plan.env_template == "/World/cells/cell_{}"
    assert plan.isolate_environments is plan.replicate_physics is False


def test_scene_legacy_replicate_physics_normalizes_into_a_copy():
    cfg = InteractiveSceneCfg(
        num_envs=2,
        env_spacing=1.0,
        clone_cfg=CloneCfg(replicate_physics=True),
        replicate_physics=False,
    )

    with pytest.warns(DeprecationWarning, match="clone_cfg.replicate_physics"):
        resolved = cfg.resolve_clone_cfg()

    assert resolved is not cfg.clone_cfg
    assert resolved.replicate_physics is False
    assert cfg.clone_cfg.replicate_physics is True


def test_replicate_validates_options_before_stage_dispatch(monkeypatch):
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

    for field in ("isolate_environments", "replicate_physics"):
        invalid_plan = replace(plan, **{field: 1})
        simulation.get_clone_plan = lambda: invalid_plan
        with pytest.raises(TypeError, match=rf"ClonePlan\.{field} must be a bool"):
            replicate_session.replicate(invalid_plan)

    assert calls == []
    assert queued
