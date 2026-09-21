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
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import CloneCfg, ClonePlan, UsdReplicateContext, clone_plan_from_env_0, make_clone_plan
from isaaclab.renderers import RenderContext, RendererCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import CuboidCfg, PinholeCameraCfg, SimulationContext


class _Context:
    replicate_priority = 0

    def __init__(self, sim):
        self.calls = sim.calls

    def replicate(self, plan):
        self.calls.append((type(self), plan))


class _RenderContext(_Context):
    pass


@pytest.fixture
def simulation(monkeypatch):
    sim = SimpleNamespace(
        physics_manager=SimpleNamespace(clone_context_type=_Context),
        clone_contexts={},
        render_context=RenderContext(),
        stage=object(),
        plan=None,
        calls=[],
    )
    sim.clone_contexts[_Context] = _Context(sim)
    sim.get_clone_plan = lambda: sim.plan
    sim.set_clone_plan = lambda plan: setattr(sim, "plan", plan)
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim)
    monkeypatch.setattr(clone_plan, "has_kit", lambda: False)
    return sim


def _plan(*context_types):
    return ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2, dtype=np.int64),
        positions=np.zeros((2, 3), dtype=np.float32),
        context_rows={context_type: (0,) for context_type in context_types},
    )


@pytest.mark.parametrize("render_context", [_Context, _RenderContext])
def test_make_clone_plan_routes_default_and_explicit_contexts(simulation, render_context):
    """Foreign rendering adds rows without overriding explicit asset physics routing."""

    class Unrelated(_Context):
        pass

    simulation.clone_contexts[Unrelated] = Unrelated(simulation)
    simulation.render_context.clone_contexts.add(render_context)
    cfg = AssetBaseCfg(
        prim_path="/World/envs/env_[^/]+/Robot", spawn=CuboidCfg(size=(1.0, 1.0, 1.0)), cloning_contexts=None
    )

    plan = make_clone_plan((cfg,), 2, 1.0)

    assert plan.context_rows == {_Context: (0,), render_context: (0,)}

    class Explicit(_Context):
        pass

    cfg.cloning_contexts = (Explicit,)
    render_rows = {_RenderContext: (0,)} if render_context is _RenderContext else {}
    assert make_clone_plan((cfg,), 2, 1.0).context_rows == {Explicit: (0,), **render_rows}
    cfg.cloning_contexts = ()
    assert make_clone_plan((cfg,), 2, 1.0).context_rows == render_rows


@pytest.mark.parametrize("global_paths", [(), ("/World/Ground",)])
def test_make_clone_plan_routes_empty_and_global_only_plans(simulation, global_paths):
    """Rendering receives empty plans; shared roots also reach active physics."""
    simulation.render_context.clone_contexts.add(_RenderContext)

    empty = make_clone_plan((), 2, 1.0, global_paths=global_paths)
    contexts = {_Context, _RenderContext} if global_paths else {_RenderContext}
    assert empty.context_rows == dict.fromkeys(contexts, ())
    assert empty.global_paths == global_paths
    simulation.plan = empty
    replicate_session.replicate(empty)
    assert {context for context, _ in simulation.calls} == contexts
    assert all(plan is empty for _, plan in simulation.calls)


@pytest.mark.parametrize("from_env_0", [False, True])
def test_camera_registers_rendering_before_planning_and_shares_the_plan(simulation, from_env_0):
    """Both public planners include camera declarations before routing, with arbitrary env roots."""
    constructed = []
    renderer_cfg = RendererCfg(
        class_type=lambda cfg: constructed.append(cfg) or object(),
        renderer_type="test",
        cloning_contexts=(_RenderContext,),
    )
    camera = CameraCfg(prim_path="/Lab/Cell[^/]+/Camera", spawn=PinholeCameraCfg(), renderer_cfg=renderer_cfg)
    ground = AssetBaseCfg(prim_path="/Lab/Ground", spawn=CuboidCfg(size=(1.0, 1.0, 1.0)))
    if from_env_0:
        plan = clone_plan_from_env_0(CloneCfg(clone_template="/Lab/Cell{}"), (camera, ground), 3, 2.0)
    else:
        plan = make_clone_plan((camera,), 3, 2.0, global_paths=(ground.prim_path,), env_template="/Lab/Cell{}")
        simulation.plan = plan

    assert constructed == [camera.renderer_cfg]
    simulation.render_context.get_renderer(camera.renderer_cfg)
    assert len(constructed) == 1
    assert plan.sources == ("/Lab/Cell0",)
    assert plan.destinations == ("/Lab/Cell{}",)
    assert plan.global_paths == (ground.prim_path,)
    assert plan.cfg_rows[id(camera)] == (0,)
    assert plan.context_rows == {_Context: (), _RenderContext: (0,)}
    replicate_session.replicate(plan)
    assert {context for context, _ in simulation.calls} == {_Context, _RenderContext}
    assert all(received is plan for _, received in simulation.calls)


def test_context_routing_requires_every_asset_row(simulation):
    """An incomplete cfg-row mapping must fail instead of silently dropping an asset."""
    cfg = SimpleNamespace(cloning_contexts=None)
    with pytest.raises(KeyError):
        clone_plan._context_rows((cfg,), {}, {0})


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


def test_replicate_dispatches_the_same_plan_in_priority_order(simulation):
    """Registered contexts receive one shared plan in backend priority order."""

    class Late(_Context):
        replicate_priority = 1

    class Early(_Context):
        replicate_priority = -1

    plan = _plan(Late, Early)
    simulation.physics_manager.clone_context_type = Late
    simulation.clone_contexts = {Late: Late(simulation), Early: Early(simulation)}
    simulation.plan = plan

    replicate_session.replicate(plan)

    assert simulation.calls == [(Early, plan), (Late, plan)]


def test_replicate_physics_false_preserves_rendering_and_usd(simulation):
    """Disabling active physics replication preserves other declared representations."""

    class Physics(_Context):
        pass

    class Usd(_Context):
        pass

    plan = _plan(Physics, UsdReplicateContext, _RenderContext)
    simulation.physics_manager.clone_context_type = Physics
    simulation.clone_contexts = {UsdReplicateContext: Usd(simulation), _RenderContext: _RenderContext(simulation)}
    simulation.plan = plan

    replicate_session.replicate(plan, replicate_physics=False)

    assert simulation.calls == [(Usd, plan), (_RenderContext, plan)]


def test_replicate_rejects_unregistered_context(simulation):
    """A routed context must register before dispatch rather than using a fallback."""
    plan = _plan(_Context)
    simulation.clone_contexts.clear()
    simulation.plan = plan

    with pytest.raises(RuntimeError, match="must be registered"):
        replicate_session.replicate(plan)
