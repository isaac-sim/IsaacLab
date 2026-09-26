# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone lifecycle routing, authoring, and dispatch without a simulator runtime."""

from types import SimpleNamespace

import numpy as np
import pytest

from pxr import Usd, UsdGeom

import isaaclab.cloner.replicate_session as replicate_session
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import CloneCfg, ReplicateSession, UsdReplicateContext, clone_plan_from_env_0, grid_transforms
from isaaclab.renderers import RenderContext, RendererCfg
from isaaclab.sensors import CameraCfg, SensorBaseCfg
from isaaclab.sim import CuboidCfg, MultiAssetSpawnerCfg, PinholeCameraCfg, SimulationContext, SphereCfg


class _Context:
    replicate_priority = 0

    def __init__(self, sim):
        self.calls = sim.calls

    def replicate(self, plan, asset_prototype_ids):
        self.calls.append((type(self), plan, asset_prototype_ids))


class _RenderContext(_Context):
    pass


@pytest.fixture
def simulation(monkeypatch):
    registry = []
    sim = SimpleNamespace(
        physics_manager=SimpleNamespace(clone_context_type=_Context),
        clone_contexts={},
        _backend_registry=registry,
        _render_context=RenderContext(registry),
        stage=Usd.Stage.CreateInMemory(),
        plan=None,
        calls=[],
    )
    sim.render_context = sim._render_context
    sim.get_or_create_backend = lambda cfg: SimulationContext.get_or_create_backend(sim, cfg)
    sim.clone_contexts[_Context] = _Context(sim)
    sim.get_clone_plan = lambda: sim.plan
    sim.set_clone_plan = lambda plan: setattr(sim, "plan", plan)
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim)
    monkeypatch.setattr(replicate_session, "has_kit", lambda: False)
    return sim


@pytest.mark.parametrize("override", [None, (), (_RenderContext,)])
@pytest.mark.parametrize("replicate_physics", [True, False])
def test_asset_routing_preserves_explicit_overrides(simulation, override, replicate_physics):
    """Rendering requirements augment asset policy; disabling physics leaves rendering active."""
    simulation.render_context.clone_contexts.add(_RenderContext)
    cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Robot", spawn=CuboidCfg(size=(1, 1, 1)), cloning_contexts=override)
    cfg.spawn.spawn_path = "/Previous/Robot"
    with ReplicateSession((cfg,), 2, 1.0, replicate_physics=replicate_physics) as session:
        assert simulation.plan is session.plan
        assert cfg.spawn.spawn_path == "/World/envs/env_0/Robot"
        assert not simulation.calls
    expected = {_RenderContext}
    if override is None and replicate_physics:
        expected.add(_Context)
    assert {context for context, _, _ in simulation.calls} == expected
    assert all(plan is session.plan and ids == (0,) for _, plan, ids in simulation.calls)


@pytest.mark.parametrize("shared", [False, True])
def test_empty_and_shared_only_worlds(simulation, shared):
    """An empty world is still a valid world; shared assets are routed once, not repeated."""
    simulation.render_context.clone_contexts.add(_RenderContext)
    assets = (AssetBaseCfg(prim_path="/World/Ground"),) if shared else ()
    with ReplicateSession(assets, 3, 2.0) as session:
        usd = simulation.clone_contexts[UsdReplicateContext]
        assert usd.global_paths == (("/World/Ground",) if shared else ())
        np.testing.assert_array_equal(session.plan.destinations, [0, 0, 0])
        np.testing.assert_array_equal(session.plan.world_prototype_starts, [0, int(shared), int(shared)])
    assert {context for context, _, _ in simulation.calls} == (
        {_Context, _RenderContext} if shared else {_RenderContext}
    )


@pytest.mark.parametrize("from_env_0", [False, True])
def test_camera_registers_before_cloning_and_shares_the_plan(simulation, from_env_0):
    """Camera requirements enter both construction workflows before dispatch."""
    constructed = []
    renderer_cfg = RendererCfg(
        class_type=lambda cfg: constructed.append(cfg) or object(),
        renderer_type="test",
        cloning_contexts=(_RenderContext,),
    )
    camera = CameraCfg(prim_path="/Lab/Cell[^/]+/Camera", spawn=PinholeCameraCfg(), renderer_cfg=renderer_cfg)
    ground = AssetBaseCfg(prim_path="/Lab/Ground", spawn=CuboidCfg(size=(1, 1, 1)))
    assets = camera, ground, SensorBaseCfg(prim_path="/Lab/Ground/Frame"), SensorBaseCfg(prim_path=camera.prim_path)
    if from_env_0:
        plan = clone_plan_from_env_0(CloneCfg(clone_template="/Lab/Cell{}"), assets, 3, 2.0)
        replicate_session.replicate(plan)
    else:
        with ReplicateSession(assets, 3, 2.0, env_template="/Lab/Cell{}") as session:
            plan = session.plan
    assert constructed == [camera.renderer_cfg]
    simulation.get_or_create_backend(camera.renderer_cfg)
    assert len(constructed) == 1
    assert plan.asset_prototypes[0] is camera and plan.asset_prototypes[1] is ground
    assert camera.spawn.spawn_path == "/Lab/Cell0/Camera"
    assert ground.spawn.spawn_path == "/Lab/Ground"
    np.testing.assert_array_equal(plan.world_prototypes, [1, 0])
    assert {context for context, _, _ in simulation.calls} == {_Context, _RenderContext}
    assert all(received is plan for _, received, _ in simulation.calls)


def test_dispatch_order_and_usd_scope(simulation):
    """USD clones only declared subtrees; all contexts receive the same topology."""

    class Late(_Context):
        replicate_priority = 1

    class Early(_Context):
        replicate_priority = -1

    cfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=CuboidCfg(size=(1, 1, 1)),
        cloning_contexts=(UsdReplicateContext, Late, Early),
    )
    with ReplicateSession((cfg,), 2, 1.0) as session:
        UsdGeom.Xform.Define(simulation.stage, cfg.spawn.spawn_path)
        UsdGeom.Camera.Define(simulation.stage, "/World/envs/env_0/UndeclaredCamera")
    assert simulation.calls == [(Early, session.plan, (0,)), (Late, session.plan, (0,))]
    assert simulation.stage.GetPrimAtPath("/World/envs/env_1/Robot")
    assert not simulation.stage.GetPrimAtPath("/World/envs/env_1/UndeclaredCamera")


def test_grid_transforms_centers_a_float32_grid():
    positions, orientations = grid_transforms(3, np.float64(2.0))
    assert positions.dtype == orientations.dtype == np.float32
    np.testing.assert_array_equal(positions, [[1, -1, 0], [1, 1, 0], [-1, -1, 0]])
    np.testing.assert_array_equal(orientations, [[0, 0, 0, 1]] * 3)


def test_multi_spawner_creates_concrete_asset_prototypes(simulation):
    """Asset alternatives become distinct definitions; their original spawner authors each once."""
    shapes = [CuboidCfg(size=(1, 1, 1)), SphereCfg(radius=1)]
    object_cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Object", spawn=MultiAssetSpawnerCfg(assets_cfg=shapes))
    ground = AssetBaseCfg(prim_path="/World/Ground")
    with ReplicateSession((object_cfg, ground), 4, 2.0) as session:
        plan = session.plan
        assert len(plan.asset_prototypes) == 3
        for cfg, shape in zip(plan.asset_prototypes[:2], object_cfg.spawn.assets_cfg, strict=True):
            assert cfg.spawn.assets_cfg[0] is shape
        assert object_cfg.spawn.spawn_paths == ["/World/envs/env_0/Object", "/World/envs/env_2/Object"]
        np.testing.assert_array_equal(plan.world_prototypes, [2, 0, 1])
        np.testing.assert_array_equal(plan.destinations, [0, 0, 1, 1])
