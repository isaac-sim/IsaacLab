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
from isaaclab.cloner import path as cloner_path
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
        templates, starts = cloner_path.get_world_prototype_asset_templates(session.plan)
        assert templates[: starts[1]] == (("/World/Ground",) if shared else ())
        np.testing.assert_array_equal(session.plan.topology.world_prototype_layout, [0, 0, 0])
        np.testing.assert_array_equal(session.plan.topology.world_prototype_starts, [0, int(shared), int(shared)])
    assert {context for context, _, _ in simulation.calls} == (
        {_Context, _RenderContext} if shared else {_RenderContext}
    )


@pytest.mark.parametrize("from_env_0", [False, True])
def test_camera_registers_before_cloning_and_shares_the_plan(simulation, from_env_0):
    """Camera requirements enter both construction workflows before dispatch."""
    constructed = []

    def factory(cfg):
        constructed.append(cfg)
        return object()

    renderer_cfg = RendererCfg(class_type=factory, renderer_type="test", cloning_contexts=(_RenderContext,))
    camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera", spawn=PinholeCameraCfg(), renderer_cfg=renderer_cfg)
    ground = AssetBaseCfg(prim_path="/Lab/Ground", spawn=CuboidCfg(size=(1, 1, 1)))
    prop = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Prop", spawn=MultiAssetSpawnerCfg(assets_cfg=[SphereCfg(radius=1)]))
    assets = camera, ground, prop, AssetBaseCfg(prim_path="/Lab/Ground/Material")
    assets += SensorBaseCfg(prim_path="/Lab/Ground/Frame"), SensorBaseCfg(prim_path=camera.prim_path)
    assets += (AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Prop/body"),)
    if from_env_0:
        plan = clone_plan_from_env_0(CloneCfg(clone_template="/Lab/Cell{}"), assets, 3, 2.0)
        replicate_session.replicate(plan)
    else:
        with ReplicateSession(assets, 3, 2.0, env_template="/Lab/Cell{}") as session:
            plan = session.plan
    assert constructed == [camera.renderer_cfg]
    simulation.get_or_create_backend(camera.renderer_cfg)
    assert len(constructed) == 1
    assert plan.asset_cfgs == assets
    assert camera.prim_path == "/Lab/Cell[^/]+/Camera"
    assert camera.spawn.spawn_path == "/Lab/Cell0/Camera"
    assert plan.env_template == "/Lab/Cell{}"
    assert ground.spawn.spawn_path == "/Lab/Ground"
    assert prop.spawn.spawn_path is None and prop.spawn.spawn_paths == ["/Lab/Cell0/Prop"]
    templates, starts = cloner_path.get_world_prototype_asset_templates(plan)
    shared = templates[: starts[1]]
    roots = [path for path, parent in zip(shared, cloner_path.get_parent_indices(shared)) if parent == -1]
    assert roots == ["/Lab/Ground"]
    np.testing.assert_array_equal(plan.topology.world_prototypes, [1, 0, 2])
    np.testing.assert_array_equal(plan.topology.world_prototype_layout, [0, 0, 0])
    assert {context for context, _, _ in simulation.calls} == {_Context, _RenderContext}
    assert all(received is plan for _, received, _ in simulation.calls)


def test_dispatch_order_and_usd_scope(simulation):
    """USD clones only declared subtrees; all contexts receive the same topology."""

    class Late(_Context):
        replicate_priority = 1

    class Early(_Context):
        replicate_priority = -1

    contexts = UsdReplicateContext, Late, Early
    cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Robot", spawn=CuboidCfg(size=(1, 1, 1)), cloning_contexts=contexts)
    with ReplicateSession((cfg,), 2, 1.0) as session:
        UsdGeom.Xform.Define(simulation.stage, cfg.spawn.spawn_path)
        UsdGeom.Camera.Define(simulation.stage, "/World/envs/env_0/UndeclaredCamera")
    assert simulation.calls == [(Early, session.plan, (0,)), (Late, session.plan, (0,))]
    assert set(vars(simulation.clone_contexts[UsdReplicateContext])) == {"stage"}
    assert simulation.stage.GetPrimAtPath("/World/envs/env_1/Robot")
    assert not simulation.stage.GetPrimAtPath("/World/envs/env_1/UndeclaredCamera")
    np.testing.assert_array_equal(session.plan.positions, grid_transforms(2, 1.0)[0])
    for world_id, position in enumerate(session.plan.positions):
        prim = simulation.stage.GetPrimAtPath(f"/World/envs/env_{world_id}")
        np.testing.assert_array_equal(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation(), position
        )


def test_grid_transforms_centers_a_float32_grid():
    positions, orientations = grid_transforms(3, np.float64(2.0))
    assert positions.dtype == orientations.dtype == np.float32
    np.testing.assert_array_equal(positions, [[1, -1, 0], [1, 1, 0], [-1, -1, 0]])
    np.testing.assert_array_equal(orientations, [[0, 0, 0, 1]] * 3)


def test_multi_spawner_creates_concrete_asset_prototypes(simulation):
    """Homogeneous planning rejects variants atomically; a session authors each alternative once."""
    shapes = [CuboidCfg(size=(1, 1, 1)), SphereCfg(radius=1)]
    object_cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Object", spawn=MultiAssetSpawnerCfg(assets_cfg=shapes))
    robot = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Robot", spawn=CuboidCfg(size=(1, 1, 1)))
    with pytest.raises(ValueError, match="single-variant"):
        clone_plan_from_env_0(CloneCfg(), (robot, object_cfg), 2, 1.0)
    assert robot.prim_path == "{ENV_REGEX_NS}/Robot" and robot.spawn.spawn_path is None
    assert object_cfg.prim_path == "{ENV_REGEX_NS}/Object" and object_cfg.spawn.spawn_paths is None
    assert simulation.plan is None

    ground = AssetBaseCfg(prim_path="/World/Ground")
    with ReplicateSession((object_cfg, ground), 4, 2.0) as session:
        plan = session.plan
        assert len(plan.asset_cfgs) == 3
        for cfg, shape in zip(plan.asset_cfgs[:2], object_cfg.spawn.assets_cfg, strict=True):
            assert cfg.spawn.assets_cfg[0] is shape
        assert object_cfg.spawn.spawn_paths == ["/World/envs/env_0/Object", "/World/envs/env_2/Object"]
        np.testing.assert_array_equal(plan.topology.world_prototypes, [2, 0, 1])
        np.testing.assert_array_equal(plan.topology.world_prototype_layout, [0, 0, 1, 1])


def test_replicate_session_clears_plan_when_asset_init_fails(simulation):
    """Failed construction releases the plan without dispatching any clone backend."""
    with pytest.raises(RuntimeError, match="asset boom"):
        with ReplicateSession((), 2, 1.0) as session:
            assert simulation.plan is session.plan
            raise RuntimeError("asset boom")
    assert simulation.plan is None and not simulation.calls
