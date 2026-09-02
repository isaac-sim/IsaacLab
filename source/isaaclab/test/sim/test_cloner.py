# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for USD cloner utilities (no PhysX dependency)."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from pxr import Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.cloner import (
    ClonePlan,
    ReplicateSession,
    UsdReplicateContext,
    grid_transforms,
    make_clone_plan,
    sequential,
    usd_replicate,
)
from isaaclab.sim import build_simulation_context
from isaaclab.sim.utils import queries

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


@pytest.fixture(params=["cpu", "cuda"])
def sim(request):
    """Provide a fresh simulation context for each test on CPU and CUDA."""
    with build_simulation_context(device=request.param, dt=0.01, add_lighting=False) as sim:
        yield sim


def test_usd_replicate_with_positions_and_mask(sim):
    """Replicate sources to selected envs and author translate ops from positions."""
    # Prepare sources under /World/template
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/A", "Xform")
    sim_utils.create_prim("/World/template/B", "Xform")

    # Prepare destination env namespaces
    num_envs = 3
    env_ids = np.arange(num_envs, dtype=np.int64)
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")

    # Map A -> env 0 and 2; B -> env 1 only
    mask = np.zeros((2, num_envs), dtype=np.bool_)
    mask[0, [0, 2]] = True
    mask[1, [1]] = True

    usd_replicate(
        sim_utils.get_current_stage(),
        sources=["/World/template/A", "/World/template/B"],
        destinations=["/World/envs/env_{}/Object/A", "/World/envs/env_{}/Object/B"],
        env_ids=env_ids,
        mask=mask,
    )

    # Validate replication and translate op
    stage = sim_utils.get_current_stage()
    assert stage.GetPrimAtPath("/World/envs/env_0/Object/A").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_0/Object/B").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_1/Object/B").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_1/Object/A").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_2/Object/A").IsValid()

    # Check xformOp:translate authored for env_2/A
    prim = stage.GetPrimAtPath("/World/envs/env_2/Object/A")
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    assert any(op.GetOpType() == UsdGeom.XformOp.TypeTranslate for op in ops)


def test_usd_replicate_context_consumes_plan(sim):
    """UsdReplicateContext consumes the same plan used by every clone backend."""
    sim_utils.create_prim("/World/template/A", "Xform")
    sim_utils.create_prim("/World/envs", "Xform")

    stage = sim_utils.get_current_stage()
    ctx = UsdReplicateContext(stage)
    plan = ClonePlan(
        sources=("/World/template/A",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.asarray([[False, True]], dtype=np.bool_),
        env_ids=np.asarray([10, 20], dtype=np.int64),
        positions=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        context_rows={UsdReplicateContext: (0,)},
    )
    ctx.replicate(plan)

    assert not stage.GetPrimAtPath("/World/envs/env_10").IsValid()
    prim = stage.GetPrimAtPath("/World/envs/env_20")
    assert prim.IsValid()
    assert tuple(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()) == (4.0, 5.0, 6.0)


def test_usd_replicate_nested_asset_preserves_local_offset_with_positions(sim):
    """Grid positions are authored on env roots but not on nested replicated assets."""
    camera_offset = (0.57, -0.8, 0.5)
    num_envs = 2
    env_ids = np.arange(num_envs, dtype=np.int64)
    positions, _ = grid_transforms(num_envs, 3.0)

    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Camera", "Camera", translation=camera_offset)

    stage = sim_utils.get_current_stage()
    usd_replicate(
        stage,
        sources=["/World/envs/env_0"],
        destinations=["/World/envs/env_{}"],
        env_ids=env_ids,
        positions=positions,
    )
    usd_replicate(
        stage,
        sources=["/World/envs/env_0/Camera"],
        destinations=["/World/envs/env_{}/Camera"],
        env_ids=env_ids,
        positions=positions,
    )

    for env_idx in range(num_envs):
        env_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}")
        assert env_prim.IsValid()
        env_translate = env_prim.GetAttribute("xformOp:translate").Get()
        assert env_translate is not None
        expected_env_pos = positions[env_idx].tolist()
        assert (env_translate[0], env_translate[1], env_translate[2]) == pytest.approx(expected_env_pos)

        camera_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/Camera")
        assert camera_prim.IsValid()
        camera_translate = camera_prim.GetAttribute("xformOp:translate").Get()
        assert camera_translate is not None
        assert (camera_translate[0], camera_translate[1], camera_translate[2]) == pytest.approx(camera_offset)


def test_disabled_fabric_change_notifies_noops_when_usdrt_unavailable(monkeypatch):
    """Fabric notice suspension no-ops when Carbonite bindings exist but ``usdrt`` does not."""
    import builtins

    from isaaclab.cloner import _fabric_notices

    class _FakeBindings:
        def validate_with(self, fabric_id: int) -> bool:
            raise AssertionError("missing usdrt should prevent fabric-id lookup")

    monkeypatch.setattr(_fabric_notices, "get_bindings", lambda: _FakeBindings())

    real_import = builtins.__import__

    def _import_without_usdrt(name, *args, **kwargs):
        if name == "usdrt":
            raise ModuleNotFoundError("No module named 'usdrt'", name="usdrt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_without_usdrt)

    with _fabric_notices.disabled_fabric_change_notifies(Usd.Stage.CreateInMemory()):
        pass


def test_usd_replicate_depth_order_parent_child(sim):
    """Replicate parent and child when provided out of order; parent should exist before child."""
    # Prepare sources
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/Parent", "Xform")
    sim_utils.create_prim("/World/template/Parent/Child", "Xform")

    # Destinations (single env)
    env_ids = np.asarray([0, 1], dtype=np.int64)
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_1", "Xform")

    # Provide child first, then parent; depth sort should handle this
    usd_replicate(
        sim_utils.get_current_stage(),
        sources=["/World/template/Parent/Child", "/World/template/Parent"],
        destinations=["/World/envs/env_{}/Parent/Child", "/World/envs/env_{}/Parent"],
        env_ids=env_ids,
    )

    stage = sim_utils.get_current_stage()
    for i in range(2):
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/Parent").IsValid()
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/Parent/Child").IsValid()


def test_usd_replicate_self_copy_skips_copy_spec(sim):
    """usd_replicate must not call Sdf.CopySpec when source and destination paths are identical."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Robot", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Robot/base_link", "Xform")
    sim_utils.create_prim("/World/envs/env_1", "Xform")

    copy_calls: list[tuple[str, str]] = []
    real_copy_spec = Sdf.CopySpec

    def capturing_copy_spec(src_layer, src_path, dst_layer, dst_path, *args):
        copy_calls.append((str(src_path), str(dst_path)))
        return real_copy_spec(src_layer, src_path, dst_layer, dst_path, *args)

    with patch.object(Sdf, "CopySpec", capturing_copy_spec):
        usd_replicate(
            stage,
            sources=["/World/envs/env_0"],
            destinations=["/World/envs/env_{}"],
            env_ids=np.asarray([0, 1], dtype=np.int64),
            mask=np.ones((1, 2), dtype=np.bool_),
        )

    assert all(src != dst for src, dst in copy_calls), f"Self-copy detected in CopySpec calls: {copy_calls}"
    assert any(dst == "/World/envs/env_1" for _, dst in copy_calls), "CopySpec was not called for env_1"


@pytest.mark.parametrize(
    "parent_paths, spawn_pattern, expected_child_paths, bad_path, match_expr",
    [
        (
            ["/World/rig_0_alpha", "/World/rig_0_beta", "/World/rig_0_gamma"],
            "/World/rig_0_[^/]*/Sensor",
            ["/World/rig_0_alpha/Sensor", "/World/rig_0_beta/Sensor", "/World/rig_0_gamma/Sensor"],
            "/World/rig_00/Sensor",
            "/World/rig_0_[^/]*",
        ),
        (
            [
                "/World/group_a/slot_0",
                "/World/group_a/slot_1",
                "/World/group_b/slot_0",
                "/World/group_b/slot_1",
            ],
            "/World/group_[^/]*/slot_[^/]*/Sensor",
            [
                "/World/group_a/slot_0/Sensor",
                "/World/group_a/slot_1/Sensor",
                "/World/group_b/slot_0/Sensor",
                "/World/group_b/slot_1/Sensor",
            ],
            "/World/group_0/slot_0/Sensor",
            "/World/group_[^/]*/slot_[^/]*",
        ),
        (
            ["/World/template/Object"],
            "/World/template/Object/proto_.*",
            ["/World/template/Object/proto_0"],
            "/World/template/Object0/proto_0",
            "/World/template/Object",
        ),
    ],
)
def test_clone_decorator_wildcard_patterns(
    sim, parent_paths, spawn_pattern, expected_child_paths, bad_path, match_expr
):
    """The @clone decorator handles two distinct wildcard patterns correctly."""
    for path in parent_paths:
        sim_utils.create_prim(path, "Xform")

    cfg = sim_utils.ConeCfg(radius=0.1, height=0.2)
    cfg.func(spawn_pattern, cfg)

    stage = sim_utils.get_current_stage()

    for child_path in expected_child_paths:
        assert stage.GetPrimAtPath(child_path).IsValid(), (
            f"Prim was not spawned at '{child_path}'. The @clone decorator may have used the wrong spawn path."
        )

    assert not stage.GetPrimAtPath(bad_path).IsValid(), (
        f"Spurious prim found at '{bad_path}'. "
        "The @clone decorator incorrectly derived the spawn path by replacing '.*' with '0'."
    )

    all_matching = sim_utils.find_matching_prims(match_expr)
    assert len(all_matching) == len(parent_paths), (
        f"Expected {len(parent_paths)} matching prims, got {len(all_matching)}. "
        "Spurious parent prims were likely created by the @clone decorator."
    )


def test_make_clone_plan_homogeneous_returns_env_root_plan(sim):
    """Homogeneous (single-variant) cfgs produce one source row at the env root."""
    cube = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Robot",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        cloning_contexts=None,
    )
    plan = make_clone_plan(
        cfgs=[cube],
        num_clones=4,
        env_spacing=1.0,
        global_paths=("/World/Ground",),
    )

    assert plan.sources == ("/World/envs/env_0",)
    assert plan.destinations == ("/World/envs/env_{}",)
    assert plan.clone_mask.shape == (1, 4)
    assert plan.clone_mask.all()
    assert plan.cfg_rows[id(cube)] == (0,)
    assert plan.global_paths == ("/World/Ground",)
    assert plan.env_ids.shape == (4,)
    assert plan.positions.shape == (4, 3)
    assert cube.spawn.spawn_path == "/World/envs/env_0/Robot"


def test_resolve_matching_prims_from_source_searches_only_plan_source(sim, monkeypatch):
    """Clone-aware regex discovery traverses its plan source, never cloned destinations."""
    stage = sim_utils.get_current_stage()
    for path in (
        "/World/envs/env_0/Robot/foo",
        "/World/envs/env_0/Robot/foo/bar",
        "/World/envs/env_1/Robot/clone_only",
    ):
        stage.DefinePrim(path, "Xform")
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2, dtype=np.int64),
        positions=np.zeros((2, 3), dtype=np.float32),
    )
    sim.set_clone_plan(plan)

    traversed_roots = []
    source_matcher = queries._iter_matching_prims_in_subtree

    def record_source_root(path_expr, root_prim):
        traversed_roots.append(root_prim.GetPath().pathString)
        return source_matcher(path_expr, root_prim)

    monkeypatch.setattr(queries, "_iter_matching_prims_in_subtree", record_source_root)
    monkeypatch.setattr(
        queries,
        "find_matching_prims",
        lambda *args, **kwargs: pytest.fail("clone-aware resolution called the unscoped stage matcher"),
    )

    matches = queries.resolve_matching_prims_from_source(r"/World/envs/env_[^/]+/Robot/[^A]+")

    assert traversed_roots == ["/World/envs/env_0/Robot"]
    assert [prim.GetPath().pathString for prim, _ in matches] == [
        "/World/envs/env_0/Robot/foo",
        "/World/envs/env_0/Robot/foo/bar",
    ]
    assert [path_expr for _, path_expr in matches] == [
        "/World/envs/env_[^/]+/Robot/foo",
        "/World/envs/env_[^/]+/Robot/foo/bar",
    ]


def test_make_clone_plan_heterogeneous_mutates_spawn_paths(sim):
    """Multi-variant spawners get per-variant spawn_paths and contribute multiple plan rows."""
    multi_cfg = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Object",
        cloning_contexts=None,
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(radius=0.1, height=0.2),
                sim_utils.SphereCfg(radius=0.1),
            ]
        ),
    )
    plain_cfg = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Robot",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        cloning_contexts=None,
    )
    plan = make_clone_plan(
        cfgs=[multi_cfg, plain_cfg],
        num_clones=4,
        env_spacing=1.0,
        global_paths=("/World/Ground",),
        clone_strategy=sequential,
    )

    assert plan.destinations == (
        "/World/envs/env_{}/Object",
        "/World/envs/env_{}/Object",
        "/World/envs/env_{}/Robot",
    )
    assert plan.cfg_rows[id(multi_cfg)] == (0, 1)
    assert plan.cfg_rows[id(plain_cfg)] == (2,)
    assert plan.global_paths == ("/World/Ground",)
    assert multi_cfg.spawn.spawn_paths == ["/World/envs/env_0/Object", "/World/envs/env_1/Object"]
    assert plain_cfg.spawn.spawn_path == "/World/envs/env_0/Robot"


def test_make_clone_plan_records_globals_outside_replication_rows(sim):
    """Global cfgs are named by the plan without becoming rows a backend might copy."""
    plan = make_clone_plan(
        cfgs=[],
        num_clones=3,
        env_spacing=1.0,
        global_paths=("/World/global/Robot", "/World/ground"),
    )

    assert plan.sources == ()
    assert plan.destinations == ()
    assert plan.clone_mask.shape == (0, 3)
    assert plan.cfg_rows == {}
    assert plan.global_paths == ("/World/global/Robot", "/World/ground")


def test_clone_plan_from_env_0_uses_flat_cfg_manifest(sim):
    """The homogeneous helper publishes exact env/global roots before construction."""
    robot = SimpleNamespace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        cloning_contexts=(UsdReplicateContext,),
    )
    sensor = SimpleNamespace(
        prim_path="{ENV_REGEX_NS}/Robot/Sensor", spawn=None, cloning_contexts=(UsdReplicateContext,)
    )
    prop = SimpleNamespace(
        prim_path="{ENV_REGEX_NS}/Prop",
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=[sim_utils.SphereCfg(radius=0.1)]),
        cloning_contexts=(UsdReplicateContext,),
    )
    light = SimpleNamespace(prim_path="/World/Light", spawn=sim_utils.DistantLightCfg(), cloning_contexts=())
    light_reference = SimpleNamespace(prim_path="/World/Light", spawn=None, cloning_contexts=())
    plan = cloner.clone_plan_from_env_0(cloner.CloneCfg(), (robot, sensor, None, prop, light, light_reference), 4, 1.0)

    assert sim.get_clone_plan() is plan
    assert plan.sources == ("/World/envs/env_0",)
    assert plan.destinations == ("/World/envs/env_{}",)
    assert plan.cfg_rows == {id(robot): (0,), id(sensor): (0,), id(prop): (0,)}
    assert plan.global_paths == ("/World/Light",)
    assert plan.clone_mask.all() and plan.clone_mask.shape == (1, 4)
    np.testing.assert_array_equal(plan.env_ids, np.arange(4, dtype=np.int64))
    assert robot.prim_path == "/World/envs/env_[^/]+/Robot"
    assert robot.spawn.spawn_path == "/World/envs/env_0/Robot"
    assert prop.spawn.spawn_path is None
    assert prop.spawn.spawn_paths == ["/World/envs/env_0/Prop"]
    assert light.spawn.spawn_path == "/World/Light"


def test_clone_plan_from_env_0_routes_plain_spawner_cfg(sim):
    """A spawned manifest entry requests USD replication without an asset-base field."""
    cfg = SimpleNamespace(prim_path="{ENV_REGEX_NS}/Decoration", spawn=sim_utils.CuboidCfg(size=(0.1,) * 3))

    plan = cloner.clone_plan_from_env_0(cloner.CloneCfg(), (cfg,), 2, 1.0)

    assert plan.context_rows[UsdReplicateContext] == (0,)


@pytest.mark.parametrize("nested", [SimpleNamespace(scene=object()), (SimpleNamespace(prim_path="/World/A"),)])
def test_clone_plan_from_env_0_rejects_cfg_tree_inputs(sim, nested):
    """The cloner accepts a flat construction manifest, never an outer cfg tree."""
    with pytest.raises(TypeError, match="directly declare prim_path"):
        cloner.clone_plan_from_env_0(cloner.CloneCfg(), (nested,), 2, 1.0)
    assert sim.get_clone_plan() is None


def test_clone_plan_from_env_0_rejects_multi_variant_spawner_atomically(sim):
    """Heterogeneous spawners use make_clone_plan and remain untouched on rejection."""
    cfg = SimpleNamespace(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.SphereCfg(radius=0.1), sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))]
        ),
    )
    with pytest.raises(ValueError, match="single-variant"):
        cloner.clone_plan_from_env_0(cloner.CloneCfg(), (cfg,), 2, 1.0)
    assert cfg.prim_path == "{ENV_REGEX_NS}/Object"
    assert cfg.spawn.spawn_paths is None
    assert sim.get_clone_plan() is None


def test_replicate_session_clears_plan_when_asset_init_fails(sim):
    """ReplicateSession clears an unconsumed plan when construction raises."""
    with pytest.raises(RuntimeError, match="asset boom"):
        with ReplicateSession(
            cfgs=[],
            num_clones=2,
            env_spacing=1.0,
        ) as session:
            assert sim.get_clone_plan() is session.plan
            raise RuntimeError("asset boom")

    assert sim.get_clone_plan() is None
