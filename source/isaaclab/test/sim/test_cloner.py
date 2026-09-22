# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cloner tests that need a live simulation context (plan publication, spawner cloning, clone-aware queries).

USD replication and plan construction are covered without the simulator in ``test/cloner``.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from types import SimpleNamespace

import numpy as np
import pytest

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.cloner import ClonePlan, ReplicateSession, UsdReplicateContext
from isaaclab.sim import build_simulation_context
from isaaclab.sim.utils import queries

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


@pytest.fixture
def sim():
    """Provide a fresh simulation context; these tests author USD only, so the device is irrelevant."""
    with build_simulation_context(device="cpu", dt=0.01, add_lighting=False) as sim:
        yield sim


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
            ["/World/group_a/slot_0", "/World/group_a/slot_1", "/World/group_b/slot_0", "/World/group_b/slot_1"],
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
    """The @clone decorator spawns under every matching parent without inventing prims from the wildcard."""
    for path in parent_paths:
        sim_utils.create_prim(path, "Xform")

    cfg = sim_utils.ConeCfg(radius=0.1, height=0.2)
    cfg.func(spawn_pattern, cfg)

    stage = sim_utils.get_current_stage()
    assert all(stage.GetPrimAtPath(child_path).IsValid() for child_path in expected_child_paths)
    assert not stage.GetPrimAtPath(bad_path).IsValid(), "the wildcard must not be replaced with '0' literally"
    assert len(sim_utils.find_matching_prims(match_expr)) == len(parent_paths), "no spurious parent prims"


def test_resolve_matching_prims_from_source_searches_only_plan_source(sim, monkeypatch):
    """Clone-aware regex discovery traverses its plan source, never cloned destinations."""
    stage = sim_utils.get_current_stage()
    for path in ("Robot/foo", "Robot/foo/bar"):
        stage.DefinePrim(f"/World/envs/env_0/{path}", "Xform")
    stage.DefinePrim("/World/envs/env_1/Robot/clone_only", "Xform")
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


def test_clone_plan_from_env_0_uses_flat_cfg_manifest(sim):
    """The homogeneous helper publishes exact env/global roots before construction."""
    robot = SimpleNamespace(prim_path="{ENV_REGEX_NS}/Robot", spawn=sim_utils.CuboidCfg(size=(0.1,) * 3))
    sensor = SimpleNamespace(prim_path="{ENV_REGEX_NS}/Robot/Sensor", spawn=None)
    prop = SimpleNamespace(
        prim_path="{ENV_REGEX_NS}/Prop",
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=[sim_utils.SphereCfg(radius=0.1)]),
        cloning_contexts=(UsdReplicateContext,),
    )
    light = SimpleNamespace(prim_path="/World/Light", spawn=sim_utils.DistantLightCfg(), cloning_contexts=())
    light_reference = SimpleNamespace(prim_path="/World/Light/Reference", spawn=None, cloning_contexts=())
    plan = cloner.clone_plan_from_env_0(cloner.CloneCfg(), (robot, sensor, prop, light, light_reference), 4, 1.0)

    assert sim.get_clone_plan() is plan
    assert plan.sources == ("/World/envs/env_0",)
    assert plan.destinations == ("/World/envs/env_{}",)
    assert plan.cfg_rows == {id(robot): (0,), id(sensor): (0,), id(prop): (0,)}
    assert plan.context_rows[UsdReplicateContext] == (0,)
    assert plan.global_paths == ("/World/Light",)
    assert plan.clone_mask.all() and plan.clone_mask.shape == (1, 4)
    np.testing.assert_array_equal(plan.env_ids, np.arange(4, dtype=np.int64))
    assert robot.prim_path == "/World/envs/env_[^/]+/Robot"
    assert robot.spawn.spawn_path == "/World/envs/env_0/Robot"
    assert prop.spawn.spawn_path is None
    assert prop.spawn.spawn_paths == ["/World/envs/env_0/Prop"]
    assert light.spawn.spawn_path == "/World/Light"


@pytest.mark.parametrize(
    ("cfgs", "error"),
    [
        ((SimpleNamespace(scene=object()),), AttributeError),
        (((SimpleNamespace(prim_path="/World/A"),),), AttributeError),
        (
            (
                SimpleNamespace(prim_path="{ENV_REGEX_NS}/Robot", spawn=sim_utils.CuboidCfg(size=(0.1,) * 3)),
                SimpleNamespace(
                    prim_path="{ENV_REGEX_NS}/Object",
                    spawn=sim_utils.MultiAssetSpawnerCfg(
                        assets_cfg=[sim_utils.SphereCfg(radius=0.1), sim_utils.CuboidCfg(size=(0.1,) * 3)]
                    ),
                ),
            ),
            ValueError,
        ),
    ],
    ids=["cfg_tree", "nested_tuple", "multi_variant"],
)
def test_clone_plan_from_env_0_rejects_invalid_manifests_atomically(sim, cfgs, error):
    """Cfg trees and heterogeneous spawners are rejected, leaving cfgs and the simulation's plan untouched."""
    with pytest.raises(error):
        cloner.clone_plan_from_env_0(cloner.CloneCfg(), cfgs, 2, 1.0)
    assert sim.get_clone_plan() is None
    for cfg in cfgs:
        if isinstance(cfg, SimpleNamespace) and hasattr(cfg, "spawn"):
            assert cfg.prim_path.startswith("{ENV_REGEX_NS}")
            assert getattr(cfg.spawn, "spawn_path", None) is None and getattr(cfg.spawn, "spawn_paths", None) is None


def test_replicate_session_clears_plan_when_asset_init_fails(sim):
    """ReplicateSession clears an unconsumed plan when construction raises."""
    with pytest.raises(RuntimeError, match="asset boom"):
        with ReplicateSession(cfgs=[], num_clones=2, env_spacing=1.0) as session:
            assert sim.get_clone_plan() is session.plan
            raise RuntimeError("asset boom")

    assert sim.get_clone_plan() is None
