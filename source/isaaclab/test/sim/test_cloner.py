# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for USD cloner utilities (no PhysX dependency)."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

from unittest.mock import patch

import numpy as np
import pytest

from pxr import Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import UsdReplicateContext, grid_transforms, make_clone_plan, usd_replicate
from isaaclab.sim import SpawnerCfg, build_simulation_context
from isaaclab.sim.utils import queries
from isaaclab.test.utils import resolve_test_sim_device

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


@pytest.fixture
def sim():
    """Provide a fresh simulation context for each test.

    The cloner utilities author USD and plan bookkeeping only, with no device branch, so one device suffices.
    """
    with build_simulation_context(device=resolve_test_sim_device(), dt=0.01, add_lighting=False) as sim:
        yield sim


def test_usd_replicate_with_positions_and_mask(sim):
    """Replicate sources only to the envs selected by the mask."""
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

    # Validate replication follows the mask
    stage = sim_utils.get_current_stage()
    assert stage.GetPrimAtPath("/World/envs/env_0/Object/A").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_0/Object/B").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_1/Object/B").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_1/Object/A").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_2/Object/A").IsValid()


def test_usd_replicate_context_consumes_plan(sim):
    """UsdReplicateContext consumes the same plan used by every clone backend."""
    sim_utils.create_prim("/World/template/A", "Cube")
    sim_utils.create_prim("/World/envs", "Xform")

    stage = sim_utils.get_current_stage()
    asset = AssetBaseCfg(prim_path="/World/envs/env_[^/]+", spawn=SpawnerCfg(spawn_path="/World/template/A"))
    positions = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    plan = make_clone_plan((asset,), ((), (0,)), 2, positions=positions)
    UsdReplicateContext(sim).replicate(plan, (0,))

    assert not stage.GetPrimAtPath("/World/envs/env_0").IsA(UsdGeom.Cube)
    prim = stage.GetPrimAtPath("/World/envs/env_1")
    assert prim.IsValid() and prim.IsA(UsdGeom.Cube)
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
    with patch.object(Sdf, "CopySpec", wraps=Sdf.CopySpec) as copy:
        for suffix in ("", "/Camera"):
            sources, destinations = ["/World/envs/env_0" + suffix], ["/World/envs/env_{}" + suffix]
            usd_replicate(stage, sources, destinations, env_ids, positions=positions)
    assert all(call.args[1] != call.args[3] for call in copy.call_args_list), "CopySpec must never copy onto itself"
    assert any(str(call.args[3]) == "/World/envs/env_1" for call in copy.call_args_list)

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

    from isaaclab.cloner import fabric_notices

    class _FakeBindings:
        def validate_with(self, fabric_id: int) -> bool:
            raise AssertionError("missing usdrt should prevent fabric-id lookup")

    monkeypatch.setattr(fabric_notices, "get_bindings", lambda: _FakeBindings())

    real_import = builtins.__import__

    def _import_without_usdrt(name, *args, **kwargs):
        if name == "usdrt":
            raise ModuleNotFoundError("No module named 'usdrt'", name="usdrt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_without_usdrt)

    with fabric_notices.disabled_fabric_change_notifies(Usd.Stage.CreateInMemory()):
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


@pytest.mark.parametrize("with_clone_plan", [True, False])
def test_resolve_matching_prims_from_source(sim, with_clone_plan):
    """Discovery returns unique source prims, preserving order and multi-instance expressions."""
    stage = sim_utils.get_current_stage()
    for path in (
        "/World/envs/env_0/Robot/foo",
        "/World/envs/env_0/Robot/foo/bar",
        "/World/envs/env_0/Robot/other",
        "/World/envs/env_0/Robot/other/bar",
        "/World/envs/env_1/Robot/clone_only",
    ):
        stage.DefinePrim(path, "Xform")
    if with_clone_plan:
        assets = (AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Robot"),)
        cloner.clone_plan_from_env_0(cloner.CloneCfg(), assets, 2, 0.0)

    matches = queries.resolve_matching_prims_from_source(r"/World/envs/env_[^/]+/Robot/[^A]+")

    # the clone-only prim under env_1 would match the expression if cloned destinations were traversed
    assert [prim.GetPath().pathString for prim, _ in matches] == [
        "/World/envs/env_0/Robot/foo",
        "/World/envs/env_0/Robot/foo/bar",
        "/World/envs/env_0/Robot/other",
        "/World/envs/env_0/Robot/other/bar",
    ]
    assert [path_expr for _, path_expr in matches] == [
        "/World/envs/env_[^/]+/Robot/foo",
        "/World/envs/env_[^/]+/Robot/foo/bar",
        "/World/envs/env_[^/]+/Robot/other",
        "/World/envs/env_[^/]+/Robot/other/bar",
    ]

    # Each bar is reachable through two matching roots; distinct paths with the same name remain distinct.
    matches = queries.resolve_matching_prims_from_source(
        r"/World/envs/env_[^/]+/Robot/.*", predicate=lambda prim: prim.GetName() == "bar", expected_num_matches=2
    )
    assert [prim.GetPath().pathString for prim, _ in matches] == [
        "/World/envs/env_0/Robot/foo/bar",
        "/World/envs/env_0/Robot/other/bar",
    ]
    assert [path_expr for _, path_expr in matches] == [
        "/World/envs/env_[^/]+/Robot/foo/bar",
        "/World/envs/env_[^/]+/Robot/other/bar",
    ]

    # Path matching does not expand the matched prims into their subtrees.
    matches = queries.resolve_matching_prims_from_source(
        r"/World/envs/env_[^/]+/Robot/(foo|other)", expected_num_matches=2
    )
    assert [(prim.GetPath().pathString, path) for prim, path in matches] == [
        ("/World/envs/env_0/Robot/foo", "/World/envs/env_[^/]+/Robot/foo"),
        ("/World/envs/env_0/Robot/other", "/World/envs/env_[^/]+/Robot/other"),
    ]

    if with_clone_plan:
        stage.DefinePrim("/World/template/Robot/foo/bar", "Xform")
        asset = AssetBaseCfg(
            prim_path="/World/envs/env_[^/]+/Robot", spawn=SpawnerCfg(spawn_path="/World/template/Robot")
        )
        sim.set_clone_plan(make_clone_plan((asset,), ((0,),), 2))

        matches = queries.resolve_matching_prims_from_source(r"/World/envs/env_[^/]+/Robot/foo", expected_num_matches=1)
        assert [(prim.GetPath().pathString, path) for prim, path in matches] == [
            ("/World/template/Robot/foo", "/World/envs/env_[^/]+/Robot/foo")
        ]
