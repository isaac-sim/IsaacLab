# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-torch tests for direct :class:`~isaaclab.cloner.ClonePlan` construction."""

from types import SimpleNamespace

import torch

import isaaclab.sim as sim_utils
from isaaclab.cloner import ClonePlan, InclusionSet, make_clone_plan, make_valid_clone_combinations, sequential


def _single_cfg(path: str):
    return SimpleNamespace(prim_path=path, spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)))


def _multi_cfg(path: str, count: int):
    return SimpleNamespace(
        prim_path=path,
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=[sim_utils.SphereCfg(radius=0.1) for _ in range(count)]),
    )


def test_clone_plan_carries_flat_contract():
    """ClonePlan stores flat source rows, destination templates, and row masks."""
    mask = torch.ones((2, 4), dtype=torch.bool)
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_0/Object"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
        clone_mask=mask,
    )

    assert plan.sources == ("/World/envs/env_0/Robot", "/World/envs/env_0/Object")
    assert plan.destinations == ("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object")
    assert plan.clone_mask is mask


def test_make_clone_plan_flattens_spawn_variants():
    """Env-scoped cfgs flatten into one row per prototype variant."""
    robot = _single_cfg("/World/envs/env_.*/Robot")
    object_cfg = _multi_cfg("/World/envs/env_.*/Object", 2)

    plan = make_clone_plan(
        [robot, object_cfg],
        num_clones=4,
        env_spacing=1.0,
        device="cpu",
        clone_strategy=sequential,
    )

    assert plan.sources == (
        "/World/envs/env_0/Robot",
        "/World/envs/env_0/Object",
        "/World/envs/env_1/Object",
    )
    assert plan.destinations == (
        "/World/envs/env_{}/Robot",
        "/World/envs/env_{}/Object",
        "/World/envs/env_{}/Object",
    )
    assert plan.clone_mask.shape == (3, 4)
    assert plan.clone_mask[0].all()
    assert plan.clone_mask[1].tolist() == [True, False, True, False]
    assert plan.clone_mask[2].tolist() == [False, True, False, True]
    assert robot.spawn.spawn_path == "/World/envs/env_0/Robot"
    assert object_cfg.spawn.spawn_paths == ["/World/envs/env_0/Object", "/World/envs/env_1/Object"]


def test_make_clone_plan_enumerates_complete_combinations():
    """Every environment gets exactly one prototype from each source group."""
    robot = _multi_cfg("/World/envs/env_.*/Robot", 2)
    object_cfg = _multi_cfg("/World/envs/env_.*/Object", 3)

    plan = make_clone_plan(
        [robot, object_cfg],
        num_clones=6,
        env_spacing=1.0,
        device="cpu",
        clone_strategy=sequential,
    )

    assert plan.clone_mask.shape == (5, 6)
    robot_rows = plan.clone_mask[:2]
    object_rows = plan.clone_mask[2:]
    assert torch.all(robot_rows.sum(dim=0) == 1)
    assert torch.all(object_rows.sum(dim=0) == 1)


def test_make_clone_plan_rejects_empty_variant_groups():
    """Each env-scoped spawner must have at least one source variant."""
    cfg = _multi_cfg("/World/envs/env_.*/Object", 0)

    try:
        make_clone_plan([cfg], num_clones=2, env_spacing=1.0, device="cpu", clone_strategy=sequential)
    except ValueError as exc:
        assert "at least one variant" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_make_valid_clone_combinations_encodes_absent_assets():
    """Clone combinations use -1 for assets absent from a legal row."""
    combos = make_valid_clone_combinations(
        ["robot", "table", "cabinet"],
        [2, 1, 1],
        [
            InclusionSet(assets=["robot", "table"], weight=1),
            InclusionSet(assets=["cabinet"], weight=1),
        ],
        device="cpu",
    )

    assert combos.tolist() == [
        [0, 0, -1],
        [1, 0, -1],
        [-1, -1, 0],
    ]


def test_make_valid_clone_combinations_keeps_unclaimed_assets_global():
    """Assets unclaimed by combinations are present in every legal row."""
    combos = make_valid_clone_combinations(
        ["robot", "floor"],
        [1, 1],
        [InclusionSet(assets=["robot"], weight=2)],
        device="cpu",
    )

    assert combos.tolist() == [[0, 0], [0, 0]]


def test_make_valid_clone_combinations_rejects_unknown_assets():
    """Clone-combination asset names must match planned scene assets."""
    try:
        make_valid_clone_combinations(
            ["robot"],
            [1],
            [InclusionSet(assets=["cabinet"], weight=1)],
            device="cpu",
        )
    except ValueError as exc:
        assert "Unknown assets" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_make_valid_clone_combinations_allows_selector_only_scene_assets():
    """Full-scene validation allows non-cloned assets in clone-combination entries."""
    combos = make_valid_clone_combinations(
        ["robot"],
        [1],
        [InclusionSet(assets=["robot", "ee_frame"], weight=1)],
        device="cpu",
        all_asset_names=["robot", "ee_frame"],
    )

    assert combos.tolist() == [[0]]


def test_make_clone_plan_rejects_invalid_valid_set_entries():
    """Valid-set entries must reference existing source variants or -1."""
    robot = _single_cfg("/World/envs/env_.*/Robot")
    invalid_set = torch.tensor([[1]], dtype=torch.long)

    try:
        make_clone_plan(
            [robot],
            num_clones=1,
            env_spacing=1.0,
            device="cpu",
            clone_strategy=sequential,
            valid_set=invalid_set,
        )
    except ValueError as exc:
        assert "outside [-1, group_size)" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_make_clone_plan_skips_absent_valid_set_entries():
    """Valid-set -1 entries leave a source row unspawned for that env."""
    robot = _single_cfg("/World/envs/env_.*/Robot")
    cabinet = _single_cfg("/World/envs/env_.*/Cabinet")
    valid_set = make_valid_clone_combinations(
        ["robot", "cabinet"],
        [1, 1],
        [
            InclusionSet(assets=["robot"], weight=1),
            InclusionSet(assets=["cabinet"], weight=0),
        ],
        device="cpu",
    )

    plan = make_clone_plan(
        [robot, cabinet],
        num_clones=3,
        env_spacing=1.0,
        device="cpu",
        clone_strategy=sequential,
        valid_set=valid_set,
    )

    assert plan.clone_mask.tolist() == [[True, True, True], [False, False, False]]
    assert robot.spawn.spawn_path == "/World/envs/env_0/Robot"
    assert cabinet.spawn.spawn_path is None
