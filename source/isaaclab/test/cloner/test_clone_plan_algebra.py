# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the cloner path/query algebra.

These exercise :mod:`isaaclab.cloner.path` and :mod:`isaaclab.cloner.query`, which are pure
string/array operations over topology and path templates. They need no stage, no
simulator and no USD, so they live outside ``test/sim/``.
"""

import subprocess
import sys

import numpy as np
import pytest

from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import make_clone_plan
from isaaclab.sim import CuboidCfg

##
# Path primitives.
##


def test_path_split():
    """Split clone destination templates around their clone slot."""
    assert cloner.path.split("/World/envs/env_{}/Robot") == ("/World/envs/env_", "/Robot")
    assert cloner.path.split("/World/scenes/{}/") == ("/World/scenes/", "")
    with pytest.raises(ValueError, match="exactly one"):
        cloner.path.split("/World/envs/env_0/Robot")
    # A second slot would survive into the suffix and break the later format call.
    with pytest.raises(ValueError, match="exactly one"):
        cloner.path.split("/World/envs/env_{}/Robot/{}")


def test_path_relative_to():
    """relative_to strips a concrete root on a boundary, or returns None."""
    root = "/World/envs/env_0/Robot"
    assert cloner.path.relative_to("/World/envs/env_0/Robot/base", root) == "/base"
    assert cloner.path.relative_to("/World/envs/env_0/Robot", root) == ""
    assert cloner.path.relative_to("/World/envs/env_0/RobotArm", root) is None
    assert cloner.path.relative_to("/World/ground", root) is None
    # The stage root is not a segment: stripping it keeps the leading slash.
    assert cloner.path.relative_to("/World/envs/env_0", "/") == "/World/envs/env_0"
    assert cloner.path.relative_to("/", "/") == ""


def test_expand_env_regex_ns_preserves_regex_quantifiers():
    """Macro expansion changes only the named macro, not braces owned by the regex."""
    path_expr = r"{ENV_REGEX_NS}/Robot/link_[0-9]{2}"

    assert cloner.expand_env_regex_ns(path_expr) == r"/World/envs/env_[^/]+/Robot/link_[0-9]{2}"
    assert cloner.expand_env_regex_ns(path_expr, "/World/scenes/scene_{}") == (
        r"/World/scenes/scene_[^/]+/Robot/link_[0-9]{2}"
    )


# (path, root) pairs spanning the ordinary cases plus the stage root and trailing slashes.
_PATH_ROOT_CASES = [
    ("/World/envs/env_0/Robot/base", "/World/envs/env_0/Robot"),
    ("/World/envs/env_0/Robot", "/World/envs/env_0/Robot"),
    ("/World/envs/env_0/RobotArm", "/World/envs/env_0/Robot"),
    ("/World/ground", "/World/envs/env_0/Robot"),
    ("/World/envs/env_0/Robot", "/World/envs/env_0/"),
    ("/World/envs/env_0", "/"),
    ("/", "/"),
    ("/World", "/World"),
]


@pytest.mark.parametrize("path, root", _PATH_ROOT_CASES)
def test_path_law_membership(path, root):
    """P1: under() holds exactly when relative_to() resolves."""
    assert cloner.path.under(path, root) == (cloner.path.relative_to(path, root) is not None)


@pytest.mark.parametrize("path, root", _PATH_ROOT_CASES)
@pytest.mark.parametrize("dst_root", ["/World/other", "/World/other/", "/"])
def test_path_law_rebase_swaps_only_the_root(path, root, dst_root):
    """P2: rebase is the destination root plus the tail, and rebasing onto the same root is a no-op."""
    tail = cloner.path.relative_to(path, root)
    rebased = cloner.path.rebase(path, root, dst_root)
    if tail is None:
        assert rebased == path
    else:
        assert rebased == ((dst_root.rstrip("/") + tail) or "/")
        assert cloner.path.rebase(path, root, root) == path


@pytest.mark.parametrize("path, root", _PATH_ROOT_CASES)
def test_path_law_no_special_cases(path, root):
    """P3: "/" is the root of every absolute path, and a trailing slash is insignificant."""
    assert cloner.path.under(path, "/")
    assert cloner.path.relative_to(path, root) == cloner.path.relative_to(path, root.rstrip("/") or "/")
    assert cloner.path.rebase(path, root, "/World/x") == cloner.path.rebase(path, root + "/", "/World/x")


def test_path_match_captures_the_clone_slot():
    """match keeps the instance the template's slot captured, which relativize discards."""
    tmpl = "/World/envs/env_{}/Robot"
    assert cloner.path.match("/World/envs/env_3/Robot/base", tmpl) == ("3", "/base")
    assert cloner.path.match("/World/envs/env_[^/]+/Robot", tmpl) == ("[^/]+", "")
    assert cloner.path.match("/World/envs/env_3/RobotArm", tmpl) is None


@pytest.mark.parametrize(
    "path_expr, template",
    [
        ("/World/envs/env_3/Robot/base", "/World/envs/env_{}/Robot"),
        ("/World/envs/env_12/Robot", "/World/envs/env_{}/Robot"),
        ("/World/scenes/0/Robot/link", "/World/scenes/{}/Robot"),
    ],
)
def test_path_law_template_split(path_expr, template):
    """P4: a match reassembles into the original path, and its suffix is what relativize returns."""
    matched = cloner.path.match(path_expr, template)
    assert matched is not None
    assert template.format(matched.instance) + matched.suffix == path_expr
    assert cloner.path.relativize(path_expr, template) == matched.suffix


def test_cloner_imports_without_kit():
    """Importing the package in a clean interpreter must not drag in pxr.

    Topology and path queries must stay independent of USD execution. Loading pxr before
    Kit boots would also bind it to the wrong USD runtime.
    """
    probe = (
        "from isaaclab.cloner import ClonePlan, query; import sys; "
        "assert not any(hasattr(query, name) for name in "
        "('get_matched_sources', 'path_to_source', 'path_to_clone', 'path_env_ids', 'iter_clones')); "
        "print(any(n == 'isaaclab.cloner.usd' or n == 'pxr' or n.startswith('pxr.') for n in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "generic clone topology and queries imported the USD backend"


@pytest.mark.parametrize("shared_assets", [(), (0,)])
def test_world_topology_preserves_repeated_assets_and_shared_world(shared_assets):
    """One prototype declaration can belong to shared and replicated worlds more than once."""
    assets = tuple(AssetBaseCfg(prim_path=path, spawn=CuboidCfg(size=(1, 1, 1))) for path in ("/Banana", "/Franka"))
    worlds = ((0, 1), (0, 1, 1), (0, 0, 1), (1,))
    plan = make_clone_plan(assets, worlds, 16, shared_assets=shared_assets)
    assert all(actual is expected for actual, expected in zip(plan.asset_prototypes, assets, strict=True))
    np.testing.assert_array_equal(plan.world_prototype_layout, np.repeat(np.arange(4), 4))
    np.testing.assert_array_equal(
        plan.world_prototype_starts,
        [0, len(shared_assets), *(len(shared_assets) + np.cumsum([len(world) for world in worlds]))],
    )
    np.testing.assert_array_equal(plan.world_prototypes, [*shared_assets, 0, 1, 0, 1, 1, 0, 0, 1, 1])
    for path_expr, asset_ids, world_ids in (
        (None, [0, 1], [-1, 0, 1, 2, 3]),
        ("/Banana", [0], ([-1] if shared_assets else []) + [0, 1, 2]),
        ("/(Banana|Franka)", [0, 1], ([-1] if shared_assets else []) + [0, 1, 2, 3]),
        ("/Missing", [], []),
    ):
        for query, expected in (
            (cloner.query.get_asset_prototypes, asset_ids),
            (cloner.query.get_world_prototypes, world_ids),
        ):
            actual = query(plan, path_expr)
            assert actual.dtype == np.int32 and actual.ndim == 1
            np.testing.assert_array_equal(actual, expected)
    for selector, asset_ids in ((0, {0}), (np.int32(1), {1}), ("/Banana", {0}), (".*", {0, 1}), ("/Missing", set())):
        expected = [-1 for asset in shared_assets if asset in asset_ids]
        expected += [world for world in range(16) for asset in worlds[world // 4] if asset in asset_ids]
        world_indices, world_starts = cloner.query.get_asset_prototype_world_index(plan, selector)
        for actual, values in (
            (world_indices, expected),
            (world_starts, [sum(index < world for index in expected) for world in range(-1, 17)]),
            (cloner.query.get_asset_prototype_unique_world_index(plan, selector), sorted(set(expected))),
        ):
            assert actual.dtype == np.int32 and actual.ndim == 1
            np.testing.assert_array_equal(actual, values)
        if isinstance(selector, str):
            actual = cloner.query.get_world_prototype_world_index(plan, selector)
            assert actual.dtype == np.int32 and actual.ndim == 1
            np.testing.assert_array_equal(actual, sorted(set(expected)))
    for prototype in range(-1, len(worlds)):
        actual = cloner.query.get_world_prototype_world_index(plan, prototype)
        assert actual.dtype == np.int32 and actual.ndim == 1
        np.testing.assert_array_equal(actual, [-1] if prototype == -1 else range(4 * prototype, 4 * (prototype + 1)))
    # Pure planning does not modify USD names or source poses.
    assert [cfg.prim_path for cfg in assets] == ["/Banana", "/Franka"]
    assert all(cfg.spawn.spawn_path is None for cfg in assets)
    # Reverse selection is indexing: retain occurrences rather than deduplicating asset IDs.
    for world_id, prototype_id in enumerate(plan.world_prototype_layout):
        start, end = plan.world_prototype_starts[prototype_id + 1 : prototype_id + 3]
        np.testing.assert_array_equal(plan.world_prototypes[start:end], worlds[world_id // 4])


def test_world_topology_weights_empty_worlds_and_invalid_membership():
    assets = tuple(AssetBaseCfg(prim_path=f"/env_[^/]+/{name}") for name in ("Banana", "Franka"))
    plan = make_clone_plan(assets, ((0,), (), (1, 1), (0,)), 6, weights=(1, 0, 2, 0))
    np.testing.assert_array_equal(plan.world_prototype_layout, [0, 0, 2, 2, 2, 2])
    np.testing.assert_array_equal(cloner.query.get_world_prototypes(plan), [-1, 0, 1, 2, 3])
    np.testing.assert_array_equal(cloner.query.get_asset_prototypes(plan, assets[0].prim_path), [0])
    np.testing.assert_array_equal(cloner.query.get_world_prototypes(plan, assets[0].prim_path), [0, 3])
    np.testing.assert_array_equal(cloner.query.get_asset_prototypes(plan, ".*/Banana"), [0])
    np.testing.assert_array_equal(cloner.query.get_asset_prototypes(plan, "/env_0/Banana"), [])
    empty = make_clone_plan((), ((),), 3)
    shared_only = make_clone_plan(assets, ((0,),), 0, shared_assets=(1, 1))
    np.testing.assert_array_equal(empty.world_prototype_starts, [0, 0, 0])
    np.testing.assert_array_equal(empty.world_prototype_layout, [0, 0, 0])
    for actual, expected in (
        (cloner.query.get_asset_prototypes(empty), []),
        (cloner.query.get_world_prototypes(empty), [-1, 0]),
        (cloner.query.get_world_prototypes(empty, ".*"), []),
        (cloner.query.get_asset_prototype_unique_world_index(empty, 0), []),
        (cloner.query.get_world_prototype_world_index(empty, 0), [0, 1, 2]),
        (cloner.query.get_world_prototype_world_index(empty, ".*"), []),
        (cloner.query.get_world_prototype_world_index(empty, -1), [-1]),
        (cloner.query.get_world_prototype_world_index(plan, 1), []),
        (cloner.query.get_world_prototype_world_index(plan, 3), []),
        (cloner.query.get_world_prototype_world_index(plan, assets[0].prim_path), [0, 1]),
        (cloner.query.get_asset_prototype_unique_world_index(shared_only, assets[1].prim_path), [-1]),
        (cloner.query.get_world_prototype_world_index(shared_only, assets[0].prim_path), []),
        (cloner.query.get_world_prototype_world_index(shared_only, assets[1].prim_path), [-1]),
    ):
        assert actual.dtype == np.int32 and actual.ndim == 1
        np.testing.assert_array_equal(actual, expected)
    for topology, selector, indices, starts in (
        (empty, ".*", [], [0, 0, 0, 0, 0]),
        (shared_only, 0, [], [0, 0]),
        (shared_only, 1, [-1, -1], [0, 2]),
    ):
        result = cloner.query.get_asset_prototype_world_index(topology, selector)
        for actual, expected in zip(result, (indices, starts), strict=True):
            assert actual.dtype == np.int32 and actual.ndim == 1
            np.testing.assert_array_equal(actual, expected)
    for members in ((2,), (-1,), (0.5,), ("0",)):
        with pytest.raises(ValueError, match="integer indices"):
            make_clone_plan(assets, (members,), 1)
