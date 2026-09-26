# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the cloner path/query algebra.

These exercise :class:`isaaclab.cloner.path` and :class:`isaaclab.cloner.query`, which are pure
string/array operations over topology and path templates. They need no stage, no
simulator and no USD, so they live outside ``test/sim/``.
"""

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import make_clone_plan
from isaaclab.sim import CuboidCfg

##
# Path primitives.
##


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
    np.testing.assert_array_equal(
        cloner.path.get_parent_indices(("/Banana/Peel", "/Franka/hand", "/Banana", "/Franka", "/Banana_1", "/")),
        [2, 3, 5, 5, 5, -1],
    )


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
@pytest.mark.parametrize("dst_root", ["/World/other", "/World/other/", "/"])
def test_path_law_rebase_swaps_only_the_root(path, root, dst_root):
    """Rebasing changes only a complete root; stage roots and trailing slashes follow the same rules."""
    tail = cloner.path.relative_to(path, root)
    rebased = cloner.path.rebase(path, root, dst_root)
    if tail is None:
        assert rebased == path
    else:
        assert rebased == ((dst_root.rstrip("/") + tail) or "/")
        assert cloner.path.rebase(path, root, root) == path
    assert cloner.path.relative_to(path, "/") is not None
    assert cloner.path.relative_to(path, root) == cloner.path.relative_to(path, root.rstrip("/") or "/")
    assert cloner.path.rebase(path, root, "/World/x") == cloner.path.rebase(path, root + "/", "/World/x")


def test_path_match_captures_the_clone_slot():
    """Match captures the instance slot and rejects ambiguous templates."""
    tmpl = "/World/envs/env_{}/Robot"
    assert cloner.path.match("/World/envs/env_3/Robot/base", tmpl) == ("3", "/base")
    assert cloner.path.match("/World/envs/env_[^/]+/Robot", tmpl) == ("[^/]+", "")
    assert cloner.path.match("/World/envs/env_3/RobotArm", tmpl) is None
    for invalid in ("/World/envs/env_0/Robot", "/World/envs/env_{}/Robot/{}"):
        with pytest.raises(ValueError, match="exactly one"):
            cloner.path.match("/World/envs/env_3/Robot", invalid)


@pytest.mark.parametrize(
    "path_expr, template",
    [
        ("/World/envs/env_3/Robot/base", "/World/envs/env_{}/Robot"),
        ("/World/envs/env_12/Robot", "/World/envs/env_{}/Robot"),
        ("/World/scenes/0/Robot/link", "/World/scenes/{}/Robot"),
    ],
)
def test_path_law_template_split(path_expr, template):
    """P4: a match reassembles into the original path."""
    matched = cloner.path.match(path_expr, template)
    assert matched is not None
    assert template.format(matched.instance) + matched.suffix == path_expr


def test_cloner_imports_without_kit():
    """Importing the package in a clean interpreter must not drag in pxr.

    Topology and path queries must stay independent of USD execution. Loading pxr before
    Kit boots would also bind it to the wrong USD runtime.
    """
    probe = (
        "from isaaclab.cloner import ClonePlan, path, query; import sys; "
        "assert all(isinstance(namespace, type) and namespace.__module__ == ClonePlan.__module__ "
        "for namespace in (path, query)); "
        "assert not any(hasattr(query, name) for name in "
        "('get_matched_sources', 'path_to_source', 'path_to_clone', 'path_env_ids', 'iter_clones')); "
        "assert not any(hasattr(path, name) for name in "
        "('get_instance_paths', 'get_shared_paths', 'iter_subtree_copies', 'under', 'relativize', 'split')); "
        "print(any(n == 'isaaclab.cloner.usd' or n == 'pxr' or n.startswith('pxr.') for n in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "generic clone topology and queries imported the USD backend"

    # Only cloning execution and its lifecycle owners may access context instances.
    for path in Path(__file__).resolve().parents[3].glob("*/isaaclab*/**/*.py"):
        if "cloner" in path.parts or path.name in {"simulation_context.py", "render_context.py"}:
            continue
        assert not any(
            isinstance(node, ast.Attribute) and node.attr == "clone_contexts"
            for node in ast.walk(ast.parse(path.read_text()))
        ), f"{path} accessed clone contexts; consumers must use the plan and path/query utilities"


@pytest.mark.parametrize("shared_assets", [(), (0,)])
def test_world_topology_preserves_repeated_assets_and_shared_world(shared_assets):
    """One prototype declaration can belong to shared and replicated worlds more than once."""
    assets = tuple(AssetBaseCfg(prim_path=path, spawn=CuboidCfg(size=(1, 1, 1))) for path in ("/Banana", "/Franka"))
    worlds = ((0, 1), (0, 1, 1), (0, 0, 1), (1,))
    plan = make_clone_plan(assets, worlds, 16, shared_assets=shared_assets)
    assert isinstance(plan.topology, cloner.PrototypeWorldTopology)
    assert set(vars(plan)) == {"topology", "asset_cfgs", "env_template", "positions"}
    assert not any(hasattr(plan, field) for field in vars(plan.topology))
    assert all(actual is expected for actual, expected in zip(plan.asset_cfgs, assets, strict=True))
    np.testing.assert_array_equal(plan.topology.world_prototype_layout, np.repeat(np.arange(4), 4))
    np.testing.assert_array_equal(
        plan.topology.world_prototype_starts,
        [0, len(shared_assets), *(len(shared_assets) + np.cumsum([len(world) for world in worlds]))],
    )
    np.testing.assert_array_equal(plan.topology.world_prototypes, [*shared_assets, 0, 1, 0, 1, 1, 0, 0, 1, 1])
    templates, starts, indices, index_starts = cloner.path.get_world_prototype_asset_templates(
        plan, include_world_indices=True
    )
    assert starts is plan.topology.world_prototype_starts
    assert cloner.path.get_world_prototype_asset_templates(plan)[0] == templates
    assert cloner.path.get_asset_prototype_paths(plan) == (
        "/Banana" if shared_assets else "/World/envs/env_0/Banana",
        "/World/envs/env_0/Franka",
    )
    for group, names in enumerate(
        (
            tuple("Banana" for _ in shared_assets),
            ("Banana", "Franka"),
            ("Banana", "Franka", "Franka_1"),
            ("Banana", "Banana_1", "Franka"),
            ("Franka",),
        )
    ):
        expected_templates = tuple(("/" if group == 0 else plan.env_template + "/") + name for name in names)
        assert templates[starts[group] : starts[group + 1]] == expected_templates
        np.testing.assert_array_equal(
            indices[index_starts[group] : index_starts[group + 1]],
            [-1] if group == 0 else range(4 * (group - 1), 4 * group),
        )
    for path_expr, asset_ids, world_ids in (
        (None, [0, 1], [-1, 0, 1, 2, 3]),
        ("/Banana", [0], ([-1] if shared_assets else []) + [0, 1, 2]),
        ("/(Banana|Franka)", [0, 1], ([-1] if shared_assets else []) + [0, 1, 2, 3]),
        ("/Missing", [], []),
    ):
        for query, expected in (
            (cloner.path.get_asset_prototypes, asset_ids),
            (cloner.path.get_world_prototypes, world_ids),
        ):
            actual = query(plan, path_expr)
            assert actual.dtype == np.int32 and actual.ndim == 1
            np.testing.assert_array_equal(actual, expected)
    # Pure planning does not modify USD names or source poses.
    assert [cfg.prim_path for cfg in assets] == ["/Banana", "/Franka"]
    assert all(cfg.spawn.spawn_path is None for cfg in assets)
    # Reverse selection is indexing: retain occurrences rather than deduplicating asset IDs.
    for world_id, prototype_id in enumerate(plan.topology.world_prototype_layout):
        start, end = plan.topology.world_prototype_starts[prototype_id + 1 : prototype_id + 3]
        np.testing.assert_array_equal(plan.topology.world_prototypes[start:end], worlds[world_id // 4])


def test_world_topology_weights_empty_worlds_and_invalid_membership():
    assets = tuple(AssetBaseCfg(prim_path=f"/env_[^/]+/{name}") for name in ("Banana", "Franka"))
    plan = make_clone_plan(assets, ((0,), (), (1, 1), (0,)), 6, weights=(1, 0, 2, 0))
    np.testing.assert_array_equal(plan.topology.world_prototype_layout, [0, 0, 2, 2, 2, 2])
    np.testing.assert_array_equal(cloner.path.get_world_prototypes(plan), [-1, 0, 1, 2, 3])
    np.testing.assert_array_equal(cloner.path.get_asset_prototypes(plan, assets[0].prim_path), [0])
    np.testing.assert_array_equal(cloner.path.get_world_prototypes(plan, assets[0].prim_path), [0, 3])
    np.testing.assert_array_equal(cloner.path.get_asset_prototypes(plan, ".*/Banana"), [0])
    np.testing.assert_array_equal(cloner.path.get_asset_prototypes(plan, "/env_0/Banana"), [])
    empty = make_clone_plan((), ((),), 3)
    shared_only = make_clone_plan(assets, ((0,),), 0, shared_assets=(1, 1))
    np.testing.assert_array_equal(empty.topology.world_prototype_starts, [0, 0, 0])
    np.testing.assert_array_equal(empty.topology.world_prototype_layout, [0, 0, 0])
    np.testing.assert_array_equal(cloner.path.get_asset_prototypes(empty), [])
    np.testing.assert_array_equal(cloner.path.get_world_prototypes(empty), [-1, 0])
    np.testing.assert_array_equal(cloner.path.get_world_prototypes(empty, ".*"), [])
    for topology, selector, indices, starts in (
        (empty.topology, 0, [], [[0, 0, 0, 0, 0]]),
        (shared_only.topology, 0, [], [[0, 0]]),
        (shared_only.topology, 1, [-1, -1], [[0, 2]]),
    ):
        result = cloner.query.get_asset_prototype_world_index(topology, selector)
        for actual, expected in zip(result, (indices, starts), strict=True):
            np.testing.assert_array_equal(actual, expected)
    for members in ((2,), (-1,), (0.5,), ("0",)):
        with pytest.raises(ValueError, match="integer indices"):
            make_clone_plan(assets, (members,), 1)
    for selector in ("/Banana", np.array([0.5]), np.array([[0]])):
        with pytest.raises(TypeError, match="integer IDs"):
            cloner.query.get_asset_prototype_world_index(plan.topology, selector)


@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda:0"])
@pytest.mark.parametrize(
    "worlds, num_worlds, shared, selectors",
    [
        (((0, 1), (0, 1, 1), (0, 0, 1), (1,)), 16, (0, 0), [0, 1, 0, 2, 3, -1]),
        (((0,), (), (1, 1)), 6, (), [0, 1, 0, 2, -1]),
        (((),), 0, (1, 1), [1, 0, 1, -1, 2]),
        (((),), 3, (), [0, -1, 0, 2]),
        (((),), 3, (), []),
    ],
)
def test_batched_world_queries(device, worlds, num_worlds, shared, selectors):
    """NumPy, Warp, and graph replay agree with direct enumeration, including empty/repeated queries."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA graph validation requires CUDA")
    plan = make_clone_plan((0, 1, 2), worlds, num_worlds, shared_assets=shared)
    compositions = (shared, *(worlds[index] for index in plan.topology.world_prototype_layout))
    layout = (-1, *plan.topology.world_prototype_layout)
    ids = np.asarray(selectors, dtype=np.int32)
    capacity = len(ids) * max(sum(map(len, compositions)), num_worlds + 1)
    topology = plan.topology if device == "numpy" else cloner.to_warp(plan.topology, device)
    assert topology.num_asset_prototypes == len(plan.asset_cfgs)
    assert set(vars(topology)) == {
        "num_asset_prototypes",
        "world_prototypes",
        "world_prototype_starts",
        "world_prototype_layout",
    }
    if device == "cpu":
        for name in ("world_prototypes", "world_prototype_starts", "world_prototype_layout"):
            assert getattr(topology, name).ptr == getattr(plan.topology, name).ctypes.data
    if device != "numpy":
        with pytest.raises(TypeError, match="do not transfer"):
            cloner.query.get_asset_prototype_world_index(plan.topology, wp.empty(0, dtype=wp.int32, device=device))
    del plan  # The Warp topology must retain borrowed host storage after its construction inputs are gone.

    for query, by_asset, unique in (
        (cloner.query.get_asset_prototype_world_index, True, False),
        (cloner.query.get_asset_prototype_unique_world_index, True, True),
        (cloner.query.get_world_prototype_world_index, False, True),
    ):
        query_ids = ids if device == "numpy" else wp.array(ids, dtype=wp.int32, device=device)
        out = (
            (np.empty(capacity, dtype=np.int32), np.empty((len(ids), num_worlds + 2), dtype=np.int64))
            if device == "numpy"
            else (
                wp.empty(capacity, dtype=wp.int32, device=device),
                wp.empty((len(ids), num_worlds + 2), dtype=wp.int64, device=device),
            )
        )
        # Warm the kernels and scan before capture. Replay must use changed device query IDs.
        query(topology, query_ids, out=out)
        graph = None
        if device.startswith("cuda"):
            with wp.ScopedCapture(device=device) as capture:
                query(topology, query_ids, out=out)
            graph = capture.graph

        for selected in (ids, ids[::-1].copy(), np.zeros_like(ids)):
            expected, boundaries = [], []
            for prototype in selected:
                starts = [len(expected)]
                for world_id, (composition, world_prototype) in enumerate(zip(compositions, layout, strict=True), -1):
                    count = composition.count(prototype) if by_asset else int(world_prototype == prototype)
                    expected.extend([world_id] * (min(count, 1) if unique else count))
                    starts.append(len(expected))
                boundaries.append(starts)
            expected_starts = np.asarray(boundaries, dtype=np.int64).reshape(len(ids), num_worlds + 2)
            if device == "numpy":
                query(topology, selected, out=out)
                indices, starts = out
                exact = query(topology, selected)
                assert exact[0].dtype == np.int32 and exact[1].dtype == np.int64
                np.testing.assert_array_equal(exact[0], expected)
                np.testing.assert_array_equal(exact[1], expected_starts)
                if len(selected):
                    for scalar in (int(selected[0]), selected[0]):
                        scalar_indices, scalar_starts = query(topology, scalar)
                        np.testing.assert_array_equal(scalar_indices, expected[: expected_starts[0, -1]])
                        np.testing.assert_array_equal(scalar_starts, expected_starts[:1])
            else:
                query_ids.assign(selected)
                if graph is None:
                    query(topology, query_ids, out=out)
                else:
                    wp.capture_launch(graph)
                indices, starts = (array.numpy() for array in out)
            np.testing.assert_array_equal(indices[: len(expected)], expected)
            np.testing.assert_array_equal(starts, expected_starts)

        # Insufficient capacity must report the required size without writing a partial result.
        if len(expected) > 1:
            if device == "numpy":
                short = (np.full(1, -99, dtype=np.int32), out[1])
            else:
                short = (wp.full(1, -99, dtype=wp.int32, device=device), out[1])
            query(topology, selected if device == "numpy" else query_ids, out=short)
            values = short[0] if device == "numpy" else short[0].numpy()
            starts = short[1] if device == "numpy" else short[1].numpy()
            np.testing.assert_array_equal(values, [-99])
            assert starts[-1, -1] == len(expected)
