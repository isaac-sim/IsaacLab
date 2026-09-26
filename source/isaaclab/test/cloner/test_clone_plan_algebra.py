# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the cloner path/query algebra.

These exercise :mod:`isaaclab.cloner.path` and :mod:`isaaclab.cloner.query`, which are pure
string/array operations over topology and native instance mappings. They need no stage, no
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


##
# Clone plans used by the query tests. Each covers a distinct shape of the source/clone
# relation, and the law tests below run over all of them.
##


def _instances(templates, variants, prototype_paths):
    return tuple(
        (index, source, template, np.flatnonzero(np.asarray(selected) == variant))
        for index, (template, selected, paths) in enumerate(zip(templates, variants, prototype_paths, strict=True))
        for variant, source in enumerate(paths)
    )


def _robot_instances(variants=(0, 0, 0, 0)):
    """One declared robot, absent in environments selecting -1."""
    return _instances(("/World/envs/env_{}/Robot",), [variants], [("/World/envs/env_0/Robot",)])


def _wide_env_id_instances():
    """Two variants of one asset over 12 envs, the second starting at a two-digit env id."""
    return _instances(
        ("/World/envs/env_{}/Object",),
        [[0] * 10 + [1, 1]],
        [("/World/envs/env_0/Object", "/World/envs/env_10/Object")],
    )


MAPPINGS = {
    "homogeneous": _robot_instances(),
    "partial_coverage": _robot_instances((0, 0, -1, 0)),
    "two_variants": _instances(
        ("/World/envs/env_{}/Object",),
        [[0, 0, 1, 0]],
        [("/World/envs/env_0/Object", "/World/envs/env_2/Object")],
    ),
    "nested_prototype": _instances(
        ("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot/wrist/Camera"),
        [[0, 0, 0, 0], [0, -1, 0, 0]],
        [("/World/envs/env_0/Robot",), ("/World/envs/env_0/Robot/wrist/Camera",)],
    ),
    "distinct_env_root": _instances(
        ("/World/scenes/{}/Robot",),
        [[0, 0]],
        [("/World/source/Robot",)],
    ),
    "wide_env_ids": _wide_env_id_instances(),
}


##
# Query operations.
##


def test_path_env_ids():
    """path_env_ids returns the environments a source-space path reaches."""
    assert cloner.query.path_env_ids(_robot_instances(), "/World/envs/env_0/Robot/base") == (0, 1, 2, 3)
    partial = _robot_instances((0, 0, -1, 0))
    assert cloner.query.path_env_ids(partial, "/World/envs/env_0/Robot/base") == (0, 1, 3)
    assert cloner.query.path_env_ids(_robot_instances(), "/World/ground") == ()
    # An unowned path has no clone; owned paths are covered by the Q1/Q2 laws below.
    assert cloner.query.path_to_clone(partial, "/World/ground", 0) is None


def test_path_to_source_nested_templates_pick_most_specific():
    """A path owned by both an ancestor and a descendant template resolves to the descendant."""
    instances = _instances(
        ("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot/ee_link/palm_link/Camera"),
        [[0, 0], [0, 0]],
        [("/World/envs/env_0/Robot",), ("/World/envs/env_0/Robot/ee_link/palm_link/Camera",)],
    )

    # The camera path matches both templates; the more specific (longer-matching) one wins.
    resolved = cloner.query.path_to_source(instances, "/World/envs/env_0/Robot/ee_link/palm_link/Camera")
    assert resolved == (
        "/World/envs/env_0/Robot/ee_link/palm_link/Camera",
        "/World/envs/env_[^/]+/Robot/ee_link/palm_link/Camera",
        "",
    )

    # A path that only the ancestor template owns still resolves against it with its suffix.
    resolved = cloner.query.path_to_source(instances, "/World/envs/env_0/Robot/base")
    assert resolved == ("/World/envs/env_0/Robot", "/World/envs/env_[^/]+/Robot", "/base")


def test_path_to_source_selects_the_declared_variant():
    """One declaration selects the correct prototype or no instance for each environment."""
    instances = _instances(
        ("/World/envs/env_{}/Object",),
        [[0, 1, 0, -1]],
        [("/World/envs/env_0/Object", "/World/envs/env_1/Object")],
    )

    # Without an env id, the first populated variant represents the asset.
    resolved = cloner.query.path_to_source(instances, "/World/envs/env_[^/]+/Object/Body/Camera")
    assert resolved == ("/World/envs/env_0/Object", "/World/envs/env_[^/]+/Object", "/Body/Camera")

    # With an env id, the variant that actually populates that env is reported.
    resolved = cloner.query.path_to_source(instances, "/World/envs/env_[^/]+/Object/Body/Camera", env_id=1)
    assert resolved == ("/World/envs/env_1/Object", "/World/envs/env_[^/]+/Object", "/Body/Camera")

    # No variant populates env 3.
    assert cloner.query.path_to_source(instances, "/World/envs/env_[^/]+/Object/Body/Camera", env_id=3) is None


def test_get_matched_sources_returns_populated_variants():
    """Return each active prototype with only the environments selecting it."""
    instances = _instances(
        ("/World/envs/env_{}/Object",),
        [[0, 0, 1, 1]],
        [("/World/envs/env_0/Object", "/World/envs/env_1/Object", None)],
    )

    matches = cloner.query.get_matched_sources(instances, "/World/envs/env_[^/]+/Object/Body/Camera")

    assert matches == [
        (
            "/World/envs/env_0/Object",
            "/World/envs/env_{}/Object",
            "/World/envs/env_0/Object/Body/Camera",
            (0, 1),
        ),
        (
            "/World/envs/env_1/Object",
            "/World/envs/env_{}/Object",
            "/World/envs/env_1/Object/Body/Camera",
            (2, 3),
        ),
    ]
    absent = tuple((index, source, template, np.empty(0, dtype=np.int64)) for index, source, template, _ in instances)
    assert cloner.query.path_to_source(absent, "/World/envs/env_[^/]+/Object/Body") is None
    assert cloner.query.get_matched_sources(absent, "/World/envs/env_[^/]+/Object/Body") == []


def test_get_matched_sources_skips_declarations_without_envs():
    """A nearer template populating no env does not hide the populated ancestor owning the path."""
    instances = _instances(
        ("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot/wrist/Camera"),
        [[0, 0, -1, -1], [-1, -1, -1, -1]],
        [("/World/envs/env_0/Robot",), ("/World/envs/env_0/Robot/wrist/Camera",)],
    )

    assert cloner.query.get_matched_sources(instances, "/World/envs/env_[^/]+/Robot/wrist/Camera") == [
        (
            "/World/envs/env_0/Robot",
            "/World/envs/env_{}/Robot",
            "/World/envs/env_0/Robot/wrist/Camera",
            (0, 1),
        )
    ]


def test_get_matched_sources_distinct_env_root():
    """The destination template need not sit under the default env root."""
    instances = MAPPINGS["distinct_env_root"]

    assert cloner.query.get_matched_sources(instances, "/World/scenes/[^/]+/Robot/base") == [
        ("/World/source/Robot", "/World/scenes/{}/Robot", "/World/source/Robot/base", (0, 1))
    ]


def test_get_matched_sources_ranks_variants_independently_of_env_id_width():
    """Regression: a variant is not ranked out because its first env id has more digits.

    Specificity is the suffix below the destination template, which does not depend on the
    env ids a row happens to populate. Ranking by the *formatted* template instead made
    ``env_10`` look more specific than ``env_0`` and silently dropped the first variant for
    any scene with more than ten envs.
    """
    instances = _wide_env_id_instances()

    matches = cloner.query.get_matched_sources(instances, "/World/envs/env_[^/]+/Object/Body")

    assert [match[0] for match in matches] == ["/World/envs/env_0/Object", "/World/envs/env_10/Object"]
    assert [match[3] for match in matches] == [tuple(range(10)), (10, 11)]


##
# Laws, checked over every native mapping above.
##


def _expected_clone(instances, path: str, env_id: int) -> str | None:
    """Independent oracle: keep the suffix below the nearest populated source."""
    owners = [
        (source, template, ids)
        for _, source, template, ids in instances
        if len(ids) and cloner.path.under(path, source)
    ]
    if not owners:
        return None
    source, template, ids = max(owners, key=lambda owner: len(owner[0]))
    return template.format(env_id) + path[len(source) :] if env_id in ids else None


def _probe_paths(instances) -> list[str]:
    return [
        source + tail
        for source in dict.fromkeys(source for _, source, _, _ in instances if source is not None)
        for tail in ("", "/base", "/link/child")
    ]


@pytest.mark.parametrize("plan_name", sorted(MAPPINGS))
def test_query_law_factorization_and_domain(plan_name):
    """Q1/Q2: clones preserve prototype suffixes and exist only in their declared worlds."""
    instances = MAPPINGS[plan_name]
    num_envs = 1 + max(int(world_id) for _, _, _, world_ids in instances for world_id in world_ids)

    for path in _probe_paths(instances):
        reached = cloner.query.path_env_ids(instances, path)
        assert set(reached) <= set(range(num_envs))

        for env_id in range(num_envs):
            clone = cloner.query.path_to_clone(instances, path, env_id)
            # Q2: a clone exists exactly in the declared destination worlds.
            assert (clone is not None) == (env_id in reached)
            if clone is None:
                continue
            assert clone == _expected_clone(instances, path, env_id)


@pytest.mark.parametrize("plan_name", sorted(MAPPINGS))
def test_query_law_round_trip(plan_name):
    """Q3: resolving a clone path returns the prototype it was cloned from.

    A clone path is concrete, so no env id has to be supplied: the clone slot names it.
    """
    instances = MAPPINGS[plan_name]

    for path in _probe_paths(instances):
        for env_id in cloner.query.path_env_ids(instances, path):
            clone = cloner.query.path_to_clone(instances, path, env_id)
            assert clone is not None

            resolved = cloner.query.path_to_source(instances, clone)
            assert resolved is not None, f"{clone} did not resolve back for env {env_id}"
            source, _glob, suffix = resolved
            assert source + suffix == path
            # Naming the env explicitly must agree with reading it out of the path.
            assert cloner.query.path_to_source(instances, clone, env_id=env_id) == resolved


def test_query_resolve_distinguishes_concrete_paths_from_wildcards():
    """A concrete clone path names its env; a wildcard expression stands for all of them.

    The clone slot is what separates the two: ``env_2`` selects the variant that populates
    env 2, while ``env_.*`` cannot and falls back to the first populated variant unless the
    caller names an env.
    """
    instances = MAPPINGS["two_variants"]
    concrete = "/World/envs/env_2/Object/base"

    # Concrete: resolves to the variant env 2 was actually cloned from.
    source, _glob, suffix = cloner.query.path_to_source(instances, concrete)
    assert source + suffix == "/World/envs/env_2/Object/base"

    # Wildcard: one-to-many, so it reports a representative variant...
    wildcard = "/World/envs/env_[^/]+/Object/base"
    source, _glob, suffix = cloner.query.path_to_source(instances, wildcard)
    assert source + suffix == "/World/envs/env_0/Object/base"

    # ...unless the caller names the env it means.
    source, _glob, suffix = cloner.query.path_to_source(instances, wildcard, env_id=2)
    assert source + suffix == "/World/envs/env_2/Object/base"


def test_query_preserves_noncontiguous_world_ids():
    """A native mapping targeting worlds (2, 5) reports those IDs, not positional indices."""
    instances = ((0, "/World/envs/env_2/Robot", "/World/envs/env_{}/Robot", np.asarray([2, 5])),)
    path = "/World/envs/env_2/Robot/base"

    assert cloner.query.path_env_ids(instances, path) == (2, 5)
    assert cloner.query.path_to_clone(instances, path, 5) == "/World/envs/env_5/Robot/base"
    # World 1 is not targeted by this mapping.
    assert cloner.query.path_to_clone(instances, path, 1) is None
    assert cloner.query.get_matched_sources(instances, "/World/envs/env_[^/]+/Robot")[0][3] == (2, 5)

    source, _glob, suffix = cloner.query.path_to_source(instances, "/World/envs/env_5/Robot/base")
    assert source + suffix == path


@pytest.mark.parametrize("env_id", [-1, 4, 99])
def test_query_rejects_env_ids_outside_the_mapping(env_id):
    """Out-of-range and negative ids resolve to nothing instead of wrapping the mask."""
    instances = _robot_instances()
    assert cloner.query.path_to_clone(instances, "/World/envs/env_0/Robot/base", env_id) is None
    assert cloner.query.path_to_source(instances, "/World/envs/env_[^/]+/Robot", env_id=env_id) is None


def test_cloner_imports_without_kit():
    """Importing the package in a clean interpreter must not drag in pxr.

    Topology and path queries must stay independent of USD execution. Loading pxr before
    Kit boots would also bind it to the wrong USD runtime.
    """
    probe = (
        "from isaaclab.cloner import ClonePlan, query; import sys; "
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
        for unique in (False, True):
            actual = cloner.query.get_asset_prototype_world_index(plan, selector, unique=unique)
            assert actual.dtype == np.int32 and actual.ndim == 1
            np.testing.assert_array_equal(actual, sorted(set(expected)) if unique else expected)
    for prototype in range(-1, len(worlds)):
        actual = cloner.query.get_world_prototype_world_index(plan, prototype)
        assert actual.dtype == np.int32 and actual.ndim == 1
        np.testing.assert_array_equal(actual, [-1] if prototype == -1 else range(4 * prototype, 4 * (prototype + 1)))
    # Pure planning does not modify USD names or source poses.
    assert [cfg.prim_path for cfg in assets] == ["/Banana", "/Franka"]
    assert all(cfg.spawn.spawn_path is None for cfg in assets)


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
        (cloner.query.get_asset_prototype_world_index(empty, ".*"), []),
        (cloner.query.get_asset_prototype_world_index(empty, 0, unique=True), []),
        (cloner.query.get_world_prototype_world_index(empty, 0), [0, 1, 2]),
        (cloner.query.get_world_prototype_world_index(empty, -1), [-1]),
        (cloner.query.get_world_prototype_world_index(plan, 1), []),
        (cloner.query.get_world_prototype_world_index(plan, 3), []),
        (cloner.query.get_asset_prototype_world_index(shared_only, 0), []),
        (cloner.query.get_asset_prototype_world_index(shared_only, 1), [-1, -1]),
        (cloner.query.get_asset_prototype_world_index(shared_only, assets[1].prim_path, unique=True), [-1]),
    ):
        assert actual.dtype == np.int32 and actual.ndim == 1
        np.testing.assert_array_equal(actual, expected)
    for members in ((2,), (-1,), (0.5,), ("0",)):
        with pytest.raises(ValueError, match="integer indices"):
            make_clone_plan(assets, (members,), 1)
