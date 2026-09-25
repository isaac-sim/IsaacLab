# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the cloner path/query algebra.

These exercise :mod:`isaaclab.cloner.path` and :mod:`isaaclab.cloner.query`, which are pure
string/array operations over a :class:`~isaaclab.cloner.ClonePlan`. They need no stage, no
simulator and no USD, so they live outside ``test/sim/``.
"""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import ClonePlan
from isaaclab.sim import MultiUsdFileCfg

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


def _plan(templates, variants, prototype_paths) -> ClonePlan:
    return ClonePlan(
        sources=tuple(
            AssetBaseCfg(
                prim_path=template.format("[^/]+"),
                spawn=MultiUsdFileCfg(usd_path=[""] * len(paths), spawn_paths=paths),
            )
            for template, paths in zip(templates, prototype_paths, strict=True)
        ),
        destinations=np.asarray(variants, dtype=np.int32),
        clone_template=templates[0].split("{}", 1)[0] + "{}",
    )


def _robot_plan(variants=(0, 0, 0, 0)) -> ClonePlan:
    """One declared robot, absent in environments selecting -1."""
    return _plan(("/World/envs/env_{}/Robot",), [variants], [("/World/envs/env_0/Robot",)])


def _wide_env_id_plan() -> ClonePlan:
    """Two variants of one asset over 12 envs, the second starting at a two-digit env id."""
    return _plan(
        ("/World/envs/env_{}/Object",),
        [[0] * 10 + [1, 1]],
        [("/World/envs/env_0/Object", "/World/envs/env_10/Object")],
    )


PLANS = {
    "homogeneous": _robot_plan(),
    "partial_coverage": _robot_plan((0, 0, -1, 0)),
    "two_variants": _plan(
        ("/World/envs/env_{}/Object",),
        [[0, 0, 1, 0]],
        [("/World/envs/env_0/Object", "/World/envs/env_2/Object")],
    ),
    "nested_prototype": _plan(
        ("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot/wrist/Camera"),
        [[0, 0, 0, 0], [0, -1, 0, 0]],
        [("/World/envs/env_0/Robot",), ("/World/envs/env_0/Robot/wrist/Camera",)],
    ),
    "distinct_env_root": _plan(
        ("/World/scenes/{}/Robot",),
        [[0, 0]],
        [("/World/source/Robot",)],
    ),
    "wide_env_ids": _wide_env_id_plan(),
}


##
# Query operations.
##


def test_path_env_ids():
    """path_env_ids returns the environments a source-space path reaches."""
    assert cloner.query.path_env_ids(_robot_plan(), "/World/envs/env_0/Robot/base") == (0, 1, 2, 3)
    partial = _robot_plan((0, 0, -1, 0))
    assert cloner.query.path_env_ids(partial, "/World/envs/env_0/Robot/base") == (0, 1, 3)
    assert cloner.query.path_env_ids(_robot_plan(), "/World/ground") == ()
    # An unowned path has no clone; owned paths are covered by the Q1/Q2 laws below.
    assert cloner.query.path_to_clone(partial, "/World/ground", 0) is None


def test_path_to_source_nested_templates_pick_most_specific():
    """A path owned by both an ancestor and a descendant template resolves to the descendant."""
    plan = _plan(
        ("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot/ee_link/palm_link/Camera"),
        [[0, 0], [0, 0]],
        [("/World/envs/env_0/Robot",), ("/World/envs/env_0/Robot/ee_link/palm_link/Camera",)],
    )

    # The camera path matches both templates; the more specific (longer-matching) one wins.
    resolved = cloner.query.path_to_source(plan, "/World/envs/env_0/Robot/ee_link/palm_link/Camera")
    assert resolved == (
        "/World/envs/env_0/Robot/ee_link/palm_link/Camera",
        "/World/envs/env_[^/]+/Robot/ee_link/palm_link/Camera",
        "",
    )

    # A path that only the ancestor template owns still resolves against it with its suffix.
    resolved = cloner.query.path_to_source(plan, "/World/envs/env_0/Robot/base")
    assert resolved == ("/World/envs/env_0/Robot", "/World/envs/env_[^/]+/Robot", "/base")


def test_path_to_source_selects_the_declared_variant():
    """One declaration selects the correct prototype or no instance for each environment."""
    plan = _plan(
        ("/World/envs/env_{}/Object",),
        [[0, 1, 0, -1]],
        [("/World/envs/env_0/Object", "/World/envs/env_1/Object")],
    )

    # Without an env id, the first populated variant represents the asset.
    resolved = cloner.query.path_to_source(plan, "/World/envs/env_[^/]+/Object/Body/Camera")
    assert resolved == ("/World/envs/env_0/Object", "/World/envs/env_[^/]+/Object", "/Body/Camera")

    # With an env id, the variant that actually populates that env is reported.
    resolved = cloner.query.path_to_source(plan, "/World/envs/env_[^/]+/Object/Body/Camera", env_id=1)
    assert resolved == ("/World/envs/env_1/Object", "/World/envs/env_[^/]+/Object", "/Body/Camera")

    # No variant populates env 3.
    assert cloner.query.path_to_source(plan, "/World/envs/env_[^/]+/Object/Body/Camera", env_id=3) is None


def test_iter_sources_yields_populated_variants():
    """Yield each active prototype with only the environments selecting it."""
    plan = _plan(
        ("/World/envs/env_{}/Object",),
        [[0, 0, 1, 1]],
        [("/World/envs/env_0/Object", "/World/envs/env_1/Object", None)],
    )

    matches = list(cloner.query.iter_sources(plan, "/World/envs/env_[^/]+/Object/Body/Camera"))

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
    absent = replace(plan, destinations=np.full((1, 4), -1, dtype=np.int32))
    assert cloner.query.path_to_source(absent, "/World/envs/env_[^/]+/Object/Body") is None
    assert not list(cloner.query.iter_sources(absent, "/World/envs/env_[^/]+/Object/Body"))


def test_iter_sources_skips_declarations_without_envs():
    """A nearer template populating no env does not hide the populated ancestor owning the path."""
    plan = _plan(
        ("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot/wrist/Camera"),
        [[0, 0, -1, -1], [-1, -1, -1, -1]],
        [("/World/envs/env_0/Robot",), ("/World/envs/env_0/Robot/wrist/Camera",)],
    )

    assert list(cloner.query.iter_sources(plan, "/World/envs/env_[^/]+/Robot/wrist/Camera")) == [
        (
            "/World/envs/env_0/Robot",
            "/World/envs/env_{}/Robot",
            "/World/envs/env_0/Robot/wrist/Camera",
            (0, 1),
        )
    ]


def test_iter_sources_distinct_env_root():
    """The destination template need not sit under the default env root."""
    plan = PLANS["distinct_env_root"]

    assert list(cloner.query.iter_sources(plan, "/World/scenes/[^/]+/Robot/base")) == [
        ("/World/source/Robot", "/World/scenes/{}/Robot", "/World/source/Robot/base", (0, 1))
    ]


def test_iter_sources_ranks_variants_independently_of_env_id_width():
    """Regression: a variant is not ranked out because its first env id has more digits.

    Specificity is the suffix below the destination template, which does not depend on the
    env ids a row happens to populate. Ranking by the *formatted* template instead made
    ``env_10`` look more specific than ``env_0`` and silently dropped the first variant for
    any scene with more than ten envs.
    """
    plan = _wide_env_id_plan()

    matches = list(cloner.query.iter_sources(plan, "/World/envs/env_[^/]+/Object/Body"))

    assert [match[0] for match in matches] == ["/World/envs/env_0/Object", "/World/envs/env_10/Object"]
    assert [match[3] for match in matches] == [tuple(range(10)), (10, 11)]


##
# Laws, checked over every plan shape above.
##


def _expected_clone(plan: ClonePlan, path: str, env_id: int) -> str | None:
    """Independent oracle: keep the suffix below the most specific selected source."""
    owners = [
        (index, source)
        for index, cfg in enumerate(plan.sources)
        for source in cfg.spawn.spawn_paths
        if cloner.path.under(path, source)
    ]
    if not owners:
        return None
    index, source = max(owners, key=lambda owner: len(owner[1]))
    variant = plan.destinations[index, env_id]
    cfg = plan.sources[index]
    if variant < 0 or cfg.spawn.spawn_paths[variant] != source:
        return None
    return cfg.prim_path.replace("[^/]+", str(env_id)) + path[len(source) :]


def _probe_paths(plan: ClonePlan) -> list[str]:
    """Source-space paths to probe: every prototype root, and prims below it."""
    return [
        source + tail
        for cfg in plan.sources
        for source in cfg.spawn.spawn_paths
        for tail in ("", "/base", "/link/child")
    ]


@pytest.mark.parametrize("plan_name", sorted(PLANS))
def test_query_law_factorization_and_domain(plan_name):
    """Q1/Q2: a clone keeps everything below the prototype root, and exists only where the plan says."""
    plan = PLANS[plan_name]
    num_envs = plan.destinations.shape[1]

    for path in _probe_paths(plan):
        reached = cloner.query.path_env_ids(plan, path)
        assert set(reached) <= set(range(num_envs))

        for env_id in range(num_envs):
            clone = cloner.query.path_to_clone(plan, path, env_id)
            # Q2: a clone exists exactly for the environments the plan reaches.
            assert (clone is not None) == (env_id in reached)
            if clone is None:
                continue
            assert clone == _expected_clone(plan, path, env_id)


@pytest.mark.parametrize("plan_name", sorted(PLANS))
def test_query_law_round_trip(plan_name):
    """Q3: resolving a clone path returns the prototype it was cloned from.

    A clone path is concrete, so no env id has to be supplied: the clone slot names it.
    """
    plan = PLANS[plan_name]

    for path in _probe_paths(plan):
        for env_id in cloner.query.path_env_ids(plan, path):
            clone = cloner.query.path_to_clone(plan, path, env_id)
            assert clone is not None

            resolved = cloner.query.path_to_source(plan, clone)
            assert resolved is not None, f"{clone} did not resolve back for env {env_id}"
            source, _glob, suffix = resolved
            assert source + suffix == path
            # Naming the env explicitly must agree with reading it out of the path.
            assert cloner.query.path_to_source(plan, clone, env_id=env_id) == resolved


def test_query_resolve_distinguishes_concrete_paths_from_wildcards():
    """A concrete clone path names its env; a wildcard expression stands for all of them.

    The clone slot is what separates the two: ``env_2`` selects the variant that populates
    env 2, while ``env_.*`` cannot and falls back to the first populated variant unless the
    caller names an env.
    """
    plan = PLANS["two_variants"]
    concrete = "/World/envs/env_2/Object/base"

    # Concrete: resolves to the variant env 2 was actually cloned from.
    source, _glob, suffix = cloner.query.path_to_source(plan, concrete)
    assert source + suffix == "/World/envs/env_2/Object/base"

    # Wildcard: one-to-many, so it reports a representative variant...
    wildcard = "/World/envs/env_[^/]+/Object/base"
    source, _glob, suffix = cloner.query.path_to_source(plan, wildcard)
    assert source + suffix == "/World/envs/env_0/Object/base"

    # ...unless the caller names the env it means.
    source, _glob, suffix = cloner.query.path_to_source(plan, wildcard, env_id=2)
    assert source + suffix == "/World/envs/env_2/Object/base"


def test_query_translates_env_ids_through_the_plan():
    """Mask columns are not env ids: a plan targeting envs (2, 5) reports 2 and 5.

    :func:`~isaaclab.cloner.replicate` formats destinations with ``env_ids[column]``, so the
    queries have to agree with it rather than reporting column indices.
    """
    plan = replace(
        _plan(("/World/envs/env_{}/Robot",), [[0, 0]], [("/World/envs/env_2/Robot",)]),
        env_ids=np.asarray([2, 5], dtype=np.int64),
    )
    path = "/World/envs/env_2/Robot/base"

    assert cloner.query.path_env_ids(plan, path) == (2, 5)
    assert cloner.query.path_to_clone(plan, path, 5) == "/World/envs/env_5/Robot/base"
    # Column indices are not environments: env 1 is not targeted by this plan.
    assert cloner.query.path_to_clone(plan, path, 1) is None
    assert next(iter(cloner.query.iter_sources(plan, "/World/envs/env_[^/]+/Robot")))[3] == (2, 5)

    source, _glob, suffix = cloner.query.path_to_source(plan, "/World/envs/env_5/Robot/base")
    assert source + suffix == path


@pytest.mark.parametrize("env_id", [-1, 4, 99])
def test_query_rejects_env_ids_outside_the_plan(env_id):
    """Out-of-range and negative ids resolve to nothing instead of wrapping the mask."""
    plan = _robot_plan()
    assert cloner.query.path_to_clone(plan, "/World/envs/env_0/Robot/base", env_id) is None
    assert cloner.query.path_to_source(plan, "/World/envs/env_[^/]+/Robot", env_id=env_id) is None


def test_cloner_imports_without_kit():
    """Importing the package in a clean interpreter must not drag in pxr.

    ``isaaclab.sim.utils.queries`` imports the cloner and the cloner's plan constructors
    import ``isaaclab.sim``, so this guards both against an import cycle and against pulling
    pxr in before Kit boots, which corrupts Kit's own USD runtime.
    """
    probe = (
        "from isaaclab.cloner import ClonePlan; import sys; "
        "print(any(n == 'pxr' or n.startswith('pxr.') for n in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "importing isaaclab.cloner pulled in pxr"
