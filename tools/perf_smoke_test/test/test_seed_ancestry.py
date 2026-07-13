# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free checks for the seed-time ancestry preflight.

The gate drops baseline samples whose ``commit_sha`` is not an ancestor of the
run's ``base_sha`` (the target branch HEAD). The seeder's preflight refuses to
seed such commits so they cannot silently produce unusable baselines.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

import seed_baselines  # noqa: E402


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _commit(work: Path, name: str) -> str:
    (work / name).write_text(f"{name}\n", encoding="utf-8")
    _git(["add", name], work)
    _git(["commit", "-m", name], work)
    return _git(["rev-parse", "HEAD"], work)


@pytest.fixture()
def repo(tmp_path: Path) -> dict[str, object]:
    """A repo where ``develop`` (c0->c1->c2) and ``feature`` (c0->f1) diverge."""
    work = tmp_path / "work"
    work.mkdir()
    _git(["init", "-b", "develop"], work)
    _git(["config", "user.email", "test@example.com"], work)
    _git(["config", "user.name", "test"], work)
    c0 = _commit(work, "c0")
    c1 = _commit(work, "c1")
    c2 = _commit(work, "c2")

    _git(["checkout", "-b", "feature", c0], work)
    f1 = _commit(work, "f1")
    _git(["checkout", "develop"], work)
    return {"work": work, "c0": c0, "c1": c1, "c2": c2, "f1": f1}


def test_is_ancestor_tracks_reachability(repo) -> None:
    """``_is_ancestor`` is true only for commits reachable from the tip."""
    work = repo["work"]
    assert seed_baselines._is_ancestor(repo["c1"], repo["c2"], work) is True
    assert seed_baselines._is_ancestor(repo["f1"], repo["c2"], work) is False


def test_resolve_branch_tip_prefers_local_when_no_remote(repo) -> None:
    """With no ``origin`` the resolver still finds the local branch tip."""
    work = repo["work"]
    assert seed_baselines._resolve_branch_tip("develop", work) == repo["c2"]
    assert seed_baselines._resolve_branch_tip("nope", work) is None


def test_filter_drops_non_ancestor_commits(repo) -> None:
    """A commit off the target branch is skipped; on-branch commits are kept."""
    work = repo["work"]
    plan = [("develop", repo["c1"]), ("develop", repo["c2"]), ("develop", repo["f1"])]

    kept = seed_baselines._filter_plan_by_ancestry(plan, work, target_sha="")

    assert kept == [("develop", repo["c1"]), ("develop", repo["c2"])]


def test_filter_strict_raises_on_non_ancestor(repo) -> None:
    """Strict mode turns an off-branch commit into a hard error."""
    work = repo["work"]
    plan = [("develop", repo["c1"]), ("develop", repo["f1"])]

    with pytest.raises(RuntimeError, match="NOT ancestors"):
        seed_baselines._filter_plan_by_ancestry(plan, work, target_sha="", strict=True)


def test_build_seed_plan_branches_ref_with_target_seeds_single_commit(repo) -> None:
    """The era-roll invocation ('<sha>:develop', count 1) seeds exactly that commit.

    The auto-era-roll orchestrator seeds a brand-new era from its boundary commit
    (HEAD) via ``branches: "<sha>:develop"``; this locks that the branches path
    resolves a raw SHA and stamps the overridden target.
    """
    work = repo["work"]
    args = argparse.Namespace(
        branches=f"{repo['c1']}:develop",
        commits="",
        commit_branch="develop",
        commit_count=1,
        target_branch="develop",
        workdir=work,
    )

    plan = seed_baselines._build_seed_plan(args)

    assert plan == [("develop", repo["c1"])]


def test_filter_uses_explicit_target_sha(repo) -> None:
    """An explicit target_sha is used verbatim for every plan entry."""
    work = repo["work"]
    plan = [("develop", repo["c1"]), ("develop", repo["f1"])]

    kept = seed_baselines._filter_plan_by_ancestry(plan, work, target_sha=repo["c2"])

    assert kept == [("develop", repo["c1"])]


def test_filter_unresolvable_tip_keeps_when_lenient_raises_when_strict(repo) -> None:
    """An unresolvable target tip warns+keeps by default but aborts under strict."""
    work = repo["work"]
    plan = [("ghost", repo["c1"])]

    kept = seed_baselines._filter_plan_by_ancestry(plan, work, target_sha="")
    assert kept == plan

    with pytest.raises(RuntimeError, match="cannot resolve tip"):
        seed_baselines._filter_plan_by_ancestry(plan, work, target_sha="", strict=True)
