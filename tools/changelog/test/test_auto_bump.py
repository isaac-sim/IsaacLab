# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end tests for :class:`cli.AutoBumpRun`.

The orchestrator owns the nightly compile + commit + push lifecycle that
used to live as inline shell in ``.github/workflows/nightly-changelog.yml``.
Tests build a tempdir source tree + bare-repo remote and exercise the full
sequence with no GitHub interaction, so the PR gate catches drift between
``cli.py``'s file-write sites and the auto-commit's staging contract.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cli
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` and return stdout (raises on non-zero)."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def _version_file(name: str) -> str:
    """Repo-relative path of ``<name>``'s version metadata file on this branch.

    Derived from :attr:`cli.Package.toml_path` rather than spelled out, so
    assertions keep passing when this file is cherry-picked to a branch with
    a different layout (``config/extension.toml`` instead of
    ``pyproject.toml``).
    """
    return cli.Package(Path("source") / name).toml_path.as_posix()


def _write_managed_pkg(source_root: Path, name: str, *, starting_version: str = "0.1.0") -> Path:
    """Build a managed package layout under ``source_root/<name>``.

    A managed package needs a version metadata file and a
    ``docs/CHANGELOG.rst`` with a parseable header. Where the version lives
    is the branch's business, not the test's: the path comes from
    :attr:`cli.Package.toml_path`, and the file's shape follows from it —
    a ``[project]`` table for ``pyproject.toml``, a bare top-level
    ``version`` for ``config/extension.toml``. That keeps this suite honest
    on both layouts without a parametrize axis that could drift from what
    ``cli.py`` on the branch actually reads.
    """
    root = source_root / name
    (root / "docs").mkdir(parents=True)
    (root / "changelog.d").mkdir(parents=True)
    (root / "docs" / "CHANGELOG.rst").write_text("Changelog\n---------\n\n", encoding="utf-8")

    toml_path = cli.Package(root).toml_path
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    if toml_path.name == "pyproject.toml":
        toml_path.write_text(
            "[build-system]\n"
            'requires = ["setuptools"]\n'
            "\n"
            "[project]\n"
            f'name = "{name}"\n'
            f'version = "{starting_version}"\n',
            encoding="utf-8",
        )
    else:
        toml_path.write_text(f'version = "{starting_version}"\n', encoding="utf-8")
    return root


def _drop_fragment(pkg_root: Path, slug: str, body: str = "Added\n^^^^^\n\n* Added thing.\n") -> Path:
    p = pkg_root / "changelog.d" / f"{slug}.rst"
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    """A tempdir with a ``source/`` tree, initialized as a git repo with a
    bare-repo ``origin`` remote and a ``develop`` branch pre-populated with
    the initial source tree.

    Returns the working-tree path. The bare remote lives at
    ``tmp_path / "origin.git"``; tests that need to inspect what landed on
    the remote can run git commands against it directly.
    """
    work = tmp_path / "work"
    work.mkdir()
    (work / "source").mkdir()

    # Bare repo acts as the upstream. The auto-bump pushes here and the
    # rebase loop fetches from here.
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "develop", str(bare)], check=True, capture_output=True)

    _git(work, "init", "-b", "develop")
    _git(work, "config", "user.name", "Test Setup")
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "remote", "add", "origin", str(bare))
    # Seed an empty commit so the bare repo has a develop tip to push to.
    _git(work, "commit", "--allow-empty", "-m", "seed")
    _git(work, "push", "origin", "develop")
    return work


def _author_log(bare: Path, branch: str = "develop") -> list[str]:
    """List commit author names on ``branch`` in the bare repo, newest first."""
    out = subprocess.run(
        ["git", "log", branch, "--format=%an"],
        cwd=bare,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_auto_bump_happy_path_commits_and_pushes(synthetic_repo: Path, tmp_path: Path):
    _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _drop_fragment(synthetic_repo / "source" / "isaaclab", "feat-x")

    rc = cli.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    ).run()

    assert rc == 0
    # The bot's commit should be on top of the develop tip on the bare remote.
    authors = _author_log(tmp_path / "origin.git")
    assert authors[0] == cli.AutoBumpRun.AUTHOR_NAME


# ---------------------------------------------------------------------------
# Regression: every file the compile wrote must be staged, whatever it is
# ---------------------------------------------------------------------------


def test_auto_bump_stages_every_file_the_compile_wrote(synthetic_repo: Path, tmp_path: Path):
    """The 2026-05-29 nightly bricked because the workflow's ``git add``
    glob carried its own idea of which files ``cli.py`` writes, and #5785
    added a write site without the paired YAML edit.

    AutoBumpRun derives the staged set from each compile's ``touched``
    return value, so there is no second list to drift. The assertion is
    deliberately expressed in terms of what the compile reports rather
    than a hardcoded file list: whatever ``Package.compile`` says it
    wrote is exactly what the commit must carry.
    """
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _drop_fragment(pkg_root, "feat-x")

    run = cli.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    )
    assert run.run() == 0

    bare = tmp_path / "origin.git"
    files = subprocess.run(
        ["git", "show", "--name-only", "--format=", "develop"],
        cwd=bare,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()

    for path in run.touched:
        assert path.relative_to(synthetic_repo).as_posix() in files
    # Sanity-check that ``touched`` was not vacuously empty and that it
    # includes the branch's version metadata file under its real name.
    assert _version_file("isaaclab") in files
    assert "source/isaaclab/docs/CHANGELOG.rst" in files


# ---------------------------------------------------------------------------
# Race-resolution: push retries on non-fast-forward
# ---------------------------------------------------------------------------


def test_auto_bump_rebases_on_non_fast_forward(synthetic_repo: Path, tmp_path: Path):
    """Simulate a human commit landing on develop between checkout and
    push. The first push fails non-fast-forward, the rebase loop fetches
    the new tip, replays the auto-commit, and the second push succeeds."""
    _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _drop_fragment(synthetic_repo / "source" / "isaaclab", "feat-x")

    bare = tmp_path / "origin.git"
    # Pre-load a "human" commit on the bare remote's develop branch by
    # cloning to a sidecar, committing, and pushing back.
    sidecar = tmp_path / "sidecar"
    subprocess.run(["git", "clone", str(bare), str(sidecar)], check=True, capture_output=True)
    _git(sidecar, "config", "user.name", "Human Dev")
    _git(sidecar, "config", "user.email", "dev@example.com")
    _git(sidecar, "commit", "--allow-empty", "-m", "human work")
    _git(sidecar, "push", "origin", "develop")

    rc = cli.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    ).run()

    assert rc == 0
    authors = _author_log(bare)
    # The bot's commit must be on top, with the human's commit one below.
    assert authors[0] == cli.AutoBumpRun.AUTHOR_NAME
    assert "Human Dev" in authors[1:]


# ---------------------------------------------------------------------------
# Race-resolution: exhausted retries raise
# ---------------------------------------------------------------------------


def test_auto_bump_raises_after_exhausting_retries(synthetic_repo: Path, tmp_path: Path, monkeypatch):
    """If every retry races against another human commit, the orchestrator
    eventually gives up rather than spinning forever."""
    _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _drop_fragment(synthetic_repo / "source" / "isaaclab", "feat-x")

    bare = tmp_path / "origin.git"
    sidecar = tmp_path / "sidecar"
    subprocess.run(["git", "clone", str(bare), str(sidecar)], check=True, capture_output=True)
    _git(sidecar, "config", "user.name", "Human Dev")
    _git(sidecar, "config", "user.email", "dev@example.com")

    # Seed a pre-existing race so the *first* push from auto-bump already
    # fails non-fast-forward. After that, every fetch-then-rebase loop
    # picks up yet another sidecar commit, so the push keeps failing.
    _git(sidecar, "commit", "--allow-empty", "-m", "initial human commit")
    _git(sidecar, "push", "origin", "develop")

    real_fetch = cli.GitRepo.fetch

    def racing_fetch(self, remote: str, ref: str) -> None:
        # Order matters: real fetch first (captures FETCH_HEAD at the
        # current tip), then sidecar pushes another commit so the bare
        # remote moves *past* what FETCH_HEAD captured. The subsequent
        # rebase replays our auto-commit onto the stale FETCH_HEAD, and
        # the next push fails non-fast-forward again. Lather, rinse, repeat.
        real_fetch(self, remote, ref)
        _git(sidecar, "commit", "--allow-empty", "-m", "another human commit")
        _git(sidecar, "push", "origin", "develop")

    monkeypatch.setattr(cli.GitRepo, "fetch", racing_fetch)

    with pytest.raises(cli.GitError):
        cli.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()


# ---------------------------------------------------------------------------
# Partial success: one package compiles, one raises; healthy commit ships,
# exit code reflects the failure
# ---------------------------------------------------------------------------


def test_auto_bump_ships_healthy_packages_when_one_fails(synthetic_repo: Path, tmp_path: Path):
    good = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    bad = _write_managed_pkg(synthetic_repo / "source", "isaaclab_assets")
    _drop_fragment(good, "feat-x")
    _drop_fragment(bad, "feat-y")
    # Break the bad package's CHANGELOG.rst header so write_changelog_entry
    # raises mid-compile. Self-heal only fixes "missing blank line"; a
    # totally absent header still raises.
    (bad / "docs" / "CHANGELOG.rst").write_text("not a real changelog\n", encoding="utf-8")

    rc = cli.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    ).run()

    assert rc == 1  # one package failed
    # The good package's bump should still have shipped to the bare remote.
    bare = tmp_path / "origin.git"
    files = subprocess.run(
        ["git", "show", "--name-only", "--format=", "develop"],
        cwd=bare,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert _version_file("isaaclab") in files
    assert "source/isaaclab_assets" not in files


# ---------------------------------------------------------------------------
# Dry run: no commits, no push, fragments preserved
# ---------------------------------------------------------------------------


def test_auto_bump_dry_run_does_not_commit(synthetic_repo: Path, tmp_path: Path):
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    frag = _drop_fragment(pkg_root, "feat-x")

    rc = cli.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        dry_run=True,
        repo_root=synthetic_repo,
    ).run()
    assert rc == 0

    bare = tmp_path / "origin.git"
    authors = _author_log(bare)
    assert cli.AutoBumpRun.AUTHOR_NAME not in authors
    # Fragment must still be on disk — dry-run is a preview, not a commit.
    assert frag.exists()


# ---------------------------------------------------------------------------
# Nothing to compile: exit clean, no commit, no push
# ---------------------------------------------------------------------------


def test_auto_bump_with_no_fragments_is_a_noop(synthetic_repo: Path, tmp_path: Path):
    _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    # No fragment dropped.

    rc = cli.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    ).run()

    assert rc == 0
    authors = _author_log(tmp_path / "origin.git")
    assert cli.AutoBumpRun.AUTHOR_NAME not in authors
