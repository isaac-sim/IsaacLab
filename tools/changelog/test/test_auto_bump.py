# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end tests for :class:`autobump.AutoBumpRun`.

The orchestrator owns the nightly compile + commit + push lifecycle that
used to live as inline shell in ``.github/workflows/nightly-changelog.yml``.
Tests build a tempdir source tree + bare-repo remote and exercise the full
sequence with no GitHub interaction, so the PR gate catches drift between
``cli.py``'s file-write sites and the auto-commit's staging contract.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import autobump
import cli
import packages
import pytest

from conftest import version_file_rel, write_version_file

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


_version_file = version_file_rel


def _write_managed_pkg(source_root: Path, name: str, *, starting_version: str = "0.1.0") -> Path:
    """Build a managed package layout under ``source_root/<name>``.

    A managed package needs a version metadata file and a
    ``docs/CHANGELOG.rst`` with a parseable header. Where the version lives
    is the branch's business, not the test's — see
    :func:`conftest.write_version_file`.
    """
    root = source_root / name
    (root / "docs").mkdir(parents=True)
    (root / "changelog.d").mkdir(parents=True)
    (root / "docs" / "CHANGELOG.rst").write_text("Changelog\n---------\n\n", encoding="utf-8")
    write_version_file(root, name, starting_version)
    return root


def _drop_fragment(pkg_root: Path, slug: str, body: str = "Added\n^^^^^\n\n* Added thing.\n") -> Path:
    p = pkg_root / "changelog.d" / f"{slug}.rst"
    p.write_text(body, encoding="utf-8")
    return p


def _commit_baseline(work: Path) -> None:
    """Commit the seeded tree so the run starts from production's state.

    The nightly runs against a fresh checkout where every manifest, changelog
    and pending fragment is already tracked — its staged diff is therefore
    made of modifications and deletions, not additions. Leaving the fixture's
    files untracked would quietly change what is under test: version diffs
    would have no ``-version`` side to read, and a consumed fragment's
    deletion would be invisible to git rather than staged.

    The baseline is pushed, not just committed, so the remote holds the
    package files before the run. Without that, a test asserting "file X is
    absent from the pushed tree" would pass vacuously whenever the run
    pushed nothing at all — which is precisely the failure such a test is
    supposed to catch.
    """
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "baseline source tree")
    _git(work, "push", "origin", "HEAD")


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
# The staging contract: every file the compile wrote must reach the commit
# ---------------------------------------------------------------------------


def test_auto_bump_stages_every_file_the_compile_wrote(synthetic_repo: Path, tmp_path: Path, monkeypatch):
    """The 2026-05-29 nightly bricked because the workflow's ``git add``
    glob carried its own idea of which files ``cli.py`` writes, and #5785
    added a write site without the paired YAML edit.

    The staged set now comes from what each compile reports it wrote, so
    there is no second list to drift. The assertion is deliberately phrased
    against that report rather than a hardcoded file list: whatever
    ``Package.compile`` says it wrote is exactly what the commit must carry,
    including write sites that do not exist yet.
    """
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _drop_fragment(pkg_root, "feat-x")
    _commit_baseline(synthetic_repo)

    # Observe the contract at its source: record what compile reports.
    reported: list[Path] = []
    real_compile = packages.Package.compile

    def recording_compile(self, **kwargs):
        compiled, touched = real_compile(self, **kwargs)
        reported.extend(touched)
        return compiled, touched

    monkeypatch.setattr(packages.Package, "compile", recording_compile)

    assert (
        autobump.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()
        == 0
    )

    files = subprocess.run(
        ["git", "show", "--name-only", "--format=", "develop"],
        cwd=tmp_path / "origin.git",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()

    assert _author_log(tmp_path / "origin.git")[0] == autobump.AutoBumpRun.AUTHOR_NAME
    assert reported, "compile reported nothing — the assertion below would be vacuous"
    for path in reported:
        assert path.relative_to(synthetic_repo).as_posix() in files
    # ...and the branch's version metadata file is in there under its real name.
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
    # Baseline lands first: it is the shared starting point both the human
    # and the nightly branch off, so it must precede the racing commit.
    _commit_baseline(synthetic_repo)

    bare = tmp_path / "origin.git"
    # Pre-load a "human" commit on the bare remote's develop branch by
    # cloning to a sidecar, committing, and pushing back.
    sidecar = tmp_path / "sidecar"
    subprocess.run(["git", "clone", str(bare), str(sidecar)], check=True, capture_output=True)
    _git(sidecar, "config", "user.name", "Human Dev")
    _git(sidecar, "config", "user.email", "dev@example.com")
    _git(sidecar, "commit", "--allow-empty", "-m", "human work")
    _git(sidecar, "push", "origin", "develop")

    rc = autobump.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    ).run()

    assert rc == 0
    authors = _author_log(bare)
    # The bot's commit must be on top, with the human's commit one below.
    assert authors[0] == autobump.AutoBumpRun.AUTHOR_NAME
    assert "Human Dev" in authors[1:]


# ---------------------------------------------------------------------------
# Race-resolution: exhausted retries raise
# ---------------------------------------------------------------------------


def test_auto_bump_raises_after_exhausting_retries(synthetic_repo: Path, tmp_path: Path, monkeypatch):
    """If every retry races against another human commit, the orchestrator
    eventually gives up rather than spinning forever."""
    _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _drop_fragment(synthetic_repo / "source" / "isaaclab", "feat-x")
    # Baseline lands before the race is staged; see the sibling test.
    _commit_baseline(synthetic_repo)

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

    real_fetch = autobump.GitRepo.fetch

    def racing_fetch(self, remote: str, ref: str) -> None:
        # Order matters: real fetch first (captures FETCH_HEAD at the
        # current tip), then sidecar pushes another commit so the bare
        # remote moves *past* what FETCH_HEAD captured. The subsequent
        # rebase replays our auto-commit onto the stale FETCH_HEAD, and
        # the next push fails non-fast-forward again. Lather, rinse, repeat.
        real_fetch(self, remote, ref)
        _git(sidecar, "commit", "--allow-empty", "-m", "another human commit")
        _git(sidecar, "push", "origin", "develop")

    monkeypatch.setattr(autobump.GitRepo, "fetch", racing_fetch)

    with pytest.raises(autobump.GitError):
        autobump.AutoBumpRun(
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

    _commit_baseline(synthetic_repo)
    rc = autobump.AutoBumpRun(
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


def test_auto_bump_dry_run_writes_nothing(synthetic_repo: Path, tmp_path: Path):
    """``--dry-run`` is documented as "compile without writing", so assert the
    writing half too. Checking only that no commit was pushed would pass a
    regression that bumped the version and prepended the entry before
    skipping the commit. A drifting lock is seeded so the lock-sync dry-run
    branch is covered in the same run."""
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab", starting_version="1.0.0")
    # Pinned behind the manifest so the sync has real drift to find. Seeded
    # in sync, it returned at its "nothing to do" guard and never reached the
    # write-suppression branch this test exists to cover.
    lock = _write_workspace_lock(synthetic_repo, ["isaaclab"], pinned={"isaaclab": "0.9.0"})
    frag = _drop_fragment(pkg_root, "feat-x")
    _commit_baseline(synthetic_repo)
    lock_before = lock.read_text(encoding="utf-8")

    rc = autobump.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        dry_run=True,
        repo_root=synthetic_repo,
    ).run()
    assert rc == 0

    assert autobump.AutoBumpRun.AUTHOR_NAME not in _author_log(tmp_path / "origin.git")
    # Nothing reached disk: no bump, no entry, no lock rewrite, no deletion.
    assert _git(synthetic_repo, "status", "--porcelain").strip() == ""
    assert lock.read_text(encoding="utf-8") == lock_before
    assert frag.exists()


# ---------------------------------------------------------------------------
# Nothing to compile: exit clean, no commit, no push
# ---------------------------------------------------------------------------


def test_auto_bump_with_no_fragments_is_a_noop(synthetic_repo: Path, tmp_path: Path):
    _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    # No fragment dropped.

    _commit_baseline(synthetic_repo)
    rc = autobump.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    ).run()

    assert rc == 0
    authors = _author_log(tmp_path / "origin.git")
    assert autobump.AutoBumpRun.AUTHOR_NAME not in authors


# ---------------------------------------------------------------------------
# uv.lock: synced from the same touched-paths manifest, no separate git add
# ---------------------------------------------------------------------------


def _write_workspace_lock(work: Path, names: list[str], *, pinned: dict[str, str] | None = None) -> Path:
    """Turn the synthetic repo into a uv workspace.

    Writes a root manifest declaring each package as an editable member and
    a ``uv.lock`` pinning them at whatever version they currently declare —
    i.e. in sync, so any drift a test observes was produced by the bump.

    ``pinned`` overrides a member's locked version to seed drift that exists
    *before* the run. A test that needs the sync to actually do something
    must use it: with everything in sync, ``LockFile.sync`` returns at its
    "nothing to do" guard and never reaches the behaviour under test.
    """
    pinned = pinned or {}
    sources = "\n".join(f'{name} = {{ path = "source/{name}", editable = true }}' for name in names)
    (work / "pyproject.toml").write_text(
        f'[project]\nname = "isaaclab-dev"\nversion = "0.0.0"\n\n[tool.uv.sources]\n{sources}\n',
        encoding="utf-8",
    )
    blocks = []
    for name in names:
        version = pinned.get(name) or packages.Package.declared_version(work / "source" / name)
        blocks.append(
            f'[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {{ editable = "source/{name}" }}\n'
        )
    lock = work / "uv.lock"
    lock.write_text('version = 1\nrequires-python = ">=3.11"\n\n' + "\n".join(blocks), encoding="utf-8")
    return lock


def test_auto_bump_syncs_and_commits_uv_lock(synthetic_repo: Path, tmp_path: Path):
    """``uv.lock`` sits at the repo root, outside ``source/``, so no
    ``git add source/...`` glob can ever reach it. Routing the sync's written
    path through the same ``touched`` manifest as every other compiler output
    is what gets it staged — there is no second list to keep in step."""
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab", starting_version="1.0.0")
    _write_workspace_lock(synthetic_repo, ["isaaclab"])
    _drop_fragment(pkg_root, "feat-x")

    _commit_baseline(synthetic_repo)
    rc = autobump.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    ).run()
    assert rc == 0

    bare = tmp_path / "origin.git"
    show = subprocess.run(
        ["git", "show", "develop", "--name-only", "--format="],
        cwd=bare,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "uv.lock" in show.split()
    # And the committed lock carries the bumped version, not the old pin.
    locked = subprocess.run(
        ["git", "show", "develop:uv.lock"],
        cwd=bare,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert 'version = "1.0.1"' in locked


def test_auto_bump_without_a_lock_still_commits(synthetic_repo: Path, tmp_path: Path):
    """The release-branch shape: changelog tooling present, no uv workspace.
    The lock sync must be a silent no-op rather than a red tile, because the
    same cli.py is cherry-picked to those branches."""
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _drop_fragment(pkg_root, "feat-x")

    _commit_baseline(synthetic_repo)
    rc = autobump.AutoBumpRun(
        branch="release/3.0.0-beta2",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    ).run()

    assert rc == 0
    authors = _author_log(tmp_path / "origin.git", branch="release/3.0.0-beta2")
    assert authors[0] == autobump.AutoBumpRun.AUTHOR_NAME


def test_auto_bump_reds_the_tile_but_still_ships_when_the_lock_needs_a_full_relock(
    synthetic_repo: Path, tmp_path: Path
):
    """A member added to the workspace without re-locking cannot be repaired
    by a version rewrite. That is a real problem worth a red tile, but it
    predates this run — wedging the commit would strand every package's
    changelog over a lock the auto-bump did not break."""
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _write_workspace_lock(synthetic_repo, ["isaaclab"])
    _drop_fragment(pkg_root, "feat-x")
    # Declare a second member that the lock has never seen.
    root_toml = synthetic_repo / "pyproject.toml"
    root_toml.write_text(
        root_toml.read_text(encoding="utf-8")
        + 'isaaclab_assets = { path = "source/isaaclab_assets", editable = true }\n',
        encoding="utf-8",
    )

    _commit_baseline(synthetic_repo)
    rc = autobump.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    ).run()

    assert rc == 1  # red tile
    bare = tmp_path / "origin.git"
    show = subprocess.run(
        ["git", "show", "develop", "--name-only", "--format="],
        cwd=bare,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    # The changelog bump still shipped; the un-repairable lock did not.
    assert _version_file("isaaclab") in show
    assert "uv.lock" not in show


# ---------------------------------------------------------------------------
# Branch-state compatibility
#
# One cli.py is cherry-picked to every branch the nightly targets, and those
# branches disagree: on some the version lives in ``pyproject.toml``, on
# others in ``config/extension.toml``; some carry a ``uv.lock``, some do not.
#
# These tests are deliberately written against the *contract* rather than
# against one branch's filenames, so this file keeps testing the truth after
# a cherry-pick instead of testing a layout it no longer runs on. Where a
# fixture needs to know the layout it asks the code under test
# (:attr:`packages.Package.toml_path`) rather than hardcoding an answer.
# ---------------------------------------------------------------------------


def _commit_body(bare: Path, branch: str = "develop") -> str:
    return subprocess.run(
        ["git", "log", "-1", "--format=%B", branch],
        cwd=bare,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def test_commit_message_names_the_package_once_per_bump(synthetic_repo: Path, tmp_path: Path):
    """The body lists one ``- <pkg>: old → new`` line per package, sourced
    from the branch's own version metadata file. Guards the case that would
    otherwise regress silently: a filename-keyed lookup that matches nothing
    on a branch whose layout differs, producing an empty ``Bumped packages:``
    section that nobody notices because the commit still lands."""
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab", starting_version="2.3.4")
    _drop_fragment(pkg_root, "feat-x")

    _commit_baseline(synthetic_repo)
    assert (
        autobump.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()
        == 0
    )

    body = _commit_body(tmp_path / "origin.git")
    assert "Bumped packages:" in body
    assert "- isaaclab: 2.3.4 → 2.3.5" in body


def test_consumed_fragments_are_deleted_in_the_commit(synthetic_repo: Path, tmp_path: Path):
    """Deleting a fragment on disk is not enough — the deletion must be staged.

    ``compile`` consumes fragments by unlinking them. If those paths are
    absent from the staged set, the commit carries the changelog entry and
    the version bump while leaving the fragments untouched on the branch.
    The next checkout restores them, the next nightly recompiles them, and
    the package gets a duplicate entry and a second bump — every night.

    The old workflow got this right only by accident, via a blanket
    ``git add --update -- source/``. With the staging set derived from the
    compile, the deletions have to be reported explicitly.
    """
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    frag = _drop_fragment(pkg_root, "feat-x")
    skip = pkg_root / "changelog.d" / "chore-ci.skip"
    skip.write_text("", encoding="utf-8")

    _commit_baseline(synthetic_repo)
    assert (
        autobump.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()
        == 0
    )

    # Gone from the working tree...
    assert not frag.exists()
    assert not skip.exists()
    # ...and gone on the branch that was pushed, which is what actually matters.
    bare = tmp_path / "origin.git"
    committed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "develop"],
        cwd=bare,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    assert "source/isaaclab/docs/CHANGELOG.rst" in committed
    assert "source/isaaclab/changelog.d/feat-x.rst" not in committed
    assert "source/isaaclab/changelog.d/chore-ci.skip" not in committed


def test_skip_only_cleanup_stages_its_deletions(synthetic_repo: Path, tmp_path: Path):
    """A package whose only pending files are ``.skip`` entries produces no
    entry and no bump, but the skip files are still consumed. That deletion
    has to be staged too, or the same "cleaned" files reappear every night."""
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    skip = pkg_root / "changelog.d" / "chore-ci.skip"
    skip.write_text("", encoding="utf-8")

    _commit_baseline(synthetic_repo)
    assert (
        autobump.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()
        == 0
    )

    committed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "develop"],
        cwd=tmp_path / "origin.git",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    # Anchor first: the package must be present on the pushed branch at all,
    # otherwise "the skip file is absent" would hold trivially for a run that
    # pushed nothing.
    assert "source/isaaclab/docs/CHANGELOG.rst" in committed
    assert "source/isaaclab/changelog.d/chore-ci.skip" not in committed


def test_multi_file_version_write_is_staged_and_reported_once(synthetic_repo: Path, tmp_path: Path, monkeypatch):
    """A branch whose ``write_version`` touches more than one file per package.

    ``develop`` and ``release/3.0.0-beta2`` each write a single file today,
    but between #5785 and #6505 ``develop`` wrote two (``extension.toml`` and
    ``pyproject.toml``), and a future branch could again. The orchestrator
    must stage *every* returned path while still reporting the package once
    in the commit body — the report is keyed on the package's canonical
    version file, not on "every file that changed".
    """
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab", starting_version="1.0.0")
    _drop_fragment(pkg_root, "feat-x")

    real_write_version = packages.Package.write_version
    # A path that is not ``toml_path`` on any branch — using a real layout's
    # filename here would collide with the actual version file on whichever
    # branch uses it, and the test would stop testing a *second* write site.
    sidecar = pkg_root / "docs" / "VERSION_SIDECAR.txt"

    def dual_write(self, new_version, *, dry_run):
        touched = real_write_version(self, new_version, dry_run=dry_run)
        if dry_run:
            return touched
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(f'version = "{new_version}"\n', encoding="utf-8")
        return [*touched, sidecar]

    monkeypatch.setattr(packages.Package, "write_version", dual_write)

    _commit_baseline(synthetic_repo)
    assert (
        autobump.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()
        == 0
    )

    bare = tmp_path / "origin.git"
    files = subprocess.run(
        ["git", "show", "develop", "--name-only", "--format="],
        cwd=bare,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    # Both files the write site reported must be in the commit.
    assert _version_file("isaaclab") in files
    assert "source/isaaclab/docs/VERSION_SIDECAR.txt" in files
    # ...but the package is still reported exactly once.
    assert _commit_body(bare).count("- isaaclab:") == 1


# ---------------------------------------------------------------------------
# Hostile-input and environment robustness
# ---------------------------------------------------------------------------


def test_non_ascii_fragment_deletion_is_staged(synthetic_repo: Path, tmp_path: Path):
    """A fragment slug with a non-ASCII character must still get its deletion staged.

    ``git ls-files`` C-quotes such paths under the default ``core.quotePath``
    (``"jos\\303\\251-fix.rst"``), and the fragment filename pattern happily
    accepts them. If the quoted form is compared against the caller's real
    path it never matches, the deletion is dropped, the fragment survives on
    the branch, and the next nightly compiles it a second time -- the exact
    duplicate-entry, double-bump failure this whole design exists to prevent.
    """
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    frag = _drop_fragment(pkg_root, "josé-fix")

    _commit_baseline(synthetic_repo)
    assert (
        autobump.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()
        == 0
    )

    assert not frag.exists()
    committed = subprocess.run(
        ["git", "-c", "core.quotePath=false", "ls-tree", "-r", "--name-only", "develop"],
        cwd=tmp_path / "origin.git",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    assert "source/isaaclab/docs/CHANGELOG.rst" in committed
    assert "source/isaaclab/changelog.d/josé-fix.rst" not in committed


def test_lock_failure_is_reported_on_every_run_not_just_the_bumping_one(synthetic_repo: Path, tmp_path: Path):
    """An unrepairable lock must keep reporting failure until a human fixes it.

    Reconciling the lock only when a package happened to bump would report the
    problem on the first night and then fall silent: the following night has
    no pending fragments, exits early, never looks at the lock, and reports
    success while the branch is still inconsistent.
    """
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _write_workspace_lock(synthetic_repo, ["isaaclab"])
    root_toml = synthetic_repo / "pyproject.toml"
    root_toml.write_text(
        root_toml.read_text(encoding="utf-8")
        + 'isaaclab_assets = { path = "source/isaaclab_assets", editable = true }\n',
        encoding="utf-8",
    )
    _drop_fragment(pkg_root, "feat-x")
    _commit_baseline(synthetic_repo)

    def run_once() -> int:
        return autobump.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()

    assert run_once() == 1, "first run: mismatch reported"
    # Second run has no fragments left to compile, but the lock is still broken.
    assert run_once() == 1, "second run must still report the unrepairable lock"


def test_push_rejection_is_detected_from_porcelain_not_prose(synthetic_repo: Path, tmp_path: Path, monkeypatch):
    """Non-fast-forward detection must not depend on git's human-readable text.

    git translates its summary output; the ``--porcelain`` status flag in
    column zero is structural. Simulating a localized git (all prose replaced)
    proves the retry decision comes from the flag.
    """
    _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _drop_fragment(synthetic_repo / "source" / "isaaclab", "feat-x")
    _commit_baseline(synthetic_repo)

    bare = tmp_path / "origin.git"
    sidecar = tmp_path / "sidecar"
    subprocess.run(["git", "clone", str(bare), str(sidecar)], check=True, capture_output=True)
    _git(sidecar, "config", "user.name", "Human Dev")
    _git(sidecar, "config", "user.email", "dev@example.com")
    _git(sidecar, "commit", "--allow-empty", "-m", "human work")
    _git(sidecar, "push", "origin", "develop")

    real_run = autobump.GitRepo._run

    def localized_run(self, *args, check=True):
        result = real_run(self, *args, check=check)
        if args and args[0] == "push":
            # Simulate a git whose human-readable text is translated: the
            # summary field and the stderr prose lose every English token the
            # old substring scan keyed on ("[rejected]", "non-fast-forward"),
            # while the structural ``!`` flag in column zero is untouched.
            result.stderr = "erreur: échec de l'envoi de certaines références\n"
            result.stdout = result.stdout.replace("[rejected]", "[rejeté]").replace("non-fast-forward", "non-accéléré")
        return result

    monkeypatch.setattr(autobump.GitRepo, "_run", localized_run)

    assert (
        autobump.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()
        == 0
    )
    authors = _author_log(bare)
    assert authors[0] == autobump.AutoBumpRun.AUTHOR_NAME
    assert "Human Dev" in authors[1:]


def test_retry_fetch_prefers_the_branch_over_a_same_named_tag(synthetic_repo: Path, tmp_path: Path):
    """The retry must rebase onto the branch, even if a tag shares its name.

    ``git fetch origin develop`` with both a tag and a branch called
    ``develop`` resolves the *tag* into ``FETCH_HEAD``. The rebase would then
    replay the auto-commit onto an unrelated commit while the push targets
    ``refs/heads/develop`` -- so the two halves of the retry would disagree
    about what "develop" means.
    """
    _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _drop_fragment(synthetic_repo / "source" / "isaaclab", "feat-x")
    _commit_baseline(synthetic_repo)

    bare = tmp_path / "origin.git"
    sidecar = tmp_path / "sidecar"
    subprocess.run(["git", "clone", str(bare), str(sidecar)], check=True, capture_output=True)
    _git(sidecar, "config", "user.name", "Human Dev")
    _git(sidecar, "config", "user.email", "dev@example.com")
    # A tag named exactly like the branch, pointing somewhere else entirely.
    _git(sidecar, "commit", "--allow-empty", "-m", "decoy commit")
    _git(sidecar, "tag", "develop")
    _git(sidecar, "push", "origin", "refs/tags/develop")
    _git(sidecar, "reset", "--hard", "HEAD~1")
    # ...and a real human commit on the branch, to force the retry path.
    _git(sidecar, "commit", "--allow-empty", "-m", "human work")
    _git(sidecar, "push", "origin", "HEAD:refs/heads/develop")

    assert (
        autobump.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()
        == 0
    )
    # Qualify the ref here too: with both a tag and a branch named ``develop``
    # in the bare repo, a bare ``git log develop`` reads the *tag* -- the same
    # ambiguity this test exists to pin down, which would otherwise make the
    # assertions inspect the wrong history.
    assert _author_log(bare, "refs/heads/develop")[0] == autobump.AutoBumpRun.AUTHOR_NAME
    # The rebase must have landed on top of the human's branch commit, not the
    # tag's decoy commit.
    subjects = subprocess.run(
        ["git", "log", "refs/heads/develop", "--format=%s"],
        cwd=bare,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split("\n")
    assert "human work" in subjects
    assert "decoy commit" not in subjects


def test_partial_compile_failure_is_rolled_back_not_committed(synthetic_repo: Path, tmp_path: Path, monkeypatch):
    """A compile that raises after writing must be undone, not shipped.

    Half a compile is a changelog entry announcing a version the manifest
    never received, over a fragment that was never consumed. Committing that
    leaves the package self-inconsistent *and* leaves the fragment on the
    branch, so the next night compiles it into a second identical entry —
    the duplicate-bump failure this whole design exists to prevent.

    Rolling back satisfies the other requirement too: the tree ends clean,
    which the rebase in the push-retry loop needs.
    """
    good = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    bad = _write_managed_pkg(synthetic_repo / "source", "isaaclab_assets")
    _drop_fragment(good, "feat-x")
    bad_fragment = _drop_fragment(bad, "feat-y")
    _commit_baseline(synthetic_repo)

    real_write_version = packages.Package.write_version

    def fail_after_changelog(self, new_version, *, dry_run):
        if self.name == "isaaclab_assets":
            raise ValueError("simulated failure after the changelog was written")
        return real_write_version(self, new_version, dry_run=dry_run)

    monkeypatch.setattr(packages.Package, "write_version", fail_after_changelog)

    assert (
        autobump.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()
        == 1  # the bad package is still a failure
    )

    # The healthy package shipped...
    files = subprocess.run(
        ["git", "show", "develop", "--name-only", "--format="],
        cwd=tmp_path / "origin.git",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    assert _version_file("isaaclab") in files

    # ...and nothing from the failed one did.
    assert not any(f.startswith("source/isaaclab_assets/") for f in files), (
        f"a half-applied compile reached the branch: {files}"
    )
    # Its fragment survives on disk, so the next run can retry it cleanly.
    assert bad_fragment.exists()
    # And its changelog is back to the committed baseline — no orphan entry.
    assert "0.1.1" not in (bad / "docs" / "CHANGELOG.rst").read_text(encoding="utf-8")

    # The tree is clean either way, which the rebase in the retry loop needs.
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=synthetic_repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert dirty == "", f"working tree left dirty after partial failure: {dirty!r}"


def test_untracked_file_that_vanished_does_not_abort_staging(synthetic_repo: Path, tmp_path: Path, monkeypatch):
    """A reported path that is gone *and* was never tracked must be skipped.

    ``git add`` fails the whole invocation on such a pathspec, which would
    take down the staging of every healthy package alongside it. This is the
    branch ``GitRepo._tracked`` exists for, and no ordinary fixture reaches
    it: fragments are committed in the baseline, so they are always tracked.
    """
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _drop_fragment(pkg_root, "feat-x")
    _commit_baseline(synthetic_repo)

    # A path that never existed in git and does not exist now.
    phantom = pkg_root / "changelog.d" / "never-tracked.rst"
    real_compile = packages.Package.compile

    def compile_reporting_a_phantom(self, **kwargs):
        compiled, touched = real_compile(self, **kwargs)
        return compiled, [*touched, phantom]

    monkeypatch.setattr(packages.Package, "compile", compile_reporting_a_phantom)

    assert (
        autobump.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()
        == 0
    )
    files = subprocess.run(
        ["git", "show", "develop", "--name-only", "--format="],
        cwd=tmp_path / "origin.git",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    assert _version_file("isaaclab") in files


def test_auto_bump_argparse_wiring(synthetic_repo: Path, monkeypatch):
    """The subcommand parses and reaches the orchestrator.

    Every other test constructs ``AutoBumpRun`` directly, so nothing else
    would notice if the parser stopped handing over what it promises.
    """
    _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    monkeypatch.setattr(cli, "REPO_ROOT", synthetic_repo)
    monkeypatch.setattr(autobump, "REPO_ROOT", synthetic_repo)

    parser = cli._build_parser()
    args = parser.parse_args(["auto-bump", "--branch", "develop", "--dry-run"])

    assert args.branch == "develop"
    assert args.dry_run is True
    assert args.remote == "origin"
    assert args.func(args, parser) == 0


def test_dirty_tree_is_refused_before_anything_is_written(synthetic_repo: Path, tmp_path: Path):
    """auto-bump owns the working tree, so it refuses to share it.

    The nightly always hands it a fresh checkout. Rather than scope every
    git operation to survive a tree it will never see, the run asserts the
    precondition once — which also turns local misuse into one obvious
    error instead of a commit that quietly carries someone else's work.
    """
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    frag = _drop_fragment(pkg_root, "feat-x")
    unrelated = synthetic_repo / "unrelated.txt"
    unrelated.write_text("original\n", encoding="utf-8")
    _commit_baseline(synthetic_repo)

    # Someone left work staged before the run.
    unrelated.write_text("MEDDLED\n", encoding="utf-8")
    _git(synthetic_repo, "add", "unrelated.txt")

    with pytest.raises(autobump.GitError, match="working tree is not clean"):
        autobump.AutoBumpRun(
            branch="develop",
            remote="origin",
            event_name="schedule",
            repo_root=synthetic_repo,
        ).run()

    # Refused before touching anything: no compile, no commit, no push.
    assert frag.exists()
    assert autobump.AutoBumpRun.AUTHOR_NAME not in _author_log(tmp_path / "origin.git")


def test_manual_compile_rolls_back_a_half_applied_package(synthetic_repo: Path, monkeypatch):
    """``cli.py compile`` must undo a compile that failed after writing.

    Same hazard as the nightly path, reached manually: an entry written over
    a fragment that was never consumed means a re-run compiles it a second
    time. Only the auto-bump path used to roll this back.
    """
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    frag = _drop_fragment(pkg_root, "feat-x")
    _commit_baseline(synthetic_repo)
    monkeypatch.setattr(cli, "REPO_ROOT", synthetic_repo)
    monkeypatch.setattr(packages.Package, "discover", classmethod(lambda cls, **kw: [packages.Package(pkg_root)]))
    monkeypatch.setattr(
        packages.Package,
        "write_version",
        lambda self, v, *, dry_run: (_ for _ in ()).throw(ValueError("boom")),
    )

    args = argparse.Namespace(package=None, all=True, fragments_dir=None, version=None, dry_run=False)
    assert cli.cmd_compile(args, argparse.ArgumentParser()) == 1

    assert frag.exists(), "fragment must survive a failed compile"
    assert "0.1.1" not in (pkg_root / "docs" / "CHANGELOG.rst").read_text(encoding="utf-8")
    assert _git(synthetic_repo, "status", "--porcelain").strip() == ""
