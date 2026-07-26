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
# Happy path
# ---------------------------------------------------------------------------


def test_auto_bump_happy_path_commits_and_pushes(synthetic_repo: Path, tmp_path: Path):
    _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _drop_fragment(synthetic_repo / "source" / "isaaclab", "feat-x")

    _commit_baseline(synthetic_repo)
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

    _commit_baseline(synthetic_repo)
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

    _commit_baseline(synthetic_repo)
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

    _commit_baseline(synthetic_repo)
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

    _commit_baseline(synthetic_repo)
    rc = cli.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    ).run()

    assert rc == 0
    authors = _author_log(tmp_path / "origin.git")
    assert cli.AutoBumpRun.AUTHOR_NAME not in authors


# ---------------------------------------------------------------------------
# uv.lock: synced from the same touched-paths manifest, no separate git add
# ---------------------------------------------------------------------------


def _write_workspace_lock(work: Path, names: list[str]) -> Path:
    """Turn the synthetic repo into a uv workspace.

    Writes a root manifest declaring each package as an editable member and
    a ``uv.lock`` pinning them at whatever version they currently declare —
    i.e. in sync, so any drift a test observes was produced by the bump.
    """
    sources = "\n".join(f'{name} = {{ path = "source/{name}", editable = true }}' for name in names)
    (work / "pyproject.toml").write_text(
        f'[project]\nname = "isaaclab-dev"\nversion = "0.0.0"\n\n[tool.uv.sources]\n{sources}\n',
        encoding="utf-8",
    )
    blocks = []
    for name in names:
        version = cli.Package.declared_version(work / "source" / name)
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
    rc = cli.AutoBumpRun(
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
    rc = cli.AutoBumpRun(
        branch="release/3.0.0-beta2",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    ).run()

    assert rc == 0
    authors = _author_log(tmp_path / "origin.git", branch="release/3.0.0-beta2")
    assert authors[0] == cli.AutoBumpRun.AUTHOR_NAME


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
    rc = cli.AutoBumpRun(
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
# (:attr:`cli.Package.toml_path`) rather than hardcoding an answer.
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
        cli.AutoBumpRun(
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
        cli.AutoBumpRun(
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
        cli.AutoBumpRun(
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

    real_write_version = cli.Package.write_version
    sidecar = pkg_root / "config" / "extension.toml"

    def dual_write(self, new_version, *, dry_run):
        touched = real_write_version(self, new_version, dry_run=dry_run)
        if dry_run:
            return touched
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(f'version = "{new_version}"\n', encoding="utf-8")
        return [*touched, sidecar]

    monkeypatch.setattr(cli.Package, "write_version", dual_write)

    _commit_baseline(synthetic_repo)
    assert (
        cli.AutoBumpRun(
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
    assert "source/isaaclab/config/extension.toml" in files
    # ...but the package is still reported exactly once.
    assert _commit_body(bare).count("- isaaclab:") == 1


def test_lock_present_but_workspace_undeclared_reds_the_tile_without_blocking(synthetic_repo: Path, tmp_path: Path):
    """A ``uv.lock`` with editable members but a root manifest that declares
    none of them. Membership cannot be reconciled by a version rewrite, so
    the run reports a failure — but the changelog bump still ships, matching
    how a failed package compile is handled."""
    pkg_root = _write_managed_pkg(synthetic_repo / "source", "isaaclab")
    _write_workspace_lock(synthetic_repo, ["isaaclab"])
    # Strip the workspace declaration, keeping the lock's member entry.
    (synthetic_repo / "pyproject.toml").write_text(
        '[project]\nname = "isaaclab-dev"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    _drop_fragment(pkg_root, "feat-x")

    _commit_baseline(synthetic_repo)
    rc = cli.AutoBumpRun(
        branch="develop",
        remote="origin",
        event_name="schedule",
        repo_root=synthetic_repo,
    ).run()

    assert rc == 1
    files = subprocess.run(
        ["git", "show", "develop", "--name-only", "--format="],
        cwd=tmp_path / "origin.git",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    assert _version_file("isaaclab") in files
    assert "uv.lock" not in files
