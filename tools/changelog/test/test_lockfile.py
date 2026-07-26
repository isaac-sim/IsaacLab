# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Version rewriting in ``uv.lock`` after a nightly bump.

The rewrite must be surgical: workspace members' own ``version`` lines and
nothing else. The blast radius of a mistake here is a corrupted lockfile
landing in an unreviewed auto-commit, so the tests pin both halves — what
does change and what must not.
"""

from __future__ import annotations

import packages
import pytest
from lockfile import LockFile
from packages import RootPackage

from conftest import write_version_file

# A miniature lock exercising every shape the rewriter must distinguish: a
# member whose version moves, a member already correct, a third-party
# package that happens to share a version string, the virtual workspace
# root, and indented ``name`` / ``version`` keys inside dependency and
# metadata tables that must be left alone.
LOCK = """\
version = 1
requires-python = ">=3.11"

[[package]]
name = "isaaclab-dev"
version = "0.0.0"
source = { virtual = "." }

[[package]]
name = "isaaclab"
version = "13.0.0"
source = { editable = "source/isaaclab" }
dependencies = [
    { name = "warp-lang" },
]

[package.metadata]
requires-dist = [
    { name = "warp-lang", specifier = "==13.0.0" },
]

[[package]]
name = "isaaclab-tasks"
version = "9.1.0"
source = { editable = "source/isaaclab_tasks" }

[[package]]
name = "warp-lang"
version = "13.0.0"
source = { registry = "https://pypi.org/simple" }
"""

ROOT_TOML = """\
[project]
name = "isaaclab-dev"
version = "0.0.0"

[tool.uv.sources]
isaaclab = { path = "source/isaaclab", editable = true }
isaaclab-tasks = { path = "source/isaaclab_tasks", editable = true }
"""


def _write_workspace(root, *, lock=LOCK, root_toml=ROOT_TOML, versions=None) -> LockFile:
    """Lay down a workspace root with a lock, a root manifest, and members.

    ``versions`` maps a member directory name to the version its own
    manifest declares; it defaults to the lock's own values (in sync).

    Member versions go through :func:`conftest.write_version_file`, so each
    one lands in whatever file this branch's ``Package`` actually reads.
    Hardcoding the filename here would make every drift assertion pass
    vacuously on a branch that keeps versions elsewhere: no target would
    resolve, and the lock would look permanently "in sync".
    """
    (root / "uv.lock").write_text(lock, encoding="utf-8")
    (root / "pyproject.toml").write_text(root_toml, encoding="utf-8")
    for name, version in (versions or {"isaaclab": "13.0.0", "isaaclab_tasks": "9.1.0"}).items():
        pkg = root / "source" / name
        pkg.mkdir(parents=True, exist_ok=True)
        write_version_file(pkg, name, version)
    return LockFile(RootPackage(root))


# ---------------------------------------------------------------------------
# The rewrite: what moves, and what must not
# ---------------------------------------------------------------------------


def test_bumped_member_version_is_rewritten():
    updated, drifts = LockFile._rewrite(LOCK, {"isaaclab": "13.1.0", "isaaclab-tasks": "9.1.0"})
    assert [(d.package, d.old, d.new) for d in drifts] == [("isaaclab", "13.0.0", "13.1.0")]
    assert 'name = "isaaclab"\nversion = "13.1.0"' in updated


def test_third_party_package_is_never_rewritten():
    """``warp-lang`` shares isaaclab's old version string. Only names present
    in ``targets`` may move, so the registry package must survive verbatim."""
    updated, _ = LockFile._rewrite(LOCK, {"isaaclab": "13.1.0"})
    assert 'name = "warp-lang"\nversion = "13.0.0"' in updated


def test_indented_dependency_rows_are_never_rewritten():
    """``    { name = "warp-lang", specifier = "==13.0.0" }`` is a dependency
    reference, not a package pin. The column-zero anchor keeps the rewriter
    off it."""
    updated, _ = LockFile._rewrite(LOCK, {"warp-lang": "99.0.0"})
    assert 'specifier = "==13.0.0"' in updated


def test_exactly_one_line_changes():
    """Byte-for-byte: a rewrite touches the version line and nothing else."""
    updated, _ = LockFile._rewrite(LOCK, {"isaaclab": "13.1.0"})
    before, after = LOCK.splitlines(), updated.splitlines()
    assert len(before) == len(after)
    assert [i for i, (b, a) in enumerate(zip(before, after)) if b != a] == [before.index('version = "13.0.0"')]


# ---------------------------------------------------------------------------
# Block scoping: a package name is not a safe key
# ---------------------------------------------------------------------------

# The same name twice: once as the editable workspace member, once as a
# registry release of the same project. Only the editable block may be
# re-pointed — moving the registry pin would leave its recorded hashes and
# source metadata describing a different version.
DUPLICATE_NAME_LOCK = """\
version = 1

[[package]]
name = "isaaclab"
version = "13.0.0"
source = { editable = "source/isaaclab" }

[[package]]
name = "isaaclab"
version = "12.4.0"
source = { registry = "https://pypi.org/simple" }
"""

DUPLICATE_NAME_ROOT_TOML = """\
[project]
name = "isaaclab-dev"
version = "0.0.0"

[tool.uv.sources]
isaaclab = { path = "source/isaaclab", editable = true }
"""


def test_registry_block_sharing_a_member_name_is_not_rewritten(tmp_path):
    lock = _write_workspace(
        tmp_path,
        lock=DUPLICATE_NAME_LOCK,
        root_toml=DUPLICATE_NAME_ROOT_TOML,
        versions={"isaaclab": "13.1.0"},
    )
    updated, drifts = lock._guarded_drift()
    assert [(d.package, d.old, d.new) for d in drifts] == [("isaaclab", "13.0.0", "13.1.0")]
    assert 'version = "13.1.0"\nsource = { editable = "source/isaaclab" }' in updated
    assert 'version = "12.4.0"\nsource = { registry = "https://pypi.org/simple" }' in updated


# ---------------------------------------------------------------------------
# Resolving what the lock should say
# ---------------------------------------------------------------------------


def test_targets_come_from_member_manifests_and_skip_the_virtual_root(tmp_path):
    """The virtual workspace root carries a ``version`` line but no editable
    source, so it must never become a rewrite target."""
    lock = _write_workspace(tmp_path, versions={"isaaclab": "13.1.0", "isaaclab_tasks": "9.1.0"})
    assert lock._target_versions() == {"isaaclab": "13.1.0", "isaaclab-tasks": "9.1.0"}


def test_member_without_a_manifest_keeps_its_locked_version(tmp_path):
    """An unmanaged or half-checked-out member must not fail a workspace-wide
    operation — it simply keeps whatever the lock already records."""
    lock = _write_workspace(tmp_path, versions={"isaaclab": "13.1.0"})
    assert lock._target_versions() == {"isaaclab": "13.1.0"}


# ---------------------------------------------------------------------------
# Guards — sync and check must be equally strict
# ---------------------------------------------------------------------------

ENTRY_POINTS = [lambda lk: lk.sync(), lambda lk: lk.check()]
ENTRY_IDS = ["sync", "check"]


@pytest.mark.parametrize("call", ENTRY_POINTS, ids=ENTRY_IDS)
def test_membership_mismatch_is_refused(tmp_path, call):
    """A declared member the lock has never seen cannot be repaired by a
    version rewrite. ``check`` must be exactly as strict as ``sync``, or the
    gate reports "in sync" for a lock that needs a full re-lock."""
    root_toml = ROOT_TOML + 'isaaclab-newton = { path = "source/isaaclab_newton", editable = true }\n'
    lock = _write_workspace(tmp_path, root_toml=root_toml)
    with pytest.raises(LockFile.MembershipMismatch, match="declared but absent"):
        call(lock)


def test_locked_member_no_longer_declared_is_refused(tmp_path):
    root_toml = ROOT_TOML.replace('isaaclab-tasks = { path = "source/isaaclab_tasks", editable = true }\n', "")
    lock = _write_workspace(tmp_path, root_toml=root_toml)
    with pytest.raises(LockFile.MembershipMismatch, match="no longer declared"):
        lock.assert_repairable()


@pytest.mark.parametrize("call", ENTRY_POINTS, ids=ENTRY_IDS)
def test_malformed_toml_raises_the_modules_own_error(tmp_path, call):
    """A parser failure must arrive as ``LockFile.Error``, not as a raw
    traceback out of whichever command happened to call in."""
    lock = _write_workspace(tmp_path, lock="this is not toml = [[[\n")
    with pytest.raises(LockFile.Error):
        call(lock)


@pytest.mark.parametrize("call", ENTRY_POINTS, ids=ENTRY_IDS)
def test_branch_without_a_lock_is_a_clean_noop(tmp_path, call):
    """Release branches cut before the uv workspace carry no ``uv.lock``.
    That must be a no-op, not a traceback — the same code is cherry-picked
    to those branches."""
    lock = LockFile(RootPackage(tmp_path))
    assert lock.exists is False
    assert call(lock) == []


# ---------------------------------------------------------------------------
# sync(): the path auto-bump actually calls
# ---------------------------------------------------------------------------


def test_sync_writes_the_lock_and_reports_the_path(tmp_path):
    lock = _write_workspace(tmp_path, versions={"isaaclab": "13.1.0", "isaaclab_tasks": "9.1.0"})
    assert lock.sync() == [tmp_path / "uv.lock"]
    assert 'name = "isaaclab"\nversion = "13.1.0"' in (tmp_path / "uv.lock").read_text(encoding="utf-8")


def test_sync_is_idempotent(tmp_path):
    """A second sync reports nothing to do.

    This is what holds the cached read honest: the first call rewrites the
    very file the cache was taken from, so a cache outliving that write
    would replay stale drift and report a change that no longer exists.
    """
    lock = _write_workspace(tmp_path, versions={"isaaclab": "13.1.0", "isaaclab_tasks": "9.1.0"})
    lock.sync()
    assert lock.sync() == []


def test_sync_of_an_in_sync_lock_touches_nothing(tmp_path):
    assert _write_workspace(tmp_path).sync() == []


def test_dry_run_reports_but_does_not_write(tmp_path):
    lock = _write_workspace(tmp_path, versions={"isaaclab": "13.1.0", "isaaclab_tasks": "9.1.0"})
    assert lock.sync(dry_run=True) == []
    assert (tmp_path / "uv.lock").read_text(encoding="utf-8") == LOCK


# ---------------------------------------------------------------------------
# Gate against the checked-in workspace
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (packages.REPO_ROOT / "uv.lock").is_file(), reason="branch carries no uv.lock")
def test_repo_lock_membership_is_repairable():
    """The real lock's member set must match the real root manifest.

    This is the assertion that fires if someone adds a package to
    ``[tool.uv.sources]`` without re-locking — catching it at PR time rather
    than at 5 AM in the auto-commit job. Deliberately does NOT assert the
    lock is version-in-sync: the nightly is what brings those into line, so
    drift there is expected between runs.
    """
    LockFile(RootPackage(packages.REPO_ROOT)).assert_repairable()
