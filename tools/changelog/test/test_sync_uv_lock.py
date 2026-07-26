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

import pytest
import sync_uv_lock

# A miniature lock exercising every shape the rewriter must distinguish:
# a member whose version moves, a member already correct, a third-party
# package that happens to share a version string, and indented ``name`` /
# ``version`` keys inside dependency tables that must be left alone.
LOCK = """\
version = 1
requires-python = ">=3.11"

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


def test_bumped_member_version_is_rewritten():
    updated, changes = sync_uv_lock.rewrite_versions(LOCK, {"isaaclab": "13.1.0", "isaaclab-tasks": "9.1.0"})
    assert changes == [("isaaclab", "13.0.0", "13.1.0")]
    assert 'name = "isaaclab"\nversion = "13.1.0"' in updated


def test_member_already_at_target_is_untouched():
    """``isaaclab-tasks`` is already at 9.1.0 — no change, no report line."""
    updated, changes = sync_uv_lock.rewrite_versions(LOCK, {"isaaclab-tasks": "9.1.0"})
    assert changes == []
    assert updated == LOCK


def test_third_party_package_is_untouched():
    """``warp-lang`` shares 13.0.0 with the stale member but is not a member."""
    updated, _ = sync_uv_lock.rewrite_versions(LOCK, {"isaaclab": "13.1.0"})
    assert 'name = "warp-lang"\nversion = "13.0.0"' in updated


def test_indented_dependency_references_are_untouched():
    """``{ name = ..., specifier = "==13.0.0" }`` rows are data, not version fields."""
    updated, _ = sync_uv_lock.rewrite_versions(LOCK, {"isaaclab": "13.1.0", "warp-lang": "99.0.0"})
    assert '{ name = "warp-lang", specifier = "==13.0.0" },' in updated


def test_name_does_not_leak_past_its_own_block():
    """A block with no ``version`` line must not claim the next block's version.

    uv emits ``name`` then ``version`` adjacently today, so this is about the
    rewriter staying correct if a block ever lacks a version rather than a
    shape currently in ``uv.lock``.
    """
    lock = '[[package]]\nname = "isaaclab"\nsource = { editable = "source/isaaclab" }\n\n[[package]]\nname = "warp-lang"\nversion = "13.0.0"\n'
    updated, changes = sync_uv_lock.rewrite_versions(lock, {"isaaclab": "13.1.0"})
    assert changes == []
    assert updated == lock


def test_only_the_intended_lines_change():
    """Nothing outside the rewritten ``version`` lines moves, byte for byte."""
    updated, changes = sync_uv_lock.rewrite_versions(LOCK, {"isaaclab": "13.1.0"})
    differing = [(a, b) for a, b in zip(LOCK.splitlines(), updated.splitlines()) if a != b]
    assert differing == [('version = "13.0.0"', 'version = "13.1.0"')]
    assert len(LOCK.splitlines()) == len(updated.splitlines())
    assert len(changes) == 1


# ---------------------------------------------------------------------------
# Membership guard — the cases that need a real ``uv lock``, not a rewrite
# ---------------------------------------------------------------------------

ROOT_TOML = """\
[tool.uv.sources]
isaaclab = { path = "source/isaaclab", editable = true }
isaaclab-tasks = { path = "source/isaaclab_tasks", editable = true }
torch = [{ index = "pytorch-cu128" }]
"""


def _write_pair(tmp_path, lock_text, root_text=ROOT_TOML):
    lock, root = tmp_path / "uv.lock", tmp_path / "pyproject.toml"
    lock.write_text(lock_text, encoding="utf-8")
    root.write_text(root_text, encoding="utf-8")
    return lock, root


def test_matching_membership_passes(tmp_path):
    lock, root = _write_pair(tmp_path, LOCK)
    sync_uv_lock.assert_lock_is_repairable(lock, root)


def test_declared_member_absent_from_lock_is_rejected(tmp_path):
    """A package added to the workspace brings new dependency edges — needs a full lock."""
    lock, root = _write_pair(tmp_path, LOCK.replace('editable = "source/isaaclab_tasks"', 'editable = "source/other"'))
    with pytest.raises(SystemExit, match="source/isaaclab_tasks"):
        sync_uv_lock.assert_lock_is_repairable(lock, root)


def test_locked_member_no_longer_declared_is_rejected(tmp_path):
    lock, root = _write_pair(
        tmp_path,
        LOCK,
        ROOT_TOML.replace('isaaclab-tasks = { path = "source/isaaclab_tasks"', 'x = { path = "source/x"'),
    )
    with pytest.raises(SystemExit, match="no longer declared"):
        sync_uv_lock.assert_lock_is_repairable(lock, root)


def test_non_editable_sources_are_not_members(tmp_path):
    """``torch`` is an index-pinned list, not a ``{path, editable}`` table."""
    assert sync_uv_lock.read_declared_members(_write_pair(tmp_path, LOCK)[1]) == {
        sync_uv_lock.REPO_ROOT / "source/isaaclab",
        sync_uv_lock.REPO_ROOT / "source/isaaclab_tasks",
    }
