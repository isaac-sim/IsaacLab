# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Git-backed tests for changelog pull-request diff collection."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cli


def _git(repo: Path, *args: str) -> None:
    """Run Git in a temporary repository."""
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_from_git_include_worktree_collects_tracked_local_changes(tmp_path, monkeypatch):
    """Local checks include tracked changes but ignore untracked files."""
    _git(tmp_path, "init", "-b", "develop")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")

    package_dir = tmp_path / "source" / "example"
    package_dir.mkdir(parents=True)
    (package_dir / "base.py").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "source/example/base.py")
    _git(tmp_path, "commit", "-m", "Add base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/develop", "HEAD")
    _git(tmp_path, "switch", "-c", "feature")

    (package_dir / "committed.py").write_text("committed\n", encoding="utf-8")
    _git(tmp_path, "add", "source/example/committed.py")
    _git(tmp_path, "commit", "-m", "Add committed change")

    (package_dir / "staged.py").write_text("staged\n", encoding="utf-8")
    _git(tmp_path, "add", "source/example/staged.py")
    (package_dir / "base.py").write_text("unstaged\n", encoding="utf-8")
    (package_dir / "changelog.d").mkdir()
    (package_dir / "changelog.d" / "untracked.skip").touch()

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    assert cli.PRDiff.from_git("develop", include_worktree=True) == cli.PRDiff(
        changed={
            "source/example/base.py",
            "source/example/committed.py",
            "source/example/staged.py",
        },
        added={
            "source/example/committed.py",
            "source/example/staged.py",
        },
    )
