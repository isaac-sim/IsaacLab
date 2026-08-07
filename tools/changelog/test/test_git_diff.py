# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Git-backed tests for changelog pull-request diff collection."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cli


def _git(repo: Path, *args: str) -> str:
    """Run Git in a temporary repository and return standard output."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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
    (package_dir / "untracked.py").write_text("untracked\n", encoding="utf-8")

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

    committed_diff = cli.PRDiff.from_git("develop")
    assert committed_diff == cli.PRDiff(
        changed={"source/example/committed.py"},
        added={"source/example/committed.py"},
    )

    local_diff = cli.PRDiff.from_git("develop", include_worktree=True)
    assert local_diff == cli.PRDiff(
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


def test_check_parser_defaults_to_develop_for_local_checks(monkeypatch):
    """Local checks can omit the base branch and request worktree changes."""
    monkeypatch.delenv("ISAACLAB_CHANGELOG_BASE_REF", raising=False)

    args = cli._build_parser().parse_args(["check", "--include-worktree"])

    assert args.base_ref == "develop"
    assert args.include_worktree is True


def test_check_parser_uses_environment_base_override(monkeypatch):
    """Release-branch contributors can override the local default base."""
    monkeypatch.setenv("ISAACLAB_CHANGELOG_BASE_REF", "release/6.0")

    args = cli._build_parser().parse_args(["check", "--include-worktree"])

    assert args.base_ref == "release/6.0"


def test_cmd_check_include_worktree_rejects_untracked_fragment(tmp_path, monkeypatch):
    """An untracked fragment cannot satisfy an uncommitted package change."""
    _git(tmp_path, "init", "-b", "develop")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")

    package_dir = tmp_path / "source" / "example"
    (package_dir / "docs").mkdir(parents=True)
    (package_dir / "changelog.d").mkdir()
    (package_dir / "pyproject.toml").write_text('[project]\nversion = "1.0.0"\n', encoding="utf-8")
    (package_dir / "docs" / "CHANGELOG.rst").write_text("Changelog\n---------\n\n", encoding="utf-8")
    (package_dir / "base.py").write_text("base\n", encoding="utf-8")
    (package_dir / "changelog.d" / ".gitkeep").touch()
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "Add base package")
    _git(tmp_path, "update-ref", "refs/remotes/origin/develop", "HEAD")
    _git(tmp_path, "switch", "-c", "feature")

    (package_dir / "uncommitted.py").write_text("change\n", encoding="utf-8")
    _git(tmp_path, "add", "source/example/uncommitted.py")
    (package_dir / "changelog.d" / "local.skip").touch()

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli.Package, "discover", classmethod(lambda cls: [cls(package_dir)]))
    parser = cli._build_parser()
    args = parser.parse_args(["check", "--include-worktree"])

    assert cli.cmd_check(args, parser) == 1
