# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import subprocess

import cli
import pytest

LOCK = 'version = 1\n\n[[package]]\nname = "pkg"\nversion = "1.0.0"\nsource = { editable = "source/pkg" }\n'


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def write(repo, files):
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with packages ``pkg`` and ``other`` whose ``develop`` is also ``origin/develop``."""
    for pkg in ("pkg", "other"):
        write(tmp_path, {f"source/{pkg}/docs/CHANGELOG.rst": "Changelog\n---------\n", f"source/{pkg}/code.py": ""})
    write(tmp_path, {"uv.lock": LOCK})
    git(tmp_path, "init", "-q", "-b", "develop")
    git(tmp_path, "add", "-A")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    git(tmp_path, "update-ref", "refs/remotes/origin/develop", "HEAD")
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    return tmp_path


def test_split_legacy_maps_sections_and_tier(tmp_path):
    path = tmp_path / "my-branch.minor.rst"
    path.write_text("Added\n^^^^^\n\n* Added x.\n  More on x.\n\nFixed\n^^^^^\n\n* Fixed y.\n")
    assert cli.split_legacy(path) == {
        "my-branch.minor": "",
        "my-branch.added.rst": "* Added x.\n  More on x.\n",
        "my-branch.fixed.rst": "* Fixed y.\n",
    }


@pytest.mark.parametrize(
    "text",
    [
        "* Fixed x without a heading.\n",
        "Notes\n^^^^^\n\n* Unknown section.\n",
        "Fixed\n^^^^^\n\n* Fixed x.\nFlush-left paragraph.\n",
        "Fixed\n^^^^^\n\n* Fixed x.\n\nFixed\n^^^^^\n\n* Fixed y.\n",
    ],
)
def test_split_legacy_rejects_malformed_fragments(tmp_path, text):
    path = tmp_path / "my-branch.rst"
    path.write_text(text)
    with pytest.raises(ValueError):
        cli.split_legacy(path)


FRAGMENTS = "source/pkg/changelog.d/"


@pytest.mark.parametrize(
    ("files", "status"),
    [
        ({FRAGMENTS + "a.fixed.rst": "* Fixed x.\n"}, 0),
        ({FRAGMENTS + "a.added.rst": "* Added x.\n", FRAGMENTS + "a.major": ""}, 0),
        ({FRAGMENTS + "a.skip": ""}, 0),
        ({FRAGMENTS + "a.minor.rst": "Added\n^^^^^\n\n* Added x.\n"}, 0),  # legacy
        ({}, 1),
        ({FRAGMENTS + "a.minor": ""}, 1),
        ({FRAGMENTS + "a.notes.rst": "* Noted x.\n"}, 1),
        ({FRAGMENTS + "a.rst": "Fixed\n^^^^^\n\n* Fixed x.\nFlush-left paragraph.\n"}, 1),  # legacy
    ],
)
def test_check_requires_a_valid_fragment_for_a_changed_package(repo, files, status):
    write(repo, {"source/pkg/code.py": "x = 1\n", **files})
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "change")
    assert cli.cmd_check(argparse.Namespace(base_ref="develop", include_worktree=False)) == status


def test_check_ignores_changes_outside_packages(repo):
    write(repo, {"tools/script.py": "x = 1\n"})
    git(repo, "add", "-A")
    assert cli.cmd_check(argparse.Namespace(base_ref="develop", include_worktree=True)) == 0


def test_compile_bumps_highest_tier_then_builds(repo, monkeypatch):
    pkg = repo / "source/pkg"
    write(repo, {FRAGMENTS + "a.rst": "Fixed\n^^^^^\n\n* Fixed x.\n", FRAGMENTS + "b.added.rst": "* Added y.\n"})
    write(repo, {FRAGMENTS + "b.minor": ""})
    calls = []
    monkeypatch.setattr(cli, "run", lambda *cmd, env=None: calls.append(cmd) or "1.1.0\n")
    assert cli.compile_package(pkg) == "1.1.0"
    assert calls == [
        ("uv", "version", "--project", pkg, "--bump", "minor", "--frozen", "--short"),
        (*cli.TOWNCRIER, "build", "--yes", "--config", cli.CONFIG, "--dir", pkg, "--version", "1.1.0"),
    ]
    assert (pkg / "changelog.d/a.fixed.rst").read_text() == "* Fixed x.\n"


def test_compile_drops_stale_markers_without_entries(repo, monkeypatch):
    write(repo, {FRAGMENTS + "a.skip": "", FRAGMENTS + "a.minor": ""})
    monkeypatch.setattr(cli, "run", lambda *cmd, env=None: pytest.fail(f"unexpected {cmd}"))
    assert cli.compile_package(repo / "source/pkg") is None
    assert list((repo / FRAGMENTS).iterdir()) == []


@pytest.mark.parametrize(
    ("relocked", "accepted"),
    [(LOCK.replace("1.0.0", "1.1.0"), True), (LOCK.replace("version = 1\n", "version = 1\nrevision = 3\n"), False)],
)
def test_sync_lock_accepts_only_version_lines(repo, monkeypatch, relocked, accepted):
    run = cli.run

    def fake_run(*cmd, env=None):
        if cmd[:2] == ("uv", "lock"):
            (repo / "uv.lock").write_text(relocked)
            return ""
        return run(*cmd, env=env)

    monkeypatch.setattr(cli, "run", fake_run)
    if accepted:
        cli.sync_lock()
    else:
        with pytest.raises(RuntimeError):
            cli.sync_lock()
    assert (repo / "uv.lock").read_text() == (relocked if accepted else LOCK)
