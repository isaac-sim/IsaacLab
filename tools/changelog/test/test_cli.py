# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import datetime
import shutil
import subprocess

import cli
import legacy
import pytest
import tomllib

CHANGELOG = "Changelog\n---------\n\n.. towncrier release notes start\n"


def lock(pkg="1.0.0", other="1.0.0", header="version = 1\n"):
    """Return a ``uv.lock`` pinning the two local packages at the given versions."""
    packages = {"other": other, "pkg": pkg}
    return header + "".join(
        f'\n[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {{ editable = "source/{name}" }}\n'
        for name, version in packages.items()
    )


FRAGMENTS = "source/pkg/changelog.d/"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def write(repo, files):
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)


def commit(repo, message):
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Packages ``pkg`` and ``other`` at version 1.0.0, locked, with ``origin/develop`` at ``HEAD``."""
    for pkg in ("pkg", "other"):
        write(
            tmp_path,
            {
                f"source/{pkg}/docs/CHANGELOG.rst": CHANGELOG,
                f"source/{pkg}/pyproject.toml": f'[project]\nname = "{pkg}"\nversion = "1.0.0"\n',
                f"source/{pkg}/changelog.d/.gitkeep": "",
                f"source/{pkg}/code.py": "",
            },
        )
    write(tmp_path, {"uv.lock": lock()})
    git(tmp_path, "init", "-q", "-b", "develop")
    commit(tmp_path, "base")
    git(tmp_path, "update-ref", "refs/remotes/origin/develop", "HEAD")
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    return tmp_path


def check(include_worktree=False):
    return cli.cmd_check(argparse.Namespace(base_ref="develop", include_worktree=include_worktree))


def test_split_legacy_maps_sections_and_tier(tmp_path):
    path = tmp_path / "my-branch.minor.rst"
    path.write_text("Added\n^^^^^\n\n* Added x.\n  More on x.\n\nFixed\n^^^^^\n\n* Fixed y.\n")
    assert legacy.split_legacy(path) == {
        "my-branch.minor": "",
        "my-branch.added.rst": "* Added x.\n  More on x.\n",
        "my-branch.fixed.rst": "* Fixed y.\n",
    }


@pytest.mark.parametrize(
    "text",
    [
        "* Fixed x without a heading.\n",
        "Notes\n^^^^^\n\n* Unknown section.\n",
        "Fixed\n^^^^^\n\n* Fixed x.\n\nFixed\n^^^^^\n\n* Fixed y.\n",
    ],
)
def test_split_legacy_rejects_malformed_fragments(tmp_path, text):
    path = tmp_path / "my-branch.rst"
    path.write_text(text)
    with pytest.raises(ValueError):
        legacy.split_legacy(path)


@pytest.mark.parametrize(
    ("files", "status"),
    [
        ({FRAGMENTS + "a.fixed.rst": "* Fixed x.\n  More on x.\n"}, 0),
        ({FRAGMENTS + "a.added.rst": "* Added x.\n", FRAGMENTS + "a.major": ""}, 0),
        ({FRAGMENTS + "a.skip": ""}, 0),
        ({FRAGMENTS + "a.minor.rst": "Added\n^^^^^\n\n* Added x.\n"}, 0),  # legacy
        ({}, 1),
        ({FRAGMENTS + "a.minor": ""}, 1),
        ({FRAGMENTS + "a.notes.rst": "* Noted x.\n"}, 1),
        ({FRAGMENTS + "a.fixed.rst": ""}, 1),
        ({FRAGMENTS + "a.fixed.rst": "* Fixed x.\nFlush-left paragraph.\n"}, 1),
        ({FRAGMENTS + "a.rst": "Fixed\n^^^^^\n\n* Fixed x.\nFlush-left paragraph.\n"}, 1),  # legacy
    ],
)
def test_check_requires_a_valid_fragment_for_a_changed_package(repo, files, status):
    write(repo, {"source/pkg/code.py": "x = 1\n", **files})
    commit(repo, "change")
    assert check() == status


def test_check_ignores_changes_outside_packages(repo):
    write(repo, {"tools/script.py": "x = 1\n"})
    git(repo, "add", "-A")
    assert check(include_worktree=True) == 0


@pytest.mark.parametrize(
    "files",
    [
        {"source/other/docs/CHANGELOG.rst": "Changelog\n---------\n"},  # marker removed
        {"source/other/pyproject.toml": '[project]\nname = "other"\nversion = "1.0.1"\n'},  # lock pin stale
    ],
)
def test_check_requires_the_marker_and_current_lock_pins(repo, files):
    write(repo, {**files, "source/other/changelog.d/a.skip": ""})
    git(repo, "add", "-A")
    assert check(include_worktree=True) == 1


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


def test_compile_refuses_to_overwrite_a_fragment_when_splitting(repo):
    write(repo, {FRAGMENTS + "a.rst": "Fixed\n^^^^^\n\n* Fixed x.\n", FRAGMENTS + "a.fixed.rst": "* Fixed y.\n"})
    with pytest.raises(ValueError):
        cli.compile_package(repo / "source/pkg")
    assert (repo / FRAGMENTS / "a.fixed.rst").read_text() == "* Fixed y.\n"


def test_compile_drops_stale_markers_without_entries(repo, monkeypatch):
    write(repo, {FRAGMENTS + "a.skip": "", FRAGMENTS + "a.minor": ""})
    monkeypatch.setattr(cli, "run", lambda *cmd, env=None: pytest.fail(f"unexpected {cmd}"))
    assert cli.compile_package(repo / "source/pkg") is None
    assert [p.name for p in (repo / FRAGMENTS).iterdir()] == [".gitkeep"]


def fake_tools(repo, monkeypatch, relocked, stray=False):
    """Replace uv and towncrier with file edits that mimic them; git still runs for real."""
    run = cli.run

    def fake_run(*cmd, env=None):
        if cmd[:2] == ("uv", "version"):
            write(repo, {"source/pkg/pyproject.toml": '[project]\nname = "pkg"\nversion = "1.0.1"\n'})
            return "1.0.1\n"
        if cmd[: len(cli.TOWNCRIER)] == cli.TOWNCRIER:
            write(repo, {"source/pkg/docs/CHANGELOG.rst": CHANGELOG + "\n1.0.1 (today)\n"})
            if stray:
                write(repo, {"notes.txt": "a file no compile output should touch"})
            (repo / FRAGMENTS / "a.fixed.rst").unlink()
            return ""
        if cmd[:2] == ("uv", "lock"):
            write(repo, {"uv.lock": relocked})
            return ""
        return run(*cmd, env=env)

    monkeypatch.setattr(cli, "run", fake_run)


def test_compile_stages_only_its_outputs(repo, monkeypatch):
    write(repo, {FRAGMENTS + "a.fixed.rst": "* Fixed x.\n"})
    commit(repo, "fragment")
    fake_tools(repo, monkeypatch, relocked=lock(pkg="1.0.1"), stray=True)
    assert cli.cmd_compile(argparse.Namespace()) == 1  # the stray notes.txt a tool wrote is reported
    staged = set(git(repo, "diff", "--cached", "--name-only").split())
    assert staged == {
        "source/pkg/pyproject.toml",
        "source/pkg/docs/CHANGELOG.rst",
        FRAGMENTS + "a.fixed.rst",
        "uv.lock",
    }


def test_compile_refuses_a_dirty_checkout(repo, monkeypatch):
    write(repo, {FRAGMENTS + "a.fixed.rst": "* Fixed x.\n", "notes.txt": "local work"})
    run = cli.run
    monkeypatch.setattr(cli, "run", lambda *cmd, env=None: run(*cmd) if cmd[0] == "git" else pytest.fail(f"ran {cmd}"))
    assert cli.cmd_compile(argparse.Namespace()) == 1
    assert (repo / "notes.txt").read_text() == "local work"


def test_compile_holds_every_bump_when_the_relock_is_refused(repo, monkeypatch):
    write(repo, {FRAGMENTS + "a.fixed.rst": "* Fixed x.\n"})
    commit(repo, "fragment")
    fake_tools(repo, monkeypatch, relocked=lock(header="version = 1\nrevision = 3\n"))
    assert cli.cmd_compile(argparse.Namespace()) == 1
    assert git(repo, "status", "--porcelain") == ""


@pytest.mark.parametrize(
    ("relocked", "accepted"),
    [(lock(pkg="1.1.0", other="1.1.0"), True), (lock(header="version = 1\nrevision = 3\n"), False)],
)
def test_sync_lock_accepts_only_version_lines(repo, monkeypatch, relocked, accepted):
    fake_tools(repo, monkeypatch, relocked)
    if accepted:
        cli.sync_lock()
    else:
        with pytest.raises(RuntimeError):
            cli.sync_lock()
    assert (repo / "uv.lock").read_text() == (relocked if accepted else lock())


def test_sync_lock_rejects_a_version_change_outside_local_packages(repo, monkeypatch):
    root = '\n[[package]]\nname = "root"\nversion = "{}"\nsource = {{ virtual = "." }}\n'
    write(repo, {"uv.lock": lock() + root.format("0.1.0")})
    commit(repo, "root package")
    fake_tools(repo, monkeypatch, relocked=lock() + root.format("0.2.0"))  # only a `version =` line moves
    with pytest.raises(RuntimeError):
        cli.sync_lock()
    assert (repo / "uv.lock").read_text() == lock() + root.format("0.1.0")


@pytest.mark.skipif(not (shutil.which("uv") and shutil.which("uvx")), reason="needs uv and uvx")
def test_compile_renders_the_changelog_with_real_uv_and_towncrier(repo, monkeypatch):
    """Runs the real ``uv version`` and towncrier with the repository config and template."""
    shutil.copytree(cli.CONFIG.parent, repo / "tools/changelog", ignore=shutil.ignore_patterns("test", "__pycache__"))
    monkeypatch.setattr(cli, "CONFIG", repo / "tools/changelog/towncrier.toml")
    old_entry = "\n1.0.0 (2026-01-01)\n~~~~~~~~~~~~~~~~~~\n\nAdded\n^^^^^\n\n* Initial.\n"
    write(
        repo,
        {
            "source/pkg/docs/CHANGELOG.rst": CHANGELOG + old_entry,
            FRAGMENTS + "b.fixed.rst": "* Fixed y.\n",
            FRAGMENTS + "a.added.rst": "* Added x.\n  More on x.\n",
            FRAGMENTS + "a.minor": "",
            FRAGMENTS + "c.skip": "",
        },
    )
    commit(repo, "fragments")
    assert cli.compile_package(repo / "source/pkg") == "1.1.0"
    today = datetime.date.today().isoformat()
    title = f"1.1.0 ({today})"
    new_entry = (
        f"\n{title}\n{'~' * len(title)}\n\nAdded\n^^^^^\n\n* Added x.\n  More on x.\n\nFixed\n^^^^^\n\n* Fixed y.\n\n"
    )
    assert (repo / "source/pkg/docs/CHANGELOG.rst").read_text() == CHANGELOG + new_entry + old_entry
    assert [p.name for p in (repo / FRAGMENTS).iterdir()] == [".gitkeep"]
    assert tomllib.loads((repo / "source/pkg/pyproject.toml").read_text())["project"]["version"] == "1.1.0"
