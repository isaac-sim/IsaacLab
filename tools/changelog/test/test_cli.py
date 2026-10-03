# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import datetime
import shutil
import subprocess

import cli
import pytest
import tomllib

CHANGELOG = "Changelog\n---------\n\n.. towncrier release notes start\n"
SOURCES = (
    '[tool.uv.sources]\nother = { path = "source/other", editable = true }\n'
    'pkg = { path = "source/pkg", editable = true }\n'
)


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
    write(tmp_path, {"uv.lock": lock(), "pyproject.toml": SOURCES})
    git(tmp_path, "init", "-q", "-b", "develop")
    commit(tmp_path, "base")
    git(tmp_path, "update-ref", "refs/remotes/origin/develop", "HEAD")
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    return tmp_path


def check(include_worktree=False, base_ref="develop"):
    return cli.cmd_check(argparse.Namespace(base_ref=base_ref, include_worktree=include_worktree))


@pytest.mark.parametrize(
    ("files", "status", "base_ref"),
    [
        ({FRAGMENTS + "a.fixed.rst": "* Fixed x.\n  More on x.\n", FRAGMENTS + "a.major": ""}, 0, "develop"),
        ({FRAGMENTS + "a.skip": ""}, 0, "upstream/develop"),
        ({FRAGMENTS + "a.minor.rst": "* Added x.\n"}, 1, "upstream/develop"),  # pre-towncrier name
        ({FRAGMENTS + "a.minor": ""}, 1, "refs/heads/develop"),  # a tier file alone is not an entry
        ({FRAGMENTS + "a.fixed.rst": ""}, 1, "commit"),
        ({FRAGMENTS + "a.fixed.rst": "* Fixed x.\nFlush-left paragraph.\n"}, 1, "develop"),
    ],
)
def test_check_requires_a_valid_fragment_for_a_changed_package(repo, files, status, base_ref):
    # Rotate supported base forms across existing cases without multiplying Git fixtures.
    git(repo, "update-ref", "refs/remotes/upstream/develop", "HEAD")
    if base_ref == "commit":
        base_ref = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "switch", "-qc", "feature")
    write(repo, {"source/pkg/code.py": "x = 1\n", **files})
    commit(repo, "change")
    assert check(base_ref=base_ref) == status


def test_check_rejects_editing_or_deleting_a_pending_fragment(repo, capsys):
    write(repo, {FRAGMENTS + "a.fixed.rst": "* Fixed x.\n", FRAGMENTS + "b.fixed.rst": "* Fixed y.\n"})
    commit(repo, "pending fragments")
    git(repo, "update-ref", "refs/remotes/origin/develop", "HEAD")
    write(repo, {FRAGMENTS + "a.fixed.rst": "* Fixed x differently.\n"})
    (repo / FRAGMENTS / "b.fixed.rst").unlink()
    git(repo, "add", "-A")
    assert check(include_worktree=True) == 1
    errors = capsys.readouterr().out
    assert FRAGMENTS + "a.fixed.rst" in errors and FRAGMENTS + "b.fixed.rst" in errors


def test_check_ignores_changes_outside_packages(repo):
    write(repo, {"tools/script.py": "x = 1\n"})
    git(repo, "add", "-A")
    assert check(include_worktree=True) == 0


@pytest.mark.parametrize(
    "files",
    [
        {"source/other/docs/CHANGELOG.rst": CHANGELOG.replace("start\n", "start here\n")},  # marker line altered
        {"source/other/pyproject.toml": '[project]\nname = "other"\nversion = "1.0.1"\n'},  # lock pin stale
        {"pyproject.toml": SOURCES + 'new = { path = "source/new", editable = true }\n'},  # package not locked
    ],
)
def test_check_requires_the_marker_and_current_lock_pins(repo, files):
    write(repo, {**files, "source/other/changelog.d/a.skip": ""})
    git(repo, "add", "-A")
    assert check(include_worktree=True) == 1


def test_compile_refuses_to_overwrite_a_fragment_when_splitting(repo):
    write(repo, {FRAGMENTS + "a.rst": "Fixed\n^^^^^\n\n* Fixed x.\n", FRAGMENTS + "a.fixed.rst": "* Fixed y.\n"})
    with pytest.raises(ValueError):
        cli.compile_package(repo / "source/pkg")
    assert (repo / FRAGMENTS / "a.fixed.rst").read_text() == "* Fixed y.\n"


def fake_tools(repo, monkeypatch, relocked, stray=False, lock_fails=False):
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
            if lock_fails:
                raise subprocess.CalledProcessError(1, cmd, stderr="resolution failed")
            return ""
        return run(*cmd, env=env)

    monkeypatch.setattr(cli, "run", fake_run)


def test_compile_stages_only_its_outputs(repo, monkeypatch):
    # ``other`` has only a leftover tier file: no release, and the file is dropped.
    write(repo, {FRAGMENTS + "a.fixed.rst": "* Fixed x.\n", "source/other/changelog.d/b.minor": ""})
    commit(repo, "fragments")
    fake_tools(repo, monkeypatch, relocked=lock(pkg="1.0.1"), stray=True)
    assert cli.cmd_compile(argparse.Namespace()) == 1  # the stray notes.txt a tool wrote is reported
    staged = set(git(repo, "diff", "--cached", "--name-only").split())
    assert staged == {
        "source/pkg/pyproject.toml",
        "source/pkg/docs/CHANGELOG.rst",
        FRAGMENTS + "a.fixed.rst",
        "source/other/changelog.d/b.minor",
        "uv.lock",
    }


def test_compile_refuses_a_dirty_checkout(repo, monkeypatch):
    write(repo, {FRAGMENTS + "a.fixed.rst": "* Fixed x.\n", "notes.txt": "local work"})
    run = cli.run
    monkeypatch.setattr(cli, "run", lambda *cmd, env=None: run(*cmd) if cmd[0] == "git" else pytest.fail(f"ran {cmd}"))
    assert cli.cmd_compile(argparse.Namespace()) == 1
    assert (repo / "notes.txt").read_text() == "local work"


ROOT = '\n[[package]]\nname = "root"\nversion = "{}"\nsource = {{ virtual = "." }}\n'


@pytest.mark.parametrize(
    ("base", "relocked", "lock_fails"),
    [
        (lock(), lock(header="version = 1\nrevision = 3\n"), False),
        (lock() + ROOT.format("0.1.0"), lock(pkg="1.0.1") + ROOT.format("0.2.0"), False),  # only versions move
        (lock(), lock(pkg="1.0.1"), True),
    ],
    ids=["re-serialized", "non-local-version-moved", "uv-lock-failed"],
)
def test_compile_aborts_when_the_relock_fails(repo, monkeypatch, base, relocked, lock_fails):
    write(repo, {"uv.lock": base, FRAGMENTS + "a.fixed.rst": "* Fixed x.\n", "source/other/changelog.d/b.skip": ""})
    commit(repo, "fragments")
    fake_tools(repo, monkeypatch, relocked, lock_fails=lock_fails)
    assert cli.cmd_compile(argparse.Namespace()) == 1
    assert git(repo, "status", "--porcelain") == ""


@pytest.mark.skipif(not (shutil.which("uv") and shutil.which("uvx")), reason="needs uv and uvx")
def test_compile_renders_the_changelog_with_real_uv_and_towncrier(repo, monkeypatch):
    """Runs the real ``uv version`` and towncrier with the repository config and template.

    An already-merged pre-towncrier ``d.major.rst`` is split, and its tier outranks ``a.minor``.
    """
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
            FRAGMENTS + "d.major.rst": "Fixed\n^^^^^\n\n* Fixed z.\n",
        },
    )
    commit(repo, "fragments")
    assert cli.compile_package(repo / "source/pkg") == "2.0.0"
    today = datetime.date.today().isoformat()
    title = f"2.0.0 ({today})"
    new_entry = (
        f"\n{title}\n{'~' * len(title)}\n\nAdded\n^^^^^\n\n* Added x.\n  More on x.\n\n"
        "Fixed\n^^^^^\n\n* Fixed y.\n* Fixed z.\n\n"
    )
    assert (repo / "source/pkg/docs/CHANGELOG.rst").read_text() == CHANGELOG + new_entry + old_entry
    assert [p.name for p in (repo / FRAGMENTS).iterdir()] == [".gitkeep"]
    assert tomllib.loads((repo / "source/pkg/pyproject.toml").read_text())["project"]["version"] == "2.0.0"
