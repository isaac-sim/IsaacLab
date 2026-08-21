# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX renderer log collector (``tools/ovrtx_log.py``).

Each behavioural test drives a nested pytest session whose "renderer" imitates the part of OVRTX that
matters here: a native writer that opens its log once and keeps the handle for the lifetime of the process
(``keep_system_alive=True``). The nested session loads the collector the same way the repo-root
``conftest.py`` does and runs under pytest's default file-descriptor capture, so what it prints is what CI
would print.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]

_INNER_CONFTEST = (
    '"""Loads the collector exactly as the repo-root conftest does."""\n\npytest_plugins = ["tools.ovrtx_log"]\n'
)

_FAKE_RENDERER = '''\
import os
import tempfile

_HANDLE = []
LOG_PATH = os.path.join(tempfile.gettempdir(), "ovrtx_renderer.log")


def _log():
    """Open the log once and keep the handle, as OVRTX does with keep_system_alive=True.

    Resolving the path the way ``OVRTXRendererCfg.log_file_path`` defaults to it is what makes this a
    stand-in for the renderer; the session under test redirects the temp directory.
    """
    if not _HANDLE:
        _HANDLE.append(open(LOG_PATH, "wb", buffering=0))
    return _HANDLE[0]


def render(tag, count=1):
    """Append ``count`` renderer log lines tagged ``tag``."""
    _log().write(f"[omni.rtx] {tag}-line\\n".encode() * count)


def reopen():
    """Drop the retained handle and open the log again, truncating it."""
    _HANDLE.clear()
    _log()
'''

_STDOUT_RENDERER = '''\
import os

_FD = []


def _fd():
    """Open pytest's stdout once and keep the descriptor, as the removed /dev/stdout override did."""
    if not _FD:
        _FD.append(os.open("/dev/stdout", os.O_WRONLY))
    return _FD[0]


def render(tag, count=1):
    """Append ``count`` renderer log lines tagged ``tag``."""
    os.write(_fd(), f"[omni.rtx] {tag}-line\\n".encode() * count)
'''


def _load_collector_module():
    """Import ``tools/ovrtx_log.py`` the way the repo-root conftest loads it."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import tools.ovrtx_log as collector

    return collector


def _inner_log_path(tmp_path: Path) -> Path:
    """Return the log the nested session writes, i.e. the renderer default under its temp directory."""
    return tmp_path / "ovrtx_renderer.log"


def _run_inner_pytest(tmp_path: Path, renderer: str, tests: str, save_dir: Path | None = None) -> bytes:
    """Run ``tests`` in a nested pytest session and return its raw output.

    The session's temp directory is redirected to ``tmp_path`` so its renderer log is this test's own file
    rather than the one a real run on this machine uses.

    Args:
        tmp_path: Directory the nested session is written to, run from, and given as its temp directory.
        renderer: Source of the fake renderer module prepended to the test module.
        tests: Source of the tests themselves.
        save_dir: Directory the session saves each test's log to, as CI asks it to. ``None`` leaves the
            environment variable unset, which is what a local run does.

    Returns:
        The nested session's stdout and stderr, undecoded, so NUL bytes survive.
    """
    (tmp_path / "conftest.py").write_text(_INNER_CONFTEST, encoding="utf-8")
    (tmp_path / "test_inner.py").write_text(renderer + textwrap.dedent(tests), encoding="utf-8")

    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    env["TMPDIR"] = str(tmp_path)
    env["PYTHONPATH"] = os.pathsep.join([str(_REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # Dropped rather than left alone: under CI this test process already has it set, and the nested
    # session would otherwise inherit that directory and save into the real job artifact.
    env.pop(_load_collector_module().LOG_DIR_ENV_VAR, None)
    if save_dir is not None:
        env[_load_collector_module().LOG_DIR_ENV_VAR] = str(save_dir)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly", "test_inner.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=600,
    )
    return result.stdout + result.stderr


def test_collector_is_loaded_into_every_session(request) -> None:
    """The repo-root conftest wires the collector in, so no suite has to wire it in itself."""
    assert request.config.pluginmanager.hasplugin("tools.ovrtx_log")


def test_retained_handle_on_pytest_stdout_writes_nul_blocks(tmp_path: Path) -> None:
    """Pin the failure this collector exists to avoid.

    Pointing the renderer at ``/dev/stdout`` hands it a second handle on the temporary file pytest captures
    file descriptor 1 into. Pytest rewinds and truncates that file between tests while the renderer's own
    offset keeps advancing, so the next native write lands past the end and the gap reads back as NUL bytes.
    """
    if os.name == "nt":
        pytest.skip("/dev/stdout is POSIX-only")

    output = _run_inner_pytest(
        tmp_path,
        _STDOUT_RENDERER,
        """
        def test_alpha():
            render("alpha", 40)
            assert False


        def test_beta():
            render("beta")
            assert False
        """,
    )

    assert b"\x00" in output


def test_replayed_log_is_free_of_nul_blocks_and_attributed_per_test(tmp_path: Path) -> None:
    """A renderer logging to the claimed file keeps pytest's capture intact and its output readable.

    The same retained-handle writer as above, pointed at the log the collector claims for the process: no
    descriptor of pytest's is shared, so nothing is written past the end of the capture file, and each
    test's failure carries the renderer output produced while it ran.
    """
    output = _run_inner_pytest(
        tmp_path,
        _FAKE_RENDERER,
        """
        def test_alpha():
            render("alpha", 40)
            assert False


        def test_beta():
            render("beta")
            assert False
        """,
    )

    assert b"\x00" not in output
    # Reported against the test's name, the same string the saved copy is named after.
    assert b"----- OVRTX renderer log: test_alpha -----" in output
    assert b"----- OVRTX renderer log: test_beta -----" in output
    # Replayed once each: a repeat would mean one test was credited with another's output.
    assert output.count(b"alpha-line") == 40
    assert output.count(b"beta-line") == 1


def test_replay_restarts_when_the_log_is_rewritten(tmp_path: Path) -> None:
    """A log re-opened mid-session is replayed whole rather than from an offset it no longer has."""
    output = _run_inner_pytest(
        tmp_path,
        _FAKE_RENDERER,
        """
        def test_alpha():
            render("alpha", 40)
            assert False


        def test_beta():
            reopen()
            render("beta")
            assert False
        """,
    )

    assert output.count(b"alpha-line") == 40
    assert output.count(b"beta-line") == 1


def test_replay_caps_a_long_log_and_reports_what_it_dropped(tmp_path: Path) -> None:
    """A verbose failing test replays a bounded tail, and says how much it left out.

    The cap is what keeps a long renderer log out of the job log and the JUnit XML, which is the problem the
    NUL blocks caused with sparse bytes rather than text.
    """
    collector = _load_collector_module()
    # Enough filler to push the head out of the reported tail, counted from the cap itself so that
    # raising the cap does not quietly turn this into a test of a log that fits inside it.
    filler_lines = collector.LOG_LIMIT_BYTES // len(b"[omni.rtx] filler-line\n") + 1

    output = _run_inner_pytest(
        tmp_path,
        _FAKE_RENDERER,
        f"""
        def test_verbose():
            render("head")
            render("filler", {filler_lines})
            render("tail")
            assert False
        """,
    )

    assert b"head-line" not in output
    assert b"tail-line" in output
    assert b"earlier bytes omitted" in output


def test_saved_log_is_written_per_test_when_a_directory_is_named(tmp_path: Path) -> None:
    """With the artifact directory named, each test's log is saved under a directory of its own name.

    This is the copy that survives the caps: the replay and the crash report both quote a bounded tail,
    so a run whose renderer log outgrows the cap is only diagnosable from the saved file.
    """
    save_dir = tmp_path / "ovrtx-logs"

    output = _run_inner_pytest(
        tmp_path,
        _FAKE_RENDERER,
        """
        def test_alpha():
            render("alpha", 40)
            assert False


        def test_beta():
            render("beta")
            assert False
        """,
        save_dir=save_dir,
    )

    saved = sorted(path.name for path in save_dir.iterdir())
    assert saved == ["test_alpha.0", "test_beta.0"]
    # The saved copy is unbounded where the replay is capped, and the replay still happens alongside it.
    assert (save_dir / "test_alpha.0" / "ovrtx_renderer.log").read_bytes().count(b"alpha-line") == 40
    assert output.count(b"alpha-line") == 40


def test_dumps_written_beside_the_log_are_saved_with_it(tmp_path: Path) -> None:
    """A crashing renderer leaves a dump next to its log, which is why the log's directory is saved.

    Neither the replay nor the crash report quotes anything but the log, so a dump reaches a diagnosis
    through this copy or not at all. Subdirectories are left alone: the log's directory is the system temp
    directory, which every other process on the machine is free to build trees in.
    """
    save_dir = tmp_path / "ovrtx-logs"
    (tmp_path / "somebody_elses_tree").mkdir()

    _run_inner_pytest(
        tmp_path,
        _FAKE_RENDERER,
        """
        def test_crashing():
            render("alpha")
            # Where a crashing renderer drops its dump: beside the log it was writing.
            with open(os.path.join(os.path.dirname(LOG_PATH), "ovrtx_crash.dmp"), "wb") as handle:
                handle.write(b"dump-bytes")
            assert False
        """,
        save_dir=save_dir,
    )

    saved = {path.name for path in (save_dir / "test_crashing.0").iterdir()}
    assert {"ovrtx_renderer.log", "ovrtx_crash.dmp"} <= saved
    assert "somebody_elses_tree" not in saved


def test_nothing_is_saved_for_a_test_that_never_builds_a_renderer(tmp_path: Path) -> None:
    """The artifact directory is named for the whole run, but most tests never write renderer output.

    Nothing creates the log until the renderer's first write, so the save has to tolerate its absence
    rather than failing the test that ran without one.
    """
    save_dir = tmp_path / "ovrtx-logs"

    output = _run_inner_pytest(
        tmp_path,
        _FAKE_RENDERER,
        """
        def test_no_renderer():
            assert True
        """,
        save_dir=save_dir,
    )

    assert b"1 passed" in output
    assert not save_dir.exists()


def test_log_left_behind_by_a_previous_owner_is_discarded(tmp_path: Path) -> None:
    """One path is shared by every session on the machine, so a session starts by claiming it.

    Without dropping what is already there, the offset recorded before the renderer rewrites the file is
    measured against a previous session's bytes, and the start of this session's log is skipped.
    """
    _inner_log_path(tmp_path).write_bytes(b"[omni.rtx] stale-line\n" * 10)

    output = _run_inner_pytest(
        tmp_path,
        _FAKE_RENDERER,
        """
        def test_alpha():
            render("alpha", 40)
            assert False
        """,
    )

    assert b"stale-line" not in output
    assert output.count(b"alpha-line") == 40


def test_missing_log_reports_nothing(tmp_path: Path) -> None:
    """Every test that never builds a renderer must add nothing to its report."""
    collector = _load_collector_module()

    assert collector.format_log_section(str(tmp_path / "absent.log"), "test_a") == ""
    (tmp_path / "empty.log").write_bytes(b"")
    assert collector.format_log_section(str(tmp_path / "empty.log"), "test_a") == ""
