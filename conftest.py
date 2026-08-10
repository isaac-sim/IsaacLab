# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Repo-root pytest configuration shared by every ``source/`` and ``scripts/`` test.

Collects the markers applied to each test and records them in the JUnit XML report so that CI
can carry them into the uploaded test artifact.

Also maintains a crash journal (see :data:`JOURNAL_ENV_VAR`). pytest writes its JUnit XML once,
at the end of the session, so a run killed before then - a Kit shutdown crash, an OOM kill, a
hard timeout - loses every verdict it had already printed. The journal records collection,
per-test start/finish, and per-test outcomes as they happen, letting ``tools/conftest.py``
rebuild a real report instead of a single synthetic ``test_execution`` entry.

Level markers (``unit`` / ``integration`` / ``benchmark``) are applied per file via a module-level ``pytestmark``
and registered in the repo-root ``pyproject.toml``. Select them with the standard ``-m`` syntax,
e.g. ``pytest -m unit source/isaaclab/test`` or ``pytest -m "not unit" source/isaaclab/test``.
"""

from __future__ import annotations

import json
import os

JOURNAL_ENV_VAR = "ISAACLAB_TEST_JOURNAL"
"""Environment variable naming the crash-journal file. Unset (the default) disables journaling."""

_MAX_LONGREPR_CHARS = 4000
"""Maximum characters of a failure representation stored on a single journal record."""


def _journal_write(record: dict) -> None:
    """Append one JSON record to the crash journal, when one is configured.

    The per-record flush is the whole point: it puts the data in the OS page cache before the
    next test starts, so a process killed by a signal cannot take down verdicts it had already
    reported. Journaling failures are swallowed — losing debug context must never fail a run.
    """
    path = os.environ.get(JOURNAL_ENV_VAR)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
    except (OSError, TypeError, ValueError):
        pass


def pytest_collection_modifyitems(config, items):
    """Record each collected test's registered markers for the JUnit report.

    Each item's marker set (taken from its ``@pytest.mark`` annotations) is recorded
    as a ``user_properties`` entry so it is emitted into the JUnit XML report as
    ``<property name="markers" value="...">``. CI's ``upload-omni-github-test-results``
    action reads that property and carries the markers into the uploaded test artifact.

    Only markers registered in the repo-root ``pyproject.toml`` (e.g. ``unit``,
    ``integration``, ``benchmark``, ``rendering``) are recorded; pytest's built-in structural marks
    (``parametrize``, ``skip``, ``usefixtures``, ...) are excluded so they do not leak
    into the artifact's ``test_type`` field.

    The same marker map is journaled alongside the collected node IDs so a crash-rebuilt
    report can reproduce both, keeping ``test_type`` consistent with a clean run.
    """
    registered = {entry.split(":", 1)[0].strip() for entry in config.getini("markers")}
    journal_markers = {}
    for item in items:
        markers = {mark.name for mark in item.iter_markers() if mark.name in registered}
        value = ",".join(sorted(markers))
        item.user_properties.append(("markers", value))
        if value:
            journal_markers[item.nodeid] = value
    _journal_write(
        {
            "event": "collected",
            "node_ids": [item.nodeid for item in items],
            "markers": journal_markers,
        }
    )


def pytest_runtest_logstart(nodeid, location):
    """Journal that a test is about to run, so a crash can be attributed to it."""
    _journal_write({"event": "start", "node_id": nodeid})


def pytest_runtest_logreport(report):
    """Journal a test phase's outcome as soon as pytest resolves it.

    Passing setup and teardown phases are skipped to keep the journal small; everything that
    decides a test's reported outcome (the call phase, plus any non-passing phase) is kept.
    """
    if report.when != "call" and report.outcome == "passed":
        return
    _journal_write(
        {
            "event": "result",
            "node_id": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "duration": getattr(report, "duration", 0.0),
            "longrepr": str(report.longrepr)[-_MAX_LONGREPR_CHARS:] if report.longrepr else "",
        }
    )


def pytest_runtest_logfinish(nodeid, location):
    """Journal that a test completed teardown.

    A node with a ``start`` and no ``finish`` is the one that was in flight when the process
    died. When every started node also finished, the crash happened during session shutdown
    and must not be blamed on the last test that ran.
    """
    _journal_write({"event": "finish", "node_id": nodeid})
