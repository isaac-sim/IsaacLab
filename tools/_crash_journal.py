# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rebuild a JUnit report from the crash journal written by the repo-root ``conftest.py``.

pytest writes its JUnit XML once, in ``pytest_sessionfinish``. A run killed before that point —
a Kit shutdown crash, an OOM kill, a hard timeout — leaves no report at all, even though every
test verdict was already printed to stdout. ``tools/conftest.py`` used to answer that by
synthesizing a single ``test_execution`` error, which discarded which tests passed, which failed,
and which one was in flight when the process died.

The repo-root ``conftest.py`` journals collection, per-test start/finish, and per-test outcomes
as they happen, flushing after each record. This module turns that journal back into a report
whose test IDs match the ones a clean run produces.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from junitparser import Error, Failure, JUnitXml, Properties, Property, Skipped, TestCase, TestSuite

JOURNAL_ENV_VAR = "ISAACLAB_TEST_JOURNAL"
"""Environment variable naming the crash journal. Must match the repo-root ``conftest.py``."""

SESSION_CRASH_CASE = "session_shutdown"
"""Case name used when a run dies after its last test finished teardown.

Blaming a shutdown crash on the last test that ran would libel a test that passed, so the failure
is reported against the session instead.
"""

_OUTCOME_PRIORITY = {"passed": 0, "skipped": 1, "failed": 2}
"""Severity ranking used to fold a test's setup/call/teardown outcomes into one verdict."""

_MAX_MESSAGE_CHARS = 500
"""Maximum length of a generated JUnit ``message`` attribute."""


@dataclass
class Journal:
    """Progress of a pytest run reconstructed from its crash journal.

    Attributes:
        collected: Node IDs in collection order; empty when the run died during collection.
        markers: Registered markers per node ID, as recorded for the JUnit ``markers`` property.
        started: Node IDs in the order they began running.
        finished: Node IDs that completed teardown.
        results: Aggregated ``{"outcome", "duration", "longrepr"}`` per node ID.
    """

    collected: list[str] = field(default_factory=list)
    markers: dict[str, str] = field(default_factory=dict)
    started: list[str] = field(default_factory=list)
    finished: set[str] = field(default_factory=set)
    results: dict[str, dict] = field(default_factory=dict)

    @property
    def culprit(self) -> str | None:
        """Node ID that was in flight when the process died, or ``None`` for a shutdown crash.

        A test that started but never finished teardown is the one that took the process down.
        When every started test also finished, nothing was running and the crash belongs to
        session shutdown.
        """
        pending = [node_id for node_id in self.started if node_id not in self.finished]
        return pending[-1] if pending else None

    @property
    def ordered_node_ids(self) -> list[str]:
        """Every known node ID, in collection order, with un-collected runs appended."""
        return self.collected + [node_id for node_id in self.started if node_id not in self.collected]


def junit_names(node_id: str) -> tuple[str, str]:
    """Split a pytest node ID into the ``(classname, name)`` pair pytest's JUnit writer emits.

    ``source/pkg/test/test_x.py::TestC::test_y[p]`` becomes
    ``("source.pkg.test.test_x.TestC", "test_y[p]")``. Matching pytest's own naming is what keeps
    a test's identity stable between crashing and clean runs, so the uploaded results stay on one
    history instead of forking into a second entry.

    Args:
        node_id: A pytest node ID, with or without ``::`` segments.

    Returns:
        The JUnit ``classname`` and ``name`` for the node.
    """
    path, _, rest = node_id.partition("::")
    module = os.path.splitext(path)[0].replace("\\", "/").replace("/", ".")
    segments = [segment for segment in rest.split("::") if segment]
    if not segments:
        return module, "test_execution"
    return ".".join([module, *segments[:-1]]), segments[-1]


def _first_line(text: str, limit: int = _MAX_MESSAGE_CHARS) -> str:
    """Return the first non-empty line of ``text``, truncated for a JUnit ``message`` attribute."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return ""


def _merge_result(results: dict[str, dict], record: dict) -> None:
    """Fold one journal ``result`` record into the aggregated outcome for its node ID.

    A test emits up to three records (setup, call, teardown) and, when retried by the ``flaky``
    plugin, one set per attempt. The aggregate keeps the most severe outcome, sums the phase
    durations, and holds the first failure text seen — so a retried test collapses to one entry
    instead of the duplicate IDs the JUnit path produces.
    """
    node_id = record.get("node_id")
    if not node_id:
        return
    entry = results.setdefault(str(node_id), {"outcome": "passed", "duration": 0.0, "longrepr": ""})
    outcome = str(record.get("outcome", "passed"))
    if _OUTCOME_PRIORITY.get(outcome, 0) > _OUTCOME_PRIORITY.get(entry["outcome"], 0):
        entry["outcome"] = outcome
    try:
        entry["duration"] += float(record.get("duration") or 0.0)
    except (TypeError, ValueError):
        pass
    if not entry["longrepr"]:
        entry["longrepr"] = str(record.get("longrepr") or "")


def read_journal(journal_file: str) -> Journal | None:
    """Parse a crash journal.

    The final line can be a partial write from the killed process, so malformed lines are skipped
    rather than failing the parse.

    Args:
        journal_file: Path to the JSONL journal.

    Returns:
        The parsed journal, or ``None`` when it is missing, unreadable, or holds no usable record.
    """
    if not os.path.exists(journal_file):
        return None

    journal = Journal()
    try:
        with open(journal_file, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                event = record.get("event")
                if event == "collected":
                    journal.collected = [str(node_id) for node_id in record.get("node_ids") or []]
                    journal.markers = {str(k): str(v) for k, v in (record.get("markers") or {}).items()}
                elif event == "start" and record.get("node_id"):
                    journal.started.append(str(record["node_id"]))
                elif event == "finish" and record.get("node_id"):
                    journal.finished.add(str(record["node_id"]))
                elif event == "result":
                    _merge_result(journal.results, record)
    except OSError:
        return None

    if not journal.collected and not journal.started:
        return None
    return journal


def create_crash_report(
    journal_file: str, suite_name: str, message: str, details: str
) -> tuple[JUnitXml, dict, str | None] | None:
    """Rebuild a JUnit report for a run that died before pytest wrote its own.

    Every test the journal saw reach a verdict is emitted under its real node ID with its real
    outcome, the test that was in flight when the process died becomes an ``<error>``, and tests
    that were collected but never reached become skips so they do not silently vanish from the
    uploaded results.

    Args:
        journal_file: Path to the JSONL journal for the crashed run.
        suite_name: Name for the generated ``<testsuite>``, normally the test file stem.
        message: Short crash description, used as the error's ``message`` attribute.
        details: Full crash context (diagnostics, captured output) stored as the error body.

    Returns:
        A ``(report, counters, culprit)`` tuple where ``culprit`` is the node ID blamed for the
        crash, or ``None`` when the crash happened at session shutdown. Returns ``None`` outright
        when the journal holds nothing usable, so the caller can fall back to a synthetic entry.
    """
    journal = read_journal(journal_file)
    if journal is None:
        return None

    culprit = journal.culprit
    suite = TestSuite(name=suite_name)
    counters = {"errors": 0, "failures": 0, "skipped": 0, "tests": 0, "time_elapsed": 0.0}

    for node_id in journal.ordered_node_ids:
        classname, name = junit_names(node_id)
        case = TestCase(name=name, classname=classname)
        marker_value = journal.markers.get(node_id)
        if marker_value:
            properties = Properties()
            properties.append(Property(name="markers", value=marker_value))
            case.append(properties)

        entry = journal.results.get(node_id)
        if node_id == culprit:
            case.time = entry["duration"] if entry else 0.0
            error = Error(message=message)
            error.text = details
            case.result = [error]
            counters["errors"] += 1
        elif entry is not None:
            case.time = entry["duration"]
            counters["time_elapsed"] += entry["duration"]
            if entry["outcome"] == "failed":
                failure = Failure(message=_first_line(entry["longrepr"]) or message)
                failure.text = entry["longrepr"]
                case.result = [failure]
                counters["failures"] += 1
            elif entry["outcome"] == "skipped":
                skip = Skipped(message=_first_line(entry["longrepr"]))
                skip.text = entry["longrepr"]
                case.result = [skip]
                counters["skipped"] += 1
        else:
            skip = Skipped(message=f"not run: session aborted at {culprit or SESSION_CRASH_CASE}")
            skip.text = message
            case.result = [skip]
            counters["skipped"] += 1
        counters["tests"] += 1
        suite.add_testcase(case)

    if culprit is None:
        # Nothing was in flight, so the process died during session shutdown.
        ordered = journal.ordered_node_ids
        case = TestCase(name=SESSION_CRASH_CASE, classname=junit_names(ordered[0])[0] if ordered else suite_name)
        error = Error(message=message)
        error.text = details
        case.result = [error]
        suite.add_testcase(case)
        counters["errors"] += 1
        counters["tests"] += 1

    report = JUnitXml()
    report.add_testsuite(suite)
    report.update_statistics()
    return report, counters, culprit
