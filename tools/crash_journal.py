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

import contextlib
import json
import os
from collections import Counter
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
"""Severity ranking used to fold a phase's repeated attempts into one verdict."""

_PHASE_ORDER = {"setup": 0, "call": 1, "teardown": 2}
"""Order pytest runs a test's phases in, used to emit their results in the order they happened."""

_MAX_MESSAGE_CHARS = 500
"""Maximum length of a generated JUnit ``message`` attribute."""


@dataclass
class Journal:
    """Progress of a pytest run reconstructed from its crash journal.

    Attributes:
        collected: Node IDs in collection order; empty when the run died during collection.
        markers: Registered markers per node ID, as recorded for the JUnit ``markers`` property.
        started: Node IDs in the order they began running, one entry per attempt.
        finished: Node IDs in the order they completed teardown, one entry per attempt.
        results: ``{"outcome", "duration", "longrepr"}`` per node ID, then per phase.
    """

    collected: list[str] = field(default_factory=list)
    markers: dict[str, str] = field(default_factory=dict)
    started: list[str] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    results: dict[str, dict[str, dict]] = field(default_factory=dict)

    @property
    def culprit(self) -> str | None:
        """Node ID that was in flight when the process died, or ``None`` for a shutdown crash.

        A test that started but never finished teardown is the one that took the process down.
        When every started test also finished, nothing was running and the crash belongs to
        session shutdown.

        Starts are matched against finishes one for one, rather than tested for membership,
        because the ``flaky`` plugin reruns a test in-process and every attempt journals its own
        start and finish. Treating a node as done the moment it finished once would let an earlier
        attempt's finish mask a crash during a later one, blaming the session for a test that is
        still running — and reporting that test as never run.
        """
        unmatched = Counter(self.finished)
        for node_id in reversed(self.started):
            if unmatched[node_id]:
                unmatched[node_id] -= 1
            else:
                return node_id
        return None

    @property
    def ordered_node_ids(self) -> list[str]:
        """Every known node ID, in collection order, with un-collected runs appended once each.

        A retried node appears in :attr:`started` once per attempt, so the appended IDs are
        de-duplicated to keep one ``<testcase>`` per test.
        """
        seen = set(self.collected)
        extra = []
        for node_id in self.started:
            if node_id not in seen:
                seen.add(node_id)
                extra.append(node_id)
        return self.collected + extra


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


def _merge_result(results: dict[str, dict[str, dict]], record: dict) -> None:
    """Fold one journal ``result`` record into its node's per-phase outcomes.

    A test emits up to three records (setup, call, teardown), each of which pytest reports
    separately: a failing ``call`` becomes a ``<failure>`` while a failing ``setup``/``teardown``
    becomes an ``<error>``. The phases are therefore kept apart rather than folded into one
    verdict, so a test that fails an assertion and then breaks its own teardown reports both,
    instead of the teardown error disappearing behind the equally severe call failure.

    Repeats *within* one phase are folded, because the ``flaky`` plugin reruns a test in-process
    and journals a fresh set of records per attempt. Those keep the most severe outcome, the
    failure text that established it, and the summed duration, so a retried test still collapses
    to one entry instead of the duplicate IDs the JUnit path produces.

    Records predating the ``when`` field are treated as ``call``.
    """
    node_id = record.get("node_id")
    if not node_id:
        return
    phases = results.setdefault(str(node_id), {})
    entry = phases.setdefault(str(record.get("when") or "call"), {"outcome": "passed", "duration": 0.0, "longrepr": ""})
    outcome = str(record.get("outcome", "passed"))
    longrepr = str(record.get("longrepr") or "")
    severity = _OUTCOME_PRIORITY.get(outcome, 0)
    if severity > _OUTCOME_PRIORITY.get(entry["outcome"], 0):
        entry["outcome"] = outcome
        entry["longrepr"] = longrepr
    elif severity == _OUTCOME_PRIORITY.get(entry["outcome"], 0) and not entry["longrepr"]:
        entry["longrepr"] = longrepr
    with contextlib.suppress(TypeError, ValueError):
        entry["duration"] += float(record.get("duration") or 0.0)


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
                    journal.finished.append(str(record["node_id"]))
                elif event == "result":
                    _merge_result(journal.results, record)
    except OSError:
        return None

    if not journal.collected and not journal.started:
        return None
    return journal


def _phase_results(
    phases: dict[str, dict], message: str, counters: dict
) -> tuple[list[tuple[Failure | Error | Skipped | None, float]], float]:
    """Turn a node's journaled phases into ``(result, duration)`` pairs, in the order they ran.

    Mirrors pytest's JUnit writer: only a failing ``call`` is a ``<failure>``, a failing
    ``setup``/``teardown`` is an ``<error>``, and a passing phase produces no element at all.

    Args:
        phases: One node's ``{phase: {"outcome", "duration", "longrepr"}}`` mapping.
        message: Fallback ``message`` attribute for a failure whose text is empty.
        counters: Running report totals, updated in place.

    Returns:
        The pairs for the phases that produced a result element, and the total duration of the
        phases that did not — which the caller charges to the node's first emitted case, so the
        emitted cases still sum to the node's total time.
    """
    results: list[tuple[Failure | Error | Skipped | None, float]] = []
    idle_duration = 0.0
    for when, entry in sorted(phases.items(), key=lambda item: _PHASE_ORDER.get(item[0], 1)):
        duration = entry["duration"]
        counters["time_elapsed"] += duration
        if entry["outcome"] == "failed":
            is_call = when == "call"
            result = (Failure if is_call else Error)(message=_first_line(entry["longrepr"]) or message)
            counters["failures" if is_call else "errors"] += 1
        elif entry["outcome"] == "skipped":
            result = Skipped(message=_first_line(entry["longrepr"]))
            counters["skipped"] += 1
        else:
            idle_duration += duration
            continue
        result.text = entry["longrepr"]
        results.append((result, duration))
    return results, idle_duration


def create_crash_report(
    journal_file: str, suite_name: str, message: str, details: str
) -> tuple[JUnitXml, dict, str | None] | None:
    """Rebuild a JUnit report for a run that died before pytest wrote its own.

    Every test the journal saw reach a verdict is emitted under its real node ID with its real
    outcome, the test that was in flight when the process died gains an ``<error>`` for the crash,
    and tests that were collected but never reached become skips so they do not silently vanish
    from the uploaded results.

    A test that produced more than one result — an assertion failure followed by a teardown that
    took the process down — is emitted as one ``<testcase>`` per result, which is how pytest's own
    writer represents the same situation ("we need to open new testcase in case we have failure in
    call and error in teardown in order to follow junit schema"). Stacking both inside a single
    ``<testcase>`` would drop one of them: the results uploader reads only the first ``<failure>``
    or ``<error>`` child of a case, so the crash and its flag would never leave the XML.

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
        phases = journal.results.get(node_id) or {}
        cases, idle_duration = _phase_results(phases, message, counters)
        if node_id == culprit:
            # The crash is reported beside whatever the test already achieved: a test that failed
            # its call phase before taking the process down keeps that failure, since overwriting
            # it would throw away the assertion and traceback the journal saved.
            error = Error(message=message)
            error.text = details
            cases.append((error, 0.0))
            counters["errors"] += 1
        elif not phases:
            skip = Skipped(message=f"not run: session aborted at {culprit or SESSION_CRASH_CASE}")
            skip.text = message
            cases.append((skip, 0.0))
            counters["skipped"] += 1
        if not cases:
            # Every phase passed, so the node is one case carrying no result element.
            cases.append((None, 0.0))
        cases[0] = (cases[0][0], cases[0][1] + idle_duration)

        marker_value = journal.markers.get(node_id)
        for result, duration in cases:
            case = TestCase(name=name, classname=classname)
            if marker_value:
                properties = Properties()
                properties.append(Property(name="markers", value=marker_value))
                case.append(properties)
            case.time = duration
            if result is not None:
                case.result = [result]
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
