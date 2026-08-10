# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``tools/_crash_journal.py``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from xml.etree import ElementTree

from _crash_journal import SESSION_CRASH_CASE, create_crash_report, junit_names, read_journal

_FILE = "source/pkg/test/test_x.py"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_journal(tmp_path: Path, records: list[dict], trailing: str = "") -> str:
    """Write ``records`` as JSONL and return the path, optionally with a partial trailing line."""
    path = tmp_path / "journal.jsonl"
    body = "".join(json.dumps(record) + "\n" for record in records)
    path.write_text(body + trailing, encoding="utf-8")
    return str(path)


def _cases(report) -> dict[str, ElementTree.Element]:
    """Return the generated cases keyed by ``classname::name``."""
    root = ElementTree.fromstring(report.tostring())
    return {f"{case.get('classname')}::{case.get('name')}": case for case in root.iter("testcase")}


def _result_tag(case: ElementTree.Element) -> str:
    """Return the local tag of a case's result child, or ``"passed"`` when it has none."""
    for child in case:
        if child.tag in ("failure", "error", "skipped"):
            return child.tag
    return "passed"


# -- junit_names ------------------------------------------------------------------------------


def test_junit_names_matches_module_path():
    assert junit_names(f"{_FILE}::test_y[a-b]") == ("source.pkg.test.test_x", "test_y[a-b]")


def test_junit_names_folds_class_into_classname():
    assert junit_names(f"{_FILE}::TestC::test_y") == ("source.pkg.test.test_x.TestC", "test_y")


def test_junit_names_handles_windows_separators():
    assert junit_names("source\\pkg\\test_x.py::test_y")[0] == "source.pkg.test_x"


def test_junit_names_without_node_part_falls_back():
    assert junit_names(_FILE) == ("source.pkg.test.test_x", "test_execution")


# -- read_journal -----------------------------------------------------------------------------


def test_read_journal_missing_file_returns_none(tmp_path):
    assert read_journal(str(tmp_path / "absent.jsonl")) is None


def test_read_journal_without_records_returns_none(tmp_path):
    assert read_journal(_write_journal(tmp_path, [])) is None


def test_read_journal_skips_partial_trailing_line(tmp_path):
    journal_file = _write_journal(
        tmp_path,
        [{"event": "collected", "node_ids": [f"{_FILE}::test_a"]}, {"event": "start", "node_id": f"{_FILE}::test_a"}],
        trailing='{"event": "res',
    )
    journal = read_journal(journal_file)
    assert journal.collected == [f"{_FILE}::test_a"]
    assert journal.started == [f"{_FILE}::test_a"]


def test_read_journal_collapses_repeated_results_to_worst_outcome(tmp_path):
    # The flaky plugin reruns a test in-process, so one node ID can emit several result records.
    journal_file = _write_journal(
        tmp_path,
        [
            {"event": "collected", "node_ids": [f"{_FILE}::test_a"]},
            {"event": "result", "node_id": f"{_FILE}::test_a", "outcome": "passed", "duration": 1.0},
            {"event": "result", "node_id": f"{_FILE}::test_a", "outcome": "failed", "duration": 2.0, "longrepr": "x"},
        ],
    )
    entry = read_journal(journal_file).results[f"{_FILE}::test_a"]
    assert entry["outcome"] == "failed"
    assert entry["duration"] == 3.0


# -- create_crash_report ----------------------------------------------------------------------


def test_create_crash_report_without_journal_returns_none(tmp_path):
    assert create_crash_report(str(tmp_path / "absent.jsonl"), "test_x", "boom", "details") is None


def test_create_crash_report_preserves_verdicts_and_blames_in_flight_test(tmp_path):
    journal_file = _write_journal(
        tmp_path,
        [
            {
                "event": "collected",
                "node_ids": [f"{_FILE}::test_a", f"{_FILE}::test_b", f"{_FILE}::test_c", f"{_FILE}::test_d"],
                "markers": {f"{_FILE}::test_a": "flaky,unit"},
            },
            {"event": "start", "node_id": f"{_FILE}::test_a"},
            {"event": "result", "node_id": f"{_FILE}::test_a", "when": "call", "outcome": "passed", "duration": 1.5},
            {"event": "finish", "node_id": f"{_FILE}::test_a"},
            {"event": "start", "node_id": f"{_FILE}::test_b"},
            {
                "event": "result",
                "node_id": f"{_FILE}::test_b",
                "when": "call",
                "outcome": "failed",
                "duration": 2.0,
                "longrepr": "assert 1 == 2\nE  AssertionError",
            },
            {"event": "finish", "node_id": f"{_FILE}::test_b"},
            # test_c starts and never finishes: it took the process down.
            {"event": "start", "node_id": f"{_FILE}::test_c"},
        ],
    )

    report, counters, culprit = create_crash_report(journal_file, "test_x", "SIGSEGV", "diagnostics")

    assert culprit == f"{_FILE}::test_c"
    cases = _cases(report)
    assert _result_tag(cases["source.pkg.test.test_x::test_a"]) == "passed"
    assert _result_tag(cases["source.pkg.test.test_x::test_b"]) == "failure"
    assert _result_tag(cases["source.pkg.test.test_x::test_c"]) == "error"
    # Collected but never reached: recorded as skipped so it does not vanish from the results.
    assert _result_tag(cases["source.pkg.test.test_x::test_d"]) == "skipped"
    assert counters == {"errors": 1, "failures": 1, "skipped": 1, "tests": 4, "time_elapsed": 3.5}


def test_create_crash_report_carries_markers_property(tmp_path):
    journal_file = _write_journal(
        tmp_path,
        [
            {"event": "collected", "node_ids": [f"{_FILE}::test_a"], "markers": {f"{_FILE}::test_a": "flaky,unit"}},
            {"event": "start", "node_id": f"{_FILE}::test_a"},
            {"event": "result", "node_id": f"{_FILE}::test_a", "when": "call", "outcome": "passed", "duration": 1.0},
        ],
    )
    report, _, _ = create_crash_report(journal_file, "test_x", "SIGSEGV", "diagnostics")
    case = _cases(report)["source.pkg.test.test_x::test_a"]
    assert [(p.get("name"), p.get("value")) for p in case.iter("property")] == [("markers", "flaky,unit")]


def test_create_crash_report_blames_session_when_every_test_finished(tmp_path):
    # Kit crashing during shutdown must not be charged to the last test, which passed.
    journal_file = _write_journal(
        tmp_path,
        [
            {"event": "collected", "node_ids": [f"{_FILE}::test_a"]},
            {"event": "start", "node_id": f"{_FILE}::test_a"},
            {"event": "result", "node_id": f"{_FILE}::test_a", "when": "call", "outcome": "passed", "duration": 1.0},
            {"event": "finish", "node_id": f"{_FILE}::test_a"},
        ],
    )

    report, counters, culprit = create_crash_report(journal_file, "test_x", "SIGSEGV", "diagnostics")

    assert culprit is None
    cases = _cases(report)
    assert _result_tag(cases["source.pkg.test.test_x::test_a"]) == "passed"
    assert _result_tag(cases[f"source.pkg.test.test_x::{SESSION_CRASH_CASE}"]) == "error"
    assert counters["errors"] == 1


def test_create_crash_report_handles_crash_during_collection(tmp_path):
    journal_file = _write_journal(tmp_path, [{"event": "collected", "node_ids": [f"{_FILE}::test_a"]}])

    report, counters, culprit = create_crash_report(journal_file, "test_x", "SIGSEGV", "diagnostics")

    assert culprit is None
    assert _result_tag(_cases(report)["source.pkg.test.test_x::test_a"]) == "skipped"
    assert counters["errors"] == 1


# -- identity stability against real pytest ---------------------------------------------------


def _run_pytest(tmp_path: Path, target: str, journal_file: Path, junit_file: Path):
    """Run a real pytest session in ``tmp_path`` with the repo-root journaling hooks loaded."""
    # Pin rootdir to tmp_path so node IDs stay relative to it, the way CI's --config-file does.
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    # Load the repo-root journaling hooks as a plugin without making tmp_path a repo checkout.
    (tmp_path / "journal_plugin.py").write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(_REPO_ROOT)!r})
            from conftest import (
                pytest_collection_modifyitems,
                pytest_runtest_logfinish,
                pytest_runtest_logreport,
                pytest_runtest_logstart,
            )
            """
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "journal_plugin", f"--junitxml={junit_file}", target],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "ISAACLAB_TEST_JOURNAL": str(journal_file),
            # Keep the baseline clean: third-party plugins must not reshape the JUnit IDs.
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
    )


def test_process_death_mid_test_still_reports_verdicts_and_names_the_culprit(tmp_path):
    """Regression test for crashes surfacing only a single synthetic entry.

    ``os._exit`` reproduces what a Kit shutdown crash or an OOM kill does to a run: the process
    dies before ``pytest_sessionfinish``, so pytest never writes its JUnit XML and every verdict
    it had already reported is lost. The journal has to carry them out instead.
    """
    (tmp_path / "source" / "pkg" / "test").mkdir(parents=True)
    (tmp_path / "source" / "pkg" / "test" / "test_x.py").write_text(
        textwrap.dedent(
            """
            import os

            def test_a():
                pass

            def test_b():
                assert 1 == 2

            def test_c():
                os._exit(1)

            def test_d():
                pass
            """
        ),
        encoding="utf-8",
    )

    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    _run_pytest(tmp_path, "source/pkg/test/test_x.py", journal_file, junit_file)

    # Precondition: this is exactly the situation where the old code had nothing to work with.
    assert not junit_file.exists()

    report, counters, culprit = create_crash_report(str(journal_file), "test_x", "SIGKILL", "diagnostics")

    assert culprit == "source/pkg/test/test_x.py::test_c"
    cases = _cases(report)
    assert _result_tag(cases["source.pkg.test.test_x::test_a"]) == "passed"
    assert _result_tag(cases["source.pkg.test.test_x::test_b"]) == "failure"
    assert _result_tag(cases["source.pkg.test.test_x::test_c"]) == "error"
    assert _result_tag(cases["source.pkg.test.test_x::test_d"]) == "skipped"
    assert counters["tests"] == 4


def test_rebuilt_ids_match_the_ids_pytest_writes_on_a_clean_run(tmp_path):
    """The rebuilt IDs must equal pytest's own, or a crashed run forks a test's history.

    Runs a real pytest session with the repo-root journaling hooks loaded, then checks that
    ``junit_names`` applied to the journaled node IDs reproduces the ``classname``/``name``
    pairs pytest itself wrote to the JUnit XML.
    """
    (tmp_path / "source" / "pkg" / "test").mkdir(parents=True)
    (tmp_path / "source" / "pkg" / "test" / "test_x.py").write_text(
        textwrap.dedent(
            """
            import pytest

            @pytest.mark.parametrize("value", ["a-b", "c"])
            def test_y(value):
                pass

            class TestC:
                def test_z(self):
                    pass
            """
        ),
        encoding="utf-8",
    )
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    result = _run_pytest(tmp_path, "source/pkg/test/test_x.py", journal_file, junit_file)
    assert result.returncode == 0, result.stdout + result.stderr

    root = ElementTree.fromstring(junit_file.read_text(encoding="utf-8"))
    pytest_ids = {(case.get("classname"), case.get("name")) for case in root.iter("testcase")}
    journal = read_journal(str(journal_file))
    rebuilt_ids = {junit_names(node_id) for node_id in journal.collected}

    assert rebuilt_ids == pytest_ids
    assert len(pytest_ids) == 3
