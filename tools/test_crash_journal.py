# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``tools/crash_journal.py``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from xml.etree import ElementTree

import pytest
from crash_journal import SESSION_CRASH_CASE, create_crash_report, junit_names, read_journal

_FILE = "source/pkg/test/test_x.py"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_journal(tmp_path: Path, records: list[dict], trailing: str = "") -> str:
    """Write ``records`` as JSONL and return the path, optionally with a partial trailing line."""
    path = tmp_path / "journal.jsonl"
    body = "".join(json.dumps(record) + "\n" for record in records)
    path.write_text(body + trailing, encoding="utf-8")
    return str(path)


def _case_list(report) -> list[ElementTree.Element]:
    """Return the generated cases in document order."""
    return list(ElementTree.fromstring(report.tostring()).iter("testcase"))


def _cases(report) -> dict[str, ElementTree.Element]:
    """Return the generated cases keyed by ``classname::name``.

    Only valid where each node emits a single case; use :func:`_cases_for` when a node is
    expected to emit one case per result.
    """
    cases = _case_list(report)
    keyed = {f"{case.get('classname')}::{case.get('name')}": case for case in cases}
    assert len(keyed) == len(cases), "duplicate case IDs: use _cases_for"
    return keyed


def _cases_for(report, case_id: str) -> list[ElementTree.Element]:
    """Return every case whose ``classname::name`` equals ``case_id``, in document order."""
    return [case for case in _case_list(report) if f"{case.get('classname')}::{case.get('name')}" == case_id]


def _result_tags(case: ElementTree.Element) -> list[str]:
    """Return the local tags of a case's result children, in document order."""
    return [child.tag for child in case if child.tag in ("failure", "error", "skipped")]


def _result_tag(case: ElementTree.Element) -> str:
    """Return the local tag of a case's first result child, or ``"passed"`` when it has none."""
    tags = _result_tags(case)
    return tags[0] if tags else "passed"


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
    # The flaky plugin reruns a test in-process, so one node ID can emit several result records
    # for the same phase. Those fold together; records for different phases do not.
    journal_file = _write_journal(
        tmp_path,
        [
            {"event": "collected", "node_ids": [f"{_FILE}::test_a"]},
            {"event": "result", "node_id": f"{_FILE}::test_a", "outcome": "passed", "duration": 1.0},
            {"event": "result", "node_id": f"{_FILE}::test_a", "outcome": "failed", "duration": 2.0, "longrepr": "x"},
        ],
    )
    # Records predating the ``when`` field are treated as ``call``.
    phases = read_journal(journal_file).results[f"{_FILE}::test_a"]
    assert list(phases) == ["call"]
    assert phases["call"]["outcome"] == "failed"
    assert phases["call"]["duration"] == 3.0


def test_read_journal_blames_a_retry_that_never_finished(tmp_path):
    # The flaky plugin reruns a test in-process, journaling one start/finish pair per attempt.
    # The first attempt's finish must not mask the second attempt's missing one, or the test that
    # took the process down is reported as never run and the session takes the blame.
    journal_file = _write_journal(
        tmp_path,
        [
            {"event": "collected", "node_ids": [f"{_FILE}::test_a", f"{_FILE}::test_b"]},
            {"event": "start", "node_id": f"{_FILE}::test_a"},
            {"event": "finish", "node_id": f"{_FILE}::test_a"},
            {"event": "start", "node_id": f"{_FILE}::test_a"},
        ],
    )
    assert read_journal(journal_file).culprit == f"{_FILE}::test_a"


def test_read_journal_blames_the_session_when_every_attempt_finished(tmp_path):
    # The counterpart: both attempts completed, so nothing was in flight and no test is at fault.
    journal_file = _write_journal(
        tmp_path,
        [
            {"event": "collected", "node_ids": [f"{_FILE}::test_a"]},
            {"event": "start", "node_id": f"{_FILE}::test_a"},
            {"event": "finish", "node_id": f"{_FILE}::test_a"},
            {"event": "start", "node_id": f"{_FILE}::test_a"},
            {"event": "finish", "node_id": f"{_FILE}::test_a"},
        ],
    )
    assert read_journal(journal_file).culprit is None


def test_read_journal_keeps_each_phase_separately(tmp_path):
    # A skipped setup records its reason; a later teardown failure must not inherit it as its
    # body, nor bury it - pytest reports the two in different JUnit buckets.
    journal_file = _write_journal(
        tmp_path,
        [
            {"event": "collected", "node_ids": [f"{_FILE}::test_a"]},
            {
                "event": "result",
                "node_id": f"{_FILE}::test_a",
                "when": "setup",
                "outcome": "skipped",
                "longrepr": "Skipped: needs a GPU",
            },
            {
                "event": "result",
                "node_id": f"{_FILE}::test_a",
                "when": "teardown",
                "outcome": "failed",
                "longrepr": "RuntimeError: teardown boom",
            },
        ],
    )
    phases = read_journal(journal_file).results[f"{_FILE}::test_a"]
    assert phases["setup"]["outcome"] == "skipped"
    assert phases["setup"]["longrepr"] == "Skipped: needs a GPU"
    assert phases["teardown"]["outcome"] == "failed"
    assert phases["teardown"]["longrepr"] == "RuntimeError: teardown boom"


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


def test_create_crash_report_keeps_the_recorded_failure_of_the_culprit(tmp_path):
    # A test that failed its call phase and then took the process down during teardown must keep
    # its assertion failure: it was journaled before the crash, so losing it hides a real defect.
    # The two land in separate cases, the way pytest splits a call failure from a teardown error.
    journal_file = _write_journal(
        tmp_path,
        [
            {"event": "collected", "node_ids": [f"{_FILE}::test_a"]},
            {"event": "start", "node_id": f"{_FILE}::test_a"},
            {
                "event": "result",
                "node_id": f"{_FILE}::test_a",
                "when": "call",
                "outcome": "failed",
                "duration": 2.0,
                "longrepr": "assert 1 == 2\nE  AssertionError",
            },
            # No finish record: the process died in teardown.
        ],
    )

    report, counters, culprit = create_crash_report(journal_file, "test_x", "SIGSEGV", "diagnostics")

    assert culprit == f"{_FILE}::test_a"
    failure_case, error_case = _cases_for(report, "source.pkg.test.test_x::test_a")
    assert _result_tags(failure_case) == ["failure"]
    assert _result_tags(error_case) == ["error"]
    assert "AssertionError" in (failure_case.find("failure").text or "")
    assert error_case.find("error").get("message") == "SIGSEGV"
    assert counters["failures"] == 1
    assert counters["errors"] == 1
    # Both cases are counted, so the orchestrator's tests - failures - errors - skipped summary
    # stays at zero passed instead of going negative.
    assert counters["tests"] == 2
    assert counters["time_elapsed"] == 2.0


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


# -- real pytest subprocess harness -----------------------------------------------------------

_TARGET = "source/pkg/test/test_x.py"
"""Path of the throwaway test module the subprocess tests collect, relative to their rootdir."""


def _write_test_module(tmp_path: Path, body: str) -> None:
    """Write ``body`` as the throwaway test module a subprocess pytest run will collect."""
    module = tmp_path / _TARGET
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(textwrap.dedent(body), encoding="utf-8")


def _write_conftest(tmp_path: Path, body: str) -> None:
    """Write ``body`` as the rootdir ``conftest.py`` of a subprocess pytest run.

    Used to inject a crash from a pytest hook, which is the only way to die at a point the test
    functions themselves cannot reach — session shutdown, or before the run loop starts.
    """
    (tmp_path / "conftest.py").write_text(textwrap.dedent(body), encoding="utf-8")


def _run_pytest(
    tmp_path: Path, target: str, journal_file: Path, junit_file: Path, *extra_args: str, ini_extra: str = ""
):
    """Run a real pytest session in ``tmp_path`` with the repo-root journaling hooks loaded."""
    # Pin rootdir to tmp_path so node IDs stay relative to it, the way CI's --config-file does.
    (tmp_path / "pytest.ini").write_text(f"[pytest]\n{ini_extra}", encoding="utf-8")
    # Load the repo-root journaling hooks as a plugin without making tmp_path a repo checkout.
    (tmp_path / "journal_plugin.py").write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(_REPO_ROOT)!r})
            from conftest import (
                pytest_collection_finish,
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
        [sys.executable, "-m", "pytest", "-p", "journal_plugin", f"--junitxml={junit_file}", target, *extra_args],
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


# -- identity stability against real pytest ---------------------------------------------------


def test_rebuilt_ids_match_the_ids_pytest_writes_on_a_clean_run(tmp_path):
    """The rebuilt IDs must equal pytest's own, or a crashed run forks a test's history.

    Runs a real pytest session with the repo-root journaling hooks loaded, then checks that
    ``junit_names`` applied to the journaled node IDs reproduces the ``classname``/``name``
    pairs pytest itself wrote to the JUnit XML.
    """
    _write_test_module(
        tmp_path,
        """
        import pytest

        @pytest.mark.parametrize("value", ["a-b", "c"])
        def test_y(value):
            pass

        class TestC:
            def test_z(self):
                pass
        """,
    )
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    result = _run_pytest(tmp_path, _TARGET, journal_file, junit_file)
    assert result.returncode == 0, result.stdout + result.stderr

    root = ElementTree.fromstring(junit_file.read_text(encoding="utf-8"))
    pytest_ids = {(case.get("classname"), case.get("name")) for case in root.iter("testcase")}
    journal = read_journal(str(journal_file))
    rebuilt_ids = {junit_names(node_id) for node_id in journal.collected}

    assert rebuilt_ids == pytest_ids
    assert len(pytest_ids) == 3


def test_deselected_tests_are_not_journaled_as_collected(tmp_path):
    """Regression test for a rebuilt report claiming tests that this pass never selected.

    ``tools/conftest.py`` splits a run into passes selected by marker and device, so journaling
    from ``pytest_collection_modifyitems`` — which runs before pytest's own ``trylast``
    deselection hook — would record the other passes' tests too. A crash would then rebuild them
    as "not run" skips, inflating the counts and duplicating node IDs the sibling pass reported.
    """
    _write_test_module(
        tmp_path,
        """
        def test_keep():
            pass

        def test_drop():
            pass
        """,
    )
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    result = _run_pytest(tmp_path, _TARGET, journal_file, junit_file, "-k", "keep")
    assert result.returncode == 0, result.stdout + result.stderr

    journal = read_journal(str(journal_file))
    assert journal.collected == [f"{_TARGET}::test_keep"]


def test_rebuilt_result_tags_match_the_tags_pytest_writes_on_a_clean_run(tmp_path):
    """A rebuilt case must land in the same JUnit bucket pytest would have put it in.

    pytest reports a failing ``call`` as ``<failure>`` but a failing ``setup``/``teardown`` as
    ``<error>``. Reading that split off the journaled phase is what keeps a crashed run's counts
    comparable to a clean one's instead of recounting broken fixtures as test failures.
    """
    _write_test_module(
        tmp_path,
        """
        import pytest

        @pytest.fixture
        def broken_setup():
            raise RuntimeError("fixture boom")

        @pytest.fixture
        def broken_teardown():
            yield
            raise RuntimeError("teardown boom")

        def test_setup_error(broken_setup):
            pass

        def test_teardown_error(broken_teardown):
            pass

        def test_call_failure():
            assert 1 == 2

        def test_ok():
            pass
        """,
    )
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    _run_pytest(tmp_path, _TARGET, journal_file, junit_file)

    root = ElementTree.fromstring(junit_file.read_text(encoding="utf-8"))
    pytest_tags = {f"{case.get('classname')}::{case.get('name')}": _result_tag(case) for case in root.iter("testcase")}
    report, _, _ = create_crash_report(str(journal_file), "test_x", "SIGKILL", "diagnostics")
    rebuilt_tags = {node_id: _result_tag(case) for node_id, case in _cases(report).items()}

    assert pytest_tags == {
        "source.pkg.test.test_x::test_setup_error": "error",
        "source.pkg.test.test_x::test_teardown_error": "error",
        "source.pkg.test.test_x::test_call_failure": "failure",
        "source.pkg.test.test_x::test_ok": "passed",
    }
    # The session case only exists on the rebuilt side: the clean run had no crash to report.
    rebuilt_tags.pop(f"source.pkg.test.test_x::{SESSION_CRASH_CASE}")
    assert rebuilt_tags == pytest_tags


def test_a_test_failing_twice_rebuilds_as_the_two_cases_pytest_writes(tmp_path):
    """A test that fails and then breaks its own teardown must keep both results.

    pytest opens a second ``<testcase>`` for the teardown error rather than stacking it beside
    the call failure ("in order to follow junit schema"), and the results uploader reads only a
    case's first ``<failure>``/``<error>`` child. Folding the two phases into one verdict - or
    into one case - therefore drops the teardown error on the floor.
    """
    _write_test_module(
        tmp_path,
        """
        import pytest

        @pytest.fixture
        def broken_teardown():
            yield
            raise RuntimeError("teardown boom")

        def test_double(broken_teardown):
            assert 1 == 2
        """,
    )
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    _run_pytest(tmp_path, _TARGET, journal_file, junit_file)

    root = ElementTree.fromstring(junit_file.read_text(encoding="utf-8"))
    pytest_tags = [_result_tags(case) for case in root.iter("testcase")]
    assert pytest_tags == [["failure"], ["error"]]

    report, _, _ = create_crash_report(str(journal_file), "test_x", "SIGKILL", "diagnostics")
    rebuilt = _cases_for(report, "source.pkg.test.test_x::test_double")
    assert [_result_tags(case) for case in rebuilt] == pytest_tags
    assert "assert 1 == 2" in (rebuilt[0].find("failure").text or "")
    assert "teardown boom" in (rebuilt[1].find("error").text or "")


# -- artificial crashes in a real pytest run --------------------------------------------------
#
# Each test below kills a real pytest subprocess at a different point in the session and checks
# what the journal can still reconstruct. ``assert not junit_file.exists()`` is the precondition
# that makes each one meaningful: pytest wrote no report, so everything asserted afterwards came
# out of the journal rather than out of pytest.


def test_process_death_mid_test_still_reports_verdicts_and_names_the_culprit(tmp_path):
    """Regression test for crashes surfacing only a single synthetic entry.

    ``os._exit`` reproduces what a Kit shutdown crash or an OOM kill does to a run: the process
    dies before ``pytest_sessionfinish``, so pytest never writes its JUnit XML and every verdict
    it had already reported is lost. The journal has to carry them out instead.
    """
    _write_test_module(
        tmp_path,
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
        """,
    )
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    _run_pytest(tmp_path, _TARGET, journal_file, junit_file)

    # Precondition: this is exactly the situation where the old code had nothing to work with.
    assert not junit_file.exists()

    report, counters, culprit = create_crash_report(str(journal_file), "test_x", "SIGKILL", "diagnostics")

    assert culprit == f"{_TARGET}::test_c"
    cases = _cases(report)
    assert _result_tag(cases["source.pkg.test.test_x::test_a"]) == "passed"
    assert _result_tag(cases["source.pkg.test.test_x::test_b"]) == "failure"
    assert _result_tag(cases["source.pkg.test.test_x::test_c"]) == "error"
    assert _result_tag(cases["source.pkg.test.test_x::test_d"]) == "skipped"
    assert counters["tests"] == 4


def test_hard_kill_mid_test_preserves_earlier_verdicts_and_markers(tmp_path):
    """A signal kill — a CI timeout or the OOM reaper — must cost no verdict already reported.

    ``SIGTERM`` is harsher than ``os._exit``: the process is torn down asynchronously by the OS,
    with no chance to run ``finally`` blocks, ``atexit`` handlers, or a buffer flush. Only the
    per-record flush in the journal can survive it. The markers have to come back as well, or the
    recovered results land under a different ``test_type`` than the same tests on a clean run.
    """
    _write_test_module(
        tmp_path,
        """
        import os
        import signal

        import pytest

        pytestmark = pytest.mark.unit

        def test_a():
            pass

        def test_b():
            os.kill(os.getpid(), signal.SIGTERM)
        """,
    )
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    _run_pytest(tmp_path, _TARGET, journal_file, junit_file, ini_extra="markers =\n    unit: unit test\n")

    assert not junit_file.exists()

    report, counters, culprit = create_crash_report(str(journal_file), "test_x", "SIGTERM", "diagnostics")

    assert culprit == f"{_TARGET}::test_b"
    cases = _cases(report)
    assert _result_tag(cases["source.pkg.test.test_x::test_a"]) == "passed"
    assert _result_tag(cases["source.pkg.test.test_x::test_b"]) == "error"
    markers = [(p.get("name"), p.get("value")) for p in cases["source.pkg.test.test_x::test_a"].iter("property")]
    assert markers == [("markers", "unit")]
    assert counters["errors"] == 1


def test_crash_during_teardown_keeps_the_failure_the_test_already_reported(tmp_path):
    """A crash in teardown must not erase the verdict the call phase already produced.

    pytest resolves — and the journal records — the call phase before teardown runs, so the
    assertion failure is already on disk when the teardown crash kills the process. Reporting only
    the crash would bury a real test failure under an infrastructure error.
    """
    _write_test_module(
        tmp_path,
        """
        import os

        import pytest

        @pytest.fixture
        def crash_on_teardown():
            yield
            os._exit(1)

        def test_a():
            pass

        def test_b(crash_on_teardown):
            assert 1 == 2
        """,
    )
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    _run_pytest(tmp_path, _TARGET, journal_file, junit_file)

    assert not junit_file.exists()

    report, counters, culprit = create_crash_report(str(journal_file), "test_x", "SIGSEGV", "diagnostics")

    assert culprit == f"{_TARGET}::test_b"
    assert _result_tag(_cases_for(report, "source.pkg.test.test_x::test_a")[0]) == "passed"
    # The assertion failure and the crash that followed it are both reported, in a case each -
    # pytest splits a call failure from a teardown error the same way, and the results uploader
    # reads only a case's first <failure>/<error>, so stacking them would drop the crash.
    failure_case, error_case = _cases_for(report, "source.pkg.test.test_x::test_b")
    assert _result_tags(failure_case) == ["failure"]
    assert _result_tags(error_case) == ["error"]
    assert "assert 1 == 2" in (failure_case.find("failure").text or "")
    assert error_case.find("error").get("message") == "SIGSEGV"
    assert counters["failures"] == 1
    assert counters["errors"] == 1


def test_crash_at_session_shutdown_is_not_blamed_on_a_test(tmp_path):
    """A crash after the last teardown belongs to the session, not to the test that ran last.

    This is the Kit-shutdown shape: every test passed and completed teardown, then the process
    died on the way out. Charging that to the last test would mark a passing test as failed, and
    it would do so for a different test every time the file order changed.
    """
    _write_test_module(
        tmp_path,
        """
        def test_a():
            pass

        def test_b():
            pass
        """,
    )
    # tryfirst runs this ahead of the junitxml plugin's own sessionfinish, so the run dies with no
    # report written - what a crash during simulator shutdown does to a real CI job.
    _write_conftest(
        tmp_path,
        """
        import os

        import pytest

        @pytest.hookimpl(tryfirst=True)
        def pytest_sessionfinish(session, exitstatus):
            os._exit(1)
        """,
    )
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    _run_pytest(tmp_path, _TARGET, journal_file, junit_file)

    assert not junit_file.exists()

    report, counters, culprit = create_crash_report(str(journal_file), "test_x", "SIGSEGV", "diagnostics")

    assert culprit is None
    cases = _cases(report)
    assert _result_tag(cases["source.pkg.test.test_x::test_a"]) == "passed"
    assert _result_tag(cases["source.pkg.test.test_x::test_b"]) == "passed"
    assert _result_tag(cases[f"source.pkg.test.test_x::{SESSION_CRASH_CASE}"]) == "error"
    assert counters["errors"] == 1
    assert counters["failures"] == 0


def test_crash_during_a_flaky_retry_is_blamed_on_the_retried_test(tmp_path):
    """A crash on a rerun belongs to the test being rerun, not to session shutdown.

    Driven by the real ``flaky`` plugin because the journal shape here is not something to guess
    at: flaky reruns the test in-process, so the node logs a start and a finish for the first
    attempt and then a second start that never finishes. It also reruns with pytest's report
    logging suppressed, so the crashing attempt leaves no outcome record at all — which is why
    mistaking it for a finished node reports the test that killed the run as "not run".
    """
    pytest.importorskip("flaky", reason="the rerun this test needs is driven by the flaky plugin")
    _write_test_module(
        tmp_path,
        """
        import os

        import pytest

        _attempts = []

        @pytest.mark.flaky(max_runs=2)
        def test_retried():
            _attempts.append(1)
            if len(_attempts) == 1:
                raise AssertionError("first attempt fails, which is what triggers the rerun")
            os._exit(1)

        def test_never_reached():
            pass
        """,
    )
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    # -p loads flaky explicitly: the harness disables plugin autoload to keep the JUnit IDs clean.
    _run_pytest(tmp_path, _TARGET, journal_file, junit_file, "-p", "flaky")

    assert not junit_file.exists()

    journal = read_journal(str(journal_file))
    # Two attempts started, only the first one finished.
    assert journal.started == [f"{_TARGET}::test_retried"] * 2
    assert journal.finished == [f"{_TARGET}::test_retried"]

    report, counters, culprit = create_crash_report(str(journal_file), "test_x", "SIGSEGV", "diagnostics")

    assert culprit == f"{_TARGET}::test_retried"
    cases = _cases(report)
    assert _result_tag(cases["source.pkg.test.test_x::test_retried"]) == "error"
    assert _result_tag(cases["source.pkg.test.test_x::test_never_reached"]) == "skipped"
    # A test was in flight, so the crash is charged to it rather than to the session.
    assert f"source.pkg.test.test_x::{SESSION_CRASH_CASE}" not in cases
    assert counters == {"errors": 1, "failures": 0, "skipped": 1, "tests": 2, "time_elapsed": 0.0}


def test_kill_before_any_test_runs_reports_every_collected_test_as_not_run(tmp_path):
    """A run killed between collection and the first test still has to account for its tests.

    This is the startup-hang shape ``tools/conftest.py`` guards against: the process is killed
    before a single verdict exists. The collected tests must come back as "not run" rather than
    disappear from the uploaded results, which would silently shrink the suite.
    """
    _write_test_module(
        tmp_path,
        """
        def test_a():
            pass

        def test_b():
            pass
        """,
    )
    # Collection has finished journaling by the time the run loop starts, so this kills the
    # session in the window where the journal knows every test but none has a verdict.
    _write_conftest(
        tmp_path,
        """
        import os
        import signal

        import pytest

        @pytest.hookimpl(tryfirst=True)
        def pytest_runtestloop(session):
            os.kill(os.getpid(), signal.SIGTERM)
        """,
    )
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    _run_pytest(tmp_path, _TARGET, journal_file, junit_file)

    assert not junit_file.exists()

    report, counters, culprit = create_crash_report(str(journal_file), "test_x", "SIGTERM", "diagnostics")

    assert culprit is None
    cases = _cases(report)
    assert _result_tag(cases["source.pkg.test.test_x::test_a"]) == "skipped"
    assert _result_tag(cases["source.pkg.test.test_x::test_b"]) == "skipped"
    assert _result_tag(cases[f"source.pkg.test.test_x::{SESSION_CRASH_CASE}"]) == "error"
    assert counters == {"errors": 1, "failures": 0, "skipped": 2, "tests": 3, "time_elapsed": 0.0}
