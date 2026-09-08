# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``tools/crash_journal.py``."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import pytest
from crash_journal import SESSION_CRASH_CASE, create_crash_report, junit_names, read_journal

_FILE = "source/pkg/test/test_x.py"
"""Node path the journals refer to, and the module the subprocess runs collect from their rootdir."""

_CLASS = "source.pkg.test.test_x"
"""JUnit ``classname`` pytest derives from :data:`_FILE`."""

_REPO_ROOT = Path(__file__).resolve().parent.parent

_HAS_FLAKY = importlib.util.find_spec("flaky") is not None
"""Whether the ``flaky`` plugin is installed to drive the in-process rerun one crash case needs."""


def _write_journal(tmp_path: Path, records: list[dict], trailing: str = "") -> str:
    """Write ``records`` as JSONL and return the path, optionally with a partial trailing line."""
    path = tmp_path / "journal.jsonl"
    body = "".join(json.dumps(record) + "\n" for record in records)
    path.write_text(body + trailing, encoding="utf-8")
    return str(path)


def _case_elements(source) -> list[ElementTree.Element]:
    """Return the ``<testcase>`` elements of a rebuilt report or a JUnit file, in document order."""
    xml = source.read_text(encoding="utf-8") if isinstance(source, Path) else source.tostring()
    return list(ElementTree.fromstring(xml).iter("testcase"))


def _result_tags(case: ElementTree.Element) -> list[str]:
    """Return the tags of a case's result children, in document order; empty when it passed."""
    return [child.tag for child in case if child.tag in ("failure", "error", "skipped")]


def _tags_by_case(source) -> dict[str, list[list[str]]]:
    """Map each ``classname::name`` to the result tags of every case carrying that ID.

    A node emits one case per result, so the value is a list of per-case tag lists: a passing test
    is ``[[]]`` and one that fails its call and then breaks its own teardown is
    ``[["failure"], ["error"]]``.
    """
    tags: dict[str, list[list[str]]] = {}
    for case in _case_elements(source):
        tags.setdefault(f"{case.get('classname')}::{case.get('name')}", []).append(_result_tags(case))
    return tags


def _cases_for(report, case_id: str) -> list[ElementTree.Element]:
    """Return every case whose ``classname::name`` equals ``case_id``, in document order."""
    return [case for case in _case_elements(report) if f"{case.get('classname')}::{case.get('name')}" == case_id]


def _markers_of(case: ElementTree.Element) -> list[tuple[str | None, str | None]]:
    """Return a case's JUnit properties as ``(name, value)`` pairs."""
    return [(prop.get("name"), prop.get("value")) for prop in case.iter("property")]


# -- junit_names ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_id, expected",
    [
        pytest.param(f"{_FILE}::test_y[a-b]", (_CLASS, "test_y[a-b]"), id="parametrized"),
        pytest.param(f"{_FILE}::TestC::test_y", (f"{_CLASS}.TestC", "test_y"), id="class_method"),
        pytest.param("source\\pkg\\test_x.py::test_y", ("source.pkg.test_x", "test_y"), id="windows_separators"),
        pytest.param(_FILE, (_CLASS, "test_execution"), id="no_node_part"),
    ],
)
def test_junit_names_reproduces_pytests_own_naming(node_id, expected):
    assert junit_names(node_id) == expected


# -- read_journal -----------------------------------------------------------------------------


@pytest.mark.parametrize("records", [None, []], ids=["missing_file", "no_records"])
def test_an_unusable_journal_yields_no_journal_and_no_report(tmp_path, records):
    journal_file = str(tmp_path / "absent.jsonl") if records is None else _write_journal(tmp_path, records)
    assert read_journal(journal_file) is None
    # The caller falls back to a synthetic entry, so no report at all - not an empty one.
    assert create_crash_report(journal_file, "test_x", "boom", "details") is None


def test_read_journal_skips_partial_trailing_line(tmp_path):
    journal_file = _write_journal(
        tmp_path,
        [{"event": "collected", "node_ids": [f"{_FILE}::test_a"]}, {"event": "start", "node_id": f"{_FILE}::test_a"}],
        trailing='{"event": "res',
    )
    journal = read_journal(journal_file)
    assert journal.collected == [f"{_FILE}::test_a"]
    assert journal.started == [f"{_FILE}::test_a"]


def test_read_journal_folds_repeats_within_a_phase_and_keeps_phases_apart(tmp_path):
    # The flaky plugin reruns a test in-process, so one node ID can emit several result records for
    # the same phase; those fold to the worst outcome, the text that established it, and the summed
    # duration. Distinct phases stay apart because pytest reports a failing setup/teardown as an
    # error but a failing call as a failure, so folding them would relabel a broken fixture.
    node_id = f"{_FILE}::test_a"
    journal_file = _write_journal(
        tmp_path,
        [
            {"event": "collected", "node_ids": [node_id]},
            {"event": "result", "node_id": node_id, "when": "setup", "outcome": "skipped", "longrepr": "needs a GPU"},
            # No ``when``: records predating that field are treated as ``call``.
            {"event": "result", "node_id": node_id, "outcome": "passed", "duration": 1.0},
            {"event": "result", "node_id": node_id, "outcome": "failed", "duration": 2.0, "longrepr": "assert 1 == 2"},
            {"event": "result", "node_id": node_id, "when": "teardown", "outcome": "failed", "longrepr": "boom"},
        ],
    )
    assert read_journal(journal_file).results[node_id] == {
        "setup": {"outcome": "skipped", "duration": 0.0, "longrepr": "needs a GPU"},
        "call": {"outcome": "failed", "duration": 3.0, "longrepr": "assert 1 == 2"},
        "teardown": {"outcome": "failed", "duration": 0.0, "longrepr": "boom"},
    }


_RETRIED = [
    {"event": "collected", "node_ids": [f"{_FILE}::test_a"]},
    {"event": "start", "node_id": f"{_FILE}::test_a"},
    {"event": "finish", "node_id": f"{_FILE}::test_a"},
    {"event": "start", "node_id": f"{_FILE}::test_a"},
]
"""A node the flaky plugin reran in-process, with the second attempt still in flight."""


@pytest.mark.parametrize(
    "records, expected_culprit",
    [
        pytest.param(_RETRIED, f"{_FILE}::test_a", id="retry_never_finished"),
        pytest.param(
            _RETRIED + [{"event": "finish", "node_id": f"{_FILE}::test_a"}], None, id="every_attempt_finished"
        ),
    ],
)
def test_culprit_matches_starts_against_finishes_one_for_one(tmp_path, records, expected_culprit):
    # Membership testing would let the first attempt's finish mask a crash during the second,
    # blaming the session for a test that was still running - and reporting that test as never run.
    assert read_journal(_write_journal(tmp_path, records)).culprit == expected_culprit


# -- create_crash_report ----------------------------------------------------------------------


def test_create_crash_report_preserves_verdicts_and_blames_the_in_flight_test(tmp_path):
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
    assert _tags_by_case(report) == {
        f"{_CLASS}::test_a": [[]],
        f"{_CLASS}::test_b": [["failure"]],
        f"{_CLASS}::test_c": [["error"]],
        # Collected but never reached: recorded as skipped so it does not vanish from the results.
        f"{_CLASS}::test_d": [["skipped"]],
    }
    # The markers come back too, or the recovered results land under a different ``test_type`` than
    # the same tests on a clean run.
    assert _markers_of(_cases_for(report, f"{_CLASS}::test_a")[0]) == [("markers", "flaky,unit")]
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
    failure_case, error_case = _cases_for(report, f"{_CLASS}::test_a")
    assert _result_tags(failure_case) == ["failure"]
    assert _result_tags(error_case) == ["error"]
    assert "AssertionError" in (failure_case.find("failure").text or "")
    assert error_case.find("error").get("message") == "SIGSEGV"
    # Both cases are counted, so the orchestrator's tests - failures - errors - skipped summary
    # stays at zero passed instead of going negative.
    assert counters == {"errors": 1, "failures": 1, "skipped": 0, "tests": 2, "time_elapsed": 2.0}


@pytest.mark.parametrize(
    "records, expected_tags",
    [
        pytest.param(
            [
                {"event": "start", "node_id": f"{_FILE}::test_a"},
                {"event": "result", "node_id": f"{_FILE}::test_a", "when": "call", "outcome": "passed"},
                {"event": "finish", "node_id": f"{_FILE}::test_a"},
            ],
            [[]],
            id="crash_after_the_last_teardown",
        ),
        pytest.param([], [["skipped"]], id="crash_during_collection"),
    ],
)
def test_create_crash_report_blames_the_session_when_nothing_was_in_flight(tmp_path, records, expected_tags):
    # Kit crashing during shutdown must not be charged to the last test, which passed - and it
    # would be a different test every time the file order changed.
    journal_file = _write_journal(tmp_path, [{"event": "collected", "node_ids": [f"{_FILE}::test_a"]}, *records])

    report, counters, culprit = create_crash_report(journal_file, "test_x", "SIGSEGV", "diagnostics")

    assert culprit is None
    assert _tags_by_case(report) == {
        f"{_CLASS}::test_a": expected_tags,
        f"{_CLASS}::{SESSION_CRASH_CASE}": [["error"]],
    }
    assert counters["errors"] == 1


# -- real pytest subprocess harness -----------------------------------------------------------


def _write_test_module(tmp_path: Path, body: str) -> None:
    """Write ``body`` as the throwaway test module a subprocess pytest run will collect."""
    module = tmp_path / _FILE
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(textwrap.dedent(body), encoding="utf-8")


def _write_conftest(tmp_path: Path, body: str) -> None:
    """Write ``body`` as the rootdir ``conftest.py`` of a subprocess pytest run.

    Used to inject a crash from a pytest hook, which is the only way to die at a point the test
    functions themselves cannot reach — session shutdown, or before the run loop starts.
    """
    (tmp_path / "conftest.py").write_text(textwrap.dedent(body), encoding="utf-8")


def _run_pytest(tmp_path: Path, journal_file: Path, junit_file: Path, *extra_args: str, ini_extra: str = ""):
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
        [sys.executable, "-m", "pytest", "-p", "journal_plugin", f"--junitxml={junit_file}", _FILE, *extra_args],
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


# -- parity with real pytest ------------------------------------------------------------------


def test_a_clean_run_rebuilds_to_the_cases_pytest_itself_wrote(tmp_path):
    """The rebuilt cases must equal pytest's own, or a crashed run forks every test's history.

    Runs a real pytest session with the repo-root journaling hooks loaded and compares the rebuilt
    report against the JUnit XML pytest wrote from the same session. Both the identity of a case
    (``classname``/``name``, which is what keeps a test on one history across a crash) and its
    bucket are checked: pytest reports a failing ``call`` as ``<failure>`` but a failing
    ``setup``/``teardown`` as ``<error>``, and it opens a second ``<testcase>`` when a test fails
    both — "in order to follow junit schema". The results uploader reads only a case's first
    ``<failure>``/``<error>`` child, so collapsing that pair would drop the teardown error.
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

        @pytest.mark.parametrize("value", ["a-b", "c"])
        def test_y(value):
            pass

        class TestC:
            def test_z(self):
                pass

        def test_setup_error(broken_setup):
            pass

        def test_teardown_error(broken_teardown):
            pass

        def test_call_failure():
            assert 1 == 2

        def test_double(broken_teardown):
            assert 1 == 2

        def test_ok():
            pass
        """,
    )
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    _run_pytest(tmp_path, journal_file, junit_file)

    assert _tags_by_case(junit_file) == {
        f"{_CLASS}::test_y[a-b]": [[]],
        f"{_CLASS}::test_y[c]": [[]],
        f"{_CLASS}.TestC::test_z": [[]],
        f"{_CLASS}::test_setup_error": [["error"]],
        f"{_CLASS}::test_teardown_error": [["error"]],
        f"{_CLASS}::test_call_failure": [["failure"]],
        f"{_CLASS}::test_double": [["failure"], ["error"]],
        f"{_CLASS}::test_ok": [[]],
    }

    report, _, _ = create_crash_report(str(journal_file), "test_x", "SIGKILL", "diagnostics")
    rebuilt = _tags_by_case(report)
    # The session case only exists on the rebuilt side: the clean run had no crash to report.
    assert rebuilt.pop(f"{_CLASS}::{SESSION_CRASH_CASE}") == [["error"]]
    assert rebuilt == _tags_by_case(junit_file)

    double = _cases_for(report, f"{_CLASS}::test_double")
    assert "assert 1 == 2" in (double[0].find("failure").text or "")
    assert "teardown boom" in (double[1].find("error").text or "")


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
    result = _run_pytest(tmp_path, journal_file, junit_file, "-k", "keep")
    assert result.returncode == 0, result.stdout + result.stderr

    assert read_journal(str(journal_file)).collected == [f"{_FILE}::test_keep"]


# -- artificial crashes in a real pytest run --------------------------------------------------


@dataclass(frozen=True)
class _CrashCase:
    """One way of killing a real pytest session, and what the journal must reconstruct from it.

    Attributes:
        module: Source of the throwaway test module the run collects.
        culprit: Node ID ``create_crash_report`` must blame, or ``None`` for a session crash.
        tags: Expected ``{case ID: [result tags per case]}`` of the rebuilt report.
        counters: Expected report totals; compared key by key, since the durations of a real run
            are not reproducible.
        conftest: Source of a rootdir ``conftest.py``, for crashes no test function can reach.
        extra_args: Extra pytest arguments.
        ini_extra: Extra ``pytest.ini`` body, used to register the markers a module applies.
        markers: Expected ``markers`` property value per case ID, checked where the module sets one.
    """

    # The sources are kept out of the repr so a failing case reports its expectations, not the
    # whole test module it wrote.
    module: str = field(repr=False)
    culprit: str | None
    tags: dict[str, list[list[str]]]
    counters: dict[str, int]
    conftest: str = field(default="", repr=False)
    extra_args: tuple[str, ...] = ()
    ini_extra: str = ""
    markers: dict[str, str] = field(default_factory=dict)


_CRASH_CASES = [
    pytest.param(
        # ``os._exit`` reproduces what a Kit shutdown crash or an OOM kill does to a run: the
        # process dies before ``pytest_sessionfinish``, so pytest never writes its JUnit XML and
        # every verdict it had already reported is lost. The journal has to carry them out instead.
        _CrashCase(
            module="""
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
            culprit=f"{_FILE}::test_c",
            tags={
                f"{_CLASS}::test_a": [[]],
                f"{_CLASS}::test_b": [["failure"]],
                f"{_CLASS}::test_c": [["error"]],
                f"{_CLASS}::test_d": [["skipped"]],
            },
            counters={"errors": 1, "failures": 1, "skipped": 1, "tests": 4},
        ),
        id="process_death_mid_test",
    ),
    pytest.param(
        # ``SIGTERM`` - a CI timeout or the OOM reaper - is harsher than ``os._exit``: the process
        # is torn down asynchronously, with no chance to run ``finally`` blocks, ``atexit``
        # handlers, or a buffer flush. Only the journal's per-record flush can survive it. The
        # markers have to come back as well, or the recovered results land under a different
        # ``test_type`` than the same tests on a clean run.
        _CrashCase(
            module="""
            import os
            import signal

            import pytest

            pytestmark = pytest.mark.unit

            def test_a():
                pass

            def test_b():
                os.kill(os.getpid(), signal.SIGTERM)
            """,
            culprit=f"{_FILE}::test_b",
            tags={f"{_CLASS}::test_a": [[]], f"{_CLASS}::test_b": [["error"]]},
            counters={"errors": 1, "failures": 0, "skipped": 0, "tests": 2},
            ini_extra="markers =\n    unit: unit test\n",
            markers={f"{_CLASS}::test_a": "unit", f"{_CLASS}::test_b": "unit"},
        ),
        id="hard_kill_mid_test",
    ),
    pytest.param(
        # pytest resolves - and the journal records - the call phase before teardown runs, so the
        # assertion failure is already on disk when the teardown crash kills the process. Reporting
        # only the crash would bury a real test failure under an infrastructure error.
        _CrashCase(
            module="""
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
            culprit=f"{_FILE}::test_b",
            tags={f"{_CLASS}::test_a": [[]], f"{_CLASS}::test_b": [["failure"], ["error"]]},
            counters={"errors": 1, "failures": 1, "skipped": 0, "tests": 3},
        ),
        id="crash_during_teardown",
    ),
    pytest.param(
        # The Kit-shutdown shape: every test passed and completed teardown, then the process died
        # on the way out. ``tryfirst`` runs the crash ahead of the junitxml plugin's own
        # ``pytest_sessionfinish``, so the run dies with no report written.
        _CrashCase(
            module="""
            def test_a():
                pass

            def test_b():
                pass
            """,
            conftest="""
            import os

            import pytest

            @pytest.hookimpl(tryfirst=True)
            def pytest_sessionfinish(session, exitstatus):
                os._exit(1)
            """,
            culprit=None,
            tags={
                f"{_CLASS}::test_a": [[]],
                f"{_CLASS}::test_b": [[]],
                f"{_CLASS}::{SESSION_CRASH_CASE}": [["error"]],
            },
            counters={"errors": 1, "failures": 0, "skipped": 0, "tests": 3},
        ),
        id="crash_at_session_shutdown",
    ),
    pytest.param(
        # Driven by the real ``flaky`` plugin because the journal shape here is not something to
        # guess at: flaky reruns the test in-process, so the node logs a start and a finish for the
        # first attempt and then a second start that never finishes. It also reruns with pytest's
        # report logging suppressed, so the crashing attempt leaves no outcome record at all -
        # which is why mistaking it for a finished node reports the test that killed the run as
        # "not run", and hands the blame to session shutdown.
        _CrashCase(
            module="""
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
            culprit=f"{_FILE}::test_retried",
            tags={f"{_CLASS}::test_retried": [["error"]], f"{_CLASS}::test_never_reached": [["skipped"]]},
            counters={"errors": 1, "failures": 0, "skipped": 1, "tests": 2},
            # -p loads flaky explicitly: the harness disables plugin autoload to keep IDs clean.
            extra_args=("-p", "flaky"),
        ),
        id="crash_during_flaky_retry",
        marks=pytest.mark.skipif(not _HAS_FLAKY, reason="the rerun this case needs is driven by the flaky plugin"),
    ),
    pytest.param(
        # The startup-hang shape ``tools/conftest.py`` guards against: collection has finished
        # journaling by the time the run loop starts, so this kills the session in the window where
        # the journal knows every test but none has a verdict. They must come back as "not run"
        # rather than disappear from the uploaded results, which would silently shrink the suite.
        _CrashCase(
            module="""
            def test_a():
                pass

            def test_b():
                pass
            """,
            conftest="""
            import os
            import signal

            import pytest

            @pytest.hookimpl(tryfirst=True)
            def pytest_runtestloop(session):
                os.kill(os.getpid(), signal.SIGTERM)
            """,
            culprit=None,
            tags={
                f"{_CLASS}::test_a": [["skipped"]],
                f"{_CLASS}::test_b": [["skipped"]],
                f"{_CLASS}::{SESSION_CRASH_CASE}": [["error"]],
            },
            counters={"errors": 1, "failures": 0, "skipped": 2, "tests": 3},
        ),
        id="kill_before_any_test_runs",
    ),
]


@pytest.mark.parametrize("case", _CRASH_CASES)
def test_a_crashed_run_is_rebuilt_from_its_journal(tmp_path, case: _CrashCase):
    """Regression test for crashes surfacing only a single synthetic ``test_execution`` entry.

    Each case kills a real pytest session at a different point and checks what the journal can
    still reconstruct. ``assert not junit_file.exists()`` is the precondition that makes them
    meaningful: pytest wrote no report, so everything asserted afterwards came out of the journal.
    """
    _write_test_module(tmp_path, case.module)
    if case.conftest:
        _write_conftest(tmp_path, case.conftest)
    journal_file = tmp_path / "journal.jsonl"
    junit_file = tmp_path / "report.xml"
    _run_pytest(tmp_path, journal_file, junit_file, *case.extra_args, ini_extra=case.ini_extra)

    assert not junit_file.exists()

    report, counters, culprit = create_crash_report(str(journal_file), "test_x", "SIGSEGV", "diagnostics")

    assert culprit == case.culprit
    assert _tags_by_case(report) == case.tags
    assert {name: counters[name] for name in case.counters} == case.counters
    for case_id, markers in case.markers.items():
        assert [_markers_of(element) for element in _cases_for(report, case_id)] == [[("markers", markers)]]
