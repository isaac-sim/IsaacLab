# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-file pytest result handling in the test orchestrator."""

from __future__ import annotations

import importlib.util
import json
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import ModuleType

import pytest


def _load_orchestrator_module() -> ModuleType:
    """Load ``tools/conftest.py`` without registering it as a pytest plugin."""
    module_path = Path(__file__).resolve().parents[4] / "tools" / "conftest.py"
    module_name = "isaaclab_test_orchestrator"
    tools_dir = str(module_path.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_empty_junit_report(report_file: str) -> None:
    """Write a valid JUnit report containing no test cases."""
    path = Path(report_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite tests="0"/></testsuites>',
        encoding="utf-8",
    )


def _write_partial_junit_report(report_file: str) -> None:
    """Write a valid JUnit report containing passing and skipped test cases."""
    path = Path(report_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            '<?xml version="1.0" encoding="utf-8"?><testsuites>'
            '<testsuite tests="2" skipped="1"><testcase classname="test_sample" name="test_present"/>'
            '<testcase classname="test_sample" name="test_skipped">'
            '<skipped message="Known unsupported case."/></testcase>'
            "</testsuite></testsuites>"
        ),
        encoding="utf-8",
    )


def _write_module_skipped_junit_report(report_file: str) -> None:
    """Write the JUnit shape produced by a module-level ``pytest.importorskip``."""
    path = Path(report_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            '<?xml version="1.0" encoding="utf-8"?><testsuites>'
            '<testsuite tests="1" skipped="1"><testcase name="">'
            '<skipped message="collection skipped"/></testcase>'
            "</testsuite></testsuites>"
        ),
        encoding="utf-8",
    )


def _write_failing_junit_report(report_file: str, name: str) -> None:
    """Write a valid JUnit report containing one failing test case."""
    path = Path(report_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            '<?xml version="1.0" encoding="utf-8"?><testsuites>'
            f'<testsuite tests="1" failures="1"><testcase classname="test_sample" name="{name}">'
            '<failure message="assert 0 == 1">assert 0 == 1</failure></testcase>'
            "</testsuite></testsuites>"
        ),
        encoding="utf-8",
    )


def _append_journal(journal_file: str, records: list[dict]) -> None:
    """Append crash-journal records the way the repo-root ``conftest.py`` does from the test process."""
    with open(journal_file, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _journaled_test(node_id: str, outcome: str) -> list[dict]:
    """Return the journal records one test that ran to completion writes."""
    return [
        {"event": "start", "node_id": node_id},
        {"event": "result", "node_id": node_id, "when": "call", "outcome": outcome},
        {"event": "finish", "node_id": node_id},
    ]


def test_exact_node_ids_selecting_zero_tests_fail(monkeypatch, tmp_path: Path) -> None:
    """Stale exact node IDs must fail independently of the subprocess exit code."""
    orchestrator = _load_orchestrator_module()
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_present():\n    pass\n", encoding="utf-8")
    report_paths: list[Path] = []

    def _capture(*_args, report_file: str, **_kwargs):
        report_paths.append(Path(report_file))
        _write_empty_junit_report(report_file)
        return 0, b"no tests selected", b"", "", 0.1, ""

    monkeypatch.setattr(orchestrator, "capture_test_output_with_timeout", _capture)
    monkeypatch.chdir(tmp_path)
    missing_node_id = f"{test_file}::test_missing"
    context = orchestrator._PassContext(
        test_file=str(test_file),
        file_name=test_file.name,
        workspace_root=str(tmp_path),
        ci_marker=None,
        timeout=10,
        startup_deadline=1,
        env={},
        inject_shard_select=False,
        pytest_targets=[missing_node_id],
    )

    report, status, was_failure = orchestrator._run_one_pass(context, k_expr=None, suffix="")

    assert report is not None
    assert status["result"] == "FAILED"
    assert status["errors"] == 1
    assert status["tests"] == 1
    assert was_failure
    assert missing_node_id in report_paths[0].read_text(encoding="utf-8")


def test_nonzero_pytest_exit_preserves_reported_tests(monkeypatch, tmp_path: Path) -> None:
    """A synthetic exit error should be appended without discarding real test cases."""
    orchestrator = _load_orchestrator_module()
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_present():\n    pass\n", encoding="utf-8")
    report_paths: list[Path] = []

    def _capture(*_args, report_file: str, **_kwargs):
        report_paths.append(Path(report_file))
        _write_partial_junit_report(report_file)
        return 2, b"interrupted after test completion", b"", "", 0.1, ""

    monkeypatch.setattr(orchestrator, "capture_test_output_with_timeout", _capture)
    monkeypatch.chdir(tmp_path)
    context = orchestrator._PassContext(
        test_file=str(test_file),
        file_name=test_file.name,
        workspace_root=str(tmp_path),
        ci_marker=None,
        timeout=10,
        startup_deadline=1,
        env={},
        inject_shard_select=False,
        pytest_targets=[str(test_file)],
    )

    report, status, was_failure = orchestrator._run_one_pass(context, k_expr=None, suffix="")

    assert report is not None
    assert status["result"] == "FAILED"
    assert status["errors"] == 1
    assert status["skipped"] == 1
    assert status["tests"] == 3
    assert was_failure
    xml = report_paths[0].read_text(encoding="utf-8")
    assert "test_present" in xml
    assert "test_skipped" in xml
    assert "pytest exited with code 2" in xml


def test_filter_deselecting_all_tests_is_not_a_failure(monkeypatch, tmp_path: Path) -> None:
    """A global filter selecting nothing should be a visible non-failing outcome."""
    orchestrator = _load_orchestrator_module()
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_present():\n    pass\n", encoding="utf-8")

    def _capture(*_args, report_file: str, **_kwargs):
        _write_empty_junit_report(report_file)
        return 5, b"no tests ran", b"", "", 0.1, ""

    monkeypatch.setattr(orchestrator, "capture_test_output_with_timeout", _capture)
    monkeypatch.chdir(tmp_path)
    context = orchestrator._PassContext(
        test_file=str(test_file),
        file_name=test_file.name,
        workspace_root=str(tmp_path),
        ci_marker=None,
        timeout=10,
        startup_deadline=1,
        env={},
        inject_shard_select=False,
        pytest_targets=[str(test_file)],
    )

    report, status, was_failure = orchestrator._run_one_pass(context, k_expr="ovphysx", suffix="")

    assert report is not None
    assert status["result"] == "passed (no tests selected)"
    assert status["errors"] == 0
    assert status["tests"] == 0
    assert not was_failure


def test_module_importorskip_is_not_a_failure(monkeypatch, tmp_path: Path) -> None:
    """A module-level collection skip should remain non-failing without filters."""
    orchestrator = _load_orchestrator_module()
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_present():\n    pass\n", encoding="utf-8")

    def _capture(*_args, report_file: str, **_kwargs):
        _write_module_skipped_junit_report(report_file)
        return 5, b"collected 0 items / 1 skipped", b"", "", 0.1, ""

    monkeypatch.setattr(orchestrator, "capture_test_output_with_timeout", _capture)
    monkeypatch.chdir(tmp_path)
    context = orchestrator._PassContext(
        test_file=str(test_file),
        file_name=test_file.name,
        workspace_root=str(tmp_path),
        ci_marker=None,
        timeout=10,
        startup_deadline=1,
        env={},
        inject_shard_select=False,
        pytest_targets=[str(test_file)],
    )

    report, status, was_failure = orchestrator._run_one_pass(context, k_expr=None, suffix="")

    assert report is not None
    assert status["result"] == "passed (module skipped)"
    assert status["errors"] == 0
    assert status["skipped"] == 1
    assert status["tests"] == 1
    assert not was_failure


@pytest.mark.parametrize(
    ("returncode", "kill_reason", "expected_result"),
    [(-11, "", "CRASHED"), (-9, "timeout", "TIMEOUT")],
)
def test_abnormal_termination_report_quotes_bounded_renderer_log(
    monkeypatch, tmp_path: Path, caplog, returncode: int, kill_reason: str, expected_result: str
) -> None:
    """A process that dies or hangs cannot replay its own renderer log, so the runner quotes it here.

    The per-test replay in ``tools/ovrtx_log.py`` runs inside the process under test, which rules it out for
    exactly the failures the renderer log explains best: a segfault, an abort, an OOM kill, or a SIGKILL
    from this runner. Only a bounded tail is quoted, so a verbose log cannot flood the report, and it is
    quoted there only: a failure that builds a report does not also spend that quota on the job log.
    """
    orchestrator = _load_orchestrator_module()
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_present():\n    pass\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    report_paths: list[Path] = []

    def _capture(cmd, timeout, env, *, startup_deadline, report_file):
        # Render verbosely, then die without writing a report.
        Path(log_path).write_text("head-line\n" + "filler-line\n" * 8000 + "tail-line\n", encoding="utf-8")
        report_paths.append(Path(report_file))
        return returncode, b"", b"", kill_reason, 12.0, ""

    log_path = str(tmp_path / "ovrtx_renderer.log")
    monkeypatch.setattr(orchestrator.ovrtx_log, "LOG_PATH", log_path)
    monkeypatch.setattr(orchestrator, "capture_test_output_with_timeout", _capture)
    monkeypatch.setattr(orchestrator, "_capture_system_diagnostics", lambda: "")
    monkeypatch.chdir(tmp_path)
    context = orchestrator._PassContext(
        test_file=str(test_file),
        file_name=test_file.name,
        workspace_root=str(tmp_path),
        ci_marker=None,
        timeout=10,
        startup_deadline=1,
        env={},
        inject_shard_select=False,
        pytest_targets=[str(test_file)],
    )

    with caplog.at_level("INFO"):
        report, status, was_failure = orchestrator._run_one_pass(context, k_expr=None, suffix="")

    assert report is not None
    assert status["result"] == expected_result
    assert was_failure
    details = report_paths[0].read_text(encoding="utf-8")
    assert "OVRTX renderer log" in details
    assert f"last {orchestrator.ovrtx_log.LOG_LIMIT_BYTES} bytes follow" in details
    assert "tail-line" in details
    assert "head-line" not in details
    assert "OVRTX renderer log" not in caplog.text


def test_shutdown_hang_after_report_is_not_a_failure(monkeypatch, tmp_path: Path) -> None:
    """A process SIGKILLed for hanging in shutdown had already written its report, so its tests still count.

    The kill says nothing about the tests: they ran, passed, and replayed their own share of the renderer
    log into a report this runner only has to read back.
    """
    orchestrator = _load_orchestrator_module()
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_present():\n    pass\n", encoding="utf-8")

    def _capture(cmd, timeout, env, *, startup_deadline, report_file):
        _write_partial_junit_report(report_file)
        return -1, b"", b"", "shutdown_hang", 30.0, ""

    monkeypatch.setattr(orchestrator.ovrtx_log, "LOG_PATH", str(tmp_path / "ovrtx_renderer.log"))
    monkeypatch.setattr(orchestrator, "capture_test_output_with_timeout", _capture)
    monkeypatch.chdir(tmp_path)
    context = orchestrator._PassContext(
        test_file=str(test_file),
        file_name=test_file.name,
        workspace_root=str(tmp_path),
        ci_marker=None,
        timeout=10,
        startup_deadline=1,
        env={},
        inject_shard_select=False,
        pytest_targets=[str(test_file)],
    )

    _report, status, was_failure = orchestrator._run_one_pass(context, k_expr=None, suffix="")

    assert status["result"] == "passed (shutdown hanged)"
    assert not was_failure


def test_crash_journal_path_is_absolute(monkeypatch, tmp_path: Path) -> None:
    """The journal path handed to the test subprocess must not depend on the current directory.

    The repo-root ``conftest.py`` reopens this path on every journal write, from inside the test
    process. A relative path would resolve against whatever directory the test happens to be in,
    so a test using ``monkeypatch.chdir`` would write its verdicts to a journal under the
    temporary directory and, once teardown restored the cwd, resume writing to this one — leaving
    a test that ran and passed looking like it was never reached.
    """
    orchestrator = _load_orchestrator_module()
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_present():\n    pass\n", encoding="utf-8")
    journal_paths: list[str] = []

    def _capture(_cmd, _timeout, env, *, report_file: str, **_kwargs):
        journal_paths.append(env[orchestrator.JOURNAL_ENV_VAR])
        _write_partial_junit_report(report_file)
        return 0, b"", b"", "", 0.1, ""

    monkeypatch.setattr(orchestrator.ovrtx_log, "LOG_PATH", str(tmp_path / "ovrtx_renderer.log"))
    monkeypatch.setattr(orchestrator, "capture_test_output_with_timeout", _capture)
    monkeypatch.chdir(tmp_path)
    context = orchestrator._PassContext(
        test_file=str(test_file),
        file_name=test_file.name,
        workspace_root=str(tmp_path),
        ci_marker=None,
        timeout=10,
        startup_deadline=1,
        env={},
        inject_shard_select=False,
        pytest_targets=[str(test_file)],
    )

    orchestrator._run_one_pass(context, k_expr=None, suffix="")

    assert len(journal_paths) == 1
    journal_path = Path(journal_paths[0])
    assert journal_path.is_absolute()
    # pytest only creates the report directory in ``pytest_sessionfinish``, which a crashed run
    # never reaches, so the directory has to exist before the subprocess starts journaling.
    assert journal_path.parent.is_dir()


def test_fresh_process_retry_crash_blames_the_test_that_was_running(monkeypatch, tmp_path: Path) -> None:
    """A retry killed mid-test must be reported as a crash, blamed on the test it was running.

    Files in ``PROCESS_FAILURE_RETRIES_BY_FILE`` get another subprocess after a test failure. That
    retry can itself be SIGKILLed or segfault, and then it writes no JUnit report at all. Without a
    journal of its own the run falls back to the first attempt's verdicts: reported ``FAILED`` with
    no crash, and no record of which test took the process down.

    The two attempts disagree on purpose — ``test_alpha`` fails first and passes on the retry,
    which dies inside ``test_beta`` — so the rebuilt report can only come from the retry's journal.
    """
    orchestrator = _load_orchestrator_module()
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_alpha():\n    pass\n\n\ndef test_beta():\n    pass\n", encoding="utf-8")
    monkeypatch.setitem(orchestrator.PROCESS_FAILURE_RETRIES_BY_FILE, test_file.name, 1)
    alpha, beta = f"{test_file}::test_alpha", f"{test_file}::test_beta"
    journals_seen: list[str | None] = []
    journal_was_stale: list[bool] = []
    report_paths: list[Path] = []

    def _capture(_cmd, _timeout, env, *, report_file: str, **_kwargs):
        journal_file = env.get(orchestrator.JOURNAL_ENV_VAR)
        journals_seen.append(journal_file)
        report_paths.append(Path(report_file))
        if journal_file is None:
            return -9, b"", b"", "", 0.2, ""
        journal_was_stale.append(Path(journal_file).exists())
        _append_journal(journal_file, [{"event": "collected", "node_ids": [alpha, beta]}])
        if len(journals_seen) == 1:
            _append_journal(journal_file, _journaled_test(alpha, "failed"))
            _append_journal(journal_file, _journaled_test(beta, "passed"))
            _write_failing_junit_report(report_file, "test_alpha")
            return 1, b"1 failed", b"", "", 0.1, ""
        # The retry gets through test_alpha, then dies inside test_beta — so pytest never reaches
        # ``pytest_sessionfinish`` and writes no report.
        _append_journal(journal_file, _journaled_test(alpha, "passed"))
        _append_journal(journal_file, [{"event": "start", "node_id": beta}])
        return -9, b"", b"", "", 0.2, ""

    monkeypatch.setattr(orchestrator.ovrtx_log, "LOG_PATH", str(tmp_path / "ovrtx_renderer.log"))
    monkeypatch.setattr(orchestrator, "capture_test_output_with_timeout", _capture)
    monkeypatch.chdir(tmp_path)
    context = orchestrator._PassContext(
        test_file=str(test_file),
        file_name=test_file.name,
        workspace_root=str(tmp_path),
        ci_marker=None,
        timeout=10,
        startup_deadline=1,
        env={},
        inject_shard_select=False,
        pytest_targets=[str(test_file)],
    )

    report, status, was_failure = orchestrator._run_one_pass(context, k_expr=None, suffix="")

    assert len(journals_seen) == 2
    assert all(journals_seen), "the fresh-process retry ran without a crash journal"
    # A journal left over from the first attempt would fold its test_alpha failure into the report.
    assert journal_was_stale == [False, False]
    assert report is not None
    assert status["result"] == "CRASHED"
    assert status["errors"] == 1
    assert status["failures"] == 0
    assert status["tests"] == 2
    assert was_failure

    cases = {case.get("name"): case for case in ElementTree.parse(report_paths[1]).getroot().iter("testcase")}
    assert sorted(cases) == ["test_alpha", "test_beta"]
    assert list(cases["test_alpha"]) == [], "test_alpha passed on the retry and must be reported as passed"
    errors = cases["test_beta"].findall("error")
    assert len(errors) == 1, "the crash must be reported against the test that was running"
    assert "SIGKILL" in errors[0].get("message")


def test_result_summary_includes_fast_failure_after_thirty_slower_files():
    """The summary must print failures even when at least 30 files ran longer."""
    orchestrator = _load_orchestrator_module()
    test_files = ["fast_failure.py", *(f"slow_pass_{index:02d}.py" for index in range(30))]
    test_status = {
        test_path: {
            "result": "FAILED" if test_path == "fast_failure.py" else "passed",
            "time_elapsed": 0.1 if test_path == "fast_failure.py" else float(index + 1),
            "wall_time": 0.1 if test_path == "fast_failure.py" else float(index + 1),
            "tests": 1,
            "failures": int(test_path == "fast_failure.py"),
            "errors": 0,
            "skipped": 0,
        }
        for index, test_path in enumerate(test_files)
    }

    summary = orchestrator._format_test_file_results(test_files, test_status, "cuda:0")

    assert "All Test File Results" in summary
    assert "Slowest 30 Test Files" not in summary
    assert "fast_failure.py" in summary
    assert all(test_path in summary for test_path in test_files)
