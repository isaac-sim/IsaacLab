# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-file pytest result handling in the test orchestrator."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[4] / "tools"
"""Repo ``tools/`` directory, holding the orchestrator and the stack-dump plugin it signals."""

posix_only = pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"),
    reason="the orchestrator's process handling and the stack-dump signal are both POSIX-only",
)
"""Skip on platforms where ``capture_test_output_with_timeout`` cannot run.

It needs ``select`` on pipes, ``os.killpg``, and ``start_new_session``, and the dump needs ``SIGUSR1``.
"""


def _load_orchestrator_module() -> ModuleType:
    """Load ``tools/run_tests.py`` under a private name, leaving any real import untouched."""
    module_path = TOOLS_DIR / "run_tests.py"
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

    # Enough filler to overrun the quota, counted from the quota itself so that raising it does not
    # quietly turn this into a test of a log that fits inside it.
    filler_lines = orchestrator.ovrtx_log.LOG_LIMIT_BYTES // len("filler-line\n") + 1

    def _capture(cmd, timeout, env, *, startup_deadline, report_file):
        # Render verbosely, then die without writing a report.
        Path(log_path).write_text("head-line\n" + "filler-line\n" * filler_lines + "tail-line\n", encoding="utf-8")
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


@pytest.mark.parametrize(
    ("returncode", "kill_reason", "expected_result"),
    [(-11, "", "CRASHED"), (-9, "timeout", "TIMEOUT")],
)
def test_abnormal_termination_saves_the_renderer_log_of_the_blamed_test(
    monkeypatch, tmp_path: Path, returncode: int, kill_reason: str, expected_result: str
) -> None:
    """The test a crash or a hang killed is the one test whose renderer output nothing else saves.

    ``tools/ovrtx_log.py`` saves from a fixture, so every test that reaches teardown leaves its output in
    the directory CI uploads -- and the test that took the process down, the only one anybody is going to
    read, leaves nothing. Both reports quote a bounded tail of the log instead, which is not the same
    thing: a hang that logs nothing after the render is diagnosed by what came before it, and that is the
    part a cap counted back from the end of the file drops first.
    """
    orchestrator = _load_orchestrator_module()
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_alpha():\n    pass\n\n\ndef test_beta():\n    pass\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    alpha, beta = f"{test_file}::test_alpha", f"{test_file}::test_beta"

    def _capture(_cmd, _timeout, env, *, report_file: str, **_kwargs):
        Path(log_path).write_text("alpha-line\nbeta-line\n", encoding="utf-8")
        journal_file = env[orchestrator.JOURNAL_ENV_VAR]
        _append_journal(journal_file, [{"event": "collected", "node_ids": [alpha, beta]}])
        _append_journal(journal_file, _journaled_test(alpha, "passed"))
        # Dies inside test_beta, so the teardown that would have saved its output never runs.
        _append_journal(journal_file, [{"event": "start", "node_id": beta}])
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

    _report, status, _was_failure = orchestrator._run_one_pass(context, k_expr=None, suffix="")

    assert status["result"] == expected_result
    # Named after the test alone, as the fixture names the directories of the tests that saved their own,
    # so the run that died is read beside the tests that survived it rather than somewhere else.
    saved = tmp_path / orchestrator.OVRTX_LOG_DIR / "test_beta.0" / "ovrtx_renderer.log"
    assert saved.read_text(encoding="utf-8") == "alpha-line\nbeta-line\n"


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


def test_startup_retry_wall_time_includes_every_attempt(monkeypatch, tmp_path: Path) -> None:
    """The reported wall time must include startup attempts discarded by a successful retry."""
    orchestrator = _load_orchestrator_module()
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_present():\n    pass\n", encoding="utf-8")
    attempts = 0

    def _capture(*_args, report_file: str, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return -1, b"", b"", "startup_hang", 8.0, ""
        _write_partial_junit_report(report_file)
        return 0, b"", b"", "", 2.0, ""

    monkeypatch.setattr(orchestrator.ovrtx_log, "LOG_PATH", str(tmp_path / "ovrtx_renderer.log"))
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

    _report, status, was_failure = orchestrator._run_one_pass(context, k_expr=None, suffix="")

    assert attempts == 2
    assert status["wall_time"] == 10.0
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


def test_artifact_paths_handed_to_the_subprocess_are_uploadable(monkeypatch, tmp_path: Path) -> None:
    """Each pass must tell the test process where to save renderer logs and stack dumps.

    ``tools/ovrtx_log.py`` saves a renderer log only when its variable names a directory, and
    ``tools/hang_dump.py`` writes no dump unless its own names a file. The reports quote only a bounded
    amount of either, so a path outside the tree CI collects leaves nothing to read past that cap. Both
    are absolute for the journal's reason: the log is saved from a fixture and the dump file is opened
    at plugin load, so a test using ``monkeypatch.chdir`` would otherwise leave either under the
    temporary directory.
    """
    orchestrator = _load_orchestrator_module()
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_present():\n    pass\n", encoding="utf-8")
    log_dirs: list[str] = []
    dump_paths: list[str] = []

    def _capture(_cmd, _timeout, env, *, report_file: str, **_kwargs):
        log_dirs.append(env[orchestrator.ovrtx_log.LOG_DIR_ENV_VAR])
        dump_paths.append(env[orchestrator.hang_dump.DUMP_PATH_ENV_VAR])
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

    assert len(log_dirs) == 1
    log_dir = Path(log_dirs[0])
    assert log_dir.is_absolute()
    # Under the reports directory, since that is the tree CI collects as a job artifact.
    assert log_dir == tmp_path / orchestrator.OVRTX_LOG_DIR

    assert len(dump_paths) == 1
    dump_path = Path(dump_paths[0])
    assert dump_path.is_absolute()
    assert dump_path.parent == tmp_path / orchestrator.HANG_DUMP_DIR
    # hang_dump.register() opens this path as the child starts, so it cannot be created later.
    assert dump_path.parent.is_dir()


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
    assert status["wall_time"] == pytest.approx(0.3)
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


@posix_only
def test_hung_process_report_names_where_it_is_stuck(monkeypatch, tmp_path: Path) -> None:
    """A hang must report the stack it is stuck in, not just that it stopped.

    Without a dump the runner escalates straight to ``SIGKILL``, which cannot be caught, and the report
    carries only system tables -- nothing that points at the hung code. Every property of the dump is
    asserted against a single hung child: each one costs a real timeout to reproduce, and they are all
    facets of the same ``pre_kill_diag``.
    """
    orchestrator = _load_orchestrator_module()
    # A marker rather than "" so the ordering assertion below has something to find.
    monkeypatch.setattr(orchestrator, "_capture_system_diagnostics", lambda: "=== SYSTEM DIAGNOSTICS BODY ===")
    # raising=False so a build without the dump still reaches the assertions below, and fails on the
    # missing stack rather than on the missing constant.
    monkeypatch.setattr(orchestrator, "HANG_DUMP_GRACE", 1, raising=False)

    # The child has to be pytest, not a bare script. pytest captures at the file-descriptor level, so it has
    # already redirected fd 2 by the time a test runs; a dump written there is discarded when the process is
    # killed. A bare script has no such capture and would pass whether or not the dump reaches a file.
    test_file = tmp_path / "test_wedges.py"
    test_file.write_text(
        "import threading\n\n\ndef test_wedges():\n    wedged_call()\n\n\ndef wedged_call():\n"
        "    threading.Event().wait()\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    # `-p hang_dump` needs tools/ importable; the orchestrator gets this from the repo-root conftest.
    env["PYTHONPATH"] = str(TOOLS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["ISAACLAB_HANG_DUMP"] = str(tmp_path / "hangdump.log")
    cmd = [sys.executable, "-m", "pytest", "-p", "hang_dump", "-p", "no:cacheprovider", str(test_file)]

    # The timeout is wall clock the test has to sit through, so it is only as long as the child needs to
    # reach the wedge -- measured at ~2.6 s spawning eight of these at once, against ~1.4 s idle.
    _returncode, _stdout, _stderr, kill_reason, _wall_time, pre_kill_diag = (
        orchestrator.capture_test_output_with_timeout(cmd, timeout=8, env=env)
    )

    assert kill_reason == "timeout"
    assert "HANG STACK DUMP" in pre_kill_diag
    assert "wedged_call" in pre_kill_diag, "the dump must name the hung call, not just that a hang happened"
    assert pre_kill_diag.count("----- dump ") > 1, "repeated dumps are what tell a wedged process from a slow one"
    # The stack must sit ahead of the system tables, which ``_get_diagnostics`` truncates off the end.
    assert pre_kill_diag.index("HANG STACK DUMP") < pre_kill_diag.index("SYSTEM DIAGNOSTICS BODY")


def test_hang_dump_plugin_is_inert_without_signal_support(monkeypatch) -> None:
    """The plugin loads on every platform, so it must no-op where the signal does not exist."""
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    import hang_dump

    monkeypatch.setattr(hang_dump, "DUMP_SIGNAL", None)

    assert hang_dump.is_supported() is False
    assert hang_dump.register() is False
    hang_dump.pytest_configure(config=None)  # must not raise
