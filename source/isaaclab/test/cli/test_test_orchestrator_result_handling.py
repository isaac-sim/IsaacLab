# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-file pytest result handling in the test orchestrator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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
