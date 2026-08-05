# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the omni-github JUnit result converter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

_MODULE_PATH = Path(__file__).with_name("junit_to_omni_github_results.py")
_RESULT_PATH = "_testoutput/test_results.json"


def _load_converter_module() -> ModuleType:
    """Load the converter module from the local GitHub action directory."""
    spec = importlib.util.spec_from_file_location("junit_to_omni_github_results", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_rows(output_dir: Path) -> list[dict[str, object]]:
    """Load converted rows from the omni-github result artifact."""
    result = json.loads((output_dir / _RESULT_PATH).read_text(encoding="utf-8"))
    return result["tests"]


def test_convert_junit_populates_github_metadata_and_failure_details(tmp_path: Path) -> None:
    """Converted JUnit rows should carry grouping, retry, and message metadata."""
    converter = _load_converter_module()
    junit_file = tmp_path / "report.xml"
    output_dir = tmp_path / "out"
    junit_file.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="3" failures="1" errors="0" skipped="1" time="2.0">
  <testcase classname="source.isaaclab.test.foo.test_sample" name="test_fails[param]" time="1.25">
    <failure message="AssertionError: expected 1, got 2">Traceback details</failure>
  </testcase>
  <testcase classname="source.isaaclab.test.foo.test_sample" name="test_skips" time="0.10">
    <skipped message="requires GPU"/>
  </testcase>
  <testcase classname="source.isaaclab.test.foo.test_sample" name="test_passes" time="0.65"/>
</testsuite>
""",
        encoding="utf-8",
    )

    converter.convert_junit(
        junit_file=junit_file,
        output_dir=output_dir,
        test_tool_id="pytest",
        test_type="pytest",
        app_platform="linux-x86_64",
        app_config="test-job",
        group_name="Docker + Tests / isaaclab_tasks [1/3]",
        retries=2,
    )

    rows = _load_rows(output_dir)
    failed, skipped, passed = rows
    assert failed["test_id"] == "source.isaaclab.test.foo.test_sample::test_fails[param]"
    assert failed["test_name"] == "test_fails[param]"
    assert failed["passed"] is False
    assert failed["duration"] == 1.25
    assert failed["group_id"] == "Docker + Tests / isaaclab_tasks [1/3]"
    assert failed["retries"] == 2
    assert failed["message"] == "AssertionError: expected 1, got 2"

    assert skipped["passed"] is False
    assert skipped["skipped"] is True
    assert skipped["skip_reason"] == "requires GPU"
    assert skipped["message"] == "requires GPU"

    assert passed["passed"] is True
    assert "message" not in passed


def test_convert_junit_marks_crashes_and_timeouts(tmp_path: Path) -> None:
    """Converted rows should surface crash and timeout messages."""
    converter = _load_converter_module()
    junit_file = tmp_path / "report.xml"
    output_dir = tmp_path / "out"
    junit_file.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="crash_suite" tests="1" failures="0" errors="1" skipped="0" time="0">
  <testcase classname="test_rendering_cartpole" name="test_execution" time="0">
    <error message="Process killed by signal 15 after timeout">diagnostics</error>
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )

    converter.convert_junit(
        junit_file=junit_file,
        output_dir=output_dir,
        test_tool_id="pytest",
        test_type="rendering-correctness",
        app_platform="linux-x86_64",
        app_config="test-job",
        group_name="Docker + Tests / environments",
        retries=0,
    )

    rows = _load_rows(output_dir)
    assert rows == [
        {
            "crash": True,
            "duration": 0.0,
            "group_id": "Docker + Tests / environments",
            "log_paths": ["_testoutput/logs/pytest-0001.log"],
            "message": "Process killed by signal 15 after timeout",
            "passed": False,
            "retries": 0,
            "test_id": "test_rendering_cartpole::test_execution",
            "test_name": "test_execution",
            "test_type": "rendering-correctness",
            "timeout": True,
        }
    ]


def test_convert_junit_writes_one_artifact_relative_log_per_test(tmp_path: Path) -> None:
    """Every row should point at its own log file that exists inside the artifact."""
    converter = _load_converter_module()
    junit_file = tmp_path / "report.xml"
    output_dir = tmp_path / "out"
    junit_file.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="rendering" tests="2" failures="1" errors="0" skipped="0" time="2">
  <testcase classname="test_rendering_cartpole" name="test_rgb" time="1"/>
  <testcase classname="test_rendering_cartpole" name="test_depth" time="1">
    <failure message="AssertionError: image mismatch">full traceback body</failure>
    <system-out>captured stdout line</system-out>
    <system-err>captured stderr line</system-err>
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )

    converter.convert_junit(
        junit_file=junit_file,
        output_dir=output_dir,
        test_tool_id="pytest",
        test_type="rendering-correctness",
        app_platform="linux-x86_64",
        app_config="test-job",
        group_name="Docker + Tests / rendering",
        retries=0,
    )

    passed, failed = _load_rows(output_dir)
    assert passed["log_paths"] == ["_testoutput/logs/pytest-0001.log"]
    assert failed["log_paths"] == ["_testoutput/logs/pytest-0002.log"]

    for row in (passed, failed):
        for log_path in row["log_paths"]:
            # omni-github resolves log_paths against the artifact root, so the path must stay
            # relative (no scheme, no leading slash, no backslash) and the file must exist.
            assert "://" not in log_path and not Path(log_path).is_absolute()
            assert "\\" not in log_path
            assert (output_dir / log_path).is_file()

    passed_log = (output_dir / passed["log_paths"][0]).read_text(encoding="utf-8")
    assert "test_id: test_rendering_cartpole::test_rgb" in passed_log
    assert "status: passed" in passed_log

    failed_log = (output_dir / failed["log_paths"][0]).read_text(encoding="utf-8")
    assert "status: failed" in failed_log
    assert "--- failure: AssertionError: image mismatch ---" in failed_log
    assert "full traceback body" in failed_log
    assert "captured stdout line" in failed_log
    assert "captured stderr line" in failed_log


def test_convert_junit_writes_skip_reason_into_the_per_test_log(tmp_path: Path) -> None:
    """Skipped rows should record the skip status and reason in their log file."""
    converter = _load_converter_module()
    junit_file = tmp_path / "report.xml"
    output_dir = tmp_path / "out"
    junit_file.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="1" failures="0" errors="0" skipped="1" time="0">
  <testcase classname="test_camera" name="test_rgb" time="0">
    <skipped message="requires GPU"/>
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )

    converter.convert_junit(
        junit_file=junit_file,
        output_dir=output_dir,
        test_tool_id="pytest",
        test_type="pytest",
        app_platform="linux-x86_64",
        app_config="test-job",
        group_name="Docker + Tests / isaaclab",
        retries=0,
    )

    (row,) = _load_rows(output_dir)
    log_text = (output_dir / row["log_paths"][0]).read_text(encoding="utf-8")
    assert "status: skipped" in log_text
    assert "--- skipped: requires GPU ---" in log_text


def test_convert_junit_appends_markers_to_test_type_with_separator(tmp_path: Path) -> None:
    """Recorded intent markers should append to the base test_type with comma separators."""
    converter = _load_converter_module()
    junit_file = tmp_path / "report.xml"
    output_dir = tmp_path / "out"
    junit_file.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="2" failures="0" errors="0" skipped="0" time="2">
  <testcase classname="test_camera" name="test_rgb" time="1">
    <properties>
      <property name="markers" value="integration,rendering"/>
    </properties>
  </testcase>
  <testcase classname="test_math" name="test_quat" time="1"/>
</testsuite>
""",
        encoding="utf-8",
    )

    converter.convert_junit(
        junit_file=junit_file,
        output_dir=output_dir,
        test_tool_id="pytest",
        test_type="pytest",
        app_platform="linux-x86_64",
        app_config="test-job",
        group_name="Docker + Tests / isaaclab",
        retries=0,
    )

    marked, unmarked = _load_rows(output_dir)
    # The base type must be separated from the first marker, not fused into "pytestintegration".
    assert marked["test_type"] == "pytest,integration,rendering"
    # Testcases without markers keep the bare base type.
    assert unmarked["test_type"] == "pytest"
