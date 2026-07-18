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
        junit_log_url="",
        comparison_images_url="",
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
        junit_log_url="",
        comparison_images_url="",
        retries=0,
    )

    rows = _load_rows(output_dir)
    assert rows == [
        {
            "crash": True,
            "duration": 0.0,
            "group_id": "Docker + Tests / environments",
            "log_paths": [],
            "message": "Process killed by signal 15 after timeout",
            "passed": False,
            "retries": 0,
            "test_id": "test_rendering_cartpole::test_execution",
            "test_name": "test_execution",
            "test_type": "rendering-correctness",
            "timeout": True,
        }
    ]


def test_convert_junit_adds_log_paths_for_junit_and_comparison_artifacts(tmp_path: Path) -> None:
    """Converted rows should point at the uploaded JUnit and comparison image artifacts."""
    converter = _load_converter_module()
    reports_dir = tmp_path
    output_dir = tmp_path / "out"
    junit_file = reports_dir / "report.xml"
    junit_log_url = "https://github.com/isaac-sim/IsaacLab/actions/runs/123/artifacts/456"
    comparison_images_url = "https://github.com/isaac-sim/IsaacLab/actions/runs/123/artifacts/789"

    junit_file.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="rendering" tests="1" failures="0" errors="0" skipped="0" time="1">
  <testcase classname="test_rendering_cartpole" name="test_rgb" time="1"/>
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
        junit_log_url=junit_log_url,
        comparison_images_url=comparison_images_url,
        retries=0,
    )

    rows = _load_rows(output_dir)
    assert rows[0]["log_paths"] == [
        junit_log_url,
        comparison_images_url,
    ]
